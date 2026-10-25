"""Learning: fit the parameters of an NMLN to data.

For one observed world, the gradient of the log-likelihood is

    ∂/∂θ log p(y | x) = ∂/∂θ Σ_k φ_k(y, x) - E_p[ ∂/∂θ Σ_k φ_k(Y, x) ],

data statistics minus model expectations. The learners differ in how they get the model expectations:

- `Contrastive` estimates them from the persistent chains of a sampling engine (`Gibbs`, `ParallelTempering`);
- `MaximumLikelihood` differentiates log Z, computed by an exact engine (`Exact`, `BeliefPropagation` on
  trees); it also learns from partially observed worlds, with the loss -log p(observed atoms | x).

`loss(batch)` is the training objective, for a hand-written loop. Learners are PyTorch Lightning modules:
`fit(data, epochs)` runs a quiet Lightning Trainer (and accepts any Trainer argument, e.g. a logger or
callbacks); a Trainer of your own works too. A batch is a tensor of worlds [B, D], or a dict with "y" [B, D],
optionally "observed" (bool mask, False on latent atoms) and "x" ({domain: inputs [B, num_constants, ...]}).
Batches are checked against the ontology's variables: by `loss` unless validate=False, and during training
in the first epoch only (the data are the same in later epochs).
"""

from __future__ import annotations

import logging
import warnings

import pytorch_lightning as pl
import torch

from .evidence import Evidence
from .inference.sampling import NoiseKernel


def unpack_batch(batch):
    """(inputs x, Evidence) of a batch."""
    if isinstance(batch, dict):
        y = batch["y"].float()
        observed = batch.get("observed")
        return batch.get("x"), Evidence(y, torch.ones_like(y, dtype=torch.bool) if observed is None else observed.bool())
    if isinstance(batch, (list, tuple)) and len(batch) == 1:
        batch = batch[0]
    return None, Evidence.complete(batch.float())


def _quiet_lightning():
    for name in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for message in (".*does not have many workers.*", ".*GPU available but not used.*", ".*LeafSpec.*",
                    ".*treespec.*", ".*litlogger.*"):
        warnings.filterwarnings("ignore", message=message)


class Learner(pl.LightningModule):
    def __init__(self, engine, lr: float):
        super().__init__()
        self.engine = engine
        self.lr = float(lr)
        self.history = []  # mean training loss of every epoch
        self._epoch_losses = []

    @property
    def model(self):
        return self.engine.model

    def _unpack(self, batch, validate):
        x, evidence = unpack_batch(batch)
        if validate:
            self.model.ontology.variables.validate(evidence, what="training example")
        return x, evidence

    def loss(self, batch, validate: bool = True) -> torch.Tensor:
        """The training loss of a batch; validate=True first checks the batch against the ontology's variables."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch, validate=self.current_epoch == 0)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._epoch_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        if self._epoch_losses:
            self.history.append(float(torch.stack(self._epoch_losses).mean()))
        self._epoch_losses = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.engine.model.parameters(), lr=self.lr)

    def fit(self, data, epochs: int = 1, accelerator: str = "auto", progress_bar: bool = False, **trainer_args):
        """Train with a PyTorch Lightning Trainer, quiet by default: no logger, checkpoints or model summary.

        Extra keyword arguments go to the Trainer (e.g. logger=..., callbacks=[...], devices=...).
        Returns the mean loss of every epoch of this call.
        """
        _quiet_lightning()
        trainer_args = dict(
            max_epochs=int(epochs), accelerator=accelerator, devices=1, logger=False, enable_checkpointing=False,
            enable_model_summary=False, enable_progress_bar=progress_bar,
        ) | trainer_args
        start = len(self.history)
        pl.Trainer(**trainer_args).fit(self, data)
        return self.history[start:]


class Contrastive(Learner):
    """Persistent contrastive divergence. The loss

        mean_s score(y_s, x) - mean_d score(y_d, x),    y_s from the persistent chains, y_d from the data,

    has gradient "model expectations minus data statistics". The data worlds can be corrupted with training
    noise (each variable resampled with probability noise_scale), which keeps the learned distribution from
    becoming too deterministic.
    """

    def __init__(self, engine, lr: float = 1e-3, noise_scale: float = 0.1, num_sweeps: int = 1):
        if not hasattr(engine, "model_samples"):
            raise TypeError("Contrastive learning needs a sampling engine (Gibbs or ParallelTempering)")
        super().__init__(engine, lr)
        self.noise = NoiseKernel(engine.ontology.variables.groups, noise_scale=noise_scale)
        self.num_sweeps = int(num_sweeps)

    def loss(self, batch, validate: bool = True):
        x, (y, observed) = self._unpack(batch, validate)
        if not bool(observed.all()):
            raise ValueError("Contrastive learning needs complete worlds; use MaximumLikelihood for partial data")
        with torch.no_grad():
            samples = self.engine.model_samples(y.shape[0], x, num_sweeps=self.num_sweeps)
            noisy = self.noise.step(y)
        return self.model.score(samples, x).mean() - self.model.score(noisy, x).mean()


class MaximumLikelihood(Learner):
    """Maximum (conditional) likelihood with an exact engine: loss = -log p(observed atoms | x)
    = log Z(x) - log Z(x, observed atoms). With complete worlds this is -log p(y | x)."""

    def __init__(self, engine, lr: float = 1e-3):
        if not engine.computes_log_z:
            raise TypeError("MaximumLikelihood needs an engine that computes log Z (Exact or BeliefPropagation)")
        super().__init__(engine, lr)

    def loss(self, batch, validate: bool = True):
        x, evidence = self._unpack(batch, validate)
        return -self.engine._log_probability(evidence, x, validate=False).mean()
