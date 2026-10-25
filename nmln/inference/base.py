"""The interface shared by all inference engines: four uses of the same distribution."""

from __future__ import annotations

import torch

from nmln.evidence import Evidence, as_evidence


def batch_size_of(x=None, evidence=None):
    """B when the inputs x or the evidence ([B, D]) carry a batch; None for a single, unbatched query."""
    if x is not None:
        return next(iter(x.values())).shape[0]
    if evidence is not None and evidence[0].ndim == 2:
        return evidence[0].shape[0]
    return None


class Inference(torch.nn.Module):
    """An inference engine for an NMLN model. Every engine answers the same questions, with the same calls:

        engine.marginals(x=None, evidence=None)       probability that each ground atom is true: [D] or [B, D]
        engine.map(x=None, evidence=None)             the most probable world: [D] or [B, D]
        engine.sample(n, x=None, evidence=None)       n worlds drawn from the model: [n, D] or [B, n, D]
        engine.log_partition(x=None, evidence=None)   log Z(x), or the log-mass of the worlds agreeing with the evidence
        engine.log_probability(evidence, x=None)      log p(evidence | x)

    x are the inputs attached to constants ({domain: [B, num_constants, ...]}); evidence fixes observed atoms
    (`Evidence`, from `Ontology.evidence` or `Ontology.observe`). Results carry a leading batch dimension B
    exactly when x or the evidence does. Engine settings (number of sweeps, chains, iterations) are given
    when the engine is built, so the calls are the same for every engine.

    Queries return plain tensors (no gradients); log_partition and log_probability stay differentiable, for
    learning. `exact` says whether the answers are exact. Exact engines (`Exact`, `BeliefPropagation` on trees) compute
    them; sampling engines (`Gibbs`, `ParallelTempering`) estimate marginals and MAP and cannot compute log Z.
    """

    exact = False  # are the answers exact?
    computes_log_z = False  # does the engine provide log_partition and log_probability?

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def ontology(self):
        return self.model.ontology

    def _evidence(self, evidence, validate: bool = True):
        """Evidence as an `Evidence` of float values and a bool mask, checked against the ontology's variables."""
        evidence = as_evidence(evidence)
        if evidence is not None and validate:
            self.ontology.variables.validate(evidence)
        return evidence

    def marginals(self, *, x=None, evidence=None):
        raise NotImplementedError

    def map(self, *, x=None, evidence=None):
        raise NotImplementedError

    def sample(self, n: int = 1, *, x=None, evidence=None):
        raise NotImplementedError

    def log_partition(self, *, x=None, evidence=None):
        raise NotImplementedError(f"{type(self).__name__} does not compute log Z; use Exact or BeliefPropagation")

    def log_probability(self, evidence, *, x=None):
        """log p(evidence | x): [] or [B]."""
        return self._log_probability(evidence, x, validate=True)

    def _log_probability(self, evidence, x, validate: bool):
        """log_probability without the check of the evidence (learners check their data themselves)."""
        raise NotImplementedError(f"{type(self).__name__} does not compute probabilities; use Exact or BeliefPropagation")


__all__ = ["Inference", "Evidence", "batch_size_of"]
