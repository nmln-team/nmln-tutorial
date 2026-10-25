"""Exact inference by enumerating every possible world (tiny languages only).

The possible worlds are all combinations of the values of the ontology's variables (see
`nmln.variables`); with the model's score of each one, log Z, marginals, MAP and exact samples follow
directly. Useful as ground truth for Gibbs sampling and belief propagation.
"""

from __future__ import annotations

import math

import torch

from .base import Inference, batch_size_of


class Exact(Inference):
    exact = True
    computes_log_z = True

    def __init__(self, model, max_worlds: int = 2 ** 20, chunk_size: int = 2 ** 15):
        super().__init__(model)
        self.chunk_size = int(chunk_size)
        variables = model.ontology.variables
        cardinalities = variables.cardinalities.tolist()
        num_worlds = math.prod(cardinalities)
        if num_worlds > max_worlds:
            raise ValueError(f"{num_worlds} possible worlds; Exact enumerates at most {max_worlds}")
        grid = torch.cartesian_prod(*[torch.arange(c) for c in cardinalities]).reshape(num_worlds, -1)
        self.register_buffer("worlds", variables.worlds(grid))  # [W, D]

    def scores(self, x=None):
        """Unnormalized log-probability of every possible world: [W], or [B, W] with inputs."""
        out = []
        for start in range(0, len(self.worlds), self.chunk_size):
            worlds = self.worlds[start:start + self.chunk_size]
            if x is not None:
                worlds = worlds.unsqueeze(0).expand(batch_size_of(x), -1, -1)
            out.append(self.model.score(worlds, x))
        return torch.cat(out, dim=-1)

    def _conditioned(self, x, evidence, validate: bool = True):
        """Scores of every world, -inf where the world disagrees with the evidence: [W] or [B, W]."""
        scores = self.scores(x)
        evidence = self._evidence(evidence, validate)
        if evidence is None:
            return scores
        values, observed = evidence.to(self.worlds.device)
        consistent = ((self.worlds == values.unsqueeze(-2)) | ~observed.unsqueeze(-2)).all(dim=-1)  # [..., W]
        return torch.where(consistent, scores, torch.full_like(scores, -math.inf))

    def log_partition(self, *, x=None, evidence=None):
        """log Z(x), or with evidence the log-mass of the worlds that agree with it: [] or [B]."""
        return torch.logsumexp(self._conditioned(x, evidence), dim=-1)

    def _log_probability(self, evidence, x, validate: bool):
        conditioned = torch.logsumexp(self._conditioned(x, evidence, validate), dim=-1)
        return conditioned - torch.logsumexp(self._conditioned(x, None), dim=-1)

    @torch.no_grad()
    def probabilities(self, *, x=None, evidence=None):
        """Probability of every possible world (in the order of `self.worlds`): [W] or [B, W]."""
        return torch.softmax(self._conditioned(x, evidence), dim=-1)

    @torch.no_grad()
    def marginals(self, *, x=None, evidence=None):
        """Probability that each ground atom is true: [D], or [B, D]."""
        return self.probabilities(x=x, evidence=evidence) @ self.worlds

    @torch.no_grad()
    def map(self, *, x=None, evidence=None):
        """The most probable world: [D], or [B, D]."""
        return self.worlds[self._conditioned(x, evidence).argmax(dim=-1)]

    @torch.no_grad()
    def sample(self, n: int = 1, *, x=None, evidence=None):
        """Independent exact samples: [n, D], or [B, n, D]."""
        logits = self._conditioned(x, evidence)
        index = torch.distributions.Categorical(logits=logits).sample((int(n),))
        samples = self.worlds[index]
        return samples.movedim(0, -2) if samples.ndim == 3 else samples
