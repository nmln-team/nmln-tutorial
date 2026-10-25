"""The model: a neural Markov logic network over the possible worlds of an ontology,

    p(y | x) ∝ exp( Σ_k φ_k(y, x) ),    φ_k = beta_k · statistic_k,

where y is a world, x the inputs attached to constants, and φ_k the potentials. The model only scores
worlds; answering questions about it is the job of an inference engine (`nmln.inference`), and fitting
its parameters the job of a learner (`nmln.learning`).
"""

from __future__ import annotations

import typing as t

import torch

from .potentials import Potential


class NMLN(torch.nn.Module):
    def __init__(self, ontology, potentials: t.Union[Potential, t.Iterable[Potential]]):
        super().__init__()
        potentials = [potentials] if isinstance(potentials, Potential) else list(potentials)
        for p in potentials:
            if not isinstance(p, Potential):
                raise TypeError(f"{type(p).__name__} is not a Potential (subclass nmln.potentials.Potential)")
        self.ontology = ontology
        self.potentials = torch.nn.ModuleList(potentials)

    def score(self, y: torch.Tensor, x=None) -> torch.Tensor:
        """Unnormalized log-probability Σ_k φ_k(y, x) of worlds y [..., D]: shape [...]."""
        total = y.new_zeros(y.shape[:-1])
        for potential in self.potentials:
            total = total + potential(y, x)
        return total

    def forward(self, y: torch.Tensor, x=None) -> torch.Tensor:
        return self.score(y, x)

    @property
    def device(self) -> torch.device:
        for tensor in list(self.parameters()) + list(self.buffers()):
            return tensor.device
        return torch.device("cpu")
