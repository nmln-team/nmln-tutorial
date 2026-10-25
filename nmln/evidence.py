"""Evidence: the atoms of a world that are observed, and their values."""

from __future__ import annotations

import typing as t

import torch


class Evidence(t.NamedTuple):
    """Observed atoms of one world ([D]) or of a batch of worlds ([B, D]).

    y:        the values of the atoms, a (partial) world (anything where an atom is not observed)
    observed: bool mask, True on the observed atoms

    Build it with `Ontology.evidence({"smokes(ann)": 1})`, `Ontology.observe(y, ["res"])`, or
    `Evidence.complete(y)` for fully observed worlds. Being a NamedTuple, it unpacks as
    `y, observed = evidence` and batches with the default DataLoader collation.
    """

    y: torch.Tensor
    observed: torch.Tensor

    @classmethod
    def complete(cls, y: torch.Tensor) -> "Evidence":
        """Every atom of y is observed."""
        return cls(y, torch.ones_like(y, dtype=torch.bool))

    @property
    def batch_size(self) -> t.Optional[int]:
        """B for batched evidence [B, D], None for the evidence of a single world [D]."""
        return self.y.shape[0] if self.y.ndim == 2 else None

    def to(self, device) -> "Evidence":
        return Evidence(self.y.to(device), self.observed.to(device))


def as_evidence(evidence) -> t.Optional[Evidence]:
    """None, an Evidence, or a (y, observed) pair -> Evidence (float values, bool mask) or None."""
    if evidence is None:
        return None
    y, observed = evidence
    y = torch.as_tensor(y).float()
    return Evidence(y, torch.as_tensor(observed).bool().expand(y.shape))
