
from __future__ import annotations

import torch


class ExchangeSampler:
    """Odd/even nearest-neighbor replica exchange over Chain objects (parallel tempering).

    Each step runs one sweep on every replica, then proposes to swap the states of neighboring replicas.
    `acceptance_rates()` reports, per neighboring pair, the fraction of proposed swaps that were accepted.
    """

    def __init__(self, chains, swap_rule):
        self.chains = list(chains)
        self.swap_rule = swap_rule
        self.parity = 0
        self.reset_statistics()

    def reset_statistics(self):
        self.accepted = torch.zeros(max(len(self.chains) - 1, 0))
        self.proposed = torch.zeros(max(len(self.chains) - 1, 0))

    def acceptance_rates(self) -> torch.Tensor:
        """Accepted / proposed swaps between replicas (r, r + 1), shape [R - 1]."""
        return self.accepted / self.proposed.clamp(min=1)

    def _active_pairs(self, device):
        n = len(self.chains)
        if n < 2:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        if n == 2:
            i = torch.tensor([0], dtype=torch.long, device=device)
            return i, i + 1
        i = torch.arange(self.parity, n - 1, 2, dtype=torch.long, device=device)
        return i, i + 1

    @torch.no_grad()
    def step(self, batch_size: int, x=None, evidence=None, device=None):
        for chain in self.chains:
            chain.step(batch_size=batch_size, x=x, evidence=evidence, device=device)

        if len(self.chains) < 2:
            return [chain.states for chain in self.chains]

        states = torch.stack([chain.states for chain in self.chains], dim=0)  # [R, B, 1, D]
        i, j = self._active_pairs(states.device)
        if len(i) > 0:
            accept = self.swap_rule(self.chains, i, j, x=x)
            for k, pair in enumerate(i.tolist()):
                self.accepted[pair] += float(accept[k].sum())
                self.proposed[pair] += accept[k].numel()
            si = states[i].clone()
            sj = states[j].clone()
            mask = accept
            while mask.ndim < si.ndim:
                mask = mask.unsqueeze(-1)
            states[i] = torch.where(mask, sj, si)
            states[j] = torch.where(mask, si, sj)
            for chain, new in zip(self.chains, states.unbind(0)):
                chain.states = new
        self.parity = 1 - self.parity
        return [chain.states for chain in self.chains]
