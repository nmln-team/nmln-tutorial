"""Persistent Markov chains: the states of a batch of chains, advanced by a kernel."""

from __future__ import annotations

import abc

import torch


def _normalize_log_prob_shape(lp, y: torch.Tensor):
    """Make log-probabilities match y.shape[:-1], restoring or broadcasting trailing singleton dimensions."""
    if not torch.is_tensor(lp):
        lp = torch.as_tensor(lp, device=y.device, dtype=y.dtype)
    target = tuple(y.shape[:-1])
    while lp.ndim > len(target) and lp.shape[-1] == 1:
        lp = lp.squeeze(-1)
    while lp.ndim < len(target):
        lp = lp.unsqueeze(-1)
    if tuple(lp.shape) != target:
        try:
            lp = torch.broadcast_to(lp, target)
        except RuntimeError as exc:
            raise RuntimeError(
                f"log_prob returned shape {tuple(lp.shape)} for worlds with leading shape {target}"
            ) from exc
    return lp


class StateInitializer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch_size: int, state_dim: int, device, dtype) -> torch.Tensor:
        raise NotImplementedError


class ZeroInitializer(StateInitializer):
    """Start every chain from the all-false world."""

    def __call__(self, batch_size, state_dim, device, dtype):
        return torch.zeros(batch_size, 1, state_dim, device=device, dtype=dtype)


class BernoulliInitializer(StateInitializer):
    """Start from random atoms (not necessarily a possible world: the first sweep fixes every variable)."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, batch_size, state_dim, device, dtype):
        return (torch.rand(batch_size, 1, state_dim, device=device) < self.p).to(dtype)


class TensorInitializer(StateInitializer):
    """Start from given worlds, e.g. data: [D] or [batch, D]."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.detach().clone()

    def __call__(self, batch_size, state_dim, device, dtype):
        x = self.tensor.to(device=device, dtype=dtype)
        x = x.view(1, 1, -1) if x.ndim == 1 else x.unsqueeze(1)
        if x.shape[-1] != state_dim:
            raise ValueError(f"initializer state dim mismatch: {x.shape[-1]} != {state_dim}")
        if x.shape[0] == 1 and batch_size != 1:
            x = x.expand(batch_size, 1, state_dim)
        if x.shape[0] != batch_size:
            raise ValueError(f"initializer has {x.shape[0]} worlds for {batch_size} chains")
        return x.clone()


class Chain:
    """A batch of persistent Markov chains, one per row: states [batch, 1, D] (worlds).

    `log_prob(y, x)` is the unnormalized log-probability the kernel samples from (e.g. the model's score,
    possibly at a temperature), for worlds y and inputs x. The chain holds no parameters: the engine that
    owns it does.
    """

    def __init__(self, state_dim: int, log_prob, kernel, initializer: StateInitializer | None = None, name: str = "chain"):
        self.state_dim = int(state_dim)
        self._log_prob = log_prob
        self.kernel = kernel
        self.initializer = initializer or ZeroInitializer()
        self.name = name
        self.states = None

    def log_prob(self, y, x=None):
        return _normalize_log_prob_shape(self._log_prob(y, x), y)

    def copy(self, name: str | None = None) -> "Chain":
        """A new chain with the same target and kernel, starting from these states."""
        chain = Chain(self.state_dim, self._log_prob, self.kernel, self.initializer, name or self.name)
        chain.states = None if self.states is None else self.states.clone()
        return chain

    def ensure_state(self, batch_size: int, device, dtype=torch.float32):
        if self.states is None or self.states.device != torch.device(device) or self.states.dtype != dtype:
            self.states = self.initializer(batch_size, self.state_dim, device, dtype)
        elif self.states.shape[0] != batch_size:
            self.resize(batch_size)
        return self.states

    @torch.no_grad()
    def resize(self, batch_size: int):
        """Change the number of chains, keeping the current states (no restart from scratch)."""
        if self.states.shape[0] > batch_size:
            self.states = self.states[:batch_size].clone()
        elif self.states.shape[0] < batch_size:
            extra = torch.randint(0, self.states.shape[0], (batch_size - self.states.shape[0],), device=self.states.device)
            self.states = torch.cat([self.states, self.states[extra]], dim=0)
        return self.states

    @torch.no_grad()
    def step(self, batch_size: int | None = None, x=None, evidence=None, device=None):
        """One sweep of the kernel. x are the inputs; evidence is (y, observed mask), [D] or [batch, D]."""
        if batch_size is None:
            if self.states is None:
                raise ValueError("batch_size is required the first time a chain is stepped")
        else:
            device = device if device is not None else (self.states.device if self.states is not None else "cpu")
            self.ensure_state(batch_size, device)
        if evidence is not None:
            values, observed = evidence
            values = values.to(self.states).expand(self.states.shape[0], self.state_dim).unsqueeze(1)
            observed = observed.to(self.states.device).expand(self.states.shape[0], self.state_dim).unsqueeze(1)
            evidence = (values, observed)
            self.states = torch.where(observed, values, self.states)
        self.states = self.kernel.step(self.states, self.log_prob, x=x, evidence=evidence)
        return self.states
