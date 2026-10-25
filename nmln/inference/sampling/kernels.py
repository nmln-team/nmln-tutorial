
"""Markov kernels over the variables of a world, in two families:

- transition kernels sample from the model: `step(states, log_prob, x, ...)` leaves the distribution defined by
  the potential invariant (blocked Gibbs, and compositions of such kernels);
- noise kernels corrupt a world independently of the model: `step(y)` draws y' from a fixed q(y' | y).
  They provide training noise and the corruption levels of parallel noising, and, being symmetric,
  can serve as proposals for Metropolis-Hastings. On their own they do not sample from the model.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch

from nmln import utils


class TransitionKernel(ABC):
    """A kernel that leaves the model distribution invariant; `log_prob` is the model's unnormalized log-probability."""

    @torch.no_grad()
    @abstractmethod
    def step(self, states, log_prob, x=None, evidence=None):
        raise NotImplementedError


class CompositeKernel(TransitionKernel):
    def __init__(self, schedule):
        self.schedule = list(schedule)

    @torch.no_grad()
    def step(self, states, log_prob, x=None, evidence=None):
        for kernel, n_steps in self.schedule:
            for _ in range(int(n_steps)):
                states = kernel.step(states, log_prob, x=x, evidence=evidence)
        return states


class BlockedGibbsKernel(TransitionKernel):
    """Gibbs sampling over the variables of the ontology: resample one variable at a time, from its
    conditional distribution given all the others (`groups` = `ontology.variables.groups`)."""

    def __init__(self, groups, shuffle: bool = True):
        self.groups = list(groups)
        self.shuffle = shuffle
        self._masks = {}

    def _masks_on(self, group, like):
        """The group's value masks [M, A, D] on the chains' device, copied there once."""
        key = (id(group), like.device, like.dtype)
        if key not in self._masks:
            self._masks[key] = group.masks.to(device=like.device, dtype=like.dtype)
        return self._masks[key]

    @staticmethod
    def _inner_candidates(current_state: torch.Tensor, block_masks: torch.Tensor):
        device = current_state.device
        block_masks = torch.as_tensor(block_masks, dtype=current_state.dtype, device=device)
        if block_masks.ndim == 1:
            block_masks = block_masks.unsqueeze(0)

        num_configs, num_variables = block_masks.shape
        batch_shape = current_state.shape[:-1]
        used_in_any_config = block_masks.sum(dim=0) > 0
        base_state = torch.where(used_in_any_config, torch.zeros_like(current_state), current_state)
        base_state = base_state.unsqueeze(-2).expand(*batch_shape, num_configs, num_variables)
        block_masks = block_masks.view(*((1,) * len(batch_shape)), num_configs, num_variables)
        return torch.where(block_masks > 0, torch.ones_like(base_state), base_state)

    @torch.no_grad()
    def step(self, states, log_prob, x=None, evidence=None):
        """One sweep over all variables, for chain states [batch, 1, D] and inputs x. `evidence` = (y, observed), both [batch, 1, D]:
        candidates that contradict an observed atom get a hard penalty, so the chain never leaves the evidence."""
        groups = self.groups if not self.shuffle else random.sample(self.groups, len(self.groups))
        for group in groups:
            for block_masks in self._masks_on(group, states):
                candidate_states = self._inner_candidates(states, block_masks)
                block_scores = log_prob(candidate_states, x)
                if block_scores.ndim > 0 and block_scores.shape[-1] == 1:
                    block_scores = block_scores.squeeze(-1)
                if evidence is not None:
                    values, observed = evidence
                    mismatch = ((candidate_states - values.unsqueeze(-2)).abs() * observed.unsqueeze(-2)).sum(-1)
                    block_scores = block_scores - utils.HARD_WEIGHT * mismatch
                flat = block_scores.reshape(-1, block_scores.shape[-1])
                sampled_idx = torch.distributions.Categorical(logits=flat).sample().reshape(*block_scores.shape[:-1], 1)
                states = utils.batched_gather(candidate_states, sampled_idx, batch_dims=2).squeeze(-2)
        return states



class NoiseKernel:
    """A noise kernel: each variable of a world is replaced, with probability noise_scale, by a value drawn
    uniformly among its values (possibly the same one). The proposal is symmetric, q(y' | y) = q(y | y').

    It does not look at the model, so it is not a TransitionKernel: used alone, it does not sample from it.
    """

    def __init__(self, groups, noise_scale: float = 0.1):
        self.noise_scale = float(noise_scale)
        self.groups = list(groups)
        self._masks = {}

    @torch.no_grad()
    def step(self, y):

        data = y.detach().clone()

        if self.noise_scale <= 0:
            return data

        for group in self.groups:
            key = (id(group), data.device, data.dtype)
            if key not in self._masks:
                self._masks[key] = (group.masks.to(data), group.unions.to(data))
            masks, unions = self._masks[key]  # [M, A, D], [M, D]

            B = data.shape[0]
            M, A, D = masks.shape
            apply = (torch.rand(M, B, device=data.device) < self.noise_scale)                # [M, B]
            choices = torch.randint(0, A, (M, B), device=data.device)               # [M, B]

            selected = masks[
                torch.arange(M, device=data.device).unsqueeze(1),
                choices,
            ]                                                                       # [M, B, D]
            selected = selected * apply.unsqueeze(-1).to(selected.dtype)

            zero_any = torch.einsum('mb,md->bd', apply.to(data.dtype), unions).clamp_(0, 1)   # [B, D]
            additions = selected.sum(dim=0).clamp_(0, 1)                                        # [B, D]
            data =  data * (1.0 - zero_any) + additions
        return data