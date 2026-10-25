"""Sampling engines: blocked Gibbs sampling and parallel tempering, for any potential.

Both keep two sets of persistent chains: training chains, which follow the model as its parameters
change (persistent contrastive divergence, `model_samples`), and sampling chains, which continue from
call to call of `sample`. `marginals` and `map` run fresh chains and estimate from what they visit.
"""

from __future__ import annotations

import torch

from .base import Inference, batch_size_of
from .sampling import BlockedGibbsKernel, Chain, ExchangeSampler, ZeroInitializer, metropolis_swap_rule


def _repeat(x, times):
    return None if x is None else {k: v.repeat_interleave(times, dim=0) for k, v in x.items()}


class Gibbs(Inference):
    """Blocked Gibbs sampling: resample one variable at a time, from its distribution given all the others.

    Args:
        model: the NMLN
        sweeps_per_sample: sweeps between the worlds returned by consecutive `sample` calls
        num_chains, burn_in, num_sweeps: how `marginals` and `map` estimate: num_chains fresh chains per query,
            averaged over num_sweeps sweeps after burn_in sweeps
        shuffle: visit the variable groups in a random order at every sweep
    """

    def __init__(self, model, sweeps_per_sample: int = 1, num_chains: int = 100, burn_in: int = 20,
                 num_sweeps: int = 50, shuffle: bool = True):
        super().__init__(model)
        self.kernel = BlockedGibbsKernel(model.ontology.variables.groups, shuffle=shuffle)
        self.sweeps_per_sample = int(sweeps_per_sample)
        self.num_chains, self.burn_in, self.num_sweeps = int(num_chains), int(burn_in), int(num_sweeps)
        self._training = None
        self._sampling = None

    # ------------------------------------------------------------------ chains

    def _new_runner(self, name: str):
        return Chain(self.ontology.num_atoms, self.model.score, self.kernel, ZeroInitializer(), name)

    def _copy_runner(self, runner):
        return runner.copy()

    def _step(self, runner, batch_size, x=None, evidence=None):
        """One step of a runner; returns the states of the target chain [batch_size, 1, D]."""
        return runner.step(batch_size, x=x, evidence=evidence, device=self.model.device)

    def _scores(self, runner, x):
        """Model scores of the target chain's states, [chains]."""
        return self.model.score(runner.states.squeeze(1), x)

    def _queries(self, n, x, evidence):
        """Chains for n samples of every query: (batch, total chains, repeated x, repeated evidence)."""
        batch = batch_size_of(x, evidence)
        if evidence is not None and evidence.batch_size is not None:
            evidence = tuple(e.repeat_interleave(n, dim=0) for e in evidence)
        return batch, (batch or 1) * n, _repeat(x, n), evidence

    # ------------------------------------------------------------------ learning

    @torch.no_grad()
    def model_samples(self, batch_size: int, x=None, num_sweeps: int = 1) -> torch.Tensor:
        """Worlds from the persistent training chains (one per data example) after more sweeps: [batch_size, D]."""
        if self._training is None:
            self._training = self._new_runner("training")
        for _ in range(max(1, int(num_sweeps))):
            states = self._step(self._training, batch_size, x)
        return states.reshape(batch_size, -1)

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def sample(self, n: int = 1, *, x=None, evidence=None):
        """n worlds from the sampling chains, after `sweeps_per_sample` more sweeps: [n, D], or [B, n, D].

        The sampling chains start from the training chains and persist across calls, so repeated calls continue
        the same trajectories (consecutive samples are correlated). Observed atoms are clamped.
        """
        return self.trace(n, self.sweeps_per_sample, x=x, evidence=evidence)[0]

    @torch.no_grad()
    def trace(self, n: int, num_sweeps: int, *, x=None, evidence=None):
        """Run num_sweeps sweeps of the sampling chains: (worlds [n, D] or [B, n, D], model score after every
        sweep [num_sweeps, ..., n]). A diagnostic of mixing; parallel tempering adds a replica dimension."""
        if num_sweeps < 1:
            raise ValueError("num_sweeps must be at least 1")
        batch, chains, xs, evidence = self._queries(int(n), x, self._evidence(evidence))
        if self._sampling is None:
            self._sampling = self._copy_runner(self._training) if self._training is not None else self._new_runner("sampling")
        trace = []
        for _ in range(int(num_sweeps)):
            states = self._step(self._sampling, chains, xs, evidence)
            trace.append(self._scores(self._sampling, xs))
        worlds = states.reshape(chains, -1).clone()
        trace = torch.stack(trace)
        if batch is not None:
            worlds = worlds.reshape(batch, int(n), -1)
            trace = trace.reshape(*trace.shape[:-1], batch, int(n))
        return worlds, trace

    @torch.no_grad()
    def _visit(self, x, evidence):
        """Run fresh chains (num_chains per query); yield their states after burn-in, [B, C, D], and the inputs."""
        batch, chains, xs, evidence = self._queries(self.num_chains, x, self._evidence(evidence))
        runner = self._new_runner("estimate")
        for t in range(self.burn_in + self.num_sweeps):
            states = self._step(runner, chains, xs, evidence)
            if t >= self.burn_in:
                yield states.reshape(batch or 1, self.num_chains, -1), xs, batch

    def marginals(self, *, x=None, evidence=None):
        """Monte Carlo estimate of the atom marginals: average over chains and sweeps after burn-in."""
        total, count = 0.0, 0
        for states, _, batch in self._visit(x, evidence):
            total, count = total + states.sum(dim=1), count + states.shape[1]
        estimate = total / count
        return estimate if batch is not None else estimate.squeeze(0)

    def map(self, *, x=None, evidence=None):
        """The most probable world among those the chains visit (an approximation of MAP)."""
        best, best_score = None, None
        for states, xs, batch in self._visit(x, evidence):
            size, chains, dim = states.shape
            scores = self.model.score(states.reshape(size * chains, dim), xs).reshape(size, chains)
            top, index = scores.max(dim=1)
            candidates = states[torch.arange(size, device=states.device), index]
            if best is None:
                best, best_score = candidates, top
            else:
                better = top > best_score
                best = torch.where(better.unsqueeze(-1), candidates, best)
                best_score = torch.where(better, top, best_score)
        return best if batch is not None else best.squeeze(0)


class ParallelTempering(Gibbs):
    """Gibbs chains at several temperatures, connected by replica exchange.

    Replica r samples from p(y)^(1 / T_r): hot replicas explore, and accepted swaps carry their states down to
    the target replica (T = 1), whose worlds are the samples. `acceptance_rates()` reports how often swaps
    between neighboring temperatures are accepted. Other arguments as for `Gibbs`.
    """

    def __init__(self, model, temperatures=(1.0, 1.5, 2.5, 4.0), **gibbs_args):
        temperatures = tuple(float(t) for t in temperatures)
        if len(temperatures) < 2:
            raise ValueError("parallel tempering needs at least two temperatures")
        if temperatures[0] != 1.0:
            raise ValueError("the first temperature must be 1.0")
        if any(a >= b for a, b in zip(temperatures, temperatures[1:])):
            raise ValueError("temperatures must be strictly increasing")
        super().__init__(model, **gibbs_args)
        self.temperatures = temperatures

    def _new_runner(self, name: str):
        chains = [
            Chain(self.ontology.num_atoms, (lambda y, x, t=t: self.model.score(y, x) / t), self.kernel,
                  ZeroInitializer(), f"{name}_T{t:g}")
            for t in self.temperatures
        ]
        return ExchangeSampler(chains, metropolis_swap_rule)

    def _copy_runner(self, runner):
        return ExchangeSampler([chain.copy() for chain in runner.chains], runner.swap_rule)

    def _step(self, runner, batch_size, x=None, evidence=None):
        return runner.step(batch_size, x=x, evidence=evidence, device=self.model.device)[0]

    def _scores(self, runner, x):
        """Model scores (at temperature 1) of every replica's states, [R, chains]."""
        return torch.stack([self.model.score(chain.states.squeeze(1), x) for chain in runner.chains])

    def acceptance_rates(self, training: bool = False) -> torch.Tensor:
        """Fraction of accepted swaps between temperatures (T_r, T_r+1), [R - 1], for the sampling (or training) chains."""
        runner = self._training if training else self._sampling
        if runner is None:
            raise ValueError("no swaps proposed yet")
        return runner.acceptance_rates()
