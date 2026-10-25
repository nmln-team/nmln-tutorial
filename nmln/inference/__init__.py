"""Inference engines: four uses of the same distribution (marginals, MAP, generation, expectations).

- `Exact` enumerates every possible world (tiny languages only);
- `BeliefPropagation` passes messages between variables; exact on tree-shaped factor graphs;
- `Gibbs` and `ParallelTempering` sample; they work with any potential.

`sampling` holds their building blocks: persistent chains, kernels and replica exchange.
"""

from .base import Inference
from .bp import BeliefPropagation
from .exact import Exact
from .gibbs import Gibbs, ParallelTempering

__all__ = ["Inference", "Exact", "BeliefPropagation", "Gibbs", "ParallelTempering"]
