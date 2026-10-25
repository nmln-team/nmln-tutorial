"""Neural Markov logic networks: probabilistic models of relational worlds with neural and symbolic potentials.

    ontology  = Ontology(domains, predicates, builtins=..., variables=[Categorical(...)])   # the language
    model     = NMLN(ontology, [potential, ...])                                           # the distribution
    engine    = Exact(model) | BeliefPropagation(model) | Gibbs(model) | ParallelTempering(model)
    learner   = MaximumLikelihood(engine) | Contrastive(engine)                            # the training objective
"""

from . import evidence, inference, learning, ontology, potentials, symbolic, utils, variables
from .evidence import Evidence
from .inference import BeliefPropagation, Exact, Gibbs, Inference, ParallelTempering
from .learning import Contrastive, MaximumLikelihood
from .model import NMLN
from .ontology import Builtin, Categorical, Domain, Ontology, Predicate
from .potentials import FragmentPotential, GNNPotential, Potential, UnaryNeuralPotential, WorldMLPPotential
from .symbolic import SymbolicPotential, normal_form

__all__ = [
    "NMLN",
    "Ontology", "Domain", "Predicate", "Builtin", "Categorical", "Evidence",
    "Potential", "UnaryNeuralPotential", "FragmentPotential", "GNNPotential", "WorldMLPPotential",
    "SymbolicPotential", "normal_form",
    "Inference", "Exact", "BeliefPropagation", "Gibbs", "ParallelTempering",
    "Contrastive", "MaximumLikelihood",
    "evidence", "ontology", "potentials", "symbolic", "inference", "learning", "variables", "utils",
]
