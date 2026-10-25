import torch

from nmln import NMLN, Domain, Exact, Gibbs, Ontology, Predicate, SymbolicPotential, WorldMLPPotential


def _smokers_model(smokers, beta=4.0):
    return NMLN(smokers, [SymbolicPotential(smokers, "smokes(X) and friends(X,Y) -> smokes(Y)", beta=beta)])


def test_training_chains_persist_and_resize():
    v = Domain("v", list(range(10)))
    ontology = Ontology([v], [Predicate("s", [v])])
    engine = Gibbs(NMLN(ontology, [WorldMLPPotential(ontology, hidden_size=8)]), sweeps_per_sample=3)
    assert engine.model_samples(16).shape == (16, 10)
    assert engine.model_samples(5).shape == (5, 10)  # a smaller last batch keeps the chains instead of restarting them
    assert engine._training.states.shape == (5, 1, 10)
    assert engine.sample(7).shape == (7, 10)
    assert engine._training.states.shape == (5, 1, 10)  # sampling does not touch the training chains


def test_gibbs_estimates_match_exact(smokers):
    model = _smokers_model(smokers)
    evidence = smokers.evidence({"smokes(ann)": 1, "friends(ann,bob)": 1})
    exact = Exact(model)
    engine = Gibbs(model, sweeps_per_sample=10, num_chains=200, num_sweeps=20)
    samples = engine.sample(2000, evidence=evidence)
    assert (samples[:, evidence.observed] == evidence.y[evidence.observed]).all()
    assert (samples.mean(0) - exact.marginals(evidence=evidence)).abs().max() < 0.05
    assert (engine.marginals(evidence=evidence) - exact.marginals(evidence=evidence)).abs().max() < 0.05
    best = engine.map(evidence=evidence)
    assert torch.allclose(model.score(best), model.score(exact.map(evidence=evidence)))
