"""Every engine answers the same calls, with the same output shapes."""

import pytest
import torch

from nmln import NMLN, BeliefPropagation, Evidence, Exact, Gibbs, ParallelTempering, SymbolicPotential

from test_inference import chain_model

ENGINES = {
    "exact": lambda m: Exact(m),
    "bp": lambda m: BeliefPropagation(m),
    "gibbs": lambda m: Gibbs(m, num_chains=8, burn_in=2, num_sweeps=3),
    "tempering": lambda m: ParallelTempering(m, temperatures=(1.0, 2.0), num_chains=8, burn_in=2, num_sweeps=3),
}


@pytest.mark.parametrize("name", ENGINES)
def test_uniform_interface_without_inputs(name):
    model = chain_model()
    model.potentials = model.potentials[1:]  # the symbolic rules only: no inputs needed
    engine, d = ENGINES[name](model), model.ontology.num_atoms
    single = model.ontology.evidence({"on(1)": 1})
    observed = torch.zeros(3, d, dtype=torch.bool)
    observed[:, :2] = True  # on(0) and on(1) observed true, in 3 queries
    batched = Evidence(torch.ones(3, d), observed)
    assert engine.marginals().shape == (d,)
    assert engine.marginals(evidence=single).shape == (d,)
    assert engine.marginals(evidence=batched).shape == (3, d)
    assert engine.map(evidence=batched).shape == (3, d)
    assert engine.sample(5).shape == (5, d)
    assert engine.sample(5, evidence=single).shape == (5, d)
    assert engine.sample(5, evidence=batched).shape == (3, 5, d)
    assert (engine.sample(5, evidence=batched)[..., :2] == 1).all()
    if engine.computes_log_z:
        assert engine.log_partition().shape == ()
        assert engine.log_probability(batched).shape == (3,)
    else:
        with pytest.raises(NotImplementedError):
            engine.log_partition()
    with pytest.raises(TypeError):
        engine.marginals(single)  # x and evidence are keyword-only


@pytest.mark.parametrize("name", ENGINES)
def test_uniform_interface_with_inputs(name):
    model = chain_model()
    engine, d = ENGINES[name](model), model.ontology.num_atoms
    x = {"nodes": torch.randn(4, 5, 3)}
    assert engine.marginals(x=x).shape == (4, d)
    assert engine.map(x=x).shape == (4, d)
    assert engine.sample(6, x=x).shape == (4, 6, d)
    if engine.computes_log_z:
        assert engine.log_partition(x=x).shape == (4,)


def test_gibbs_trace_shapes(smokers):
    model = NMLN(smokers, [SymbolicPotential(smokers, "smokes(X) and friends(X,Y) -> smokes(Y)", beta=4.0)])
    worlds, trace = Gibbs(model).trace(7, num_sweeps=4)
    assert worlds.shape == (7, smokers.num_atoms) and trace.shape == (4, 7)
    tempering = ParallelTempering(model, temperatures=(1.0, 2.0, 4.0))
    worlds, trace = tempering.trace(7, num_sweeps=4, evidence=smokers.observe(torch.zeros(2, smokers.num_atoms), ["smokes"]))
    assert worlds.shape == (2, 7, smokers.num_atoms) and trace.shape == (4, 3, 2, 7)
    assert tempering.acceptance_rates().shape == (2,)
