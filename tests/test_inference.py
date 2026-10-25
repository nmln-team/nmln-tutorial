import pytest
import torch

from nmln import (NMLN, BeliefPropagation, Builtin, Domain, Exact, Ontology, Predicate, SymbolicPotential,
                  UnaryNeuralPotential)

from conftest import ADDITION_RULE, addition_language, sum_evidence


def test_exact_binomial_matches_closed_form():
    n = 6
    v = Domain("v", list(range(n)))
    ontology = Ontology([v], [Predicate("s", [v])])
    exact = Exact(NMLN(ontology, [SymbolicPotential(ontology, "s(X)", beta=1.5)]))  # beta * fraction of true atoms
    assert len(exact.worlds) == 2 ** n
    # max-ent with one statistic: independent atoms with logit beta / n
    assert torch.allclose(exact.marginals(), torch.full((n,), torch.sigmoid(torch.tensor(1.5 / n)).item()))
    assert torch.allclose(exact.log_partition(), n * torch.nn.functional.softplus(torch.tensor(1.5 / n)))


def chain_model():
    nodes = Domain("nodes", list(range(5)), input_shape=(3,))
    ontology = Ontology([nodes], [Predicate("on", [nodes])],
                        builtins=[Builtin("next", [nodes, nodes], fn=lambda i, j: j == i + 1)])
    return NMLN(ontology, [
        UnaryNeuralPotential(ontology, "on", torch.nn.Linear(3, 1)),
        SymbolicPotential(ontology, "next(X,Y) -> (on(X) or not on(Y))", beta=2.0, normalize="sum"),
        SymbolicPotential(ontology, "next(X,Y) and on(X) -> on(Y)", beta=-1.0, normalize="sum"),
    ])


def test_bp_is_exact_on_trees():
    model = chain_model()
    bp, exact = BeliefPropagation(model), Exact(model)
    assert bp.is_tree and bp.exact
    x = {"nodes": torch.randn(4, 5, 3)}
    evidence = model.ontology.evidence({"on(2)": 1})
    assert torch.allclose(bp.log_partition(x=x), exact.log_partition(x=x), atol=1e-4)
    assert torch.allclose(bp.log_probability(evidence, x=x), exact.log_probability(evidence, x=x), atol=1e-4)
    assert torch.allclose(bp.marginals(x=x), exact.marginals(x=x), atol=1e-5)
    assert torch.equal(bp.map(x=x, evidence=evidence), exact.map(x=x, evidence=evidence))


def test_bp_samples_exactly_on_trees():
    model = chain_model()
    bp, exact = BeliefPropagation(model), Exact(model)
    x = {"nodes": torch.randn(2, 5, 3)}
    samples = bp.sample(4000, x=x)  # [2, 4000, D]
    assert samples.shape == (2, 4000, model.ontology.num_atoms)
    frequencies = (samples.unsqueeze(-2) == exact.worlds).all(-1).float().mean(1)  # empirical world frequencies
    assert (frequencies - exact.probabilities(x=x)).abs().max() < 0.03
    clamped = bp.sample(50, x=x, evidence=model.ontology.evidence({"on(0)": 1, "on(4)": 0}))
    assert (clamped[..., 0] == 1).all() and (clamped[..., 4] == 0).all()


def test_bp_rebuilds_when_potentials_change(smokers):
    model = NMLN(smokers, [SymbolicPotential(smokers, "friends(X,Y) -> friends(Y,X)", beta=1.0)])
    bp = BeliefPropagation(model)
    before = bp.marginals()[smokers.atom_index("smokes(ann)")]
    model.potentials.append(SymbolicPotential(smokers, "smokes(ann)", hard=True))
    after = bp.marginals()[smokers.atom_index("smokes(ann)")]
    assert abs(before.item() - 0.5) < 1e-5 and after.item() > 0.999
    assert torch.allclose(bp.marginals(), Exact(model).marginals(), atol=1e-4)


def addition_model(n_digits):
    ontology = addition_language(n_digits)
    network = torch.nn.Linear(5, 10)
    return NMLN(ontology, [
        UnaryNeuralPotential(ontology, "digit", network),
        SymbolicPotential(ontology, ADDITION_RULE, hard=True),
        SymbolicPotential(ontology, "carry(0,0)", hard=True),
    ]), network


def test_bp_is_exact_on_one_digit_addition():
    model, network = addition_model(1)
    bp, exact = BeliefPropagation(model), Exact(model)
    assert bp.is_tree
    x = {"images": torch.randn(3, 2, 5)}
    evidence = sum_evidence(model.ontology, 1, [7, 12, 18])
    log_p = bp.log_probability(evidence, x=x)
    assert torch.allclose(log_p, exact.log_probability(evidence, x=x), atol=1e-4)
    assert torch.allclose(bp.marginals(x=x, evidence=evidence), exact.marginals(x=x, evidence=evidence), atol=1e-5)
    (-log_p).sum().backward()
    assert network.weight.grad.abs().sum() > 0


def test_bp_on_the_carry_chain():
    model, _ = addition_model(4)
    bp = BeliefPropagation(model)
    assert bp.is_tree and len(bp.factor_scopes) == 4  # one factor per column
    x = {"images": torch.randn(2, 8, 5)}
    evidence = sum_evidence(model.ontology, 4, [1234 + 8766, 4321 + 1111])
    digit = model.ontology.indices("digit")
    powers = 10 ** torch.arange(4)

    def totals(worlds):
        digits = worlds[..., digit].argmax(-1)  # a0..a3, b0..b3
        return (digits[..., :4] * powers).sum(-1) + (digits[..., 4:] * powers).sum(-1)

    # the MAP world, and every sampled world, given the images and the sum is a correct addition
    assert totals(bp.map(x=x, evidence=evidence)).tolist() == [10000, 5432]
    assert (totals(bp.sample(20, x=x, evidence=evidence)) == torch.tensor([[10000], [5432]])).all()


def test_loopy_bp_is_flagged(smokers):
    model = NMLN(smokers, [SymbolicPotential(smokers, "smokes(X) and friends(X,Y) -> smokes(Y)", beta=3.0)])
    with pytest.warns(UserWarning, match="approximate"):
        bp = BeliefPropagation(model, max_iters=50, damping=0.3)
    assert not bp.is_tree and not bp.exact and bp.computes_log_z
    assert (bp.marginals() - Exact(model).marginals()).abs().max() < 0.05
    with pytest.raises(NotImplementedError):
        bp.sample(3)
    with pytest.raises(ValueError):
        BeliefPropagation(model, strict=True)


def test_impossible_evidence_is_rejected():
    model, _ = addition_model(1)
    two_digits = model.ontology.evidence({"digit(a0,3)": 1, "digit(a0,5)": 1})
    with pytest.raises(ValueError, match=r"digit\(a0,·\)"):
        Exact(model).marginals(x={"images": torch.randn(1, 2, 5)}, evidence=two_digits)


def test_queries_return_plain_tensors_and_describe(smokers):
    model = NMLN(smokers, [SymbolicPotential(smokers, "friends(X,Y) -> friends(Y,X)", beta=1.0)])
    for engine in [Exact(model), BeliefPropagation(model)]:
        assert not engine.marginals().requires_grad and not engine.map().requires_grad
        assert engine.log_partition().requires_grad  # stays differentiable, for learning
    two_digits, _ = addition_model(2)
    lines = BeliefPropagation(two_digits).describe().splitlines()
    assert lines[0].startswith("10 variables, 2 factors") and "exact" in lines[0]
    assert lines[1] == "  factor 0: digit(a0,·), digit(b0,·), res(0,·), carry(0,·), carry(1,·)"
