import itertools

import pytest
import torch

from nmln import utils
from nmln.symbolic import SymbolicPotential, normal_form, to_deeplog

from conftest import ADDITION_RULE, addition_language


def test_translation():
    assert to_deeplog("smokes(X) and friends(X,Y) -> smokes(Y)") == (
        "not smokes(X)_boolean or not friends(X,Y)_boolean or smokes(Y)_boolean"
    )
    assert to_deeplog("not (a(X) and b(X)) -> (c(X) or not d(X))") == (
        "(a(X)_boolean and b(X)_boolean) or c(X)_boolean or not d(X)_boolean"
    )
    assert to_deeplog("a or b and c") == "a_boolean or (b_boolean and c_boolean)"
    assert to_deeplog("a -> b -> c") == "not a_boolean or not b_boolean or c_boolean"
    assert to_deeplog("nothing(X) # a comment") == "nothing(X)_boolean"
    assert normal_form("smokes(X) and friends(X,Y) -> smokes(Y)") == "not smokes(X) or not friends(X,Y) or smokes(Y)"


def _brute_count(world, ontology):
    people = ontology.domains["people"].constants
    count = 0
    for x, y in itertools.product(people, repeat=2):
        s_x = world[ontology.atom_index(f"smokes({x})")]
        f_xy = world[ontology.atom_index(f"friends({x},{y})")]
        s_y = world[ontology.atom_index(f"smokes({y})")]
        count += float((not s_x) or (not f_xy) or s_y)
    return count


def test_counts_match_brute_force(smokers):
    ontology = smokers
    potential = SymbolicPotential(ontology, "smokes(X) and friends(X,Y) -> smokes(Y)", normalize="sum")
    assert potential.num_groundings == 9
    worlds = (torch.rand(20, ontology.num_atoms) < 0.5).float()
    assert potential(worlds).tolist() == [_brute_count(w, ontology) for w in worlds]
    # leading dimensions are kept, as Gibbs sampling needs
    assert potential(worlds.view(4, 5, 1, -1)).shape == (4, 5, 1)


def test_mean_beta_constants_and_hard(smokers):
    ontology = smokers
    worlds = (torch.rand(8, ontology.num_atoms) < 0.5).float()
    soft = SymbolicPotential(ontology, "friends(X,bob) -> smokes(X)", beta=2.0)
    hard = SymbolicPotential(ontology, "friends(X,bob) -> smokes(X)", hard=True)
    truth = soft.evaluate_groundings(worlds)
    assert truth.shape == (8, 3)
    assert torch.allclose(soft(worlds), 2.0 * truth.mean(-1))
    assert torch.allclose(hard(worlds), -utils.HARD_WEIGHT * (1 - truth).sum(-1))
    assert not hard.beta.requires_grad and soft.beta.requires_grad


def test_errors(smokers):
    ontology = smokers
    with pytest.raises(KeyError):
        SymbolicPotential(ontology, "smokes(X) -> drinks(X)")
    with pytest.raises(KeyError):
        SymbolicPotential(ontology, "friends(X,dave)")
    with pytest.raises(SyntaxError):
        SymbolicPotential(ontology, "smokes(X) and -> smokes(Y)")


def test_builtins_prune_groundings():
    ontology = addition_language(2)
    rule = SymbolicPotential(ontology, ADDITION_RULE, hard=True)
    # first/second/next pick the two images at each position, add picks (S, D) from (A, B, C)
    assert rule.num_groundings == 2 * 10 * 10 * 2
    assert rule.groundings(1) == [
        "first(a0,0) and second(b0,0) and next(0,1) and digit(a0,0) and digit(b0,0) and carry(0,0) "
        "and add(0,0,0,0,0) -> res(0,0) and carry(1,0)"
    ]
    # world atoms only: built-ins are not part of the world
    assert rule.grounding_atoms.max() < ontology.num_atoms
    assert ontology.num_atoms == 4 * 10 + 3 * 10 + 3 * 2


def test_deeplog_backend_agrees(smokers):
    pytest.importorskip("deeplog")
    worlds = (torch.rand(50, smokers.num_atoms) < 0.5).float()
    for rule in ["smokes(X) and friends(X,Y) -> smokes(Y)", "not (smokes(X) and friends(X,Y)) or friends(Y,X)"]:
        torch_rule = SymbolicPotential(smokers, rule, normalize="sum")
        deeplog_rule = SymbolicPotential(smokers, rule, normalize="sum", backend="deeplog")
        assert torch.equal(torch_rule(worlds), deeplog_rule(worlds))
