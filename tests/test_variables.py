import pytest
import torch

from nmln.ontology import Categorical, Domain, Ontology, Predicate


def test_undeclared_predicates_are_boolean(smokers):
    variables = smokers.variables
    assert len(variables) == smokers.num_atoms
    assert variables.cardinalities.tolist() == [2] * smokers.num_atoms
    worlds = (torch.rand(10, smokers.num_atoms) < 0.5).float()
    assert torch.equal(variables.worlds(variables.configs(worlds)), worlds)
    assert variables.is_valid(worlds).all()


def _molecules():
    atoms = Domain("atoms", list(range(4)))
    types, bonds = ["C", "N", "O"], ["SINGLE", "DOUBLE"]
    return Ontology(
        [atoms],
        [Predicate(t, [atoms]) for t in types] + [Predicate(b, [atoms, atoms]) for b in bonds],
        variables=[Categorical(types), Categorical(bonds, optional=True, symmetric=True, irreflexive=True)],
    )


def test_molecule_variables_roundtrip():
    ontology = _molecules()
    variables = ontology.variables
    assert len(variables) == 4 + 6  # one type per atom, one bond-or-none per unordered pair
    assert variables.cardinalities.tolist() == [3] * 4 + [3] * 6
    assert variables.name(0) == "C|N|O(0)" and variables.name(4) == "SINGLE|DOUBLE(0,1)"
    assert variables.groups[1].values == ["SINGLE", "DOUBLE", "none"]
    configs = torch.stack([torch.randint(0, int(c), (7,)) for c in variables.cardinalities], dim=-1)
    worlds = variables.worlds(configs)
    assert torch.equal(variables.configs(worlds), configs)
    assert variables.is_valid(worlds).all()
    # bonds are symmetric and never reflexive
    single = worlds[:, ontology.indices("SINGLE")]
    assert torch.equal(single, single.transpose(-1, -2))
    assert single.diagonal(dim1=-2, dim2=-1).sum() == 0
    # two types for one atom, or a self-bond, is not a possible world
    bad = worlds[:2].clone()
    bad[0, ontology.indices("C")] = 1.0
    bad[1, ontology.atom_index("SINGLE", 2, 2)] = 1.0
    assert (variables.configs(bad[:1])[0, :4] == -1).all()
    assert not variables.is_valid(bad).any()


def test_categorical_over_an_argument():
    images = Domain("images", ["a0", "b0"])
    value = Domain("value", list(range(10)))
    ontology = Ontology([images, value], [Predicate("digit", [images, value])],
                        variables=[Categorical("digit", over="value")])
    group = ontology.variables.groups[0]
    assert group.names == ["digit(a0,·)", "digit(b0,·)"] and group.values == list(range(10))
    assert group.masks.shape == (2, 10, ontology.num_atoms)


def test_declaration_errors():
    images = Domain("images", ["a0", "b0"])
    with pytest.raises(ValueError):
        Categorical("smokes")  # a single value: Boolean predicates need no declaration
    with pytest.raises(ValueError):
        Ontology([images], [Predicate("f", [images, images])], variables=[Categorical("f", over="images")])
    with pytest.raises(Exception):
        Ontology([images], [Predicate("p", [images]), Predicate("q", [images])],
                 variables=[Categorical(["p", "q"]), Categorical(["q", "p"])])


def test_allowed_values(smokers):
    variables = smokers.variables
    evidence = smokers.evidence({"smokes(ann)": 1, "friends(ann,bob)": 0})
    allowed = variables.allowed(evidence.y, evidence.observed)  # per group [M, 2]: (true, false)
    ann = variables.variable_of_atom[smokers.atom_index("smokes(ann)")]
    friends = variables.variable_of_atom[smokers.atom_index("friends(ann,bob)")]
    group, m = variables.group_of(int(ann))
    assert allowed[group][m].tolist() == [True, False]
    group, m = variables.group_of(int(friends))
    assert allowed[group][m].tolist() == [False, True]
    assert sum(int(a.all(-1).sum()) for a in allowed) == smokers.num_atoms - 2
