import pytest
import torch

from nmln import Domain, Ontology, Predicate


def test_atoms_and_positions(smokers):
    assert smokers.num_atoms == 3 + 9
    assert smokers.atom_index("friends(ann,bob)") == smokers.atom_index("friends", "ann", "bob") == 3 + 1
    assert [smokers.atom(i) for i in range(smokers.num_atoms)][:4] == ["smokes(ann)", "smokes(bob)", "smokes(carl)", "friends(ann,ann)"]
    for i in range(smokers.num_atoms):
        assert smokers.atom_index(smokers.atom(i)) == i
    assert smokers.indices("friends").shape == (3, 3)
    with pytest.raises(KeyError):
        smokers.atom_index("drinks(ann)")
    with pytest.raises(KeyError):
        smokers.atom_index("smokes(dave)")


def test_worlds_from_atoms(smokers):
    world = smokers.world(["smokes(ann)", "friends(ann,bob)"])
    assert smokers.true_atoms(world) == ["smokes(ann)", "friends(ann,bob)"]
    tables = smokers.as_tables(torch.stack([world, 1 - world]))
    assert tables["smokes"].shape == (2, 3) and tables["friends"].shape == (2, 3, 3)
    assert tables["friends"][0, 0, 1] == 1 and tables["friends"][1, 0, 1] == 0


def test_evidence_and_masks(smokers):
    mask = smokers.mask(["smokes"])
    assert mask.sum() == 3 and mask[:3].all()
    worlds = torch.zeros(4, smokers.num_atoms)
    evidence = smokers.observe(worlds, ["friends"])
    assert evidence.observed.shape == (4, smokers.num_atoms) and evidence.observed[:, 3:].all()
    assert evidence.batch_size == 4
    evidence.observed[0, 3] = False  # each example has a mask of its own
    assert evidence.observed[1, 3]
    evidence = smokers.evidence({"smokes(bob)": 1, "friends(bob,carl)": 0})
    assert evidence.observed.sum() == 2 and evidence.y[smokers.atom_index("smokes(bob)")] == 1
    assert evidence.batch_size is None


def test_integer_constants_and_summary():
    pos = Domain("pos", list(range(3)))
    ontology = Ontology([pos], [Predicate("carry", [pos])])
    assert ontology.atom_index("carry", 2) == ontology.atom_index("carry(2)") == 2
    assert "3 ground atoms" in repr(ontology)


def test_fragments(smokers):
    fragments = smokers.fragment_indices(2)
    # 3 subsets of 2 people, 2 renamings each, 2 smokes atoms + 4 friends atoms per fragment
    assert fragments.shape == (3, 2, 6)
    first = [smokers.atom(int(i)) for i in fragments[0, 1]]
    assert first == ["smokes(bob)", "smokes(ann)", "friends(bob,bob)", "friends(bob,ann)", "friends(ann,bob)", "friends(ann,ann)"]
