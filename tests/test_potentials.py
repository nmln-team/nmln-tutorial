import itertools

import pytest
import torch

from nmln import NMLN, Domain, FragmentPotential, GNNPotential, Ontology, Predicate, WorldMLPPotential
from nmln.potentials import Potential


def _rename(ontology, worlds, permutation):
    """The same worlds with the constants renamed by a permutation."""
    constants = next(iter(ontology.domains.values())).constants
    renamed = torch.zeros_like(worlds)
    for i in range(ontology.num_atoms):
        name, args = ontology.atom(i).split("(")
        args = [constants[permutation[constants.index(a)]] for a in args.rstrip(")").split(",")]
        renamed[..., ontology.atom_index(name, *args)] = worlds[..., i]
    return renamed


def test_fragment_and_gnn_potentials_ignore_names():
    atoms = Domain("atoms", ["a", "b", "c", "d"])
    ontology = Ontology([atoms], [Predicate("C", [atoms]), Predicate("O", [atoms]), Predicate("bond", [atoms, atoms])])
    worlds = (torch.rand(5, ontology.num_atoms) < 0.5).float()
    for potential in [FragmentPotential(ontology, k=2, hidden_sizes=(16, 8)), FragmentPotential(ontology, k=3, hidden_sizes=(16, 8)),
                      GNNPotential(ontology, ["C", "O"], ["bond"], hidden_size=16)]:
        for permutation in itertools.islice(itertools.permutations(range(4)), 1, 6):
            renamed = _rename(ontology, worlds, permutation)
            assert torch.allclose(potential(worlds), potential(renamed), atol=1e-5), type(potential).__name__
    world_mlp = WorldMLPPotential(ontology, hidden_size=16)
    assert not torch.allclose(world_mlp(worlds), world_mlp(_rename(ontology, worlds, (1, 0, 2, 3))))


def test_potential_protocol(smokers):
    class Count(Potential):
        def statistic(self, y, x=None):
            return y.sum(-1)

    potential = Count(beta=2.0)
    worlds = (torch.rand(3, 2, smokers.num_atoms) < 0.5).float()
    assert torch.allclose(potential(worlds), 2.0 * worlds.sum(-1))
    model = NMLN(smokers, [potential, Count(beta=-1.0, learn_beta=False)])
    assert torch.allclose(model.score(worlds), worlds.sum(-1))
    assert [p.requires_grad for p in model.parameters()] == [True, False]
    with pytest.raises(TypeError):
        NMLN(smokers, [torch.nn.Linear(3, 1)])
