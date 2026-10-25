import pytest
import torch

from nmln import (NMLN, BeliefPropagation, Contrastive, Domain, Exact, Gibbs, MaximumLikelihood, Ontology, Predicate,
                  SymbolicPotential, UnaryNeuralPotential)


def _binomial(n=4):
    v = Domain("v", list(range(n)))
    ontology = Ontology([v], [Predicate("s", [v])])
    return ontology, NMLN(ontology, [SymbolicPotential(ontology, "s(X)", beta=0.0)])


def test_maximum_likelihood_matches_moments():
    """With one statistic, maximum likelihood is moment matching: beta = n * logit(frequency of true atoms)."""
    ontology, model = _binomial()
    data = (torch.rand(400, ontology.num_atoms) < 0.8).float()
    learner = MaximumLikelihood(Exact(model), lr=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(300):  # the training loop, written out
        loss = learner.loss(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    frequency = data.mean()
    expected = ontology.num_atoms * torch.log(frequency / (1 - frequency))
    assert abs(model.potentials[0].beta.item() - float(expected)) < 0.05


def test_contrastive_loss_and_fit():
    ontology, model = _binomial()
    data = (torch.rand(64, ontology.num_atoms) < 0.8).float()
    learner = Contrastive(Gibbs(model), lr=0.05, noise_scale=0.0)
    loss = learner.loss(data)
    loss.backward()
    assert torch.isfinite(loss) and model.potentials[0].beta.grad is not None
    history = learner.fit(torch.utils.data.DataLoader(data, batch_size=16), epochs=30, accelerator="cpu")
    assert len(history) == 30
    assert model.potentials[0].beta.item() > 2.0  # moves towards 4 * logit(0.8) = 5.5
    with pytest.raises(ValueError):
        learner.loss({"y": data, "observed": torch.zeros_like(data, dtype=torch.bool)})


def test_learners_check_their_engine_and_data():
    ontology, model = _binomial()
    with pytest.raises(TypeError):
        Contrastive(Exact(model))
    with pytest.raises(TypeError):
        MaximumLikelihood(Gibbs(model))
    images = Domain("images", ["a"])
    value = Domain("value", list(range(3)))
    from nmln import Categorical
    one_hot = Ontology([images, value], [Predicate("digit", [images, value])], variables=[Categorical("digit", over="value")])
    learner = MaximumLikelihood(Exact(NMLN(one_hot, [SymbolicPotential(one_hot, "digit(a,0)")])))
    with pytest.raises(ValueError, match="training example 1"):
        learner.loss(torch.tensor([[1.0, 0, 0], [1.0, 1.0, 0]]))  # the second world shows two digits


def test_maximum_likelihood_with_latent_atoms():
    images = Domain("images", ["a", "b"], input_shape=(4,))
    ontology = Ontology([images], [Predicate("bright", [images])])
    network = torch.nn.Linear(4, 1)
    learner = MaximumLikelihood(BeliefPropagation(NMLN(ontology, [UnaryNeuralPotential(ontology, "bright", network)])))
    batch = {"x": {"images": torch.randn(6, 2, 4)}, "y": torch.ones(6, 2), "observed": torch.tensor([[True, False]] * 6)}
    loss = learner.loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and network.weight.grad is not None


def test_a_trainer_of_your_own():
    pl = pytest.importorskip("pytorch_lightning")
    ontology, model = _binomial()
    data = (torch.rand(32, ontology.num_atoms) < 0.8).float()
    learner = MaximumLikelihood(Exact(model), lr=0.1)
    pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
               enable_model_summary=False).fit(learner, torch.utils.data.DataLoader(data, batch_size=8))
    assert model.potentials[0].beta.item() > 0.1 and len(learner.history) == 2


def test_fit_validates_the_first_epoch_only(monkeypatch):
    ontology, model = _binomial()
    calls = []
    original = type(ontology.variables).validate
    monkeypatch.setattr(type(ontology.variables), "validate", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    data = torch.utils.data.DataLoader((torch.rand(40, ontology.num_atoms) < 0.5).float(), batch_size=10)
    learner = MaximumLikelihood(Exact(model), lr=0.1)
    learner.fit(data, epochs=3, accelerator="cpu")
    assert len(calls) == 4  # the 4 batches of the first epoch
    learner.loss(next(iter(data)), validate=False)
    assert len(calls) == 4
