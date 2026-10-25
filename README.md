# Neural Markov Logic Networks

`nmln` is a small PyTorch library for neural Markov logic networks: probability distributions over
relational worlds,

    p(y | x) ∝ exp( Σ_k β_k · statistic_k(y, x) ),

whose potentials can be symbolic (first-order rules, as in Markov logic) or neural (networks over
fragments of the world, over the whole world graph, or over inputs such as images attached to constants).

## Install

Python 3.10 or later, with PyTorch, PyTorch Lightning and NumPy.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[tutorial,test]"
```

Optional: `SymbolicPotential(..., backend="deeplog")` compiles rules with
[DeepLog](https://github.com/ML-KULeuven/deeplog) (Python 3.12+, installed from source with
`pip install -e path/to/deeplog`).

On macOS with the python.org installer, downloading MNIST can fail with `CERTIFICATE_VERIFY_FAILED`.
Run `Install Certificates.command` from the Python folder in Applications, or point Python at certifi's
certificates: `export SSL_CERT_FILE=$(python -m certifi)`.

## A first model

```python
import torch
import nmln

# The language: constants, predicates, and the possible worlds they define (here 3 + 9 Boolean atoms).
people = nmln.Domain("people", ["ann", "bob", "carl"])
ontology = nmln.Ontology([people], [nmln.Predicate("smokes", [people]), nmln.Predicate("friends", [people, people])])

# The model: one soft rule, a Markov logic network.
rule = nmln.SymbolicPotential(ontology, "smokes(X) and friends(X,Y) -> smokes(Y)", beta=2.0)
model = nmln.NMLN(ontology, [rule])

# Inference: the four uses of the distribution, with an exact engine on this tiny language.
engine = nmln.Exact(model)
evidence = ontology.evidence({"smokes(ann)": 1, "friends(ann,bob)": 1})
marginals = engine.marginals(evidence=evidence)
print(f"P(smokes(bob) | evidence) = {marginals[ontology.atom_index('smokes(bob)')]:.3f}")
print(ontology.true_atoms(engine.map(evidence=evidence)))

# Learning: maximum likelihood on data, here sampled from the model itself.
data = engine.sample(500)
history = nmln.MaximumLikelihood(engine, lr=0.1).fit(torch.utils.data.DataLoader(data, batch_size=50), epochs=20, accelerator="cpu")
```

## The pieces

| | |
|---|---|
| `Ontology(domains, predicates, builtins=..., variables=...)` | The language. A world is a binary vector over ground atoms; `atom_index`, `atom`, `world`, `true_atoms`, `as_tables` and `mask` move between atoms and vectors. |
| `Evidence(y, observed)` | Observed atoms, from `ontology.evidence({"smokes(ann)": 1})` or `ontology.observe(y, ["res"])`. |
| `Categorical(...)` | Declares how atoms group into random variables: which digit an image shows, which bond links two atoms. Undeclared predicates are Boolean. |
| `Builtin(...)` | Fixed background predicates (arithmetic, the layout of an example), used only to ground rules. |
| `NMLN(ontology, potentials)` | The model: it scores worlds. |
| Potentials | `SymbolicPotential` (rules), `UnaryNeuralPotential` (neural predicates over inputs), `FragmentPotential` (MLP on anonymized fragments), `GNNPotential` (graph network), `WorldMLPPotential` (MLP on the whole world). `nmln.normal_form(rule)` shows how a rule is evaluated. Subclass `Potential` and implement `statistic(y, x)` for your own. |
| Inference | `Exact` (enumeration), `BeliefPropagation` (exact on trees, including sampling; `describe()` lists its factors), `Gibbs`, `ParallelTempering`. One interface for all: `marginals(x=, evidence=)`, `map(...)`, `sample(n, x=, evidence=)`; exact engines also `log_partition(...)` and `log_probability(evidence, x=)`. Results have a batch dimension exactly when `x` or the evidence does; engine settings go to the constructor. |
| Learning | `MaximumLikelihood` (with `Exact` or `BeliefPropagation`; handles latent atoms) and `Contrastive` (with `Gibbs` or `ParallelTempering`). Learners are PyTorch Lightning modules: `fit(data, epochs, accelerator, **trainer_args)` runs a quiet Trainer and returns the loss per epoch, or use a Trainer of your own; `loss(batch)` serves a hand-written loop. |

## Examples

Run from the repository root:

```bash
python -m examples.binomial.train                                                  # independent atoms, Gibbs
python examples/smoker-friends/smoker_friends.py                                   # a relational world
python examples/mnist-addition/mnist_addition.py --digits 1 2 4 15 --epochs 5      # neural predicates + a program, exact BP
python examples/molecules/molecules.py --steer-oxygen                              # molecule generation, parallel tempering
```

## Tests

```bash
python -m pytest
```
