import pytest
import torch

from nmln import Builtin, Categorical, Domain, Evidence, Ontology, Predicate

ADDITION_RULE = """
    first(X,I) and second(Y,I) and next(I,J)
    and digit(X,A) and digit(Y,B) and carry(I,C) and add(A,B,C,S,D)
    -> res(I,S) and carry(J,D)
"""


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture
def smokers():
    people = Domain("people", ["ann", "bob", "carl"])
    return Ontology([people], [Predicate("smokes", [people]), Predicate("friends", [people, people])])


def addition_language(n_digits: int, input_size: int = 5):
    images = Domain("images", [f"a{i}" for i in range(n_digits)] + [f"b{i}" for i in range(n_digits)],
                    input_shape=(input_size,))
    pos = Domain("pos", list(range(n_digits + 1)))
    value = Domain("value", list(range(10)))
    bit = Domain("bit", [0, 1])
    return Ontology(
        [images, pos, value, bit],
        [Predicate("digit", [images, value]), Predicate("res", [pos, value]), Predicate("carry", [pos, bit])],
        builtins=[
            Builtin("first", [images, pos], facts=[(f"a{i}", i) for i in range(n_digits)]),
            Builtin("second", [images, pos], facts=[(f"b{i}", i) for i in range(n_digits)]),
            Builtin("next", [pos, pos], fn=lambda i, j: j == i + 1),
            Builtin("add", [value, value, bit, value, bit], fn=lambda a, b, c, s, d: a + b + c == s + 10 * d),
        ],
        variables=[Categorical("digit", over="value"), Categorical("res", over="value"), Categorical("carry", over="bit")],
    )


def sum_evidence(ontology, n_digits, totals):
    """Observed sum digits: res(i, .) for every position below n_digits, and the final carry."""
    worlds = torch.stack([
        ontology.world([f"res({i},{total // 10 ** i % 10})" for i in range(n_digits)]
                       + [f"carry({n_digits},{total // 10 ** n_digits})"])
        for total in totals
    ])
    observed = ontology.mask(["res", "carry"])
    observed[ontology.indices("res")[n_digits]] = False  # res(N, .) is not part of the sum here
    observed[ontology.indices("carry")[:n_digits].reshape(-1)] = False
    return Evidence(worlds, observed.expand(worlds.shape))
