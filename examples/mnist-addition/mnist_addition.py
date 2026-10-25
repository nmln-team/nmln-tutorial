"""MNIST addition with an NMLN: neural predicates plus the addition program, trained with exact BP.

Every example shows two N-digit numbers as 2N MNIST images, and only their sum is observed. The NMLN
puts two kinds of potentials on the same world:

- a neural predicate: a CNN reads the image attached to each constant and scores digit(image, value);
- the addition program, as hard rules in the carry representation: one rule per column links the two
  digits and the incoming carry to the result digit and the outgoing carry.

Each grounding of the column rule reads five variables, {digit(a_i), digit(b_i), carry(i), res(i), carry(i+1)},
so the factor graph is a chain of columns: belief propagation is exact and costs O(N). The model is
trained by maximizing the exact likelihood of the observed sums, -log p(sum | images), and learns to
recognize digits without ever seeing a digit label.

Usage (from the repository root):
    python examples/mnist-addition/mnist_addition.py --digits 1 2 4 15 --out examples/mnist-addition/results.md
"""

import argparse
import time
from pathlib import Path

import torch
import torchvision

import nmln
from nmln.ontology import Builtin, Categorical, Domain, Ontology, Predicate
from nmln.potentials import UnaryNeuralPotential

from dataset import MNISTAddition, to_number

PROGRAM = [
    # column i: the digits of the two numbers and the incoming carry give the result digit and the outgoing carry
    """first(X,I) and second(Y,I) and next(I,J)
       and digit(X,A) and digit(Y,B) and carry(I,C) and add(A,B,C,S,D)
       -> res(I,S) and carry(J,D)""",
    # nothing is carried into the first column
    "carry(0,0)",
    # the last carry is the leading digit of the sum
    "last(I) and carry(I,C) and same(C,V) -> res(I,V)",
]


def build_language(n_digits: int):
    """Constants a0..a{N-1}, b0..b{N-1} carry the images; the sum has N + 1 digits res(0..N, .)."""
    images = Domain(
        "images", [f"a{i}" for i in range(n_digits)] + [f"b{i}" for i in range(n_digits)], input_shape=(1, 28, 28)
    )
    pos = Domain("pos", list(range(n_digits + 1)))
    value = Domain("value", list(range(10)))
    bit = Domain("bit", [0, 1])
    ontology = Ontology(
        [images, pos, value, bit],
        [Predicate("digit", [images, value]), Predicate("res", [pos, value]), Predicate("carry", [pos, bit])],
        builtins=[
            Builtin("first", [images, pos], facts=[(f"a{i}", i) for i in range(n_digits)]),
            Builtin("second", [images, pos], facts=[(f"b{i}", i) for i in range(n_digits)]),
            Builtin("next", [pos, pos], fn=lambda i, j: j == i + 1),
            Builtin("last", [pos], fn=lambda i: i == n_digits),
            Builtin("add", [value, value, bit, value, bit], fn=lambda a, b, c, s, d: a + b + c == s + 10 * d),
            Builtin("same", [bit, value], fn=lambda c, v: c == v),
        ],
        # the variables: which digit each image shows, which digit each position of the sum has, which carry
        variables=[Categorical("digit", over="value"), Categorical("res", over="value"), Categorical("carry", over="bit")],
    )
    return ontology


class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5), torch.nn.MaxPool2d(2), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 5), torch.nn.MaxPool2d(2), torch.nn.ReLU(),
            torch.nn.Flatten(), torch.nn.Linear(32 * 4 * 4, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.net(images)


def build_model(n_digits: int):
    """The NMLN (a neural predicate plus the program) and its inference engine, exact belief propagation."""
    ontology = build_language(n_digits)
    model = nmln.NMLN(
        ontology,
        [UnaryNeuralPotential(ontology, "digit", DigitClassifier())]
        + [nmln.SymbolicPotential(ontology, rule, hard=True) for rule in PROGRAM],
    )
    return ontology, nmln.BeliefPropagation(model)


@torch.no_grad()
def evaluate(engine, loader, ontology, n_digits):
    """Accuracy of the digits and of the sums, reading both off the MAP world given the images only."""
    engine.eval()
    digit_index, res_index = ontology.indices("digit"), ontology.indices("res")  # [2N, 10], [N + 1, 10]
    digits_ok = sums_ok = count = 0
    for batch in loader:
        world = engine.map(x={"images": batch["x"]["images"].to(engine.model.device)}).cpu()
        digits = world[:, digit_index].argmax(-1)
        sums = to_number(world[:, res_index].argmax(-1))
        digits_ok += int((digits == batch["digits"]).sum())
        sums_ok += int((sums == batch["sum"]).sum())
        count += len(sums)
    return digits_ok / (count * 2 * n_digits), sums_ok / count


def run(n_digits, args, train_mnist, test_mnist):
    torch.manual_seed(args.seed)
    ontology, bp = build_model(n_digits)
    print(
        f"\nN={n_digits}: {ontology.num_atoms} atoms, {len(bp.variables)} variables, "
        f"{bp.symbolic_terms[0].num_groundings} column groundings, "
        f"{len(bp.factor_scopes)} factors over {sorted({len(s) for s in bp.factor_scopes})} variables, tree: {bp.is_tree}"
    )
    train = MNISTAddition(train_mnist, ontology, n_digits, seed=args.seed)
    test = MNISTAddition(test_mnist, ontology, n_digits, seed=args.seed + 1)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256)

    start = time.time()
    nmln.MaximumLikelihood(bp, lr=args.lr).fit(train_loader, epochs=args.epochs, accelerator=args.accelerator)
    seconds = time.time() - start
    digit_accuracy, sum_accuracy = evaluate(bp, test_loader, ontology, n_digits)
    print(f"N={n_digits}: digit accuracy {digit_accuracy:.4f}, sum accuracy {sum_accuracy:.4f}, "
          f"{len(train)} training sums, {seconds:.0f}s")
    return dict(n=n_digits, train_sums=len(train), test_sums=len(test), digit=digit_accuracy, sum=sum_accuracy,
                seconds=seconds, expected=digit_accuracy ** (2 * n_digits))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--digits", type=int, nargs="+", default=[1], help="numbers of digits N to run")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32, help="sums per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", default="auto", help="cpu, mps, gpu or auto")
    parser.add_argument("--data-dir", default=str(Path(__file__).with_name("data")))
    parser.add_argument("--out", help="write the results table (markdown) to this file")
    args = parser.parse_args()

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_mnist = torchvision.datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_mnist = torchvision.datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    results = [run(n, args, train_mnist, test_mnist) for n in args.digits]
    lines = [
        f"MNIST addition, exact BP, {args.epochs} epoch(s) over all 60k training images, batch {args.batch_size} sums, "
        f"lr {args.lr}, seed {args.seed}, accelerator {args.accelerator}.",
        "",
        "| N | training sums | test sums | digit accuracy | sum accuracy | digit acc.^(2N) | training time |",
        "|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['n']} | {r['train_sums']} | {r['test_sums']} | {r['digit']:.4f} | {r['sum']:.4f} | "
        f"{r['expected']:.4f} | {r['seconds']:.0f}s |"
        for r in results
    ]
    print("\n" + "\n".join(lines))
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
