"""MNIST addition data: two N-digit numbers shown as 2N MNIST images; only their sum is observed."""

import torch
from torch.utils.data import Dataset


def to_number(digits: torch.Tensor) -> torch.Tensor:
    """Digits [..., N], least significant first, to the numbers they write."""
    powers = 10 ** torch.arange(digits.shape[-1], device=digits.device)
    return (digits * powers).sum(-1)


class MNISTAddition(Dataset):
    """Every example uses 2N distinct MNIST images: a0..a{N-1} write the first number and
    b0..b{N-1} the second, least significant digit first.

    An item is a dict with
        "x":        {"images": [2N, 1, 28, 28]}, the inputs of the image constants a0..b{N-1}
        "y":        a world [D] in which the atoms res(i, s_i) of the sum's digits are true
        "observed": [D] mask, True on the res atoms: the sum is the only supervision
        "digits":   [2N] the true digit labels (only used for evaluation)
        "sum":      the true sum
    """

    def __init__(self, mnist, ontology, n_digits: int, seed: int = 0):
        self.mnist = mnist
        self.n_digits = n_digits
        order = torch.randperm(len(mnist), generator=torch.Generator().manual_seed(seed))
        usable = len(order) // (2 * n_digits) * (2 * n_digits)
        self.groups = order[:usable].view(-1, 2 * n_digits)
        self.ontology = ontology

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        images, labels = zip(*(self.mnist[j] for j in self.groups[i].tolist()))
        labels = torch.tensor(labels)
        total = to_number(labels[:self.n_digits]) + to_number(labels[self.n_digits:])
        sum_digits = [int(total) // 10 ** k % 10 for k in range(self.n_digits + 1)]
        world = self.ontology.world(f"res({k},{d})" for k, d in enumerate(sum_digits))
        observed = self.ontology.mask(["res"])  # the sum is the only supervision: True on every res atom
        return {"x": {"images": torch.stack(images)}, "y": world, "observed": observed, "digits": labels, "sum": total}
