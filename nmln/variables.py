"""The random variables of a world.

A world is a binary vector over the ground atoms, but its random variables are groups of atoms. The
ontology declares them (`nmln.ontology.Categorical`): which digit an image shows (one atom of
digit(a0, .) is true), which type an atom of a molecule has, which bond, if any, links two atoms. Atoms
of predicates that appear in no declaration are independent Boolean variables. Atoms of declared
predicates that belong to no variable (e.g. bond(i, i) when bonds are irreflexive) are always false.

The possible worlds are all the combinations of the variables' values. Gibbs sampling resamples one
variable at a time; exact enumeration and belief propagation use them as their variables.
"""

from __future__ import annotations

from functools import cached_property

import torch


class VariableGroup:
    """The variables made by one declaration: M variables, each with A values, each over K atoms.

    atoms:    LongTensor [M, K], the atoms of each variable
    patterns: [M, A, K], the truth values of those atoms for each value of the variable
    names:    a readable name per variable, e.g. "digit(a0,·)"
    values:   a readable name per value, e.g. 0..9, or ["SINGLE", ..., "none"]
    """

    def __init__(self, declaration, atoms, patterns, names, values, num_atoms):
        self.declaration = declaration
        self.atoms = atoms
        self.patterns = patterns
        self.names = names
        self.values = values
        self.num_atoms = num_atoms

    def __len__(self):
        return len(self.atoms)

    @cached_property
    def masks(self) -> torch.Tensor:
        """Each value of each variable as a mask over the whole world: [M, A, D] (used by Gibbs sampling)."""
        masks = torch.zeros(*self.patterns.shape[:2], self.num_atoms)
        masks.scatter_(2, self.atoms.unsqueeze(1).expand_as(self.patterns), self.patterns)
        return masks

    @cached_property
    def unions(self) -> torch.Tensor:
        """The atoms of each variable as a mask over the world: [M, D]."""
        unions = torch.zeros(len(self.atoms), self.num_atoms)
        unions.scatter_(1, self.atoms, 1.0)
        return unions


class Variables:
    def __init__(self, groups, num_atoms):
        self.groups = list(groups)
        self.num_atoms = num_atoms
        self.atoms = [g.atoms for g in self.groups]  # per group: [M, K]
        self.patterns = [g.patterns for g in self.groups]  # per group: [M, A, K]
        self.offsets, cardinalities = [], []
        variable_of_atom = torch.full((num_atoms,), -1, dtype=torch.long)
        for group in self.groups:
            first = len(cardinalities)
            if bool((variable_of_atom[group.atoms] >= 0).any()):
                raise ValueError("an atom belongs to two variables")
            variable_of_atom[group.atoms] = (first + torch.arange(len(group))).unsqueeze(1).expand_as(group.atoms)
            self.offsets.append(first)
            cardinalities.extend([group.patterns.shape[1]] * len(group))
        self.cardinalities = torch.tensor(cardinalities, dtype=torch.long)
        self.variable_of_atom = variable_of_atom
        self.num_variables = len(cardinalities)

    def __len__(self):
        return self.num_variables

    def group_of(self, variable: int):
        """(group index, index of the variable within its group)."""
        for group in reversed(range(len(self.offsets))):
            if variable >= self.offsets[group]:
                return group, variable - self.offsets[group]
        raise IndexError(variable)

    def variable(self, variable: int):
        """Atoms [K] and value patterns [A, K] of one variable."""
        group, m = self.group_of(variable)
        return self.atoms[group][m], self.patterns[group][m]

    def name(self, variable: int) -> str:
        group, m = self.group_of(variable)
        return self.groups[group].names[m]

    def configs(self, y: torch.Tensor) -> torch.Tensor:
        """The value of every variable in worlds y [..., D]; -1 where y matches none of its values."""
        out = []
        for atoms, patterns in zip(self.atoms, self.patterns):
            values = y[..., atoms]  # [..., M, K]
            match = (values.unsqueeze(-2) == patterns.to(y)).all(dim=-1)  # [..., M, A]
            index = match.float().argmax(dim=-1)
            out.append(torch.where(match.any(dim=-1), index, torch.full_like(index, -1)))
        return torch.cat(out, dim=-1)

    def is_valid(self, y: torch.Tensor) -> torch.Tensor:
        """Whether worlds y [..., D] are possible worlds: every variable takes one of its values, and atoms
        outside every variable are false."""
        outside = (self.variable_of_atom < 0).to(y.device)
        return (self.configs(y) >= 0).all(dim=-1) & (y[..., outside] == 0).all(dim=-1)

    def validate(self, evidence, what: str = "evidence"):
        """Raise ValueError if observed atoms describe no possible world: no value of some variable agrees with
        them, or an atom that belongs to no variable (always false) is observed true. One device sync per call."""
        values, observed = evidence
        values = values.reshape(-1, self.num_atoms)
        observed = observed.reshape(-1, self.num_atoms).bool()
        impossible = [~allowed.any(dim=-1) for allowed in self.allowed(values, observed)]  # per group [B, M]
        outside = (self.variable_of_atom < 0).to(values.device)
        wrong = observed[:, outside] & (values[:, outside] > 0.5)  # [B, atoms outside every variable]
        if not bool(torch.stack([i.any() for i in impossible] + [wrong.any()]).any()):
            return
        for group, bad in zip(self.groups, impossible):
            if bool(bad.any()):
                b, m = bad.nonzero()[0].tolist()
                raise ValueError(f"{what} {b}: the observed atoms of {group.names[m]} match none of its values {group.values}")
        b = int(wrong.any(dim=-1).nonzero()[0])
        raise ValueError(f"{what} {b}: an atom that is always false (it belongs to no variable) is observed true")

    def worlds(self, configs: torch.Tensor) -> torch.Tensor:
        """Worlds [..., D] from the values of all variables [..., num_variables]."""
        y = torch.zeros(*configs.shape[:-1], self.num_atoms, device=configs.device)
        for offset, atoms, patterns in zip(self.offsets, self.atoms, self.patterns):
            index = configs[..., offset:offset + len(atoms)]  # [..., M]
            values = patterns.to(y.device)[torch.arange(len(atoms), device=y.device), index]  # [..., M, K]
            y[..., atoms.reshape(-1).to(y.device)] = values.reshape(*values.shape[:-2], -1)
        return y

    def allowed(self, y: torch.Tensor, observed: torch.Tensor):
        """Per group, which values agree with the observed atoms: list of bool [..., M, A]."""
        out = []
        for atoms, patterns in zip(self.atoms, self.patterns):
            values = y[..., atoms].unsqueeze(-2)  # [..., M, 1, K]
            seen = observed[..., atoms].unsqueeze(-2)
            out.append(((patterns.to(y) == values) | ~seen).all(dim=-1))
        return out

    def atom_marginals(self, beliefs) -> torch.Tensor:
        """Atom probabilities [..., D] from per-group variable beliefs [..., M, A] (probabilities)."""
        lead = beliefs[0].shape[:-2]
        p = torch.zeros(*lead, self.num_atoms, device=beliefs[0].device, dtype=beliefs[0].dtype)
        for atoms, patterns, b in zip(self.atoms, self.patterns, beliefs):
            values = torch.einsum("...ma,mak->...mk", b, patterns.to(b))
            p[..., atoms.reshape(-1).to(p.device)] = values.reshape(*lead, -1)
        return p
