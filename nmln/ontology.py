"""The relational language: domains of constants, predicates, built-ins, and the variables of a world.

A world y is a binary vector with one entry per ground atom (the Herbrand base), in a fixed order:
predicate by predicate, and within a predicate in row-major order of its arguments. The ontology
maps atoms to positions and back, builds worlds from atoms, and declares how atoms group into
random variables (`Categorical`).
"""

import bisect
import typing as t
from collections import OrderedDict
from itertools import combinations, permutations, product

import numpy as np
import torch

from nmln.evidence import Evidence
from nmln.variables import VariableGroup, Variables


def _parse_atom(atom: str):
    """'friends(ann, bob)' -> ('friends', ['ann', 'bob'])."""
    atom = atom.strip().removesuffix(".")
    if "(" not in atom:
        return atom, []
    name, args = atom.split("(", maxsplit=1)
    args = args.strip().removesuffix(")")
    return name.strip(), [a.strip() for a in args.split(",")] if args.strip() else []


class Domain:
    def __init__(self, name: str, constants: t.List[t.Any], input_shape: t.Optional[t.Tuple[int, ...]] = None):
        """
        A finite set of constants (individuals).

        Args:
            name: name of the domain
            constants: the constants (strings or numbers)
            input_shape: if given, every constant carries an observed input of this shape (e.g. an image), which
                changes from example to example. A batch provides them as x[name], [B, num_constants, *input_shape].
        """
        self.name = name
        self.constants = list(constants)
        self.input_shape = None if input_shape is None else tuple(input_shape)
        self.constant_name_to_id = {str(c): i for i, c in enumerate(self.constants)}
        if len(self.constant_name_to_id) != len(self.constants):
            raise ValueError(f"Domain {name} has duplicate constants.")

    @property
    def num_constants(self) -> int:
        return len(self.constants)

    def index(self, constant) -> int:
        """Position of a constant, given as itself or as its string form."""
        try:
            return self.constant_name_to_id[str(constant)]
        except KeyError as e:
            raise KeyError(f"Domain {self.name} has no constant {constant!r}.") from e

    def __repr__(self) -> str:
        return f"Domain({self.name!r}, {self.num_constants} constants)"


class Predicate:
    def __init__(self, name: str, domains: t.List[Domain]):
        """A predicate (property or relation): its ground atoms are the entries of the world.

        Args:
            name: the unique name of the predicate
            domains: the domain of each argument, in order
        """
        self.name = name
        self.domains = list(domains)

    @property
    def arity(self) -> int:
        return len(self.domains)

    @property
    def num_groundings(self) -> int:
        return int(np.prod([d.num_constants for d in self.domains]))

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(d.name for d in self.domains)})"


class Categorical:
    def __init__(
        self,
        predicates: t.Union[str, t.List[str]],
        over: t.Optional[t.Union[str, int]] = None,
        optional: bool = False,
        symmetric: bool = False,
        irreflexive: bool = False,
    ):
        """
        Declares categorical random variables made of ground atoms: in every possible world, exactly one atom of
        each variable is true (at most one with optional=True).

            Categorical("digit", over="value")
                one variable per image: which constant of `value` it shows, digit(a0, 0) ... digit(a0, 9)
            Categorical(["C", "N", "O"])
                one variable per argument tuple: which of the predicates is true, C(3) or N(3) or O(3)
            Categorical(["SINGLE", "DOUBLE"], optional=True, symmetric=True, irreflexive=True)
                one variable per unordered pair of atoms: which bond links them, if any

        Atoms of predicates that appear in no declaration are independent Boolean variables.

        Args:
            predicates: one predicate name, or several with the same domains
            over: with a single predicate, the argument (domain name or position) whose constants are the values
            optional: add the value "none", where all the atoms are false
            symmetric: binary predicates: p(i, j) and p(j, i) are one variable (always equal)
            irreflexive: binary predicates: no variable for p(i, i); those atoms are always false
        """
        self.predicates = [predicates] if isinstance(predicates, str) else list(predicates)
        self.over = over
        self.optional = bool(optional)
        self.symmetric = bool(symmetric)
        self.irreflexive = bool(irreflexive)
        if not self.predicates:
            raise ValueError("Categorical needs at least one predicate")
        if over is not None and len(self.predicates) != 1:
            raise ValueError("Categorical(..., over=...) takes a single predicate")
        if over is not None and (symmetric or irreflexive):
            raise ValueError("symmetric and irreflexive are only supported without over=")
        if over is None and len(self.predicates) == 1 and not optional:
            raise ValueError(
                f"Categorical({self.predicates[0]!r}) would have a single value. Boolean predicates need no "
                "declaration; give over=... to choose among the constants of an argument."
            )

    def __repr__(self) -> str:
        options = [f"over={self.over!r}"] if self.over is not None else []
        options += [name for name in ("optional", "symmetric", "irreflexive") if getattr(self, name)]
        return f"Categorical({self.predicates!r}{''.join(', ' + o for o in options)})"

    def groups(self, ontology: "Ontology") -> t.List[VariableGroup]:
        """The variables this declaration makes in an ontology."""
        for name in self.predicates:
            if name not in ontology.predicates:
                raise KeyError(f"{self!r}: unknown predicate {name}")
        predicates = [ontology.predicates[name] for name in self.predicates]
        if self.over is not None:
            return [self._over_group(ontology, predicates[0])]
        return self._choice_groups(ontology, predicates)

    def _over_group(self, ontology, p):
        if isinstance(self.over, int):
            dim = self.over
        else:
            dims = [i for i, d in enumerate(p.domains) if d.name == self.over]
            if len(dims) != 1:
                raise ValueError(f"{self!r}: {p.name} has {len(dims)} arguments in domain {self.over!r}; give a position")
            dim = dims[0]
        index = np.moveaxis(ontology.indices(p.name).numpy(), dim, -1)
        others = [d for i, d in enumerate(p.domains) if i != dim]
        atoms = index.reshape(-1, index.shape[-1])  # [M, A]: value a of variable m is atom atoms[m, a]
        names = []
        for ids in product(*[range(d.num_constants) for d in others]):
            args = [str(d.constants[i]) for d, i in zip(others, ids)]
            args.insert(dim, "·")
            names.append(f"{p.name}({','.join(args)})")
        values = list(p.domains[dim].constants)
        patterns = np.broadcast_to(np.eye(atoms.shape[1]), (len(atoms),) + (atoms.shape[1],) * 2)
        return self._group(atoms, patterns, names, values, ontology)

    def _choice_groups(self, ontology, predicates):
        domains = [d.name for d in predicates[0].domains]
        for p in predicates[1:]:
            if [d.name for d in p.domains] != domains:
                raise ValueError(f"{self!r}: the predicates must have the same domains")
        if (self.symmetric or self.irreflexive) and (len(domains) != 2 or domains[0] != domains[1]):
            raise ValueError(f"{self!r}: symmetric and irreflexive need binary predicates over one domain")
        p0 = predicates[0]
        indices = [ontology.indices(p.name).numpy() for p in predicates]
        cells = list(product(*[range(d.num_constants) for d in p0.domains]))
        if self.irreflexive:
            cells = [c for c in cells if c[0] != c[1]]
        if self.symmetric:
            cells = [c for c in cells if c[0] <= c[1]]
        diagonal = [c for c in cells if self.symmetric and c[0] == c[1]]
        off_diagonal = [c for c in cells if not (self.symmetric and c[0] == c[1])]

        groups = []
        for part in (off_diagonal, diagonal):  # symmetric: a diagonal atom stands alone, p(i,j) and p(j,i) go together
            if not part:
                continue
            atoms, names = [], []
            for c in part:
                mirror = [c, c[::-1]] if self.symmetric and c[0] != c[1] else [c]
                atoms.append([int(index[m]) for index in indices for m in mirror])
                args = ",".join(str(d.constants[i]) for d, i in zip(p0.domains, c))
                names.append(f"{'|'.join(self.predicates)}({args})" if args else "|".join(self.predicates))
            atoms = np.asarray(atoms)  # [M, P * len(mirror)]
            per_predicate = atoms.shape[1] // len(predicates)
            patterns = np.kron(np.eye(len(predicates)), np.ones((1, per_predicate)))  # [P, K]
            patterns = np.broadcast_to(patterns, (len(atoms),) + patterns.shape)
            values = [True] if len(predicates) == 1 else list(self.predicates)
            groups.append(self._group(atoms, patterns, names, values, ontology))
        return groups

    def _group(self, atoms, patterns, names, values, ontology):
        atoms = torch.as_tensor(np.asarray(atoms), dtype=torch.long)
        patterns = torch.as_tensor(np.array(patterns), dtype=torch.float32)
        values = list(values)
        if self.optional:
            patterns = torch.cat([patterns, torch.zeros(patterns.shape[0], 1, patterns.shape[2])], dim=1)
            values.append(False if values == [True] else "none")
        return VariableGroup(self, atoms, patterns, names, values, ontology.num_atoms)


class Builtin:
    def __init__(
        self,
        name: str,
        domains: t.List[Domain],
        fn: t.Optional[t.Callable[..., bool]] = None,
        facts: t.Optional[t.Iterable[t.Tuple]] = None,
    ):
        """
        A predicate whose interpretation is fixed: the same in every possible world (background knowledge,
        e.g. arithmetic or the layout of an example). Built-ins are not random variables: they are not in the
        Herbrand base, data never provide them, and samplers never see them. SymbolicPotential uses them only
        when grounding a rule, to drop the groundings that are true in every world.

        Args:
            name: the unique name of the built-in
            domains: a positional list of domains
            fn: fn(*constants) -> bool, called with the constants themselves (e.g. ints)
            facts: alternatively, the tuples of constants for which the built-in is true
        """
        if (fn is None) == (facts is None):
            raise ValueError(f"Builtin {name}: give exactly one of fn or facts.")
        self.name = name
        self.domains = list(domains)
        self.fn = fn
        self.facts = None if facts is None else [tuple(f) for f in facts]

    @property
    def arity(self) -> int:
        return len(self.domains)

    def table(self) -> np.ndarray:
        """Truth value for every tuple of constants, a bool array of shape [|D1|, ..., |Dk|]."""
        shape = [d.num_constants for d in self.domains]
        table = np.zeros(shape, dtype=bool)
        if self.fn is not None:
            for ids in product(*[range(n) for n in shape]):
                table[ids] = bool(self.fn(*[d.constants[i] for d, i in zip(self.domains, ids)]))
        else:
            for fact in self.facts:
                if len(fact) != self.arity:
                    raise ValueError(f"Builtin {self.name}: fact {fact} does not have arity {self.arity}.")
                table[tuple(d.index(c) for d, c in zip(self.domains, fact))] = True
        return table

    def __repr__(self) -> str:
        return f"Builtin {self.name}({', '.join(d.name for d in self.domains)})"


class Ontology:
    def __init__(
        self,
        domains: t.List[Domain],
        predicates: t.List[Predicate],
        builtins: t.Optional[t.List[Builtin]] = None,
        variables: t.Optional[t.List[Categorical]] = None,
    ):
        """
        A multi-sorted, function-free first-order language, and the space of possible worlds it defines.

        Args:
            domains: the domains of the language
            predicates: the predicates of the language (their ground atoms form the world y)
            builtins: predicates with a fixed interpretation, only used to ground rules (not part of y)
            variables: Categorical declarations: how ground atoms group into random variables. Atoms of
                predicates in no declaration are independent Boolean variables.
        """
        self.domains = OrderedDict()
        for d in domains:
            if d.name in self.domains:
                raise ValueError(f"Domain {d.name} already exists")
            self.domains[d.name] = d

        self.predicates = OrderedDict()
        self._indices = {}
        self._starts = []  # first atom of each predicate, in order
        start = 0
        for p in predicates:
            if p.name in self.predicates:
                raise ValueError(f"Predicate {p.name} already exists")
            for d in p.domains:
                if self.domains.get(d.name) is not d:
                    raise ValueError(f"Predicate {p.name} uses domain {d.name}, which is not in the ontology")
            self.predicates[p.name] = p
            shape = [d.num_constants for d in p.domains]
            self._indices[p.name] = torch.arange(start, start + p.num_groundings).reshape(shape)
            self._starts.append(start)
            start += p.num_groundings
        self.num_atoms = start

        self.builtins = OrderedDict()
        for b in builtins or []:
            if b.name in self.predicates or b.name in self.builtins:
                raise ValueError(f"Predicate {b.name} already exists")
            for d in b.domains:
                if self.domains.get(d.name) is not d:
                    raise ValueError(f"Builtin {b.name} uses domain {d.name}, which is not in the ontology")
            self.builtins[b.name] = b

        self.variable_declarations = list(variables or [])
        declared, groups = set(), []
        for declaration in self.variable_declarations:
            again = declared & set(declaration.predicates)
            if again:
                raise ValueError(f"Predicates {sorted(again)} appear in two Categorical declarations")
            declared |= set(declaration.predicates)
            groups += declaration.groups(self)
        for name in self.predicates:
            if name not in declared:
                groups += Categorical(name, optional=True).groups(self)  # an independent Boolean atom
        self.variables = Variables(groups, self.num_atoms)

    def __repr__(self) -> str:
        lines = [
            f"Ontology: {self.num_atoms} ground atoms, {len(self.variables)} variables",
            "  domains:    " + ", ".join(f"{n} ({d.num_constants})" for n, d in self.domains.items()),
            "  predicates: " + ", ".join(map(repr, self.predicates.values())),
        ]
        if self.builtins:
            lines.append("  built-ins:  " + ", ".join(f"{b.name}/{b.arity}" for b in self.builtins.values()))
        if self.variable_declarations:
            lines.append("  variables:  " + ", ".join(map(repr, self.variable_declarations)))
        return "\n".join(lines)

    # ------------------------------------------------------------------ atoms and positions

    @property
    def input_domains(self):
        """The domains whose constants carry observed inputs."""
        return OrderedDict((n, d) for n, d in self.domains.items() if d.input_shape is not None)

    def indices(self, predicate: str) -> torch.Tensor:
        """The positions of a predicate's atoms in the world, LongTensor of shape [|D1|, ..., |Dk|]."""
        return self._indices[predicate].clone()

    def atom_index(self, atom: t.Union[str, t.Any], *constants) -> int:
        """Position of an atom in the world: atom_index("friends(ann,bob)") or atom_index("friends", "ann", "bob")."""
        if constants or "(" not in str(atom):
            name, args = str(atom), list(constants)
        else:
            name, args = _parse_atom(atom)
        if name not in self.predicates:
            raise KeyError(f"unknown predicate {name}")
        p = self.predicates[name]
        if len(args) != p.arity:
            raise ValueError(f"{name} takes {p.arity} arguments, got {len(args)}")
        return int(self._indices[name][tuple(d.index(c) for d, c in zip(p.domains, args))])

    def atom(self, index: int) -> str:
        """The atom at a position of the world, e.g. 'friends(ann,bob)'."""
        index = int(index)
        if not 0 <= index < self.num_atoms:
            raise IndexError(index)
        k = bisect.bisect_right(self._starts, index) - 1
        p = list(self.predicates.values())[k]
        ids = np.unravel_index(index - self._starts[k], [d.num_constants for d in p.domains])
        args = [str(d.constants[i]) for d, i in zip(p.domains, ids)]
        return f"{p.name}({','.join(args)})" if args else p.name

    # ------------------------------------------------------------------ worlds

    def world(self, atoms: t.Iterable[str]) -> torch.Tensor:
        """The world [D] in which exactly the given atoms are true."""
        y = torch.zeros(self.num_atoms)
        for atom in atoms:
            y[self.atom_index(atom)] = 1.0
        return y

    def true_atoms(self, y: torch.Tensor) -> t.List[str]:
        """The atoms that are true in a world [D]."""
        if y.ndim != 1:
            raise ValueError("true_atoms takes a single world [D]")
        return [self.atom(i) for i in torch.nonzero(y > 0.5).flatten().tolist()]

    def as_tables(self, y: torch.Tensor) -> t.Dict[str, torch.Tensor]:
        """A world [..., D] as one table per predicate: name -> [..., |D1|, ..., |Dk|]."""
        return OrderedDict((name, y[..., index.to(y.device)]) for name, index in self._indices.items())

    def mask(self, predicates: t.Iterable[str]) -> torch.Tensor:
        """Bool [D], True on all the atoms of the given predicates."""
        mask = torch.zeros(self.num_atoms, dtype=torch.bool)
        for name in [predicates] if isinstance(predicates, str) else predicates:
            mask[self._indices[name].reshape(-1)] = True
        return mask

    def observe(self, y: torch.Tensor, predicates: t.Iterable[str]) -> Evidence:
        """Evidence that observes the atoms of the given predicates in worlds y [..., D]."""
        return Evidence(y, self.mask(predicates).to(y.device).expand(y.shape).clone())

    def evidence(self, atoms: t.Dict[str, t.Union[bool, int]]) -> Evidence:
        """Evidence from atom values, e.g. {"smokes(ann)": 1, "friends(ann,bob)": 0}."""
        y = torch.zeros(self.num_atoms)
        observed = torch.zeros(self.num_atoms, dtype=torch.bool)
        for atom, value in atoms.items():
            index = self.atom_index(atom)
            y[index], observed[index] = float(value), True
        return Evidence(y, observed)

    # ------------------------------------------------------------------ fragments

    def fragment_indices(self, k: int) -> torch.Tensor:
        """Anonymized fragments of k constants (one-domain languages): LongTensor [F, k!, L].

        For every subset of k constants (F of them) and every renaming of those constants to 1..k (k! of them),
        the positions of the L atoms whose arguments all lie in the subset, in a fixed order.
        """
        if len(self.domains) != 1:
            raise ValueError("fragments are defined for languages with a single domain")
        n = next(iter(self.domains.values())).num_constants
        fragments = []
        for subset in combinations(range(n), k):
            renamings = []
            for order in permutations(subset):
                atoms = []
                for p in self.predicates.values():
                    index = self._indices[p.name].numpy()
                    for axis in range(p.arity):
                        index = np.take(index, order, axis=axis)
                    atoms.extend(index.reshape(-1))
                renamings.append(atoms)
            fragments.append(renamings)
        return torch.as_tensor(np.array(fragments), dtype=torch.long)
