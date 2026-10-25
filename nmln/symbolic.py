"""Symbolic potentials: first-order rules, grounded here and evaluated with DeepLog.

A rule such as ``smokes(X) and friends(X,Y) -> smokes(Y)`` defines a statistic of a world: how many
of its groundings are true. Variables (names starting with an uppercase letter or ``_``) range over
the domains of the ontology; everything else inside an atom is a constant.

Rules use ``->``, ``and``, ``or``, ``not``, parentheses and ``#`` comments. Precedence, from
strongest to weakest: ``not``, ``and``, ``or``, ``->`` (which groups from the right). DeepLog has no
implication and only negates atoms, so :func:`to_deeplog` rewrites a rule into an equivalent
formula with negations pushed down to the atoms.

Grounding happens once, when the potential is built:

1. every variable gets the domain of the argument positions where it appears;
2. built-ins (``Ontology(builtins=...)``) are looked up for each binding. A built-in literal that
   makes the whole rule true (e.g. a false built-in in the body of an implication) drops that
   grounding: it is true in every world;
3. the surviving groundings read only world atoms; every one is evaluated in a batch, with the Boolean
   semantics of the connectives: min for `and`, max for `or`, 1 - v for a negated atom. On 0/1 worlds
   this is exact; it is also differentiable almost everywhere.

The evaluation is plain PyTorch by default. `backend="deeplog"` compiles the same formula with DeepLog
instead (https://github.com/ML-KULeuven/deeplog), which must then be installed.
"""

from __future__ import annotations

import re
import typing as t

import numpy as np
import torch

from nmln import utils
from nmln.potentials import Potential

_TOKEN = re.compile(r"\s*(->|\(|\)|(?!(?:and|or|not)\b)[A-Za-z0-9_]+\s*\([^()]*\)|[A-Za-z0-9_]+)")
_KEYWORDS = {"and", "or", "not"}
_DEEPLOG_SYMBOL = re.compile(r"^[^_()\s,]+(\([^()]+\))?$")


# --------------------------------------------------------------------------- rule syntax


def _tokenize(text: str) -> t.List[str]:
    text = re.sub(r"#[^\n]*", "", text).strip()
    tokens, pos = [], 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            raise SyntaxError(f"cannot parse the rule at: {text[pos:pos + 30]!r}")
        tokens.append(match.group(1))
        pos = match.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return tokens


def _atom(token: str):
    if "(" not in token:
        return token, ()
    name, args = token.split("(", 1)
    args = args.rstrip().removesuffix(")")
    return name.strip(), tuple(a.strip() for a in args.split(",")) if args.strip() else ()


def parse_rule(text: str):
    """Parse a rule into a tree of ("atom", name, args), ("not", node), ("and"/"or", [nodes])."""
    tokens = _tokenize(text)
    if not tokens:
        raise SyntaxError("empty rule")

    def peek(i):
        return tokens[i] if i < len(tokens) else None

    def implication(i):
        lhs, i = disjunction(i)
        if peek(i) == "->":
            rhs, i = implication(i + 1)
            return ("or", [("not", lhs), rhs]), i
        return lhs, i

    def chain(op, sub):
        def parse(i):
            nodes = []
            node, i = sub(i)
            nodes.append(node)
            while peek(i) == op:
                node, i = sub(i + 1)
                nodes.append(node)
            return (nodes[0] if len(nodes) == 1 else (op, nodes)), i

        return parse

    def unary(i):
        token = peek(i)
        if token == "not":
            node, i = unary(i + 1)
            return ("not", node), i
        if token == "(":
            node, i = implication(i + 1)
            if peek(i) != ")":
                raise SyntaxError("missing ')' in rule")
            return node, i + 1
        if token is None or token in _KEYWORDS or token in {")", "->"}:
            raise SyntaxError(f"expected an atom, found {token!r}")
        name, args = _atom(token)
        return ("atom", name, args), i + 1

    conjunction = chain("and", unary)
    disjunction = chain("or", conjunction)
    node, i = implication(0)
    if i != len(tokens):
        raise SyntaxError(f"unexpected {' '.join(tokens[i:])!r} in rule")
    return node


def to_nnf(node, negate: bool = False):
    """Negation normal form: ("lit", (name, args), negated) leaves under flattened "and"/"or" nodes."""
    kind = node[0]
    if kind == "atom":
        return ("lit", (node[1], node[2]), negate)
    if kind == "not":
        return to_nnf(node[1], not negate)
    op = {"and": "or", "or": "and"}[kind] if negate else kind
    children = []
    for child in (to_nnf(c, negate) for c in node[1]):
        children.extend(child[1] if child[0] == op else [child])
    return (op, children)


def _atom_text(atom) -> str:
    name, args = atom
    return f"{name}({','.join(args)})" if args else name


def _render(nnf, names, structure: str | None = "boolean") -> str:
    if nnf[0] == "lit":
        return ("not " if nnf[2] else "") + names[nnf[1]] + (f"_{structure}" if structure else "")
    parts = [_render(c, names, structure) if c[0] == "lit" else f"({_render(c, names, structure)})" for c in nnf[1]]
    return f" {nnf[0]} ".join(parts)


def _leaves(nnf, out=None):
    out = {} if out is None else out
    if nnf[0] == "lit":
        out.setdefault(nnf[1], None)
    else:
        for child in nnf[1]:
            _leaves(child, out)
    return list(out)


def normal_form(rule: str) -> str:
    """The rule as it is evaluated: implications removed, negations pushed down to the atoms.

    >>> normal_form("smokes(X) and friends(X,Y) -> smokes(Y)")
    'not smokes(X) or not friends(X,Y) or smokes(Y)'
    """
    nnf = to_nnf(parse_rule(rule))
    return _render(nnf, {a: _atom_text(a) for a in _leaves(nnf)}, structure=None)


def to_deeplog(rule: str, structure: str = "boolean") -> str:
    """The normal form of a rule in DeepLog's syntax, as compiled by backend="deeplog".

    >>> to_deeplog("smokes(X) and friends(X,Y) -> smokes(Y)")
    'not smokes(X)_boolean or not friends(X,Y)_boolean or smokes(Y)_boolean'
    """
    nnf = to_nnf(parse_rule(rule))
    return _render(nnf, {a: _atom_text(a) for a in _leaves(nnf)}, structure)


def _is_variable(arg: str) -> bool:
    return arg[:1].isupper() or arg[:1] == "_"


# --------------------------------------------------------------------------- evaluation


class _Formula(torch.nn.Module):
    """Evaluates a formula in negation normal form on leaf values [N, L]: min for and, max for or."""

    def __init__(self, nnf, leaves):
        super().__init__()
        self.nnf = nnf
        self.column = {leaf: i for i, leaf in enumerate(leaves)}

    def _evaluate(self, node, values):
        if node[0] == "lit":
            v = values[:, self.column[node[1]]]
            return 1 - v if node[2] else v
        children = torch.stack([self._evaluate(child, values) for child in node[1]], dim=-1)
        return children.amin(dim=-1) if node[0] == "and" else children.amax(dim=-1)

    def forward(self, values):
        return self._evaluate(self.nnf, values).unsqueeze(-1)


_BATCHED_KLAY = None


def _compile_deeplog(nnf, leaves):
    """Compile the rule into a DeepLog module mapping leaf values [N, L] to truth values [N, 1]."""
    global _BATCHED_KLAY
    try:
        from deeplog import DeepLogModuleFactory, SymTensor, parse_formula_to_module
        from deeplog.circuit.klay_circuit import KlayCircuit
        from deeplog.module import reshape_input
        from deeplog.symbol import str_to_symbol
    except ImportError as e:
        raise ImportError("backend='deeplog' needs DeepLog: pip install -e path/to/deeplog") from e

    if _BATCHED_KLAY is None:
        class BatchedKlayCircuit(KlayCircuit):
            # Klay circuits evaluate a whole batch natively; DeepLog's default per-row vmap is ~100x slower.
            def _to_backend_module(self):
                module, name, _ = super()._to_backend_module()
                return module, name, False

        _BATCHED_KLAY = BatchedKlayCircuit

    texts = [_atom_text(a) for a in leaves]
    if not all(_DEEPLOG_SYMBOL.match(s) for s in texts):
        texts = [f"l{i}" for i in range(len(leaves))]
    names = dict(zip(leaves, texts))
    factory = DeepLogModuleFactory()  # one factory per formula: factories keep their circuits
    factory.register_circuit_factory("boolean", lambda: _BATCHED_KLAY("boolean"))
    module = parse_formula_to_module(_render(nnf, names), factory=factory)
    symbols = [str_to_symbol(f"_({s},boolean)") for s in texts]
    return reshape_input(module, SymTensor(symbols))


# --------------------------------------------------------------------------- potential


class SymbolicPotential(Potential):
    """A first-order rule as a potential.

    Soft rule (the Markov logic case): phi(y) = average (``normalize="mean"``) or number
    (``normalize="sum"``) of true groundings, weighted by a learnable beta.
    Hard rule (``hard=True``): every false grounding costs ``utils.HARD_WEIGHT`` in log-space, so worlds
    that violate the rule have (numerically) zero probability.

    Args:
        ontology: the language; world predicates are random, built-ins are fixed
        rule: e.g. ``"smokes(X) and friends(X,Y) -> smokes(Y)"``
        beta: initial weight (ignored for hard rules)
        hard: the rule must hold in every grounding
        normalize: "mean" or "sum" of the true groundings (soft rules)
        learn_beta: whether beta is trained (soft rules)
        max_groundings: refuse to build larger intermediate grounding tables
        backend: "torch" (default) evaluates the rule in plain PyTorch, "deeplog" compiles it with DeepLog
    """

    def __init__(
        self,
        ontology,
        rule: str,
        beta: float = 1.0,
        hard: bool = False,
        normalize: str = "mean",
        learn_beta: bool = True,
        max_groundings: int = 10_000_000,
        backend: str = "torch",
    ):
        super().__init__(utils.HARD_WEIGHT if hard else beta, learn_beta=learn_beta and not hard)
        if normalize not in ("mean", "sum"):
            raise ValueError("normalize must be 'mean' or 'sum'")
        self.rule = " ".join(re.sub(r"#[^\n]*", "", rule).split())
        self.hard = bool(hard)
        self.normalize = normalize

        nnf = to_nnf(parse_rule(rule))
        leaves = _leaves(nnf)
        self.deeplog_formula = _render(nnf, {a: _atom_text(a) for a in leaves})

        world = [a for a in leaves if a[0] in ontology.predicates]
        builtin = [a for a in leaves if a[0] in ontology.builtins]
        unknown = [a[0] for a in leaves if a[0] not in ontology.predicates and a[0] not in ontology.builtins]
        if unknown:
            raise KeyError(f"unknown predicates in rule: {sorted(set(unknown))}")
        if not world:
            raise ValueError("the rule reads no world atom: it is the same in every world")

        self.variables, self._domains = self._type_variables(world + builtin, ontology)
        tables = {b.name: b.table() for b in ontology.builtins.values() if b.name in {a[0] for a in builtin}}
        disjuncts = nnf[1] if nnf[0] == "or" else [nnf]
        filters = [d for d in disjuncts if d[0] == "lit" and d[1][0] in ontology.builtins]
        bindings = self._ground(filters, tables, ontology, max_groundings)
        num_groundings = len(next(iter(bindings.values()))) if bindings else 1

        def columns(atom, domains):
            return tuple(
                bindings[a] if _is_variable(a) else np.full(num_groundings, d.index(a))
                for a, d in zip(atom[1], domains)
            )

        atoms = [
            np.asarray(ontology.indices(a[0]).numpy())[columns(a, ontology.predicates[a[0]].domains)]
            if a[1] else np.full(num_groundings, int(ontology.indices(a[0]).numpy()))
            for a in world
        ]
        values = [tables[a[0]][columns(a, ontology.builtins[a[0]].domains)] for a in builtin]
        self.register_buffer("grounding_atoms", torch.as_tensor(np.stack(atoms, -1), dtype=torch.long).reshape(num_groundings, len(world)))
        self.register_buffer("builtin_values", torch.as_tensor(np.stack(values, -1) if values else np.zeros((num_groundings, 0)), dtype=torch.float32))
        self.register_buffer("bindings", torch.as_tensor(
            np.stack([bindings[v] for v in self.variables], -1) if self.variables else np.zeros((num_groundings, 0)),
            dtype=torch.long,
        ))
        self._world_leaves, self._builtin_leaves = world, builtin
        if backend == "torch":
            self.module = _Formula(nnf, world + builtin)
        elif backend == "deeplog":
            self.module = _compile_deeplog(nnf, world + builtin)
        else:
            raise ValueError("backend must be 'torch' or 'deeplog'")
        self.backend = backend

    @staticmethod
    def _type_variables(atoms, ontology):
        domains = {}
        for name, args in atoms:
            signature = (ontology.predicates.get(name) or ontology.builtins[name]).domains
            if len(args) != len(signature):
                raise ValueError(f"{name} takes {len(signature)} arguments, the rule gives {len(args)}")
            for arg, domain in zip(args, signature):
                if not _is_variable(arg):
                    domain.index(arg)  # raises if the constant is not in the domain
                elif domains.setdefault(arg, domain) is not domain:
                    raise ValueError(f"variable {arg} is used with domains {domains[arg].name} and {domain.name}")
        return list(domains), domains

    def _ground(self, filters, tables, ontology, max_groundings):
        """Enumerate the bindings of the variables, applying the built-in filters as early as possible."""
        rows: t.Dict[str, np.ndarray] = {}
        size = 1

        def extend(variables):
            nonlocal rows, size
            for v in variables:
                n = self._domains[v].num_constants
                if size * n > max_groundings:
                    raise ValueError(
                        f"grounding {self.rule!r} needs more than {max_groundings} bindings; "
                        "add built-ins that restrict the variables or raise max_groundings"
                    )
                rows = {k: np.repeat(col, n) for k, col in rows.items()}
                rows[v] = np.tile(np.arange(n), size)
                size *= n

        def unbound(literal):
            return list(dict.fromkeys(a for a in literal[1][1] if _is_variable(a) and a not in rows))

        remaining = list(filters)
        while remaining:
            literal = min(remaining, key=lambda l: np.prod([self._domains[v].num_constants for v in unbound(l)]))
            remaining.remove(literal)
            extend(unbound(literal))
            name, args = literal[1]
            domains = ontology.builtins[name].domains
            value = tables[name][tuple(
                rows[a] if _is_variable(a) else np.full(size, d.index(a)) for a, d in zip(args, domains)
            )]
            keep = value if literal[2] else ~value  # keep the groundings where this literal is false
            rows = {k: col[keep] for k, col in rows.items()}
            size = int(keep.sum())
        extend([v for v in self.variables if v not in rows])
        return rows

    @property
    def num_groundings(self) -> int:
        return self.grounding_atoms.shape[0]

    def groundings(self, limit: int | None = 10) -> t.List[str]:
        """The rule's groundings as text (after dropping the ones built-ins make true)."""
        out = []
        for row in self.bindings[:limit].tolist():
            binding = {v: str(self._domains[v].constants[i]) for v, i in zip(self.variables, row)}
            text = self.rule
            for v in sorted(binding, key=len, reverse=True):
                text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(v)}(?![A-Za-z0-9_])", binding[v], text)
            out.append(text)
        return out

    def evaluate_groundings(self, y: torch.Tensor, groundings: torch.Tensor | None = None) -> torch.Tensor:
        """Truth value of every grounding in worlds y [..., D]: [..., G] (or [..., len(groundings)])."""
        atoms = self.grounding_atoms if groundings is None else self.grounding_atoms[groundings]
        values = y[..., atoms]  # [..., G, L]
        if self.builtin_values.shape[-1]:
            fixed = self.builtin_values if groundings is None else self.builtin_values[groundings]
            values = torch.cat([values, fixed.to(values).expand(*values.shape[:-1], fixed.shape[-1])], dim=-1)
        lead = values.shape[:-1]
        if values.numel() == 0:
            return values.new_zeros(lead)
        return self.module(values.reshape(-1, values.shape[-1])).reshape(lead)

    def aggregate(self, truth: torch.Tensor) -> torch.Tensor:
        """The statistic from grounding truth values [..., g] (any subset of the groundings)."""
        if self.hard:
            return (truth - 1).sum(dim=-1)  # minus the number of false groundings
        total = truth.sum(dim=-1)
        return total / max(self.num_groundings, 1) if self.normalize == "mean" else total

    def count_weights(self):
        """(a, b) with beta * aggregate(truth) == a * truth.sum(-1) + b * truth.shape[-1]: the weighted potential
        of any subset of groundings from its number of true groundings and its size."""
        if self.hard:
            return self.beta, -self.beta
        scale = 1.0 / max(self.num_groundings, 1) if self.normalize == "mean" else 1.0
        return self.beta * scale, torch.zeros_like(self.beta)

    def statistic(self, y, x=None):
        return self.aggregate(self.evaluate_groundings(y))

    def extra_repr(self) -> str:
        kind = "hard" if self.hard else f"normalize={self.normalize!r}"
        return f"{self.rule!r}, groundings={self.num_groundings}, {kind}"
