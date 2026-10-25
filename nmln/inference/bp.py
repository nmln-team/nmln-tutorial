"""Belief propagation over the variables of an NMLN (see `nmln.variables`).

The potentials are translated into a factor graph internally:

- a `UnaryNeuralPotential` gives one unary factor per input constant: the network's logits;
- a `SymbolicPotential` is split into groups of groundings that read atoms of the same variables;
  each group becomes one factor over those variables, tabulated once over all their configurations (built-ins
  were already used to drop the groundings that are always true);
- evidence (observed atoms) becomes unary factors that rule out inconsistent configurations.

Sum-product gives log Z (Bethe free energy) and marginals, max-product gives MAP. When the factor
graph is a tree (`is_tree`), all of them are exact; otherwise BP runs loopy and is approximate.

Factors whose positions lie in the same variable groups (hence have equal shapes) are stacked into one
tensor, so a step of message passing is a few large tensor operations instead of one per message.
Variables that belong to a single factor (leaves, e.g. the digits of the addition) are first summed out
of all factors at once; message passing then runs on the remaining, much smaller tables (the carry
chain), and a last batched step sends messages back to the leaves. On a tree, messages are sent level
by level from a central root: up to the root, then back down.
"""

from __future__ import annotations

import math
import warnings

import torch

from nmln import utils
from nmln.potentials import UnaryNeuralPotential

from .base import Inference, batch_size_of


def _reducer(mode: str):
    """log-sum-exp (sum-product) or max (max-product) over several dimensions. The summed dimensions are moved
    last and flattened: one reduction over the last dimension is several times faster, notably on GPUs."""
    def reduce(t, dims):
        keep = [d for d in range(t.ndim) if d not in dims]
        flat = t.permute(keep + list(dims)).reshape(*[t.shape[d] for d in keep], -1)
        return torch.logsumexp(flat, dim=-1) if mode == "sum" else flat.amax(dim=-1)

    return reduce


class _AddAlong(torch.autograd.Function):
    """joint [B or 1, F, A1, ..., Ak] + message [B, F, A] broadcast along one table dimension.

    Same as the broadcast addition, but its backward sums the gradient over the other dimensions with one
    flattened reduction (autograd's default multi-dimension sum is very slow on some GPUs)."""

    @staticmethod
    def forward(ctx, joint, message, dim):
        shape = [message.shape[0], message.shape[1]] + [1] * (joint.ndim - 2)
        shape[dim] = message.shape[-1]
        ctx.dim, ctx.joint_shape = dim, joint.shape
        return joint + message.reshape(shape)

    @staticmethod
    def backward(ctx, grad):
        others = [d for d in range(2, grad.ndim) if d != ctx.dim]
        flat = grad.permute([0, 1, ctx.dim] + others).reshape(grad.shape[0], grad.shape[1], grad.shape[ctx.dim], -1)
        grad_joint = grad if grad.shape == ctx.joint_shape else grad.sum(dim=0, keepdim=True)
        return grad_joint, flat.sum(dim=-1), None


class BeliefPropagation(Inference):
    computes_log_z = True  # exactly on trees, with the Bethe approximation otherwise

    def __init__(
        self,
        model,
        max_iters: int | None = None,
        tol: float = 1e-6,
        damping: float = 0.0,
        max_table_size: int = 2 ** 20,
        strict: bool = False,
    ):
        super().__init__(model)
        self.variables = model.ontology.variables
        self.tol = float(tol)
        self.damping = float(damping)
        self.max_iters = int(max_iters) if max_iters is not None else len(self.variables) + 1
        self.max_table_size = int(max_table_size)
        self.strict = bool(strict)
        self._build()

    def _build(self):
        """(Re)build the factor graph from the model's current potentials, on the model's device."""
        from nmln.symbolic import SymbolicPotential

        model, max_table_size, device = self.model, self.max_table_size, self.model.device
        for name in [n for n in self._buffers if n.startswith("_")]:
            del self._buffers[name]

        terms = list(model.potentials)
        unsupported = [type(p).__name__ for p in terms if not isinstance(p, (UnaryNeuralPotential, SymbolicPotential))]
        if unsupported:
            raise TypeError(
                f"BeliefPropagation supports UnaryNeuralPotential and SymbolicPotential, not {unsupported}; "
                "use Gibbs sampling for other potentials"
            )
        self.unary_terms = [p for p in terms if isinstance(p, UnaryNeuralPotential)]
        self.symbolic_terms = [p for p in terms if isinstance(p, SymbolicPotential)]

        variables = self.variables
        self.num_groups = len(variables.atoms)  # variable groups, one or two per declaration
        for g, (atoms, patterns) in enumerate(zip(variables.atoms, variables.patterns)):
            self.register_buffer(f"_atoms_{g}", atoms.to(device))
            self.register_buffer(f"_patterns_{g}", patterns.to(device))
        self._location = [variables.group_of(b) for b in range(len(variables))]  # variable -> (group, index)
        cardinality = [p.shape[1] for p in variables.patterns]

        # Tabulate: (term, scope, number of true groundings per configuration, number of groundings).
        self.factor_specs, contributions = [], []
        for k, term in enumerate(self.symbolic_terms):
            variables_read = variables.variable_of_atom[term.grounding_atoms.cpu()]  # [G, L]
            groups = {}
            for g, row in enumerate(variables_read.tolist()):
                groups.setdefault(tuple(sorted({b for b in row if b >= 0})), []).append(g)
            for scope, groundings in groups.items():
                truth = self._tabulate(term, scope, groundings, max_table_size)
                contributions.append((k, scope, truth.sum(-1), len(groundings)))
                self.factor_specs.append((k, scope))

        # Factors over the same variables, or over a subset of a larger factor's variables, are merged into one:
        # parallel factors would otherwise look like cycles.
        scopes = {scope for _, scope in self.factor_specs if len(scope) >= 2}
        self.factor_scopes = sorted(s for s in scopes if not any(set(s) < set(t) for t in scopes))
        self._members = [[] for _ in range(len(variables))]
        for f, scope in enumerate(self.factor_scopes):
            for i, v in enumerate(scope):
                self._members[v].append((f, i))

        # Stack the factors whose positions lie in the same variable groups and have their leaves (variables in no
        # other factor) at the same positions.
        stacks = {}
        for f, scope in enumerate(self.factor_scopes):
            key = (tuple(self._location[b][0] for b in scope), tuple(len(self._members[b]) == 1 for b in scope))
            stacks.setdefault(key, []).append(f)
        self._signatures = [signature for signature, _ in stacks]
        self._leaves = [[i for i, leaf in enumerate(leaves) if leaf] for _, leaves in stacks]
        self._core = [[i for i, leaf in enumerate(leaves) if not leaf] for _, leaves in stacks]
        self._position = {}  # factor -> (stack, row)
        num_terms = len(self.symbolic_terms)
        for s, ((signature, _), factors) in enumerate(stacks.items()):
            for row, f in enumerate(factors):
                self._position[f] = (s, row)
            self.register_buffer(
                f"_vars_{s}", torch.tensor([[self._location[b][1] for b in self.factor_scopes[f]] for f in factors], device=device)
            )
            self.register_buffer(f"_counts_{s}", torch.zeros(num_terms, len(factors), *[cardinality[g] for g in signature], device=device))
            self.register_buffer(f"_sizes_{s}", torch.zeros(num_terms, len(factors), device=device))
        for g in range(self.num_groups):
            num_vars = len(variables.atoms[g])
            self.register_buffer(f"_unary_counts_{g}", torch.zeros(num_terms, num_vars, cardinality[g], device=device))
            self.register_buffer(f"_unary_sizes_{g}", torch.zeros(num_terms, num_vars, device=device))
            degree = torch.zeros(num_vars, device=device)
            for b in range(variables.offsets[g], variables.offsets[g] + num_vars):
                degree[b - variables.offsets[g]] = len(self._members[b])
            self.register_buffer(f"_degree_{g}", degree)
        self.register_buffer("_constant_counts", torch.zeros(num_terms, device=device))
        self.register_buffer("_constant_sizes", torch.zeros(num_terms, device=device))

        for k, scope, counts, size in contributions:
            if len(scope) == 0:
                self._constant_counts[k] += counts
                self._constant_sizes[k] += size
            elif len(scope) == 1:
                g, m = self._location[scope[0]]
                getattr(self, f"_unary_counts_{g}")[k, m] += counts
                getattr(self, f"_unary_sizes_{g}")[k, m] += size
            else:
                f = next(f for f, t in enumerate(self.factor_scopes) if set(scope) <= set(t))
                s, row = self._position[f]
                shape = [int(variables.cardinalities[b]) if b in scope else 1 for b in self.factor_scopes[f]]
                getattr(self, f"_counts_{s}")[k, row] += counts.reshape(shape)
                getattr(self, f"_sizes_{s}")[k, row] += size

        self._tree = self._check_tree()
        self._schedule = self._tree_schedule() if self._tree else [self._send(
            [(f, self._core[self._position[f][0]]) for f in range(len(self.factor_scopes))]
        )]
        self._schedule_on = {}
        self._built_for = tuple(id(p) for p in model.potentials)
        if not self._tree:
            message = "the factor graph has cycles: loopy belief propagation, log Z, marginals and MAP are approximate"
            if self.strict:
                raise ValueError(message)
            warnings.warn(message)

    def _current(self):
        """Rebuild when potentials were added to or removed from the model since the factor graph was built."""
        if tuple(id(p) for p in self.model.potentials) != self._built_for:
            self._build()

    # ------------------------------------------------------------------ factor graph

    @torch.no_grad()
    def _tabulate(self, term, scope, groundings, max_table_size):
        """Truth value of each grounding for every configuration of the scope: [A1, ..., Ak, g]."""
        cardinalities = [int(self.variables.cardinalities[b]) for b in scope]
        size = math.prod(cardinalities)
        if size > max_table_size:
            raise ValueError(
                f"a factor of {term.rule!r} spans {len(scope)} variables with {size} joint configurations "
                f"(max_table_size={max_table_size}); rewrite the rule so each grounding reads fewer variables"
            )
        configs = torch.cartesian_prod(*[torch.arange(c) for c in cardinalities]).reshape(size, -1)
        worlds = torch.zeros(size, self.variables.num_atoms, device=term.grounding_atoms.device)
        for j, b in enumerate(scope):
            atoms, patterns = self.variables.variable(b)
            worlds[:, atoms] = patterns.to(worlds)[configs[:, j]]
        truth = term.evaluate_groundings(worlds, torch.tensor(groundings, device=worlds.device))
        return truth.reshape(*cardinalities, len(groundings))

    def _weights(self):
        """Per symbolic term, (a, b) such that its log-factor is a * (#true groundings) + b * (#groundings)."""
        if not self.symbolic_terms:
            zero = torch.zeros(0, device=self._constant_counts.device)
            return zero, zero
        a, b = zip(*(term.count_weights() for term in self.symbolic_terms))
        return torch.cat(a), torch.cat(b)

    def _tables(self, weights):
        a, b = weights
        tables = []
        for s in range(len(self._signatures)):
            counts, sizes = getattr(self, f"_counts_{s}"), getattr(self, f"_sizes_{s}")
            constant = torch.tensordot(b, sizes, dims=1)
            tables.append(torch.tensordot(a, counts, dims=1) + constant.view(-1, *[1] * (counts.ndim - 2)))
        return tables  # per stack [F, A1, ..., Ak]

    def factors(self):
        """The factor graph: list of (scope, log-table [A1, ..., Ak]) for the factors over 2+ variables."""
        tables = self._tables(self._weights())
        return [(scope, tables[self._position[f][0]][self._position[f][1]]) for f, scope in enumerate(self.factor_scopes)]

    def _check_tree(self) -> bool:
        parent = list(range(len(self.variables) + len(self.factor_scopes)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for f, scope in enumerate(self.factor_scopes):
            for v in scope:
                a, b = find(v), find(len(self.variables) + f)
                if a == b:
                    return False
                parent[a] = b
        return True

    def describe(self) -> str:
        """The factor graph in words: one line per factor, with the variables it connects."""
        self._current()
        kind = "a tree: BP is exact" if self._tree else "it has cycles: loopy BP is approximate"
        lines = [f"{len(self.variables)} variables, {len(self.factor_scopes)} factors over 2 or more variables; {kind}"]
        for f, scope in enumerate(self.factor_scopes):
            lines.append(f"  factor {f}: " + ", ".join(self.variables.name(v) for v in scope))
        return "\n".join(lines)

    @property
    def is_tree(self) -> bool:
        """True if the factor graph (variables and factors over 2+ variables) has no cycles: BP is exact."""
        return self._tree

    def _depths(self, root):
        """Breadth-first from a variable: factor -> (depth, position of its parent variable), and the variables reached."""
        depth, seen, frontier, d = {}, {root}, [root], 0
        while frontier:
            nxt = []
            for v in frontier:
                for f, i in self._members[v]:
                    if f in depth:
                        continue
                    depth[f] = (d, i)
                    for j, u in enumerate(self.factor_scopes[f]):
                        if j != i and u not in seen:
                            seen.add(u)
                            nxt.append(u)
            frontier, d = nxt, d + 1
        return depth, seen

    def _tree_schedule(self):
        """Levels of a two-pass schedule on a tree: every factor sends to its parent (deepest level first),
        then to its children (root first). Each component is rooted at a variable of minimal height."""
        depth, done = {}, set()
        self._roots, self._sampling_order = [], []  # for exact sampling: roots, then (factor, parent position)
        for start in range(len(self.variables)):
            if start in done or not self._members[start]:
                continue
            _, component = self._depths(start)
            done |= component
            inner = sorted(v for v in component if len(self._members[v]) >= 2)
            if not inner:  # a single factor: only leaves, nothing to pass
                self._roots.append(start)
                self._sampling_order += list(self._depths(start)[0].items())
                continue
            candidates = inner if len(inner) <= 2000 else inner[:1]
            root = min(candidates, key=lambda v: max(d for d, _ in self._depths(v)[0].values()))
            depth.update(self._depths(root)[0])
            self._roots.append(root)
            self._sampling_order += list(self._depths(root)[0].items())
        self._sampling_order = [(f, i) for f, (_, i) in self._sampling_order]
        height = max((d for d, _ in depth.values()), default=-1) + 1

        def children(f, parent):
            return [j for j in self._core[self._position[f][0]] if j != parent]

        up = [self._send([(f, [i]) for f, (d, i) in depth.items() if d == level]) for level in reversed(range(height))]
        down = [self._send([(f, children(f, i)) for f, (d, i) in depth.items() if d == level]) for level in range(height)]
        return [step for step in up + down if step]

    def _send(self, sends):
        """One step: per stack, the rows that send, which positions they send to, and the positions involved."""
        by_stack = {}
        for f, positions in sends:
            if not positions:
                continue
            s, row = self._position[f]
            by_stack.setdefault(s, []).append((row, positions))
        step = []
        for s, items in sorted(by_stack.items()):
            commit = torch.zeros(len(items), len(self._signatures[s]), dtype=torch.bool)
            for n, (_, positions) in enumerate(items):
                commit[n, positions] = True
            rows = torch.tensor([row for row, _ in items])
            positions = [i for i in range(commit.shape[1]) if commit[:, i].any()]
            full = [bool(commit[:, i].all()) for i in range(commit.shape[1])]
            step.append((s, rows, commit, positions, full))
        return step

    def _schedule_for(self, device):
        if device not in self._schedule_on:
            self._schedule_on[device] = [
                [(s, rows.to(device), commit.to(device), positions, full) for s, rows, commit, positions, full in step]
                for step in self._schedule
            ]
        return self._schedule_on[device]

    # ------------------------------------------------------------------ evidence and unary terms

    def _unary(self, x, weights):
        """Unary log-potentials per variable group, [B, M, A], from neural terms and single-variable groundings."""
        a, b = weights
        neural = None
        if self.unary_terms:
            if x is None:
                raise ValueError("this model has neural potentials: pass the inputs x")
            neural = sum(term.beta * term.atom_weights(x) for term in self.unary_terms)  # [B, D]
        unary = []
        for g in range(self.num_groups):
            counts, sizes = getattr(self, f"_unary_counts_{g}"), getattr(self, f"_unary_sizes_{g}")
            u = (torch.tensordot(a, counts, dims=1) + torch.tensordot(b, sizes, dims=1).unsqueeze(-1)).unsqueeze(0)
            if neural is not None:
                atoms, patterns = getattr(self, f"_atoms_{g}"), getattr(self, f"_patterns_{g}")
                u = u + torch.einsum("bmk,mak->bma", neural[:, atoms], patterns)
            unary.append(u)
        constant = (a * self._constant_counts).sum() + (b * self._constant_sizes).sum()
        return unary, constant

    def _with_evidence(self, unary, evidence):
        if evidence is None:
            return unary
        device = self._constant_counts.device
        y = evidence.y.to(device).reshape(-1, evidence.y.shape[-1])
        observed = evidence.observed.to(device).reshape(y.shape)
        out = []
        for g, u in enumerate(unary):
            atoms, patterns = getattr(self, f"_atoms_{g}"), getattr(self, f"_patterns_{g}")
            values, seen = y[:, atoms].unsqueeze(-2), observed[:, atoms].unsqueeze(-2)  # [B, M, 1, K]
            allowed = ((patterns == values) | ~seen).all(dim=-1)  # [B, M, A]
            out.append(u - utils.HARD_WEIGHT * (~allowed).float())
        return out

    # ------------------------------------------------------------------ message passing

    def _incoming(self, s, rows, beliefs, messages):
        """Variable-to-factor messages into the given rows of stack s, one [B, F, A_i] per position."""
        variables = getattr(self, f"_vars_{s}")[rows]
        return [
            torch.log_softmax(beliefs[g][:, variables[:, i]] - messages[s][i][:, rows], dim=-1)
            for i, g in enumerate(self._signatures[s])
        ]

    @staticmethod
    def _joint(table, incoming):
        """Factor table times its incoming messages: [B, F, A1, ..., Ak]."""
        joint = table.unsqueeze(0)
        for i, message in enumerate(incoming):
            joint = _AddAlong.apply(joint, message, 2 + i)
        return joint

    def _run(self, unary, mode: str):
        """Message passing; returns (unary, beliefs per variable group, factor tables, full joints per stack)."""
        reduce = _reducer(mode)
        batch = max(u.shape[0] for u in unary)
        unary = [u.expand(batch, *u.shape[1:]) for u in unary]
        tables = self._tables(self._weights())
        messages = [
            [unary[g].new_zeros(batch, tables[s].shape[0], tables[s].shape[1 + i]) for i, g in enumerate(signature)]
            for s, signature in enumerate(self._signatures)
        ]
        variables = [getattr(self, f"_vars_{s}") for s in range(len(self._signatures))]

        def beliefs():
            out = list(unary)
            for s, signature in enumerate(self._signatures):
                for i, g in enumerate(signature):
                    out[g] = out[g].index_add(1, variables[s][:, i], messages[s][i])
            return out

        # 1. Sum the leaves out of every factor at once: what is left are tables over the core positions.
        reduced = []
        for s, table in enumerate(tables):
            joint = table.unsqueeze(0)
            for i in self._leaves[s]:
                leaf = torch.log_softmax(unary[self._signatures[s][i]][:, variables[s][:, i]], dim=-1)
                joint = _AddAlong.apply(joint, leaf, 2 + i)
            reduced.append(reduce(joint, [2 + i for i in self._leaves[s]]) if self._leaves[s] else joint)

        # 2. Pass messages between core variables, on the reduced tables.
        schedule = self._schedule_for(self._constant_counts.device)
        belief = beliefs()
        iterations = len(schedule) if self._tree else (self.max_iters if self.factor_scopes else 0)
        for iteration in range(iterations):
            step = schedule[iteration] if self._tree else schedule[0]
            change = 0.0
            for s, rows, commit, positions, full in step:
                core, signature = self._core[s], self._signatures[s]
                incoming = {
                    i: torch.log_softmax(belief[signature[i]][:, variables[s][rows, i]] - messages[s][i][:, rows], dim=-1)
                    for i in core
                }
                joint = reduced[s][:, rows]
                for r, i in enumerate(core):
                    joint = _AddAlong.apply(joint, incoming[i], 2 + r)
                for i in positions:
                    r = core.index(i)
                    dims = [2 + q for q in range(len(core)) if q != r]
                    new = torch.log_softmax((reduce(joint, dims) if dims else joint) - incoming[i], dim=-1)
                    old = messages[s][i][:, rows]
                    if self.damping:
                        new = torch.log_softmax((1 - self.damping) * new + self.damping * old, dim=-1)
                    if not full[i]:
                        new = torch.where(commit[:, i].view(1, -1, 1), new, old)
                    if not self._tree:
                        change = max(change, float((new - old).detach().abs().max()))
                    messages[s][i] = messages[s][i].index_copy(1, rows, new)
                    belief[signature[i]] = belief[signature[i]].index_add(1, variables[s][rows, i], new - old)
            if not self._tree and change < self.tol:
                break
        else:
            if self.factor_scopes and not self._tree:
                warnings.warn("loopy belief propagation did not converge; results are approximate")

        # 3. Messages to the leaves, and the full joints (needed by the Bethe free energy).
        joints = []
        for s, table in enumerate(tables):
            rows = torch.arange(table.shape[0], device=table.device)
            incoming = self._incoming(s, rows, belief, messages)
            joint = self._joint(table, incoming)
            for i in self._leaves[s]:
                dims = [2 + j for j in range(len(incoming)) if j != i]
                messages[s][i] = torch.log_softmax(reduce(joint, dims) - incoming[i], dim=-1)
            joints.append(joint)
        return unary, beliefs(), tables, joints

    def _bethe(self, unary, beliefs, tables, joints, constant):
        """Bethe approximation of log Z (exact on trees)."""
        log_z = constant
        for table, joint in zip(tables, joints):
            log_b = torch.log_softmax(joint.flatten(2), dim=-1)
            log_z = log_z + (log_b.exp() * (table.flatten(1).unsqueeze(0) - log_b)).sum(dim=(-1, -2))
        for g, (u, belief) in enumerate(zip(unary, beliefs)):
            log_b = torch.log_softmax(belief, dim=-1)
            b = log_b.exp()
            degree = getattr(self, f"_degree_{g}").view(1, -1, 1)
            log_z = log_z + (b * u).sum(dim=(-1, -2)) + ((degree - 1) * b * log_b).sum(dim=(-1, -2))
        return log_z

    def _output(self, value, batch):
        return value.squeeze(0) if batch is None else value.expand(batch, *value.shape[1:])

    # ------------------------------------------------------------------ public API

    @property
    def exact(self) -> bool:
        """Exact on tree-shaped factor graphs; approximate (loopy BP) otherwise."""
        return self._tree

    def _prepare(self, x, evidence, validate: bool = True):
        self._current()
        evidence = self._evidence(evidence, validate)
        unary, constant = self._unary(x, self._weights())
        return batch_size_of(x, evidence), unary, constant, evidence

    def log_partition(self, *, x=None, evidence=None):
        """log Z(x), or with evidence the log-mass of the worlds that agree with it: [] or [B]."""
        batch, unary, constant, evidence = self._prepare(x, evidence)
        return self._output(self._bethe(*self._run(self._with_evidence(unary, evidence), "sum"), constant), batch)

    def _log_probability(self, evidence, x, validate: bool):
        """log p(evidence | x) = log Z(x, evidence) - log Z(x): [] or [B]."""
        batch, unary, constant, evidence = self._prepare(x, evidence, validate)
        free = self._bethe(*self._run(unary, "sum"), constant)
        clamped = self._bethe(*self._run(self._with_evidence(unary, evidence), "sum"), constant)
        return self._output(clamped - free, batch)

    @torch.no_grad()
    def variable_marginals(self, *, x=None, evidence=None):
        """Belief of every variable, per group: list of [B, M, A] probabilities."""
        _, unary, _, evidence = self._prepare(x, evidence)
        _, beliefs, _, _ = self._run(self._with_evidence(unary, evidence), "sum")
        return [torch.softmax(b, dim=-1) for b in beliefs]

    @torch.no_grad()
    def marginals(self, *, x=None, evidence=None):
        """Probability that each ground atom is true: [D], or [B, D]."""
        beliefs = self.variable_marginals(x=x, evidence=evidence)
        return self._output(self.variables.atom_marginals(beliefs), batch_size_of(x, evidence))

    @torch.no_grad()
    def map(self, *, x=None, evidence=None):
        """Most probable world (max-product): [D], or [B, D]."""
        batch, unary, _, evidence = self._prepare(x, evidence)
        _, beliefs, _, _ = self._run(self._with_evidence(unary, evidence), "max")
        configs = torch.cat([b.argmax(dim=-1) for b in beliefs], dim=-1)  # [B, num_variables], in variable order
        return self._output(self.variables.worlds(configs), batch)

    @torch.no_grad()
    def sample(self, n: int = 1, *, x=None, evidence=None):
        """Exact, independent samples (tree-shaped factor graphs): [n, D], or [B, n, D].

        After message passing, a root is drawn from its marginal; then, going down the tree, the other variables
        of every factor are drawn jointly from their distribution given the variable above them.
        """
        self._current()
        if not self._tree:
            raise NotImplementedError("exact sampling needs a tree-shaped factor graph; use Gibbs sampling")
        batch, unary, _, evidence = self._prepare(x, evidence)
        n = int(n)
        unary = self._with_evidence(unary, evidence)
        unary = [u.repeat_interleave(n, dim=0) if u.shape[0] > 1 else u.expand(n, *u.shape[1:]) for u in unary]
        _, beliefs, _, joints = self._run(unary, "sum")
        size = beliefs[0].shape[0]
        values = torch.full((size, len(self.variables)), -1, dtype=torch.long, device=beliefs[0].device)

        def draw(v):
            g, m = self._location[v]
            values[:, v] = torch.distributions.Categorical(logits=beliefs[g][:, m]).sample()

        for root in self._roots:
            draw(root)
        for f, parent in self._sampling_order:
            s, row = self._position[f]
            scope = self.factor_scopes[f]
            joint = joints[s][:, row]
            joint = joint.movedim(parent + 1, 1)  # [N, A_parent, the others...]
            index = values[:, scope[parent]].view(size, 1, *[1] * (joint.ndim - 2)).expand(size, 1, *joint.shape[2:])
            conditional = joint.gather(1, index).squeeze(1).reshape(size, -1)
            choice = torch.distributions.Categorical(logits=conditional).sample()
            others = [j for j in range(len(scope)) if j != parent]
            for j in reversed(others):
                cardinality = int(self.variables.cardinalities[scope[j]])
                values[:, scope[j]] = choice % cardinality
                choice = choice // cardinality
        for v in (values[0] < 0).nonzero().flatten().tolist():  # variables in no factor
            draw(v)
        worlds = self.variables.worlds(values)
        return worlds.reshape(-1, n, worlds.shape[-1]) if batch is not None else worlds
