"""Potentials: the features of an NMLN, p(y | x) ∝ exp(Σ_k beta_k · statistic_k(y, x)).

Every potential implements `statistic(y, x=None)`: one value per world y [..., D] (x are the inputs
attached to constants, when the language has any). Calling a potential returns beta * statistic.

- `UnaryNeuralPotential`: a neural predicate, one factor per input constant (e.g. a CNN reading an image);
- `FragmentPotential`: an MLP on every anonymized fragment of k constants, averaged over fragments;
- `GNNPotential`: a relational graph network on the world seen as a graph;
- `WorldMLPPotential`: an MLP on the whole world vector;
- `nmln.symbolic.SymbolicPotential`: a first-order rule, as in Markov logic.
"""

import typing as t

import torch


class Potential(torch.nn.Module):
    """Base class: subclasses implement `statistic(y, x=None)`; `forward` multiplies it by beta."""

    def __init__(self, beta: float = 1.0, learn_beta: bool = True):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor([float(beta)]), requires_grad=learn_beta)

    def statistic(self, y: torch.Tensor, x=None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, y: torch.Tensor, x=None) -> torch.Tensor:
        value = self.statistic(y, x)
        if value.shape == y.shape[:-1] + (1,):
            value = value.squeeze(-1)
        return self.beta[0] * value


class UnaryNeuralPotential(Potential):
    """Unary neural potential (neural predicate): connects observed inputs to the world.

    For a predicate p(C, V), where the constants of the domain C carry inputs (Domain(..., input_shape=...))
    and V is a finite domain of values,

        statistic(y, x) = sum_c sum_v  y[p(c, v)] * f(x_c)[v],

    i.e. one factor per constant c, over the atoms p(c, .). Declared as Categorical(p, over=V), these atoms
    form one categorical variable and f(x_c) are its logits. For a unary predicate p(C), f returns a single
    logit per constant: statistic(y, x) = sum_c y[p(c)] * f(x_c).

    The same network f is shared by all constants (the k=1 case of anonymization). Being linear in y is no
    restriction: on a one-hot variable, every function of y is linear in y. Beta is fixed to 1 (the scale is
    absorbed in the logits).
    """

    def __init__(self, ontology, predicate: str, network: torch.nn.Module):
        super().__init__(beta=1.0, learn_beta=False)
        p = ontology.predicates[predicate]
        domain = p.domains[0]
        if domain.input_shape is None:
            raise ValueError(
                f"The first domain of {predicate}, {domain.name}, has no input_shape: its constants carry no inputs."
            )
        if p.arity > 2:
            raise ValueError("UnaryNeuralPotential supports p(C) and p(C, V) predicates.")
        self.predicate = predicate
        self.input_domain = domain.name
        self.network = network
        self.register_buffer("atom_index", ontology.indices(predicate).reshape(domain.num_constants, -1))  # [n, V]
        self.num_atoms = ontology.num_atoms
        self._cache = None

    def logits(self, x) -> torch.Tensor:
        """Network outputs for every constant, [B, n, V]."""
        if x is None:
            raise ValueError(f"{type(self).__name__} on {self.predicate} needs the inputs x")
        inputs = x[self.input_domain]
        batch, n = inputs.shape[:2]
        versions = tuple(q._version for q in self.network.parameters())
        if not torch.is_grad_enabled() and self._cache is not None:
            cached_inputs, cached_versions, cached = self._cache
            if cached_inputs is inputs and cached_versions == versions:
                return cached
        out = self.network(inputs.reshape(batch * n, *inputs.shape[2:])).reshape(batch, n, -1)
        if out.shape[-1] != self.atom_index.shape[-1]:
            raise ValueError(
                f"network returns {out.shape[-1]} logits per constant, {self.predicate} needs {self.atom_index.shape[-1]}"
            )
        # Gibbs evaluates the potential many times on the same inputs: cache when no gradient is needed.
        self._cache = (inputs, versions, out) if not torch.is_grad_enabled() else None
        return out

    def atom_weights(self, x) -> torch.Tensor:
        """The statistic as a weight per ground atom, [B, D]: statistic(y, x) = sum_i y_i * w_i."""
        logits = self.logits(x)
        weights = logits.new_zeros(logits.shape[0], self.num_atoms)
        weights[:, self.atom_index.reshape(-1)] = logits.reshape(logits.shape[0], -1)
        return weights

    def statistic(self, y, x=None):
        logits = self.logits(x)  # [B, n, V]
        atoms = y[..., self.atom_index]  # [B, ..., n, V]
        logits = logits.view(logits.shape[0], *([1] * (atoms.ndim - 3)), *logits.shape[1:])
        return (atoms * logits).sum(dim=(-1, -2))


class FragmentPotential(Potential):
    """A neural potential over fragments of k constants (the original NMLN potential).

    1. Fragments: for every subset S of k constants, the fragment y_S keeps the atoms whose arguments all lie
       in S (`Ontology.fragment_indices`).
    2. Anonymization: each of the k! renamings of S to 1..k gives a vector; the same MLP scores every one and
       the scores are summed, so the fragment's score does not depend on the names of its constants.
    3. Global statistic: the average of the fragment scores over all subsets.

    Single-domain languages. MLP: Linear(L, h1) - ReLU - Linear(h1, h2) - ReLU - Linear(h2, 1), with
    (h1, h2) = (150, 50) as in the molecule experiments of the thesis.
    """

    def __init__(self, ontology, k: int, hidden_sizes: t.Tuple[int, int] = (150, 50), beta: float = 1.0):
        super().__init__(beta)
        self.k = k
        self.register_buffer("fragments", ontology.fragment_indices(k))  # [F, k!, L]
        h1, h2 = hidden_sizes
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.fragments.shape[-1], h1), torch.nn.ReLU(),
            torch.nn.Linear(h1, h2), torch.nn.ReLU(),
            torch.nn.Linear(h2, 1),
        )

    def statistic(self, y, x=None):
        scores = self.mlp(y[..., self.fragments]).squeeze(-1)  # [..., F, k!]
        return scores.sum(dim=-1).mean(dim=-1)


class _RelationalGraphConv(torch.nn.Module):
    """One R-GCN layer with basis decomposition (Schlichtkrull et al., 2018):
    h_i' = sum_r sum_{j in N_r(i)} W_r h_j / c_{i,r} + W_0 h_i, with W_r = sum_b a_{rb} V_b."""

    def __init__(self, in_size: int, out_size: int, num_relations: int, num_bases: int):
        super().__init__()
        bases = min(num_bases, num_relations)
        self.basis = torch.nn.Parameter(torch.empty(bases, in_size, out_size))
        self.coeffs = torch.nn.Parameter(torch.empty(num_relations, bases))
        self.self_loop = torch.nn.Linear(in_size, out_size)
        for b in range(bases):
            torch.nn.init.xavier_uniform_(self.basis[b])
        torch.nn.init.xavier_uniform_(self.coeffs)

    def forward(self, h, adjacency):  # h [..., N, H], adjacency [..., N, N, R]
        weights = torch.einsum("rb,bio->rio", self.coeffs, self.basis)
        degree = adjacency.sum(dim=-2).clamp(min=1.0)
        adjacency = adjacency / degree.unsqueeze(-2)
        messages = torch.einsum("...ijr,...jk->...irk", adjacency, h)
        return torch.einsum("...irk,rko->...io", messages, weights) + self.self_loop(h)


class GNNPotential(Potential):
    """A graph neural network on the world seen as a graph (R-GCN message passing, then sum pooling).

    Nodes are the constants of a single domain. Unary predicates give the node features (e.g. atom types);
    binary predicates are edge relations (e.g. bond types), plus one relation for "no edge". The score is
    invariant to renaming the constants, like a fragment potential, but looks at the whole world.
    """

    def __init__(
        self,
        ontology,
        node_predicates: t.List[str],
        edge_predicates: t.List[str],
        hidden_size: int = 64,
        num_layers: int = 2,
        num_bases: int = 2,
        beta: float = 1.0,
    ):
        super().__init__(beta)
        names = list(node_predicates) + list(edge_predicates)
        domains = {d.name for name in names for d in ontology.predicates[name].domains}
        if len(domains) != 1:
            raise ValueError("GNNPotential needs node and edge predicates over one domain")
        for name in node_predicates:
            if ontology.predicates[name].arity != 1:
                raise ValueError(f"node predicate {name} must be unary")
        for name in edge_predicates:
            if ontology.predicates[name].arity != 2:
                raise ValueError(f"edge predicate {name} must be binary")
        self.num_nodes = ontology.domains[domains.pop()].num_constants
        nodes = (torch.stack([ontology.indices(n) for n in node_predicates], -1) if node_predicates
                 else torch.zeros(self.num_nodes, 0, dtype=torch.long))
        self.register_buffer("node_index", nodes)  # [N, P]
        self.register_buffer("edge_index", torch.stack([ontology.indices(e) for e in edge_predicates], -1))  # [N, N, E]
        relations = len(edge_predicates) + 1
        self.embed = torch.nn.Linear(max(len(node_predicates), 1), hidden_size, bias=False)
        self.layers = torch.nn.ModuleList(
            [_RelationalGraphConv(hidden_size, hidden_size, relations, num_bases) for _ in range(num_layers)]
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU(), torch.nn.Linear(hidden_size // 2, 1)
        )

    def statistic(self, y, x=None):
        if self.node_index.shape[-1]:
            nodes = y[..., self.node_index]
        else:
            nodes = y.new_ones(*y.shape[:-1], self.num_nodes, 1)
        edges = y[..., self.edge_index]  # [..., N, N, E]
        no_edge = (1.0 - edges.sum(dim=-1)).clamp(min=0.0) * (1.0 - torch.eye(self.num_nodes, device=y.device))
        adjacency = torch.cat([edges, no_edge.unsqueeze(-1)], dim=-1)
        h = self.embed(nodes)
        for layer in self.layers:
            h = torch.relu(layer(h, adjacency))
        return self.readout(h).squeeze(-1).sum(dim=-1)


class WorldMLPPotential(Potential):
    """An MLP on the whole world vector. The most general neural potential: it knows nothing about constants,
    so, unlike FragmentPotential and GNNPotential, it is not invariant to renaming them."""

    def __init__(self, ontology, hidden_size: int = 64, beta: float = 1.0):
        super().__init__(beta)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(ontology.num_atoms, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, 1)
        )

    def statistic(self, y, x=None):
        return self.mlp(y).squeeze(-1)
