"""Tiny molecule-generation example: SMILES -> graphs -> NMLN with a GNN potential -> graphs.

Molecules are written in Kekulé form (aromatic rings as alternating single and double bonds), so every
bond has an integer order. See tutorial/04_molecules.ipynb for a step-by-step version.
"""
import argparse
import random
from pathlib import Path
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
import nmln

torch.manual_seed(0)
random.seed(0)  # Gibbs sweeps visit the variables in a random order
N_ATOMS = 6
ATOM_TYPES = ["C", "N", "O", "F", "S", "Cl"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE"]
VALENCE_LAMBDA = 30.0
CONNECTED_LAMBDA = 10.0
OXYGEN_STEERING_LAMBDA = 10.0
MIN_OXYGEN_ATOMS = 3
RDKIT_BONDS = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
}


MAX_VALENCE = {"C": 4, "N": 3, "O": 2, "F": 1, "S": 6, "Cl": 1}
BOND_ORDER = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}


class ValencePotential(nmln.Potential):
    """Give low scores to graphs that exceed an atom's maximum valence.
    We only optimize an upper bound on valence, since we are not modeling
    explicitely the hydrogenons. A custom potential: subclass Potential and
    implement statistic(y, x); the fixed beta is the penalty per unit of excess."""

    def __init__(self, ontology, n_atoms, penalty):
        super().__init__(beta=penalty, learn_beta=False)
        allowed_valence = torch.zeros(ontology.num_atoms, n_atoms)
        used_valence = torch.zeros(ontology.num_atoms, n_atoms)

        for atom in range(n_atoms):
            for atom_type, maximum in MAX_VALENCE.items():
                allowed_valence[ontology.atom_index(atom_type, atom), atom] = maximum

            for other in range(n_atoms):
                if atom == other:
                    continue
                for bond_type, order in BOND_ORDER.items():
                    used_valence[ontology.atom_index(bond_type, atom, other), atom] = order

        self.register_buffer("allowed_valence", allowed_valence)
        self.register_buffer("used_valence", used_valence)

    def statistic(self, graph, x=None):
        excess = torch.relu(graph @ self.used_valence - graph @ self.allowed_valence)
        return -excess.sum(dim=-1)


class ConnectedPotential(nmln.Potential):
    """Give low scores to graphs that fall apart into several pieces: the statistic is -1 when
    some atom cannot be reached from atom 0 along bonds, 0 otherwise."""

    def __init__(self, ontology, n_atoms, penalty):
        super().__init__(beta=penalty, learn_beta=False)
        self.n_atoms = n_atoms
        bonds = torch.stack([ontology.indices(bond_type) for bond_type in BOND_TYPES], dim=-1)
        self.register_buffer("bond_indices", bonds)  # [n_atoms, n_atoms, bond types]

    def statistic(self, graph, x=None):
        adjacency = graph[..., self.bond_indices].sum(dim=-1) + torch.eye(self.n_atoms, device=graph.device)
        reachable = torch.linalg.matrix_power(adjacency, self.n_atoms - 1)[..., 0, :] > 0
        return -(~reachable).any(dim=-1).float()


class OxygenSteeringPotential(nmln.Potential):
    """At generation time, prefer graphs with enough oxygen atoms."""

    def __init__(self, ontology, n_atoms, minimum, penalty):
        super().__init__(beta=penalty, learn_beta=False)
        indices = [ontology.atom_index("O", atom) for atom in range(n_atoms)]
        self.register_buffer("oxygen_indices", torch.tensor(indices))
        self.minimum = float(minimum)

    def statistic(self, graph, x=None):
        oxygen_count = graph[..., self.oxygen_indices].sum(dim=-1)
        return -torch.relu(self.minimum - oxygen_count)

# The ontology says that every graph has 6 atoms, atom types, and bond types.
# Its variables: each atom has exactly one type; each pair of atoms has zero or one bond type.
atoms = nmln.Domain("atoms", [str(i) for i in range(N_ATOMS)])
atom_predicates = [nmln.Predicate(name, [atoms]) for name in ATOM_TYPES]
bond_predicates = [nmln.Predicate(name, [atoms, atoms]) for name in BOND_TYPES]
ontology = nmln.Ontology(
    [atoms],
    atom_predicates + bond_predicates,
    variables=[
        nmln.Categorical(ATOM_TYPES),
        nmln.Categorical(BOND_TYPES, optional=True, symmetric=True, irreflexive=True),
    ],
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--steer-oxygen",
    action="store_true",
    help=f"during generation, prefer molecules with at least {MIN_OXYGEN_ATOMS} oxygen atoms",
)
args = parser.parse_args()

def smiles_to_graph(smiles):
    """Turn one SMILES string into NMLN's flat graph representation (in Kekulé form)."""
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule.GetNumHeavyAtoms() == N_ATOMS
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    graph = torch.zeros(ontology.num_atoms)

    for atom in molecule.GetAtoms():
        graph[ontology.atom_index(atom.GetSymbol(), atom.GetIdx())] = 1

    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        name = str(bond.GetBondType())
        graph[ontology.atom_index(name, i, j)] = 1
        graph[ontology.atom_index(name, j, i)] = 1

    return graph

def graph_to_molecule(graph):
    """Convert one generated graph to an RDKit molecule."""
    molecule = Chem.RWMol()

    for i in range(N_ATOMS):
        atom_type = next(
            name for name in ATOM_TYPES if graph[ontology.atom_index(name, i)]
        )
        molecule.AddAtom(Chem.Atom(atom_type))

    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):
            for bond_type in BOND_TYPES:
                if graph[ontology.atom_index(bond_type, i, j)]:
                    molecule.AddBond(i, j, RDKIT_BONDS[bond_type])

    molecule = molecule.GetMol()
    try:
        Chem.SanitizeMol(molecule)
        return molecule if len(Chem.GetMolFrags(molecule)) == 1 else None
    except Chem.rdchem.MolSanitizeException:
        return None

# Read the second CSV column. All rows contain exactly 6 heavy atoms.
data_file = Path(__file__).with_name("chembl6.csv")
smiles = [line.strip().split(",")[1] for line in data_file.read_text().splitlines()[1:]]
dataset = torch.stack([smiles_to_graph(s) for s in smiles])

# The GNN learns the structure of molecules; valence and connectivity are fixed knowledge.
# Oxygen steering is not part of training.
model = nmln.NMLN(ontology, [
    nmln.GNNPotential(ontology, ATOM_TYPES, BOND_TYPES, hidden_size=32),
    ValencePotential(ontology, N_ATOMS, penalty=VALENCE_LAMBDA),
    ConnectedPotential(ontology, N_ATOMS, penalty=CONNECTED_LAMBDA),
])
# 20 sweeps between the samples of consecutive sample() calls; or nmln.Gibbs(model)
engine = nmln.ParallelTempering(model, temperatures=[1.0, 1.5, 2.5, 4.0], sweeps_per_sample=20)

# The persistent chains make the gradient noisy: a smaller learning rate at the end averages it out.
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
learner = nmln.Contrastive(engine, lr=1e-3, noise_scale=0.1)
learner.fit(loader, epochs=20)
learner.lr = 3e-4
learner.fit(loader, epochs=10)

# Add the optional energy only after training, before sampling.
if args.steer_oxygen:
    oxygen_steering = OxygenSteeringPotential(
        ontology,
        N_ATOMS,
        minimum=MIN_OXYGEN_ATOMS,
        penalty=OXYGEN_STEERING_LAMBDA,
    ).to(model.device)
    model.potentials.append(oxygen_steering)

# Generate valid molecules and save an image with some of them.
RDLogger.DisableLog("rdApp.error")
generated_molecules = []
generated_smiles = set()
num_sampled, num_valid = 0, 0
for _ in range(100):
    generated = engine.sample(32)
    for graph in generated:
        molecule = graph_to_molecule(graph)
        smiles = Chem.MolToSmiles(molecule) if molecule is not None else None
        num_sampled, num_valid = num_sampled + 1, num_valid + (molecule is not None)
        if smiles is not None and smiles not in generated_smiles:
            generated_molecules.append(molecule)
            generated_smiles.add(smiles)
    if len(generated_molecules) >= 20:
        break

image = Draw.MolsToGridImage(
    generated_molecules[:20],
    legends=[
        f"O atoms: {sum(atom.GetSymbol() == 'O' for atom in molecule.GetAtoms())}\n"
        f"{Chem.MolToSmiles(molecule)}"
        for molecule in generated_molecules[:20]
    ],
    molsPerRow=4,
    subImgSize=(250, 200),
)
mode = "steered_oxygen" if args.steer_oxygen else "unsteered"
image_path = Path(__file__).with_name(f"generated_molecules_{mode}.png")
image.save(image_path)
matching = sum(
    sum(atom.GetSymbol() == "O" for atom in molecule.GetAtoms()) >= MIN_OXYGEN_ATOMS
    for molecule in generated_molecules[:20]
)
print(
    f"{num_valid} of {num_sampled} sampled graphs are valid molecules. "
    f"Saved {len(generated_molecules[:20])} {mode} molecules to {image_path}; "
    f"{matching} have at least {MIN_OXYGEN_ATOMS} oxygen atoms"
)
