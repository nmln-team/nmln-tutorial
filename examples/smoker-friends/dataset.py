import numpy as np
from torch.utils.data import Dataset


class FriendsSmokesGenerator:
    def __init__(self, ontology, weight_rule1=2.0, weight_rule2=0.5):
        self.ontology = ontology
        self.weight1 = weight_rule1
        self.weight2 = weight_rule2
        domain_map = ontology.domains["people"].constant_name_to_id
        self.id_to_name = {v: k for k, v in domain_map.items()}
        self.n = len(self.id_to_name)
        self.num_atoms = ontology.num_atoms
        self.triplets = []
        #Smokes(x) AND Friend(x, y) => Smokes(y)
        #Smokes(x)
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                name_i = self.id_to_name[i]
                name_j = self.id_to_name[j]
                id_smokes_i = ontology.atom_index(f"smokes({name_i})")
                id_friend_ij = ontology.atom_index(f"friendOf({name_i},{name_j})")
                id_smokes_j = ontology.atom_index(f"smokes({name_j})")
                self.triplets.append((id_smokes_i, id_friend_ij, id_smokes_j))


    def calculate_energy(self, world):
        score = 0
        for (s_i, f_ij, s_j) in self.triplets:
            val_s_i = world[s_i]
            val_f_ij = world[f_ij]
            val_s_j = world[s_j]

            # Rule 1 is not satisfied if (Smokes_i=1 AND Friend=1 AND Smokes_j=0)
            if val_s_i == 1 and val_f_ij == 1 and val_s_j == 0:
                score -= self.weight1  #penalty rule 1
            if val_s_i == 0:
                score -= self.weight2  #penalty rule 2
            if val_s_j == 0:
                score -= self.weight2  #penalty rule2

        return score

    def generate_dataset(self, num_samples=1000, burn_in=1000):
        samples = []
        current_world = np.random.randint(0, 2, size=self.num_atoms)
        current_score = self.calculate_energy(current_world)

        #Gibbs sampling
        for step in range(num_samples * 10 + burn_in):
            #flip a random bit
            idx_to_flip = np.random.randint(0, self.num_atoms)

            #create new point
            proposal_world = current_world.copy()
            proposal_world[idx_to_flip] = 1 - proposal_world[idx_to_flip]

            proposal_score = self.calculate_energy(proposal_world)

            #accept or reject based on metropolis rule Prob = exp(new_score - old_score)
            acceptance_prob = np.exp(proposal_score - current_score)

            if np.random.rand() < acceptance_prob:
                current_world = proposal_world
                current_score = proposal_score

            # save the sample after 10 burns sample (for independence)
            if step >= burn_in and step % 10 == 0:
                samples.append(current_world.copy())
                if len(samples) >= num_samples:
                    break

        return np.array(samples, dtype=np.float32)

def check_rule_satisfaction_rule1(samples, ontology, names):
    violations = 0
    total_relevant_cases = 0
    n = len(names)
    for world in samples:
        for i in range(n):
            for j in range(n):
                if i == j: continue
                name_i, name_j = names[i], names[j]
                idx_s_i = ontology.atom_index(f"smokes({name_i})")
                idx_f_ij = ontology.atom_index(f"friendOf({name_i},{name_j})")
                idx_s_j = ontology.atom_index(f"smokes({name_j})")
                s_i = world[idx_s_i]
                f_ij = world[idx_f_ij]
                s_j = world[idx_s_j]
                if s_i > 0.5 and f_ij > 0.5:
                    total_relevant_cases += 1
                    if s_j < 0.5:
                        violations += 1

    return violations, total_relevant_cases


def check_rule_satisfaction_rule2(samples, ontology, names):
    violations = 0
    total_relevant_cases = 0
    n = len(names)
    for world in samples:
        for i in range(n):
            name_i = names[i]
            idx_s_i = ontology.atom_index(f"smokes({name_i})")
            s_i = world[idx_s_i]
            if s_i<0.5:
                violations += 1
            total_relevant_cases += 1
    return violations, total_relevant_cases

# Wrapper per PyTorch
class SyntheticDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]