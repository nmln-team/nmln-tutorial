import numpy as np
import nmln
from torch.utils.data import DataLoader
from dataset import FriendsSmokesGenerator, SyntheticDataset, check_rule_satisfaction_rule1, check_rule_satisfaction_rule2

list_names = ["ann", "bob", "carl", "dave", "eve", "frank"]
n = len(list_names)

people = nmln.Domain("people", list_names)
smokes = nmln.Predicate("smokes", [people])
friendOf = nmln.Predicate("friendOf", [people, people])
o = nmln.Ontology(domains=[people], predicates=[smokes, friendOf])

generator = FriendsSmokesGenerator(ontology=o, weight_rule1=0.1, weight_rule2=0.01)
raw_data = generator.generate_dataset(num_samples=200)
dataset = SyntheticDataset(raw_data)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

print(f"ORIGINAL DATA:")
real_viol, real_tot = check_rule_satisfaction_rule1(raw_data, o, list_names)
real_perc = (1 - real_viol / real_tot) * 100 if real_tot > 0 else 0
print(f"All possible cases: {real_tot}, Violation for rule 1: {real_viol}")
print(f"Percentage of rule's satisfaction for rule 1: {real_perc:.2f}%")
real_viol, real_tot = check_rule_satisfaction_rule2(raw_data, o, list_names)
real_perc = (1 - real_viol / real_tot) * 100 if real_tot > 0 else 0
print(f"All possible cases: {real_tot}, Violation for rule 2: {real_viol}")
print(f"Percentage of rule's satisfaction for rule 2: {real_perc:.2f}%")

model = nmln.NMLN(o, [nmln.WorldMLPPotential(o, hidden_size=64)])
engine = nmln.Gibbs(model, sweeps_per_sample=20)
nmln.Contrastive(engine, lr=1e-4).fit(dataloader, epochs=100)

model_samples = engine.sample(200).cpu().numpy()

print(f"NMLN results:")
model_viol, model_tot = check_rule_satisfaction_rule1(model_samples, o, list_names)
model_perc = (1 - model_viol / model_tot) * 100 if model_tot > 0 else 0
print(f"All possible cases: {model_tot}, Violations for rule 1: {model_viol}")
print(f"Percentage of rule's satisfaction for rule 1: {model_perc:.2f}%")
model_viol, model_tot = check_rule_satisfaction_rule2(model_samples, o, list_names)
model_perc = (1 - model_viol / model_tot) * 100 if model_tot > 0 else 0
print(f"All possible cases: {model_tot}, Violations for rule 2: {model_viol}")
print(f"Percentage of rule's satisfaction for rule 2: {model_perc:.2f}%")