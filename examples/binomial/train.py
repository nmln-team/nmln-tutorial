import torch
import nmln
from examples.binomial.dataset import NMLNBinomialDataset, plot_binomial

n = 10  # number of constants

v = nmln.Domain("v", ["%d" % i for i in range(n)])
s = nmln.Predicate("s", [v])
o = nmln.Ontology(domains=[v], predicates=[s])

model = nmln.NMLN(o, [nmln.WorldMLPPotential(o, hidden_size=20)])
engine = nmln.Gibbs(model, sweeps_per_sample=20)

dataset = NMLNBinomialDataset(n=n, p=0.4, size=10000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
nmln.Contrastive(engine, lr=1e-4).fit(dataloader, epochs=1000, accelerator="cpu")

samples = engine.sample(1000).numpy()
plot_binomial(n, dataset.samples, title="Data Binomial Distribution")
plot_binomial(n, samples, title="Sample Binomial Distribution")
