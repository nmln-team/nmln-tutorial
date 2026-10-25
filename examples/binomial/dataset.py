
# Creating a simple binomial distribution example using just numpy
import numpy as np
import torch
import random



def plot_binomial( n, samples, title="Plot of Binomial Distribution"):
    binom_samples = np.sum(samples, axis=1)
    counts = np.zeros(n)
    for i in range(n):
        counts[i] = np.sum(binom_samples == i)

    import matplotlib.pyplot as plt
    plt.bar(range(n), counts / np.sum(counts))
    plt.xlabel("Number of successes")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


class NMLNBinomialDataset(torch.utils.data.Dataset):
    def __init__(self, n, p, size):

        super().__init__()
        self.n = n
        self.p = p
        self.size = size
        self.samples = np.array([[ int(random.random() < p) for _ in range(n)] for _ in range(size)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]







if __name__ == "__main__":
    n = 10  # number of constants
    p = 0.2  # probability of success
    size = 1000  # number of samples



    dataset = NMLNBinomialDataset(n, p, size)
    plot_binomial(n, dataset.samples)
