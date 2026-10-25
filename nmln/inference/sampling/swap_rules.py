import torch


def metropolis_swap_rule(chains, i, j, x=None):
    """Metropolis acceptance of swapping the states of chains i[k] and j[k] (each with its own target).

    x are the inputs of the worlds. Returns a boolean tensor with shape [num_pairs, batch, 1].
    """
    masks = []
    for a, b in zip(i.tolist(), j.tolist()):
        ca, cb = chains[a], chains[b]
        ya, yb = ca.states, cb.states
        log_alpha = ca.log_prob(yb, x) + cb.log_prob(ya, x) - ca.log_prob(ya, x) - cb.log_prob(yb, x)
        u = torch.rand_like(log_alpha).clamp_min(1e-12)
        masks.append(torch.log(u) < log_alpha)
    return torch.stack(masks, dim=0)
