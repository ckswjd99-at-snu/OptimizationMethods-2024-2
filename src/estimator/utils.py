import torch


def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two gradient dicts.
    """

    a_flat = torch.cat([p.view(-1) for p in a.values()])
    b_flat = torch.cat([p.view(-1) for p in b.values()])

    return torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat))
    