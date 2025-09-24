import torch
import torch.nn.functional as F

def mahalanobis_dist_from_features(outputs: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Euclidean distance between all row vectors in a given tensor.
    This is needed for triplet mining within a batch.

    Args:
        outputs (torch.Tensor): A tensor of shape (N, D) where N is the number of
                                items and D is the feature dimension.

    Returns:
        torch.Tensor: A symmetric distance matrix of shape (N, N).
    """
    norms_sq = torch.sum(outputs ** 2, dim=1, keepdim=True)
    dists_sq = norms_sq + norms_sq.t() - 2.0 * torch.mm(outputs, outputs.t())
    dists_sq = F.relu(dists_sq) # Ensure non-negative values before sqrt
    dists = torch.sqrt(dists_sq)
    return dists

def mahalanobis_dist_from_vectors(q_vec: torch.Tensor, g_vecs: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean distance between a single query vector and a batch of
    gallery vectors. This is used during evaluation.

    Args:
        q_vec (torch.Tensor): The query feature vector, shape (D,).
        g_vecs (torch.Tensor): The batch of gallery feature vectors, shape (N, D).

    Returns:
        torch.Tensor: A tensor of shape (N,) with distances.
    """
    q = q_vec.unsqueeze(0) if q_vec.dim() == 1 else q_vec
    q_norm = torch.sum(q**2, dim=1, keepdim=True)
    g_norm = torch.sum(g_vecs**2, dim=1, keepdim=True)
    dists_sq = q_norm + g_norm.t() - 2.0 * torch.mm(q, g_vecs.t())
    return torch.sqrt(F.relu(dists_sq)).squeeze(0)
