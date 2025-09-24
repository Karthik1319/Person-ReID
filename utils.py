import numpy as np
from typing import Tuple

def compute_ap_cmc(index: np.ndarray, good_index: np.ndarray, junk_index: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculates Average Precision (AP) and Cumulative Matching Characteristics (CMC).

    Args:
        index (np.ndarray): A 1D array of ranked gallery indices (size G).
        good_index (np.ndarray): A 1D array of indices for 'good' gallery matches.
        junk_index (np.ndarray): A 1D array of indices for 'junk' gallery matches to be ignored.

    Returns:
        A tuple containing:
        - ap (float): The Average Precision score.
        - cmc (np.ndarray): A binary vector of size G for the CMC curve.
    """
    num_good = len(good_index)
    if num_good == 0:
        return 0.0, np.zeros(len(index), dtype=int)

    mask_junk = np.in1d(index, junk_index, invert=True)
    index_clean = index[mask_junk]
    
    good_mask = np.in1d(index_clean, good_index)
    positions = np.where(good_mask)[0]
    
    if positions.size == 0:
        return 0.0, np.zeros(len(index), dtype=int)

    cmc = np.zeros(len(index), dtype=int)

    first_match_idx_in_clean = positions[0]
    original_rank_of_first_match = np.where(index == index_clean[first_match_idx_in_clean])[0][0]
    cmc[original_rank_of_first_match:] = 1
    
    total_precision = np.sum((np.arange(len(positions)) + 1) / (positions + 1))
    ap = total_precision / num_good
    return ap, cmc

def evaluate_ranked_list(ranking: np.ndarray, gt_labels: np.ndarray, gt_cams: np.ndarray, q_label: int, q_cam: int) -> Tuple[float, np.ndarray]:
    """
    Computes Average Precision (AP) and CMC for a single query.
    """
    matches = np.where(gt_labels == q_label)[0]
    good_index = matches[gt_cams[matches] != q_cam]
    junk_mask_1 = gt_cams[matches] == q_cam
    junk_index_1 = matches[junk_mask_1]
    junk_index_2 = np.where(gt_labels == -1)[0]
    junk_index = np.concatenate([junk_index_1, junk_index_2])
    return compute_ap_cmc(ranking, good_index, junk_index)
