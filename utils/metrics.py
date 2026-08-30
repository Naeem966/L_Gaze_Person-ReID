import numpy as np
import torch

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Standard Market-1501 / DukeMTMC evaluation metric calculation.
    Computes mAP and Rank-1, Rank-5, Rank-10 accuracies.
    Excludes same-identity same-camera query-gallery matches.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc and mAP
    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        # Truncate or pad cmc curve to max_rank
        cmc_slice = cmc[:max_rank]
        if len(cmc_slice) < max_rank:
            cmc_slice = np.pad(cmc_slice, (0, max_rank - len(cmc_slice)), mode='edge')

        all_cmc.append(cmc_slice)
        num_valid_q += 1.

        # compute average precision (AP)
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.array(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        return 0.0, np.zeros(max_rank, dtype=np.float32)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return mAP, all_cmc
