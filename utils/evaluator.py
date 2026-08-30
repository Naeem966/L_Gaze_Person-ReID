import torch
import torch.nn.functional as F
import numpy as np
from .metrics import eval_func

class R1_mAP_Evaluator:
    """
    Evaluator for zero-target-domain evaluation.
    """
    def __init__(self, num_query=0, max_rank=50, feat_norm=True):
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    @torch.no_grad()
    def evaluate(self, model, dataloader, device):
        model.eval()
        feats = []
        pids = []
        camids = []

        for imgs, pid, camid, _ in dataloader:
            imgs = imgs.to(device)
            # Forward test pass: DFS and LDT are completely off
            feat = model(imgs)
            if self.feat_norm:
                feat = F.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu())
            pids.extend(pid.numpy() if isinstance(pid, torch.Tensor) else pid)
            camids.extend(camid.numpy() if isinstance(camid, torch.Tensor) else camid)

        feats = torch.cat(feats, dim=0)
        pids = np.asarray(pids)
        camids = np.asarray(camids)

        if self.num_query > 0:
            q_feats = feats[:self.num_query]
            q_pids = pids[:self.num_query]
            q_camids = camids[:self.num_query]

            g_feats = feats[self.num_query:]
            g_pids = pids[self.num_query:]
            g_camids = camids[self.num_query:]
        else:
            # Interleave per identity so query and gallery share identical PIDs across different cameras
            q_indices = []
            g_indices = []
            unique_pids = np.unique(pids)
            for u_pid in unique_pids:
                idx = np.where(pids == u_pid)[0]
                n_half = max(1, len(idx) // 2)
                q_indices.extend(idx[:n_half])
                g_indices.extend(idx[n_half:])
                
            q_indices = np.array(q_indices)
            g_indices = np.array(g_indices)

            q_feats, g_feats = feats[q_indices], feats[g_indices]
            q_pids, g_pids = pids[q_indices], pids[g_indices]
            q_camids, g_camids = camids[q_indices], camids[g_indices]

        # Pairwise Cosine Distance matrix
        distmat = 1.0 - torch.mm(q_feats, g_feats.t()).numpy()

        mAP, cmc = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)
        
        results = {
            'mAP': mAP * 100.0,
            'Rank-1': cmc[0] * 100.0 if len(cmc) > 0 else 0.0,
            'Rank-5': cmc[4] * 100.0 if len(cmc) > 4 else (cmc[-1] * 100.0 if len(cmc) > 0 else 0.0),
            'Rank-10': cmc[9] * 100.0 if len(cmc) > 9 else (cmc[-1] * 100.0 if len(cmc) > 0 else 0.0)
        }
        return results
