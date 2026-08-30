import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Batch-hard Triplet Loss with Euclidean distance.
    
    L_tri = sum_i max(0, d(f_{BN,i}, f_{BN,i}^+) - d(f_{BN,i}, f_{BN,i}^-) + m)
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)
        
        # Compute pairwise Euclidean distance matrix: d(x, y) = sqrt(||x||^2 + ||y||^2 - 2 x.y)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # Mask for positive and negative pairs
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

        # Hardest positive distance: max distance among same identity
        dist_ap = []
        # Hardest negative distance: min distance among different identity
        dist_an = []

        for i in range(n):
            dist_ap.append(dist[i][is_pos[i]].max().unsqueeze(0))
            dist_an.append(dist[i][is_neg[i]].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
