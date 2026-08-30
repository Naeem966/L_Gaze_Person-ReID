import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenDiversityLoss(nn.Module):
    """
    Pairwise Token Diversity Loss over Learnable Domain Tokens.
    
    L_div = sum_{i < j} max(0, cos(c_i, c_j) - delta)
    with margin delta = 0.0
    """
    def __init__(self, delta=0.0):
        super(TokenDiversityLoss, self).__init__()
        self.delta = delta

    def forward(self, token_bank):
        """
        Args:
            token_bank: tensor of shape (K, D) containing all domain tokens
        Returns:
            scalar loss value
        """
        K = token_bank.size(0)
        if K <= 1:
            return torch.tensor(0.0, device=token_bank.device)

        # Normalize tokens to unit vectors
        tokens_norm = F.normalize(token_bank, p=2, dim=1)  # (K, D)

        # Compute pairwise cosine similarity matrix
        cos_sim = torch.matmul(tokens_norm, tokens_norm.t())  # (K, K)

        # Extract strictly upper-triangular elements (i < j)
        triu_indices = torch.triu_indices(K, K, offset=1, device=token_bank.device)
        pair_cos = cos_sim[triu_indices[0], triu_indices[1]]

        # Hinge loss with margin delta
        hinge_loss = F.relu(pair_cos - self.delta)

        # Sum over all unordered pairs
        loss = hinge_loss.sum()
        return loss
