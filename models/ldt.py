import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableDomainTokenBank(nn.Module):
    """
    Learnable Domain Tokens (LDT) Bank.
    
    Bank of K = 50 learnable vectors, each initialized c_k ~ N(0, 0.02^2 I_{768}).
    Stored in the same 768-dimensional space as backbone features.
    """
    def __init__(self, num_tokens=50, feat_dim=768, init_std=0.02):
        super(LearnableDomainTokenBank, self).__init__()
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        
        # Initialize K = 50 learnable vectors
        tokens = torch.randn(num_tokens, feat_dim) * init_std
        self.tokens = nn.Parameter(tokens)

    def sample_tokens(self, batch_size):
        """
        Samples a token uniformly at random for each item in the mini-batch.
        Returns:
            sampled_tokens: tensor of shape (batch_size, feat_dim)
            indices: sampled indices of shape (batch_size,)
        """
        indices = torch.randint(0, self.num_tokens, (batch_size,), device=self.tokens.device)
        sampled_tokens = self.tokens[indices]
        return sampled_tokens, indices

    def get_bank(self):
        """
        Returns the entire static token bank tensor of shape (K, feat_dim).
        """
        return self.tokens
