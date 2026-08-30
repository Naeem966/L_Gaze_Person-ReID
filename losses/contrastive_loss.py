import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EntropyWeightedContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super(EntropyWeightedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def compute_entropy_weights(self, f_n, classifier, bnneck, targets, num_classes):
        """
        Computes uncertainty weight w_i = 1 - H(p(f_{n,i})) / log(C_src)
        classifier and bnneck are evaluated in no_grad mode (forward-only pass).
        """
        with torch.no_grad():
            f_n_bn = bnneck(f_n)
            logits = classifier(f_n_bn)
            probs = F.softmax(logits, dim=1)
            # Entropy H(p) = - sum_c p_c log(p_c + 1e-12)
            log_probs = torch.log(probs + 1e-12)
            entropy = - (probs * log_probs).sum(dim=1)
            
            max_entropy = math.log(max(num_classes, 2))
            weights = 1.0 - (entropy / max_entropy)
            weights = torch.clamp(weights, min=0.0, max=1.0)
        return weights

    def forward(self, f_s, f_n, targets, classifier, bnneck, num_classes):
        """
        Args:
            f_s: source pre-BN features of shape (B, D)
            f_n: hallucinated/stylized pre-BN features of shape (B, D)
            targets: identity labels of shape (B,)
            classifier: classification layer W_id
            bnneck: BNNeck layer
            num_classes: total number of source identity classes C_src
        """
        B = f_s.size(0)
        
        # Normalize features for cosine similarity
        f_s_norm = F.normalize(f_s, p=2, dim=1)
        f_n_norm = F.normalize(f_n, p=2, dim=1)

        # Compute entropy weights w_i
        w_i = self.compute_entropy_weights(f_n, classifier, bnneck, targets, num_classes)

        # Positive pair similarity for each sample i: sim(f_{s,i}, f_{n,i})
        pos_sim = (f_s_norm * f_n_norm).sum(dim=1) / self.temperature  # (B,)

        # Cross-sample similarity matrix: sim(f_{s,i}, f_{n,j}) for all i, j
        sim_matrix = torch.matmul(f_s_norm, f_n_norm.t()) / self.temperature  # (B, B)

        # Mask for cross-identity pairs: y_j != y_i
        targets_expand = targets.unsqueeze(0).expand(B, B)
        cross_id_mask = targets_expand.ne(targets_expand.t())  # True where y_j != y_i

        # Compute log-sum-exp denominator over cross-identity pairs
        # Replace non-cross-identity entries with very negative values so exp is ~0
        neg_sim_matrix = sim_matrix.masked_fill(~cross_id_mask, -1e9)
        
        # log_denom = log( sum_{j: y_j != y_i} exp(sim(f_{s,i}, f_{n,j}) / tau) )
        log_denom = torch.logsumexp(neg_sim_matrix, dim=1)  # (B,)

        # InfoNCE loss per anchor i
        loss_i = - (pos_sim - log_denom)

        # Entropy-weighted average over batch
        weighted_loss = (w_i * loss_i).mean()
        return weighted_loss
