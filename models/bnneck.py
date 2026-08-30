import torch
import torch.nn as nn

class BNNeckHead(nn.Module):
    """
    BNNeck and Classification Head for Person Re-ID.
    
    f_BN = BN(z_cls)
    logits = W_id * f_BN
    
    Separates BN-normalized space (for metric learning and classification)
    from pre-BN space (where DFS operates).
    """
    def __init__(self, in_dim=768, num_classes=1000):
        super(BNNeckHead, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        
        self.bottleneck = nn.BatchNorm1d(in_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        
        # Initialize weights
        self.bottleneck.apply(self._weights_init_kaiming)
        self.classifier.apply(self._weights_init_classifier)

    def _weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm1d') != -1:
            if m.weight is not None:
                m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def _weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.001)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z_cls):
        """
        Args:
            z_cls: pre-BN feature vector from backbone, shape (B, in_dim)
        Returns:
            f_BN: BN-normalized feature vector, shape (B, in_dim)
            logits: identity classification logits, shape (B, num_classes)
        """
        f_BN = self.bottleneck(z_cls)
        logits = self.classifier(f_BN)
        return f_BN, logits

    def bn_only(self, features):
        """Forward BNNeck pass only."""
        return self.bottleneck(features)

    def classify_only(self, f_BN):
        """Forward classification pass only."""
        return self.classifier(f_BN)
