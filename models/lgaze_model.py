import torch
import torch.nn as nn
from .vit_backbone import ViTBackbone
from .ldt import LearnableDomainTokenBank
from .dfs import DiffusionFeatureStylizer
from .bnneck import BNNeckHead

class LGazeModel(nn.Module):
    """
    Complete L-Gaze Framework.
    Combines:
    - ViT-B/16 Backbone (with S12 stride and SIE support)
    - Learnable Domain Token (LDT) Bank (K = 50)
    - Diffusion-Based Feature Stylizer (DFS)
    - BNNeck Head & Classifier
    """
    def __init__(
        self,
        num_classes=1000,
        img_size=(256, 128),
        stride=12,
        num_tokens=50,
        feat_dim=768,
        dfs_hidden_dim=512,
        T=200,
        T_prime=10,
        sigma=0.3,
        num_cameras=0,
        num_viewpoints=0,
        sie_coeff=3.0
    ):
        super(LGazeModel, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # 1. Feature Extraction Backbone
        self.backbone = ViTBackbone(
            img_size=img_size,
            stride=stride,
            embed_dim=feat_dim,
            num_cameras=num_cameras,
            num_viewpoints=num_viewpoints,
            sie_coeff=sie_coeff
        )
        
        # 2. Learnable Domain Token Bank
        self.ldt_bank = LearnableDomainTokenBank(
            num_tokens=num_tokens, feat_dim=feat_dim
        )
        
        # 3. Diffusion-Based Feature Stylizer
        self.dfs = DiffusionFeatureStylizer(
            feat_dim=feat_dim,
            hidden_dim=dfs_hidden_dim,
            T=T,
            T_prime=T_prime,
            sigma=sigma
        )
        
        # 4. BNNeck & Classifier Head
        self.head = BNNeckHead(in_dim=feat_dim, num_classes=num_classes)

    def forward(self, x, camera_ids=None, viewpoint_ids=None):
        """
        Standard forward pass for testing/inference.
        DFS and LDT are completely disabled at test time (0 extra FLOPs / latency).
        """
        z_cls = self.backbone(x, camera_ids=camera_ids, viewpoint_ids=viewpoint_ids)
        f_BN = self.head.bn_only(z_cls)
        return f_BN

    def forward_train(self, x, camera_ids=None, viewpoint_ids=None):
        """
        Full training forward pass (Phase 3 Joint Optimization).
        
        Returns:
            z_cls: pre-BN source features f_s (B, 768)
            f_BN: BN-normalized source features (B, 768)
            logits: identity classification logits (B, C_src)
            f_n: DFS-hallucinated stylized features (B, 768)
            sampled_tokens: sampled domain tokens (B, 768)
            token_bank: static token bank (K, 768)
        """
        B = x.size(0)
        
        # 1. Extract source features f_s = z_cls
        z_cls = self.backbone(x, camera_ids=camera_ids, viewpoint_ids=viewpoint_ids)
        
        # 2. BNNeck & classification
        f_BN, logits = self.head(z_cls)
        
        # 3. Sample domain tokens c_n ~ Uniform(C)
        sampled_tokens, _ = self.ldt_bank.sample_tokens(B)
        
        # 4. Synthesize stylized features f_n via DFS unrolled DDIM chain
        f_n = self.dfs(z_cls, sampled_tokens)
        
        return {
            'z_cls': z_cls,
            'f_BN': f_BN,
            'logits': logits,
            'f_n': f_n,
            'sampled_tokens': sampled_tokens,
            'token_bank': self.ldt_bank.get_bank()
        }

    def forward_dfs_warmup(self, f_aug):
        """Phase 2a DFS reconstruction pass with null token c = 0."""
        return self.dfs.reconstruct(f_aug)
