import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding with configurable stride.
    """
    def __init__(self, img_size=(256, 128), patch_size=16, stride=12, in_chans=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        
        # Calculate grid size
        self.grid_size = (
            (img_size[0] - self.patch_size[0]) // self.stride[0] + 1,
            (img_size[1] - self.patch_size[1]) // self.stride[1] + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.stride
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

class SideInformationEmbedding(nn.Module):
    """
    Side Information Embedding (SIE) module from TransReID
    Adds learned camera and viewpoint embeddings to patch embeddings with coefficient lambda_SIE = 3.0.
    """
    def __init__(self, num_cameras=15, num_viewpoints=5, embed_dim=768, sie_coeff=3.0):
        super(SideInformationEmbedding, self).__init__()
        self.sie_coeff = sie_coeff
        self.num_cameras = num_cameras
        self.num_viewpoints = num_viewpoints
        
        if num_cameras > 0:
            self.cam_embed = nn.Parameter(torch.zeros(num_cameras, embed_dim))
            nn.init.normal_(self.cam_embed, std=0.02)
        else:
            self.cam_embed = None

        if num_viewpoints > 0:
            self.view_embed = nn.Parameter(torch.zeros(num_viewpoints, embed_dim))
            nn.init.normal_(self.view_embed, std=0.02)
        else:
            self.view_embed = None

    def forward(self, patch_embeds, camera_ids=None, viewpoint_ids=None):
        # patch_embeds: (B, N, D)
        sie = 0
        if self.cam_embed is not None and camera_ids is not None:
            cam_e = self.cam_embed[camera_ids].unsqueeze(1)  # (B, 1, D)
            sie = sie + cam_e
            
        if self.view_embed is not None and viewpoint_ids is not None:
            view_e = self.view_embed[viewpoint_ids].unsqueeze(1)  # (B, 1, D)
            sie = sie + view_e
            
        if isinstance(sie, torch.Tensor):
            patch_embeds = patch_embeds + self.sie_coeff * sie
        return patch_embeds

class ViTBackbone(nn.Module):
    """
    Vision Transformer (ViT-B/16) backbone with S12 stride support, SIE, and positional embedding interpolation.
    """
    def __init__(
        self,
        img_size=(256, 128),
        patch_size=16,
        stride=12,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_cameras=0,
        num_viewpoints=0,
        sie_coeff=3.0
    ):
        super(ViTBackbone, self).__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, stride=stride, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # SIE Module
        self.sie = SideInformationEmbedding(
            num_cameras=num_cameras,
            num_viewpoints=num_viewpoints,
            embed_dim=embed_dim,
            sie_coeff=sie_coeff
        )

        # Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_embed(self, pos_embed, new_grid_size):
        """Bicubic interpolation of positional embeddings when resolution or stride changes."""
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        
        dim = pos_embed.shape[-1]
        n_patches = patch_pos.shape[1]
        old_size = int(math.sqrt(n_patches))
        
        if old_size * old_size == n_patches and old_size != new_grid_size[0]:
            patch_pos = patch_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=new_grid_size, mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
            return torch.cat([cls_pos, patch_pos], dim=1)
        return pos_embed

    def forward(self, x, camera_ids=None, viewpoint_ids=None):
        """
        Args:
            x: input image tensor (B, 3, H, W)
            camera_ids: optional tensor of camera metadata (B,)
            viewpoint_ids: optional tensor of viewpoint metadata (B,)
        Returns:
            z_cls: output [CLS] token vector (B, 768)
        """
        B = x.shape[0]
        patch_embeds = self.patch_embed(x)  # (B, N, D)

        # Apply Side Information Embedding (SIE)
        if camera_ids is not None or viewpoint_ids is not None:
            patch_embeds = self.sie(patch_embeds, camera_ids, viewpoint_ids)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x_seq = torch.cat((cls_tokens, patch_embeds), dim=1)  # (B, 1 + N, D)

        # Add positional embedding
        x_seq = x_seq + self.pos_embed

        # Transformer blocks
        feat_seq = self.blocks(x_seq)
        feat_seq = self.norm(feat_seq)

        # Extract [CLS] token output
        z_cls = feat_seq[:, 0]
        return z_cls
