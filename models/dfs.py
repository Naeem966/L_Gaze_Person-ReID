import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""
    def __init__(self, dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim

    def forward(self, timesteps):
        # timesteps shape: (B,)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ResidualMLPBlock(nn.Module):
    """Residual MLP Block with FiLM conditioning and timestep injection."""
    def __init__(self, hidden_dim=512):
        super(ResidualMLPBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.time_emb_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, gamma, beta, t_emb):
        # x: (B, hidden_dim), gamma: (B, hidden_dim), beta: (B, hidden_dim), t_emb: (B, hidden_dim)
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.gelu(out + self.time_emb_proj(t_emb))
        out = self.fc2(out)
        out = self.ln2(out)
        
        # Apply FiLM modulation: gamma * out + beta
        out = gamma * out + beta
        out = F.gelu(out + residual)
        return out

class DenoiserMLP(nn.Module):
    """
    Lightweight MLP-based residual denoiser D_{\theta_{DFS}}(h_t, t; c_n).
    Input projection: FC(768 -> 512)
    4 Residual MLP blocks (hidden_dim = 512) with FiLM conditioning and timestep embedding
    Output projection: FC(512 -> 768)
    FiLM networks: gamma = FC(768 -> 512), beta = FC(768 -> 512)
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_blocks=4):
        super(DenoiserMLP, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gamma_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.out_proj = nn.Linear(hidden_dim, in_dim)
        
        # Initialize output projection near zero for residual identity start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h_t, timesteps, c_n):
        """
        Args:
            h_t: noisy feature vector at step t, shape (B, 768)
            timesteps: integer timesteps, shape (B,)
            c_n: conditioning domain token, shape (B, 768)
        Returns:
            hat_epsilon_t: predicted noise estimate, shape (B, 768)
        """
        t_emb = self.time_embed(timesteps)
        gamma = self.gamma_net(c_n)
        beta = self.beta_net(c_n)
        
        x = self.in_proj(h_t)
        for block in self.blocks:
            x = block(x, gamma, beta, t_emb)
            
        hat_epsilon = self.out_proj(x)
        return hat_epsilon

class DiffusionFeatureStylizer(nn.Module):
    """
    Diffusion-Based Feature Stylizer (DFS).
    Synthesizes domain-shifted features in latent space via a unrolled T' = 10 step DDIM chain.
    Noise schedule: linear beta from 1e-4 to 1e-2 for T = 200.
    Initial noise scale: sigma = 0.3 (t_eff approx 60).
    """
    def __init__(self, feat_dim=768, hidden_dim=512, T=200, T_prime=10, sigma=0.3, beta_start=1e-4, beta_end=1e-2):
        super(DiffusionFeatureStylizer, self).__init__()
        self.feat_dim = feat_dim
        self.T = T
        self.T_prime = T_prime
        self.sigma = sigma
        
        # Denoiser network
        self.denoiser = DenoiserMLP(in_dim=feat_dim, hidden_dim=hidden_dim, num_blocks=4)
        
        # Precompute DDIM noise schedule
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # Effective timestep for sigma = 0.3: sqrt(1 - alpha_bar_{t_eff}) = 0.3 => alpha_bar_{t_eff} = 0.91
        # Find index t_eff
        target_alpha_bar = 1.0 - (sigma ** 2)
        t_eff = torch.argmin(torch.abs(alphas_cumprod - target_alpha_bar)).item() + 1
        self.t_eff = max(t_eff, T_prime)
        
        # Sequence of timesteps for unrolled T' DDIM steps
        # From t_eff down to 1 with stride
        step_stride = self.t_eff / float(T_prime)
        timesteps = [max(1, int(round(self.t_eff - i * step_stride))) for i in range(T_prime)]
        self.timesteps_seq = timesteps

    def ddim_step(self, h_t, t_val, t_prev_val, c_n):
        """
        Single DDIM deterministic reverse step (eta = 0).
        hat_h_0 = (h_t - sqrt(1 - alpha_bar_t) * hat_epsilon_t) / sqrt(alpha_bar_t)
        h_{t-1} = sqrt(alpha_bar_{t-1}) * hat_h_0 + sqrt(1 - alpha_bar_{t-1}) * hat_epsilon_t
        """
        B = h_t.size(0)
        t_tensor = torch.full((B,), t_val, device=h_t.device, dtype=torch.long)
        
        # Get alpha_bar for current and previous timesteps
        alpha_bar_t = self.alphas_cumprod[t_val - 1]
        alpha_bar_prev = self.alphas_cumprod[t_prev_val - 1] if t_prev_val > 0 else torch.tensor(1.0, device=h_t.device)
        
        # Predict noise
        hat_epsilon_t = self.denoiser(h_t, t_tensor, c_n)
        
        # Predict clean signal hat_h_0
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        hat_h_0 = (h_t - sqrt_one_minus_alpha_bar_t * hat_epsilon_t) / (sqrt_alpha_bar_t + 1e-12)
        
        # Next state h_{t-1}
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)
        h_prev = sqrt_alpha_bar_prev * hat_h_0 + sqrt_one_minus_alpha_bar_prev * hat_epsilon_t
        return h_prev

    def forward(self, f_s, c_n):
        """
        Args:
            f_s: source feature vector, shape (B, 768)
            c_n: conditioning domain token vector, shape (B, 768)
        Returns:
            f_n: stylized domain-shifted feature vector, shape (B, 768)
        """
        B = f_s.size(0)
        
        # Initial noisy state: h_{T'} = f_s + epsilon, epsilon ~ N(0, sigma^2 I)
        epsilon = torch.randn_like(f_s) * self.sigma
        h_t = f_s + epsilon
        
        # Unroll T' DDIM reverse diffusion steps
        for i in range(len(self.timesteps_seq)):
            t_val = self.timesteps_seq[i]
            t_prev_val = self.timesteps_seq[i + 1] if (i + 1) < len(self.timesteps_seq) else 0
            h_t = self.ddim_step(h_t, t_val, t_prev_val, c_n)
            
        f_n = h_t
        return f_n

    def reconstruct(self, f_aug):
        """
        Reconstruction pass for Phase 2a warm-up using null token c = 0.
        f_n = f_aug + Delta f(f_aug, 0)
        """
        B = f_aug.size(0)
        null_c = torch.zeros_like(f_aug)
        return self.forward(f_aug, null_c)
