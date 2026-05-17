import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ProbCrossAttention(nn.Module):
    """
    Simple and stable probabilistic cross-attention:
      - keys/values each have mean + variance
      - softplus to keep variance positive
      - attention scores adjusted by key uncertainty
    """
    def __init__(self, dim, beta: float = 2.35, gate_init: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * 2)  # mean + logvar
        self.v_proj = nn.Linear(dim, dim * 2)  # mean + logvar
        self.out_proj = nn.Linear(dim, dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.eps = 1e-6
        self.gate = nn.Parameter(torch.tensor(gate_init))  # Initial residual mix
        self.beta = beta

    def forward(self, query, context, sample=True, num_samples=1):
        B, Tq, C = query.shape
        _, Tk, _ = context.shape

        Q = self.q_proj(query)  # [B, Tq, C]

        # Keys: mean + variance
        K_out = self.k_proj(context)
        K_mu, K_logvar = K_out[..., :C], K_out[..., C:]
        K_mu = self.norm_k(K_mu)
        K_var = F.softplus(K_logvar) + self.eps  # positive

        # Values: mean + variance
        V_out = self.v_proj(context)
        V_mu, V_logvar = V_out[..., :C], V_out[..., C:]
        V_mu = self.norm_v(V_mu)
        V_var = F.softplus(V_logvar) + self.eps  # positive

        # Attention scores
        scale = math.sqrt(C)
        mean_scores = torch.matmul(Q, K_mu.transpose(1, 2)) / scale

        var_penalty = torch.matmul(Q.pow(2), K_var.transpose(1, 2)) / C
        scores = mean_scores - self.beta * torch.sqrt(var_penalty)
    
        attn_weights = F.softmax(scores, dim=-1)

        eps = torch.randn_like(V_var)
        V_sample = V_mu + torch.sqrt(V_var) * eps
        out = torch.matmul(attn_weights, V_sample) # average over samples

        gate = torch.sigmoid(self.gate)
        proj_out = self.out_proj(out)
        fused = gate * proj_out + (1 - gate) * query

        return fused

class TwoWayTransformerLayer(nn.Module):
    def __init__(self, embed_dim, beta=2.35, gate_init=0.0):
        super().__init__()
        self.cross_attn_img_to_txt = ProbCrossAttention(embed_dim, beta, gate_init)
        self.cross_attn_txt_to_img = ProbCrossAttention(embed_dim, beta, gate_init)

    def forward(self, img_tokens, txt_tokens):
        img_tokens = self.cross_attn_img_to_txt(img_tokens, txt_tokens)
        txt_tokens = self.cross_attn_txt_to_img(txt_tokens, img_tokens)
        return img_tokens, txt_tokens

class PVL_Adapter(nn.Module):
    def __init__(self,
                 in_channels_vis: int,
                 in_channels_txt: int,
                 adapter_channels: int,
                 beta: float,
                 gate_init: int):
        
        super().__init__()

        # Down projection
        self.proj_vis_down = nn.Sequential(nn.Linear(in_channels_vis, adapter_channels, bias=False))
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)

        # Up projection
        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

        # Cross-modal interaction
        self.two_way = TwoWayTransformerLayer(adapter_channels, beta, gate_init)

    def forward(self, vis, text):

        v = self.proj_vis_down(vis)
        t = self.proj_txt_down(text)

        v_fused, t_fused = self.two_way(v, t)

        vis_out = self.proj_vis_up(v_fused)
        txt_out = self.proj_txt_up(t_fused)

        return vis_out, txt_out
# ============================================================
# Extra modules for improved MedCLIPSeg:
# Prompt-Guided CBAM + Lightweight PVL + Boundary-aware decoder
# ============================================================

class LightweightCrossAttention(nn.Module):
    """
    Lightweight cross-attention used inside the lightweight PVL adapter.
    It returns only a residual delta, so the caller can do:
        tokens = tokens + delta
    """
    def __init__(self, dim, num_heads=4, dropout=0.0, gate_init=-2.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, query_tokens, context_tokens):
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(context_tokens)

        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.out(out)

        return torch.sigmoid(self.gate) * out


class LightweightPVLAdapter(nn.Module):
    """
    Lightweight replacement for the original PVL_Adapter.

    Input:
        vis_tokens:  [B, Nv, Cvis]
        txt_tokens:  [B, Nt, Ctxt]

    Output:
        vis_delta:   [B, Nv, Cvis]
        txt_delta:   [B, Nt, Ctxt]
    """
    def __init__(
        self,
        in_channels_vis: int,
        in_channels_txt: int,
        adapter_channels: int,
        beta: float = 2.35,
        gate_init: float = -2.0,
        num_heads: int = 4,
    ):
        super().__init__()

        self.proj_vis_down = nn.Linear(in_channels_vis, adapter_channels, bias=False)
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)

        self.vis_from_txt = LightweightCrossAttention(
            adapter_channels,
            num_heads=num_heads,
            gate_init=gate_init,
        )

        self.txt_from_vis = LightweightCrossAttention(
            adapter_channels,
            num_heads=num_heads,
            gate_init=gate_init,
        )

        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

    def forward(self, vis_tokens, txt_tokens):
        v = self.proj_vis_down(vis_tokens)
        t = self.proj_txt_down(txt_tokens)

        v_delta = self.vis_from_txt(v, t)
        t_delta = self.txt_from_vis(t, v)

        vis_delta = self.proj_vis_up(v_delta)
        txt_delta = self.proj_txt_up(t_delta)

        return vis_delta, txt_delta


class PromptGuidedCBAM(nn.Module):
    """
    Prompt-Guided CBAM for ViT patch tokens.

    It applies:
    1. prompt-guided channel attention
    2. spatial attention on patch grid
    3. gated residual update

    Input:
        vis_tokens: [B, N, C]
        txt_tokens: [B, T, Ctxt]

    Works with tokens that either include CLS token or not.
    """
    def __init__(
        self,
        vis_channels: int,
        txt_channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
        gate_init: float = -2.0,
    ):
        super().__init__()

        hidden = max(vis_channels // reduction, 32)

        self.txt_proj = nn.Linear(txt_channels, vis_channels)

        self.channel_mlp = nn.Sequential(
            nn.Linear(vis_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, vis_channels),
        )

        padding = kernel_size // 2
        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.norm = nn.LayerNorm(vis_channels)

    def _split_cls_patch(self, x):
        B, N, C = x.shape

        root = int(math.sqrt(N))
        if root * root == N:
            return None, x, root, root

        root = int(math.sqrt(N - 1))
        if root * root == (N - 1):
            return x[:, :1, :], x[:, 1:, :], root, root

        return None, x, None, None

    def forward(self, vis_tokens, txt_tokens):
        cls_token, patch_tokens, h, w = self._split_cls_patch(vis_tokens)

        # prompt context: [B, Cvis]
        txt_context = txt_tokens.mean(dim=1)
        txt_context = self.txt_proj(txt_context)

        # channel attention
        avg_pool = patch_tokens.mean(dim=1)
        max_pool = patch_tokens.max(dim=1).values

        channel_attn = torch.sigmoid(
            self.channel_mlp(avg_pool + txt_context)
            + self.channel_mlp(max_pool + txt_context)
        ).unsqueeze(1)

        patch_tokens_ca = patch_tokens * channel_attn

        # spatial attention, only if patch tokens form a square grid
        if h is not None and w is not None:
            B, Np, C = patch_tokens_ca.shape

            feat = patch_tokens_ca.transpose(1, 2).reshape(B, C, h, w)

            avg_map = feat.mean(dim=1, keepdim=True)
            max_map = feat.max(dim=1, keepdim=True).values

            spatial_attn = torch.sigmoid(
                self.spatial(torch.cat([avg_map, max_map], dim=1))
            )

            feat = feat * spatial_attn
            patch_tokens_sa = feat.flatten(2).transpose(1, 2)
        else:
            patch_tokens_sa = patch_tokens_ca

        alpha = torch.sigmoid(self.gate)
        patch_out = patch_tokens + alpha * (patch_tokens_sa - patch_tokens)

        if cls_token is not None:
            out = torch.cat([cls_token, patch_out], dim=1)
        else:
            out = patch_out

        return self.norm(out)


class BoundaryAwareDecoder(nn.Module):
    """
    Boundary-aware text-conditioned decoder.

    Input:
        seg_feats:      [B, C, h, w]
        text_features:  [B, C]

    Output:
        mask_logits:    [B, H, W]
        edge_logits:    [B, H, W]
    """
    def __init__(self, embed_dim: int, num_upscale: int = 2, gate_init: float = -2.0):
        super().__init__()

        groups = 8 if embed_dim % 8 == 0 else 1

        self.text_gate = nn.Linear(embed_dim, embed_dim)

        blocks = []
        for _ in range(num_upscale):
            blocks.extend([
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
                nn.GroupNorm(groups, embed_dim),
            ])

        self.up_blocks = nn.Sequential(*blocks)

        self.mask_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

        self.boundary_refine = nn.Sequential(
            nn.Conv2d(embed_dim + 2, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, 1, kernel_size=1),
        )

        self.edge_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, seg_feats, text_features, out_size):
        B, C, _, _ = seg_feats.shape

        text_gate = torch.sigmoid(self.text_gate(text_features)).view(B, C, 1, 1)
        x = seg_feats * (1.0 + text_gate)

        x = self.up_blocks(x)

        mask_logits = self.mask_head(x)
        edge_logits = self.edge_head(x)

        correction = self.boundary_refine(
            torch.cat([x, mask_logits, edge_logits], dim=1)
        )

        mask_logits = mask_logits + torch.sigmoid(self.edge_gate) * correction

        if isinstance(out_size, int):
            out_size = (out_size, out_size)

        mask_logits = F.interpolate(
            mask_logits,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )

        edge_logits = F.interpolate(
            edge_logits,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )

        return mask_logits.squeeze(1), edge_logits.squeeze(1)
