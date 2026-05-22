import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Improved Probabilistic Vision-Language adapter.
#
# This is a DROP-IN replacement for trainers/layers.py in HealthX-Lab/MedCLIPSeg.
# It keeps the original public API (PVL_Adapter(vis, text) -> (vis_out, txt_out))
# so the rest of the repo does not need to change, BUT:
#
#   1. The probabilistic K/V now carry an explicit KL term toward a unit prior,
#      with the "free bits" trick to stop posterior collapse. The original code
#      sampled V once with NO regularizer, so the variance was unconstrained.
#      Call `adapter.collect_kl()` after the forward pass and add it to your loss
#      (weighted by `kl_weight`, ideally annealed). See runner notes at bottom.
#
#   2. Multi-head probabilistic attention (set n_heads>1). The original was
#      effectively single-head full-dim.
#
#   3. True Monte-Carlo inference. The original accepted num_samples but ignored
#      it. Now `num_samples>1` averages the output AND the module exposes the
#      per-pixel sample variance via `adapter.last_uncertainty` for your maps.
#
# Every change is behind a flag so you can run the ablations a reviewer expects:
#   - use_kl=False              -> original-style unregularized variance
#   - n_heads=1                 -> original single-head
#   - num_samples=1 (inference) -> original single-sample
#   - learnable_beta=False      -> original fixed beta
# ---------------------------------------------------------------------------


class ProbCrossAttention(nn.Module):
    """
    Probabilistic cross-attention with variational K/V, optional multi-head,
    optional learnable confidence penalty (beta), KL regularization with free
    bits, and true Monte-Carlo sampling at inference.
    """
    def __init__(self,
                 dim,
                 beta: float = 2.35,
                 gate_init: float = 0.0,
                 n_heads: int = 1,
                 use_kl: bool = True,
                 free_bits: float = 0.5,
                 learnable_beta: bool = False):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_kl = use_kl
        self.free_bits = free_bits

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * 2)  # mean + logvar
        self.v_proj = nn.Linear(dim, dim * 2)  # mean + logvar
        self.out_proj = nn.Linear(dim, dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.eps = 1e-6
        self.gate = nn.Parameter(torch.tensor(gate_init))

        if learnable_beta:
            # softplus(raw_beta) keeps it positive; init so softplus(raw)~beta
            init = math.log(math.expm1(max(beta, 1e-3)))
            self.raw_beta = nn.Parameter(torch.tensor(init))
            self.learnable_beta = True
        else:
            self.register_buffer("_beta_const", torch.tensor(beta))
            self.learnable_beta = False

        # filled in during forward, read by the adapter / trainer
        self._kl = None
        self.last_uncertainty = None

    @property
    def beta(self):
        if self.learnable_beta:
            return F.softplus(self.raw_beta)
        return self._beta_const

    def _split_heads(self, x):
        # x: [B, T, C] -> [B, H, T, head_dim]
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        # x: [B, H, T, head_dim] -> [B, T, C]
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    @staticmethod
    def _kl_unit_gaussian(mu, var, free_bits):
        # KL( N(mu,var) || N(0,1) ) per element, then free-bits clamp + mean.
        # 0.5 * (var + mu^2 - 1 - log var)
        logvar = torch.log(var)
        kl = 0.5 * (var + mu.pow(2) - 1.0 - logvar)
        if free_bits > 0:
            kl = torch.clamp(kl, min=free_bits)
        return kl.mean()

    def forward(self, query, context, sample=True, num_samples=1):
        B, Tq, C = query.shape
        _, Tk, _ = context.shape

        Q = self.q_proj(query)

        K_out = self.k_proj(context)
        K_mu, K_logvar = K_out[..., :C], K_out[..., C:]
        K_mu = self.norm_k(K_mu)
        K_var = F.softplus(K_logvar) + self.eps

        V_out = self.v_proj(context)
        V_mu, V_logvar = V_out[..., :C], V_out[..., C:]
        V_mu = self.norm_v(V_mu)
        V_var = F.softplus(V_logvar) + self.eps

        # KL term (toward unit Gaussian) for both K and V
        if self.use_kl and self.training:
            kl_k = self._kl_unit_gaussian(K_mu, K_var, self.free_bits)
            kl_v = self._kl_unit_gaussian(V_mu, V_var, self.free_bits)
            self._kl = kl_k + kl_v
        else:
            self._kl = query.new_zeros(())

        # ---- multi-head split ----
        Qh = self._split_heads(Q)            # [B,H,Tq,hd]
        K_mu_h = self._split_heads(K_mu)     # [B,H,Tk,hd]
        K_var_h = self._split_heads(K_var)
        V_mu_h = self._split_heads(V_mu)
        V_var_h = self._split_heads(V_var)

        scale = math.sqrt(self.head_dim)
        mean_scores = torch.matmul(Qh, K_mu_h.transpose(-1, -2)) / scale
        var_penalty = torch.matmul(Qh.pow(2), K_var_h.transpose(-1, -2)) / self.head_dim
        scores = mean_scores - self.beta * torch.sqrt(var_penalty + self.eps)
        attn_weights = F.softmax(scores, dim=-1)

        # ---- true Monte-Carlo over Values ----
        # at train time 1 sample (reparam) is enough for the gradient;
        # at eval time num_samples>1 gives a stable mean + an uncertainty map.
        n = num_samples if (not self.training) else 1
        outs = []
        for _ in range(max(1, n)):
            eps = torch.randn_like(V_var_h)
            V_sample = V_mu_h + torch.sqrt(V_var_h) * eps
            outs.append(torch.matmul(attn_weights, V_sample))   # [B,H,Tq,hd]
        stacked = torch.stack(outs, dim=0)                       # [n,B,H,Tq,hd]
        out_h = stacked.mean(0)

        # per-query uncertainty = variance across MC samples (averaged over dim)
        if n > 1:
            unc = stacked.var(0).mean(dim=1).mean(dim=-1)        # [B,Tq]
            self.last_uncertainty = unc.detach()
        else:
            self.last_uncertainty = None

        out = self._merge_heads(out_h)        # [B,Tq,C]
        gate = torch.sigmoid(self.gate)
        proj_out = self.out_proj(out)
        fused = gate * proj_out + (1 - gate) * query
        return fused

    def collect_kl(self):
        return self._kl if self._kl is not None else torch.zeros((), device=self.gate.device)


class TwoWayTransformerLayer(nn.Module):
    def __init__(self, embed_dim, beta=2.35, gate_init=0.0,
                 n_heads=1, use_kl=True, free_bits=0.5, learnable_beta=False):
        super().__init__()
        self.cross_attn_img_to_txt = ProbCrossAttention(
            embed_dim, beta, gate_init, n_heads, use_kl, free_bits, learnable_beta)
        self.cross_attn_txt_to_img = ProbCrossAttention(
            embed_dim, beta, gate_init, n_heads, use_kl, free_bits, learnable_beta)

    def forward(self, img_tokens, txt_tokens, num_samples=1):
        img_tokens = self.cross_attn_img_to_txt(img_tokens, txt_tokens, num_samples=num_samples)
        txt_tokens = self.cross_attn_txt_to_img(txt_tokens, img_tokens, num_samples=num_samples)
        return img_tokens, txt_tokens

    def collect_kl(self):
        return self.cross_attn_img_to_txt.collect_kl() + self.cross_attn_txt_to_img.collect_kl()


class PVL_Adapter(nn.Module):
    def __init__(self,
                 in_channels_vis: int,
                 in_channels_txt: int,
                 adapter_channels: int,
                 beta: float,
                 gate_init: int,
                 # --- new optional args; defaults reproduce improved behaviour ---
                 n_heads: int = 4,
                 use_kl: bool = True,
                 free_bits: float = 0.5,
                 learnable_beta: bool = True,
                 num_samples_eval: int = 8):
        super().__init__()

        self.proj_vis_down = nn.Sequential(nn.Linear(in_channels_vis, adapter_channels, bias=False))
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)
        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

        # adapter_channels must be divisible by n_heads
        if adapter_channels % n_heads != 0:
            n_heads = 1
        self.num_samples_eval = num_samples_eval

        self.two_way = TwoWayTransformerLayer(
            adapter_channels, beta, gate_init,
            n_heads=n_heads, use_kl=use_kl,
            free_bits=free_bits, learnable_beta=learnable_beta)

    def forward(self, vis, text):
        v = self.proj_vis_down(vis)
        t = self.proj_txt_down(text)
        ns = self.num_samples_eval if not self.training else 1
        v_fused, t_fused = self.two_way(v, t, num_samples=ns)
        vis_out = self.proj_vis_up(v_fused)
        txt_out = self.proj_txt_up(t_fused)
        return vis_out, txt_out

    def collect_kl(self):
        """Call after forward; add kl_weight * adapter.collect_kl() to your loss."""
        return self.two_way.collect_kl()

    @property
    def uncertainty(self):
        """Per-query MC variance from the last eval forward (image->text branch)."""
        return self.two_way.cross_attn_img_to_txt.last_uncertainty
