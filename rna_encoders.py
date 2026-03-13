"""
Factored encoder modules for composition in MoE and pretraining models.

Both encoders expose the same encode() API:
    token_states : (B, L, d)
    pooled       : (B, d)
    aux_dict     : dict (empty for sequence; geometry internals for bender)

This separation lets RNAMoEMRLModel compose them without coupling to the
full model classes (RNABenderModel, RNATransformerBaseline), and lets
pretrain_bender.py save/load only the geometry encoder state dict.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from rna_bender import (
    VOCAB_SIZE, PAD_ID, BACKBONE_OFFSETS,
    RNABenderLayer,
)


# ─── Shared pooling ────────────────────────────────────────────────────────────

def _pool(
    h:          torch.Tensor,          # (B, L, d)
    seq_mask:   torch.Tensor,          # (B, L) bool
    pool_attn:  Optional[nn.Linear],   # linear(d → 1) or None for mean
) -> torch.Tensor:                     # (B, d)
    if pool_attn is not None:
        scores  = pool_attn(h).squeeze(-1)
        scores  = scores.masked_fill(~seq_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (weights.unsqueeze(-1) * h).sum(dim=1)
    mf = seq_mask.float()
    return (h * mf.unsqueeze(-1)).sum(1) / mf.sum(1, keepdim=True).clamp(min=1)


# ─── Shared weight initialisation ─────────────────────────────────────────────

def _init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()


# ─── Sequence encoder ─────────────────────────────────────────────────────────

class RNASequenceEncoder(nn.Module):
    """
    Standard pre-LN Transformer encoder.  Sequence only — no edge inputs.

    Returns:
        token_states : (B, L, d)
        pooled       : (B, d)
        aux_dict     : {}   (empty — no geometry)
    """

    def __init__(
        self,
        vocab_size:  int           = VOCAB_SIZE,
        max_len:     int           = 256,
        model_dim:   int           = 128,
        num_layers:  int           = 4,
        num_heads:   int           = 8,
        ff_dim:      Optional[int] = None,
        dropout:     float         = 0.1,
        pooling:     str           = 'attention',
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * model_dim

        self.model_dim = model_dim

        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_len, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = model_dim,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder   = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.ln_f      = nn.LayerNorm(model_dim)
        self.pool_attn = nn.Linear(model_dim, 1) if pooling == 'attention' else None
        self.drop      = nn.Dropout(dropout)
        _init_weights(self)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(
        self,
        input_ids: torch.Tensor,   # (B, L)
        seq_mask:  torch.Tensor,   # (B, L) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h    = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        h    = self.encoder(h, src_key_padding_mask=~seq_mask)
        h    = self.ln_f(h)
        return h, _pool(h, seq_mask, self.pool_attn), {}

    def forward(self, input_ids, seq_mask):
        return self.encode(input_ids, seq_mask)


# ─── Bender geometry encoder ──────────────────────────────────────────────────

class RNABenderEncoder(nn.Module):
    """
    Grassmann geometry encoder.  Processes sequence + graph edges.

    Returns:
        token_states : (B, L, d)
        pooled       : (B, d)
        aux_dict     : {kappa_list, p_bb1_list, p_struct_list}
    """

    def __init__(
        self,
        vocab_size:   int             = VOCAB_SIZE,
        max_len:      int             = 256,
        model_dim:    int             = 128,
        num_layers:   int             = 4,
        reduced_dim:  int             = 32,
        ff_dim:       Optional[int]   = None,
        dropout:      float           = 0.1,
        pooling:      str             = 'attention',
        offsets:      Tuple[int, ...] = BACKBONE_OFFSETS,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * model_dim

        self.model_dim = model_dim

        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_len, model_dim)
        self.blocks    = nn.ModuleList([
            RNABenderLayer(model_dim, reduced_dim, ff_dim, dropout, offsets)
            for _ in range(num_layers)
        ])
        self.ln_f      = nn.LayerNorm(model_dim)
        self.pool_attn = nn.Linear(model_dim, 1) if pooling == 'attention' else None
        self.drop      = nn.Dropout(dropout)
        _init_weights(self)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(
        self,
        input_ids:  torch.Tensor,   # (B, L)
        edge_idx:   torch.Tensor,   # (B, L, K)
        edge_feat:  torch.Tensor,   # (B, L, K, E)
        seq_mask:   torch.Tensor,   # (B, L) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h    = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        p_bb1_list, kappa_list, p_struct_list = [], [], []
        for block in self.blocks:
            h, p_bb1, kappa, p_struct = block(h, edge_idx, edge_feat, seq_mask)
            p_bb1_list.append(p_bb1)
            kappa_list.append(kappa)
            p_struct_list.append(p_struct)

        h = self.ln_f(h)
        return h, _pool(h, seq_mask, self.pool_attn), {
            'kappa_list':    kappa_list,
            'p_bb1_list':    p_bb1_list,
            'p_struct_list': p_struct_list,
        }

    def forward(self, input_ids, edge_idx, edge_feat, seq_mask):
        return self.encode(input_ids, edge_idx, edge_feat, seq_mask)


__all__ = ['RNASequenceEncoder', 'RNABenderEncoder']