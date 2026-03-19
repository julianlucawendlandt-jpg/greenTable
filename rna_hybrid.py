"""
RNA Hybrid Model: Geometry Encoder → Structure Bottleneck → MRL Predictor

Two-stage design:

  Stage A — Geometry encoder + Structure bottleneck
    BenderEncoder produces H_geom, pair_logits P, SS logits S, MFE scalar,
    and last-layer curvature K.  These are compressed into an interpretable
    per-token bottleneck B_tok (R^{struct_bottleneck_dim}) and a global
    summary B_glob (R^{glob_bottleneck_dim}):

        pair_degree_i    = Σ_j σ(P_ij)               (soft pairing count)
        partner_ctx_i    = Σ_j softmax(P_i·)_j H_geom_j   (attending to partners)
        ss_prob_i        = softmax(S_i)               (3-class SS distribution)
        curv_i           = MLP(K_i)                   (compressed curvature)
        b_i = MLP([H_geom_i ∥ partner_ctx_i ∥ pair_degree_i ∥ ss_prob_i ∥ curv_i])

        B_tok  (B, L, struct_bottleneck_dim)
        B_glob (B, glob_bottleneck_dim) = MLP(AttnPool(B_tok) ∥ mean_degree ∥ mfe_hat)

    This is the model's inferred structure-aware abstraction of the sequence.

  Stage B — MRL predictor
    A small sequence encoder produces H_seq.
    A cross-attention bridge injects structure into sequence representations:

        H' = CrossAttn(Q=H_seq, K=V=B_tok)    (attend to structure)
        H~ = H_seq + W(H')                     (residual fusion)

    Final prediction:
        y = MLP([AttnPool(H~) ∥ B_glob])

Training phases (what is actually implemented)
    Phase 1  Pretrain BenderPretrainModel on folding via pretrain_bender.py.
             Checkpoint saves: geom_encoder, pair_head, ss_head, mfe_head, mfe_pool.
             NOT saved: struct_bottleneck, global_bottleneck (hybrid-specific; random).

    Phase 2  Build RNAHybridModel, load Phase 1 checkpoint.
             freeze_encoder_and_heads() freezes the pretrained components only.
             struct_bottleneck, global_bottleneck, and Stage B train freely on MRL.
             (--freeze_geom_epochs N)

    Phase 3  unfreeze_encoder_and_heads(); full model trains jointly with
             differential LR: pretrained components at base_lr × geom_lr_scale,
             Stage B at base_lr.  (--geom_lr_scale 0.1)

    Note: freeze_stage_a() additionally freezes struct_bottleneck and
    global_bottleneck.  Only safe if those were ALSO pretrained (currently
    they never are — pretrain_bender.py does not produce them).  The default
    training loop uses freeze_encoder_and_heads(), not freeze_stage_a().

Forward API
    Accepts same dict API as RNABenderModel and RNAMoEMRLModel — both
    edge_idx/edge_feat (folding collate) and edge_index/edge_attrs (UTR collate).

Parameter budget note
    Defaults (geom_dim=128, r=32, layers=4, seq_dim=128, layers=2) produce ~2.3M
    params.  For a matched-budget comparison run param_count.py to find configs
    near 850k–900k (e.g. geom layers=2, r=8 + seq layers=1).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from rna_encoders import RNABenderEncoder, RNASequenceEncoder, _pool, _init_weights
from rna_bender import (
    VOCAB_SIZE, PAD_ID, SS_IGNORE_IDX, N_EDGE_FEATS,
    PairMapHead, _consistency_loss,
)


# ─── Structure bottleneck (Stage A output compression) ────────────────────────

class StructureBottleneck(nn.Module):
    """
    Per-token bottleneck that compresses geometry-rich representations into a
    smaller, interpretable vector b_i ∈ R^{bottleneck_dim}.

    For each position i, the bottleneck concatenates:
        H_geom_i         — raw geometry-aware hidden state
        partner_context  — H_geom weighted by predicted pair probabilities
        pair_degree      — scalar soft pairing count  Σ_j σ(P_ij)
        ss_prob          — 3-class softmax SS distribution
        curv_compressed  — curvature K_i passed through a small MLP

    The result is fused by a 2-layer MLP into the final b_i.
    """

    def __init__(
        self,
        geom_dim:       int,
        reduced_dim:    int,      # geom encoder's reduced_dim (for plu_dim)
        bottleneck_dim: int = 64,
        curv_out:       int = 16,
        mode:           str = 'full',  # 'full' | 'simple'
    ):
        """
        mode='full'   : fuse [H_geom ∥ partner_ctx ∥ pair_degree ∥ ss_prob ∥ curv]
        mode='simple' : fuse [H_geom ∥ partner_ctx ∥ curv]
                        omits pair_degree and ss_prob; useful when those heads
                        are not yet pretrained or as a cleaner ablation baseline.
        """
        super().__init__()
        self.mode = mode
        plu_dim = reduced_dim * (reduced_dim - 1) // 2

        # Compress last-layer curvature from plu_dim → curv_out
        self.curv_mlp = nn.Sequential(
            nn.Linear(plu_dim, max(curv_out * 2, plu_dim // 2)),
            nn.GELU(),
            nn.Linear(max(curv_out * 2, plu_dim // 2), curv_out),
        )

        # Fuse all per-token features
        # 'full'  : H_geom + partner_ctx + pair_degree(1) + ss_prob(3) + curv
        # 'simple': H_geom + partner_ctx + curv
        fuse_in = (geom_dim + geom_dim + 1 + 3 + curv_out
                   if mode == 'full'
                   else geom_dim + geom_dim + curv_out)
        fuse_h  = bottleneck_dim * 2
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, fuse_h),
            nn.GELU(),
            nn.Linear(fuse_h, bottleneck_dim),
        )
        _init_weights(self)

    def forward(
        self,
        H_geom:      torch.Tensor,   # (B, L, geom_dim)
        pair_logits: torch.Tensor,   # (B, L, L)
        ss_logits:   torch.Tensor,   # (B, L, 3)
        kappa_last:  torch.Tensor,   # (B, L, plu_dim)
        seq_mask:    torch.Tensor,   # (B, L) bool
    ) -> torch.Tensor:               # (B, L, bottleneck_dim)
        mf = seq_mask.float()        # (B, L)

        # Mask diagonal (self-pairing) and padding before computing pair-derived features.
        # Without this, degree and partner context pick up trivial self-similarity signal.
        L = pair_logits.shape[-1]
        diag = torch.eye(L, device=pair_logits.device, dtype=torch.bool).unsqueeze(0)
        clean = pair_logits.masked_fill(diag | ~seq_mask.unsqueeze(1), -1e4)

        # Partner context: softmax(P_i·)_j * H_geom_j
        partner_weights = torch.softmax(clean, dim=-1)                   # (B, L, L)
        partner_weights = torch.nan_to_num(partner_weights, nan=0.0)
        partner_ctx     = torch.bmm(partner_weights, H_geom)             # (B, L, geom_dim)

        # Compressed curvature
        curv = self.curv_mlp(kappa_last)                                 # (B, L, curv_out)

        if self.mode == 'full':
            # Pair degree: Σ_j σ(P_ij) over valid j (diagonal already masked)
            pair_degree = torch.sigmoid(clean).sum(-1, keepdim=True)    # (B, L, 1)
            # SS probability distribution
            ss_prob = torch.softmax(ss_logits, dim=-1)                   # (B, L, 3)
            fuse_in = torch.cat([H_geom, partner_ctx, pair_degree, ss_prob, curv], dim=-1)
        else:  # 'simple'
            fuse_in = torch.cat([H_geom, partner_ctx, curv], dim=-1)

        B_tok = self.fuse_mlp(fuse_in)                                   # (B, L, bottleneck_dim)

        return B_tok * mf.unsqueeze(-1)   # zero out padding


class GlobalBottleneck(nn.Module):
    """
    Summarises B_tok into a fixed-size global vector B_glob (R^{glob_dim}).

    Concatenates:
        AttnPool(B_tok)    — attention-weighted global structure summary
        mean_pair_degree   — scalar average pairing probability
        mfe_hat            — predicted free energy scalar

    Then passed through a 2-layer MLP.
    """

    def __init__(self, tok_dim: int = 64, glob_dim: int = 128):
        super().__init__()
        self.pool_attn = nn.Linear(tok_dim, 1)
        self.glob_mlp  = nn.Sequential(
            nn.Linear(tok_dim + 2, glob_dim * 2),
            nn.GELU(),
            nn.Linear(glob_dim * 2, glob_dim),
        )
        _init_weights(self)

    def forward(
        self,
        B_tok:       torch.Tensor,   # (B, L, tok_dim)
        pair_logits: torch.Tensor,   # (B, L, L)
        mfe_hat:     torch.Tensor,   # (B,)
        seq_mask:    torch.Tensor,   # (B, L) bool
    ) -> torch.Tensor:               # (B, glob_dim)
        mf = seq_mask.float()

        # Attention pool over B_tok
        pooled = _pool(B_tok, seq_mask, self.pool_attn)   # (B, tok_dim)

        # Mean pair degree — same diagonal+padding masking as StructureBottleneck
        # so the global summary is consistent with the per-token one.
        L    = pair_logits.shape[-1]
        diag = torch.eye(L, device=pair_logits.device, dtype=torch.bool).unsqueeze(0)
        clean = pair_logits.masked_fill(diag | ~seq_mask.unsqueeze(1), -1e4)
        pair_probs       = torch.sigmoid(clean)
        mean_pair_degree = (pair_probs * mf.unsqueeze(1)).sum(-1)        # (B, L)
        mean_pair_degree = (mean_pair_degree * mf).sum(-1) / mf.sum(-1).clamp(min=1)  # (B,)

        x = torch.cat([pooled, mean_pair_degree.unsqueeze(-1), mfe_hat.unsqueeze(-1)], dim=-1)
        return self.glob_mlp(x)                                          # (B, glob_dim)


# ─── Cross-attention bridge (Stage B structure injection) ─────────────────────

class CrossAttentionBridge(nn.Module):
    """
    Injects structure information (B_tok) into sequence representations (H_seq)
    via multi-head cross-attention.

        Q from H_seq  (seq_dim)
        K, V from B_tok  (tok_dim → projected to seq_dim internally by nn.MHA)

    Uses pre-LN residual:
        H_tilde = H_seq + out_proj(drop(MHA(LN(H_seq), B_tok, B_tok)))
    """

    def __init__(
        self,
        seq_dim:    int,
        tok_dim:    int,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.ln       = nn.LayerNorm(seq_dim)
        # nn.MultiheadAttention handles Q/K/V projections; kdim/vdim for cross-attn
        self.mha      = nn.MultiheadAttention(
            embed_dim     = seq_dim,
            num_heads     = num_heads,
            kdim          = tok_dim,
            vdim          = tok_dim,
            batch_first   = True,
            dropout       = dropout,
        )
        self.out_proj = nn.Linear(seq_dim, seq_dim)
        self.drop     = nn.Dropout(dropout)
        _init_weights(self)

    def forward(
        self,
        H_seq:    torch.Tensor,   # (B, L, seq_dim)
        B_tok:    torch.Tensor,   # (B, L, tok_dim)
        seq_mask: torch.Tensor,   # (B, L) bool  True = valid
    ) -> torch.Tensor:            # (B, L, seq_dim)
        # Pre-LN on query
        H_norm = self.ln(H_seq)
        # key_padding_mask: True marks positions to IGNORE (i.e. padding)
        H_cross, _ = self.mha(H_norm, B_tok, B_tok, key_padding_mask=~seq_mask)
        return H_seq + self.drop(self.out_proj(H_cross))   # residual


# ─── Full hybrid model ─────────────────────────────────────────────────────────

class RNAHybridModel(nn.Module):
    """
    Two-stage RNA model for MRL prediction.

    Stage A: geometry encoder + interpretable structure bottleneck
    Stage B: sequence encoder + cross-attention bridge from structure to sequence

    Supports:
      - pretrained geometry encoder via load_pretrained_geom()
      - Stage A freeze/unfreeze via freeze_geom_encoder() / unfreeze_geom_encoder()
      - differential LR for Stage A via get_optimizer_groups()
      - same edge-name aliases as MoEMRLModel (edge_idx/edge_feat or edge_index/edge_attrs)
    """

    def __init__(
        self,
        vocab_size:           int            = VOCAB_SIZE,
        max_len:              int            = 256,
        # Stage A — geometry encoder
        geom_dim:             int            = 128,
        geom_num_layers:      int            = 4,
        geom_reduced_dim:     int            = 32,
        geom_ff_dim:          Optional[int]  = None,
        geom_max_len:         Optional[int]  = None,    # override if ckpt differs
        # Stage A — bottleneck
        struct_bottleneck_dim: int           = 64,
        glob_bottleneck_dim:   int           = 128,
        curv_out:              int           = 16,
        bottleneck_mode:       str           = 'full',  # 'full' | 'simple'
        # Stage B — sequence encoder
        seq_dim:              int            = 128,
        seq_num_layers:       int            = 2,
        seq_num_heads:        int            = 8,
        seq_ff_dim:           Optional[int]  = None,
        cross_attn_heads:     int            = 4,
        # Shared
        dropout:              float          = 0.1,
        pooling:              str            = 'attention',
        num_libraries:        int            = 0,
        # Stage A auxiliary loss weights
        lambda_pair:          float          = 0.1,
        lambda_ss:            float          = 0.1,
        lambda_mfe:           float          = 0.01,
        lambda_curv:          float          = 0.01,
        lambda_cons:          float          = 0.0,
    ):
        super().__init__()
        self.lambda_pair = lambda_pair
        self.lambda_ss   = lambda_ss
        self.lambda_mfe  = lambda_mfe
        self.lambda_curv = lambda_curv
        self.lambda_cons = lambda_cons

        _geom_max_len = geom_max_len or max_len

        # ── Stage A — geometry encoder ─────────────────────────────────────────
        self.geom_encoder = RNABenderEncoder(
            vocab_size  = vocab_size,
            max_len     = _geom_max_len,
            model_dim   = geom_dim,
            num_layers  = geom_num_layers,
            reduced_dim = geom_reduced_dim,
            ff_dim      = geom_ff_dim,
            dropout     = dropout,
            pooling     = pooling,
        )
        # Stage A auxiliary heads (pair map, SS, MFE)
        self.pair_head = PairMapHead(geom_dim)
        self.ss_head   = nn.Linear(geom_dim, 3)
        self.mfe_head  = nn.Linear(geom_dim, 1)
        self.mfe_pool  = nn.Linear(geom_dim, 1)   # attention pool for MFE

        # ── Stage A — structure bottleneck ─────────────────────────────────────
        self.struct_bottleneck = StructureBottleneck(
            geom_dim       = geom_dim,
            reduced_dim    = geom_reduced_dim,
            bottleneck_dim = struct_bottleneck_dim,
            curv_out       = curv_out,
            mode           = bottleneck_mode,
        )
        self.global_bottleneck = GlobalBottleneck(
            tok_dim  = struct_bottleneck_dim,
            glob_dim = glob_bottleneck_dim,
        )

        # ── Stage B — sequence encoder ─────────────────────────────────────────
        self.seq_encoder = RNASequenceEncoder(
            vocab_size = vocab_size,
            max_len    = max_len,
            model_dim  = seq_dim,
            num_layers = seq_num_layers,
            num_heads  = seq_num_heads,
            ff_dim     = seq_ff_dim,
            dropout    = dropout,
            pooling    = pooling,
        )

        # ── Cross-attention bridge ──────────────────────────────────────────────
        self.cross_attn   = CrossAttentionBridge(
            seq_dim   = seq_dim,
            tok_dim   = struct_bottleneck_dim,
            num_heads = cross_attn_heads,
            dropout   = dropout,
        )
        self.seq_pool_attn = nn.Linear(seq_dim, 1)

        # ── MRL prediction head ─────────────────────────────────────────────────
        head_in  = seq_dim + glob_bottleneck_dim
        head_h   = head_in // 2
        self.mrl_head = nn.Sequential(
            nn.Linear(head_in, head_h),
            nn.GELU(),
            nn.Linear(head_h, 1),
        )

        # ── Library conditioning (MRL cross-library) ───────────────────────────
        if num_libraries > 0:
            self.lib_emb: Optional[nn.Embedding] = nn.Embedding(num_libraries, seq_dim)
        else:
            self.lib_emb = None

        self.drop = nn.Dropout(dropout)
        self._init_new_params()

    def _init_new_params(self):
        """Initialise Stage B modules + heads (encoders handle their own init)."""
        for m in [self.mrl_head, self.cross_attn, self.struct_bottleneck,
                  self.global_bottleneck, self.pair_head, self.ss_head,
                  self.mfe_head, self.mfe_pool, self.seq_pool_attn]:
            _init_weights(m)
        if self.lib_emb is not None:
            nn.init.normal_(self.lib_emb.weight, std=0.02)
        # Set to True by load_pretrained_geom() when structure heads are loaded.
        # Used by train_utr.py to decide which freeze method to call.
        self._heads_loaded: bool = False

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Stage A freeze helpers ─────────────────────────────────────────────────

    def _stage_a_modules(self) -> List[nn.Module]:
        """All Stage A modules — used for LR grouping (see get_optimizer_groups)."""
        return [self.geom_encoder, self.pair_head, self.ss_head, self.mfe_head,
                self.mfe_pool, self.struct_bottleneck, self.global_bottleneck]

    def _pretrained_modules(self) -> List[nn.Module]:
        """Modules that are actually saved in the pretrain checkpoint.

        Excludes struct_bottleneck and global_bottleneck, which are hybrid-specific
        and always start from random initialisation.
        """
        return [self.geom_encoder, self.pair_head, self.ss_head,
                self.mfe_head, self.mfe_pool]

    # ── Freeze helpers (use these in training loops) ───────────────────────────

    def freeze_geom_encoder(self):
        """Freeze only the geometry encoder backbone.

        Safe when structure heads are NOT pretrained — only the backbone carries
        transferred knowledge; heads are still learning from aux supervision.
        """
        for p in self.geom_encoder.parameters():
            p.requires_grad = False

    def unfreeze_geom_encoder(self):
        """Unfreeze geometry encoder backbone."""
        for p in self.geom_encoder.parameters():
            p.requires_grad = True

    def freeze_encoder_and_heads(self):
        """Freeze geom_encoder + pair_head + ss_head + mfe_head + mfe_pool.

        CORRECT freeze to use after load_pretrained_geom() loads all head state
        dicts.  struct_bottleneck and global_bottleneck remain trainable — they
        are NOT in the pretrain checkpoint and must learn from scratch on MRL.

        This is what --freeze_geom_epochs triggers in train_utr.py when heads
        were loaded from the checkpoint.
        """
        for mod in self._pretrained_modules():
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_encoder_and_heads(self):
        """Unfreeze geom_encoder + structure heads (reverse of freeze_encoder_and_heads)."""
        for mod in self._pretrained_modules():
            for p in mod.parameters():
                p.requires_grad = True

    def freeze_stage_a(self):
        """Freeze ALL of Stage A: pretrained modules + struct_bottleneck + global_bottleneck.

        WARNING: struct_bottleneck and global_bottleneck are NOT saved in the
        pretrain checkpoint — they always start random.  Freezing them at epoch 0
        means they can never learn.  Only call this if you have separately
        pretrained those modules (e.g. via a full hybrid folding pretraining run
        that is not currently implemented in this codebase).

        For the standard workflow, use freeze_encoder_and_heads() instead.
        """
        for mod in self._stage_a_modules():
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_stage_a(self):
        """Unfreeze ALL of Stage A (reverse of freeze_stage_a)."""
        for mod in self._stage_a_modules():
            for p in mod.parameters():
                p.requires_grad = True

    def get_optimizer_groups(self, base_lr: float, geom_lr_scale: float = 0.1) -> List[Dict]:
        """
        Three-group LR split matching the actual pretraining status of each module:

            pretrained  (geom_encoder + pair/ss/mfe heads) → base_lr × geom_lr_scale
            bottleneck  (struct_bottleneck + global_bottleneck) → base_lr
            Stage B     (seq_encoder + cross_attn + mrl_head + …) → base_lr

        struct_bottleneck and global_bottleneck are always randomly initialised
        (they are not in the pretrain checkpoint), so they should train at the
        full base_lr alongside Stage B — not at the reduced pretrained LR.
        """
        pretrained_ids  = {id(p) for mod in self._pretrained_modules()
                           for p in mod.parameters()}
        bottleneck_mods = [self.struct_bottleneck, self.global_bottleneck]
        bottleneck_ids  = {id(p) for mod in bottleneck_mods for p in mod.parameters()}

        pretrained_params = [p for p in self.parameters() if id(p) in pretrained_ids]
        bottleneck_params = [p for p in self.parameters() if id(p) in bottleneck_ids]
        stage_b_params    = [p for p in self.parameters()
                             if id(p) not in pretrained_ids and id(p) not in bottleneck_ids]
        return [
            {'params': stage_b_params,    'lr': base_lr,                  'name': 'stage_b'},
            {'params': bottleneck_params, 'lr': base_lr,                  'name': 'bottleneck'},
            {'params': pretrained_params, 'lr': base_lr * geom_lr_scale,  'name': 'pretrained'},
        ]

    # ── Checkpoint loading ─────────────────────────────────────────────────────

    def load_pretrained_geom(self, path: str, strict: bool = False) -> Tuple:
        """
        Load pretrained geometry encoder weights from pretrain_bender.py checkpoint.

        Also loads pair_head and ss_head if saved in the checkpoint (requires
        pretrain_bender.py to save pair_head_state_dict / ss_head_state_dict).
        When those are available, Stage A has fully consistent structure heads and
        freeze_stage_a() can safely be used.  Without them, only the encoder
        backbone is loaded and heads start fresh — use freeze_geom_encoder() instead.
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd   = (ckpt.get('geom_encoder_state_dict')
                or ckpt.get('encoder_state_dict')
                or ckpt)
        missing, unexpected = self.geom_encoder.load_state_dict(sd, strict=strict)
        n = self.geom_encoder.get_num_params()
        print(f'  Loaded pretrained geom encoder ({n:,} params) from {path}')
        if missing:
            print(f'    missing  : {missing[:4]}{"..." if len(missing) > 4 else ""}')
        if unexpected:
            print(f'    unexpected: {unexpected[:4]}{"..." if len(unexpected) > 4 else ""}')

        # Load auxiliary structure heads if the checkpoint contains them.
        # pretrain_bender.py saves these when trained with the updated script.
        heads_loaded = []
        for attr, key in [('pair_head', 'pair_head_state_dict'),
                          ('ss_head',   'ss_head_state_dict'),
                          ('mfe_head',  'mfe_head_state_dict'),
                          ('mfe_pool',  'mfe_pool_state_dict')]:
            if key in ckpt:
                getattr(self, attr).load_state_dict(ckpt[key], strict=False)
                heads_loaded.append(attr)

        if 'pair_head' in heads_loaded and 'ss_head' in heads_loaded:
            self._heads_loaded = True
            print(f'    Also loaded structure heads: {heads_loaded}')
            print(f'    → use freeze_encoder_and_heads() to protect pretrained weights.')
            print(f'    → do NOT use freeze_stage_a(): struct_bottleneck/global_bottleneck '
                  f'are not in this checkpoint and must stay trainable.')
        else:
            self._heads_loaded = False
            print(f'    Note: structure heads not in checkpoint — heads start fresh.')
            print(f'    → use freeze_geom_encoder() (backbone only).')

        return missing, unexpected

    # ── Loss computation ───────────────────────────────────────────────────────

    def _compute_loss(
        self,
        out:         Dict,
        labels:      torch.Tensor,
        seq_mask:    torch.Tensor,
        pair_targets: Optional[torch.Tensor] = None,  # (B, L, L) float, folding/BPP
        ss_labels:    Optional[torch.Tensor] = None,  # (B, L) int, SS class indices
        mfe_labels:   Optional[torch.Tensor] = None,  # (B,) float, MFE targets
    ) -> torch.Tensor:
        """
        Full hybrid loss:
            L = L_MRL + λ_pair·L_pair + λ_ss·L_ss + λ_mfe·L_mfe
                      + λ_curv·L_curv + λ_cons·L_cons

        Stage A auxiliary supervision (pair/SS/MFE) is only applied when the
        corresponding targets are provided.  Without it, pair_head/ss_head are
        only indirectly trained through the bottleneck → MRL path, which is
        much weaker.  Pass targets via aux_struct=True (ss_labels, mfe_labels)
        or rnastralign pair_targets.
        """
        loss = F.mse_loss(out['task_logits'], labels.float())

        # ── Stage A auxiliary supervision ──────────────────────────────────────

        # Pair-map BCE  (folding: always available; UTR: only if pair_targets passed)
        if self.lambda_pair > 0 and pair_targets is not None:
            pair_logits = out['pair_logits']             # (B, L, L)
            mf          = seq_mask.float()
            pair_mask   = mf.unsqueeze(1) * mf.unsqueeze(2)   # (B, L, L)
            n_valid     = pair_mask.sum().clamp(min=1)
            pair_loss   = F.binary_cross_entropy_with_logits(
                pair_logits, pair_targets.float(), reduction='none'
            )
            loss = loss + self.lambda_pair * (pair_loss * pair_mask).sum() / n_valid

        # Per-token SS cross-entropy  (available when aux_struct=True)
        if self.lambda_ss > 0 and ss_labels is not None:
            ss_logits = out['ss_logits']                 # (B, L, 3)
            valid     = ss_labels != SS_IGNORE_IDX
            if valid.any():
                loss = loss + self.lambda_ss * F.cross_entropy(
                    ss_logits[valid], ss_labels[valid]
                )

        # MFE regression  (available when aux_struct=True)
        if self.lambda_mfe > 0 and mfe_labels is not None:
            loss = loss + self.lambda_mfe * F.mse_loss(
                out['mfe_pred'], mfe_labels.float()
            )

        # ── Geometry regularisers ──────────────────────────────────────────────

        # Curvature regularisation (smoothness prior on geometry latent flow)
        if self.lambda_curv > 0:
            kappa_list = out.get('kappa_list', [])
            if kappa_list:
                mf      = seq_mask.float()
                n_valid = mf.sum().clamp(min=1)
                loss_curv = sum(
                    (k.pow(2).sum(-1) * mf).sum() / n_valid for k in kappa_list
                ) / len(kappa_list)
                loss = loss + self.lambda_curv * loss_curv

        # Backbone–pairing consistency
        if self.lambda_cons > 0:
            p_bb1_list    = out.get('p_bb1_list', [])
            p_struct_list = out.get('p_struct_list', [])
            edge_feat     = out.get('edge_feat')
            if p_bb1_list and p_struct_list and edge_feat is not None:
                loss = loss + self.lambda_cons * _consistency_loss(
                    p_bb1_list[-1], p_struct_list[-1], edge_feat
                )

        return loss

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:    torch.Tensor,
        seq_mask:     torch.Tensor,
        # Folding collate: edge_idx / edge_feat
        edge_idx:     Optional[torch.Tensor] = None,
        edge_feat:    Optional[torch.Tensor] = None,
        # UTR collate: edge_index / edge_attrs (accepted as aliases)
        edge_index:   Optional[torch.Tensor] = None,
        edge_attrs:   Optional[torch.Tensor] = None,
        edge_mask:    Optional[torch.Tensor] = None,
        labels:       Optional[torch.Tensor] = None,
        library_ids:  Optional[torch.Tensor] = None,
        # Auxiliary supervision targets
        pair_targets: Optional[torch.Tensor] = None,  # (B, L, L) — folding or BPP
        ss_labels:    Optional[torch.Tensor] = None,  # (B, L)     — aux_struct mode
        mfe_labels:   Optional[torch.Tensor] = None,  # (B,)       — aux_struct mode
        **kwargs,
    ) -> Dict:
        """
        Returns a dict with:
            task_logits  : (B,)          predicted MRL
            pair_logits  : (B, L, L)     Stage A pair map
            ss_logits    : (B, L, 3)     Stage A SS logits
            mfe_pred     : (B,)          Stage A MFE prediction
            B_tok        : (B, L, d_btok) per-token structure bottleneck
            B_glob       : (B, d_bglob)  global structure bottleneck
            kappa_list   : list of (B, L, plu) per-layer curvature
            p_bb1_list   : list of (B, L, plu) backbone Plücker
            p_struct_list: list of (B, L, K, plu) structural edge Plücker
            edge_feat    : forwarded (for loss helpers)
            loss         : scalar MSE + regularisers (only when labels provided)
        """
        # Normalise edge name aliases
        eidx  = edge_idx  if edge_idx  is not None else edge_index
        efeat = edge_feat if edge_feat is not None else edge_attrs

        # ── Stage A: geometry encoder ──────────────────────────────────────────
        H_geom, geom_pool, geom_aux = self.geom_encoder.encode(
            input_ids, eidx, efeat, seq_mask
        )
        kappa_list    = geom_aux['kappa_list']
        p_bb1_list    = geom_aux['p_bb1_list']
        p_struct_list = geom_aux['p_struct_list']
        kappa_last    = kappa_list[-1] if kappa_list else H_geom.new_zeros(
            H_geom.shape[0], H_geom.shape[1], 0)

        pair_logits, _ = self.pair_head(H_geom, seq_mask)   # (B, L, L)
        ss_logits      = self.ss_head(H_geom)                # (B, L, 3)
        mfe_hat        = self.mfe_head(
            _pool(H_geom, seq_mask, self.mfe_pool)
        ).squeeze(-1)                                        # (B,)

        # ── Stage A: structure bottleneck ──────────────────────────────────────
        B_tok  = self.struct_bottleneck(
            H_geom, pair_logits, ss_logits, kappa_last, seq_mask
        )                                                    # (B, L, d_btok)
        B_glob = self.global_bottleneck(
            B_tok, pair_logits, mfe_hat, seq_mask
        )                                                    # (B, d_bglob)

        # ── Stage B: sequence encoder ──────────────────────────────────────────
        H_seq, _, _ = self.seq_encoder.encode(input_ids, seq_mask)  # (B, L, seq_dim)

        # Cross-attention: inject structure into sequence representations
        H_tilde = self.cross_attn(H_seq, B_tok, seq_mask)   # (B, L, seq_dim)

        # Pool Stage B output
        h_pool = _pool(H_tilde, seq_mask, self.seq_pool_attn)  # (B, seq_dim)
        if self.lib_emb is not None and library_ids is not None:
            h_pool = h_pool + self.lib_emb(library_ids)

        # Prediction
        y = self.mrl_head(
            self.drop(torch.cat([h_pool, B_glob], dim=-1))
        ).squeeze(-1)                                        # (B,)

        out: Dict = {
            'task_logits':   y,
            'pair_logits':   pair_logits,
            'ss_logits':     ss_logits,
            'mfe_pred':      mfe_hat,
            'B_tok':         B_tok,
            'B_glob':        B_glob,
            'kappa_list':    kappa_list,
            'p_bb1_list':    p_bb1_list,
            'p_struct_list': p_struct_list,
            'edge_feat':     efeat,
        }

        if labels is not None:
            out['loss'] = self._compute_loss(
                out, labels, seq_mask,
                pair_targets=pair_targets,
                ss_labels=ss_labels,
                mfe_labels=mfe_labels,
            )

        return out


__all__ = [
    'RNAHybridModel',
    'StructureBottleneck',
    'GlobalBottleneck',
    'CrossAttentionBridge',
]
