"""
Structure-pretraining for the RNA Bender geometry encoder.

Trains RNABenderEncoder on RNA secondary structure prediction, then saves
its weights for downstream loading into RNAMoEMRLModel.

Objectives (all external losses from rna_fold.py):
    L_pair  — BCE on base-pair map  (primary)
    L_ss    — per-token SS cross-entropy
    L_curv  — curvature regularisation

No MLM.  The pretraining signal is entirely structural geometry.

Output checkpoint:
    {
        'geom_encoder_state_dict': ...,
        'epoch': int,
        'val_metrics': {pair_f1, ...},
        'cfg': dict,
    }

Usage:
    python pretrain_bender.py \\
        --data rnastralign/data.json --data_format json \\
        --max_len 256 --epochs 60 --batch_size 32 \\
        --output_dir pretrain_bender_out

    # Then use in MoE downstream training:
    python train_utr.py --task mrl --data eGFP_U1.csv \\
        --model_type moe \\
        --pretrained_geom_encoder pretrain_bender_out/best_geom_encoder.pt \\
        --freeze_geom_epochs 5 --geom_lr_scale 0.1
"""

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rna_encoders import RNABenderEncoder, _pool
from rna_bender import VOCAB_SIZE, PAD_ID, PairMapHead
from rna_fold import (
    RNAstralignDataset, collate_rnastralign,
    folding_loss, aggregate_structure_metrics, structure_metrics,
    random_family_split,
)
from train_utr import WarmupCosineScheduler


# ─── Pretraining model ────────────────────────────────────────────────────────

class BenderPretrainModel(nn.Module):
    """
    RNABenderEncoder + pair map head + SS head + MFE head for structure pretraining.

    All three heads are saved alongside the encoder in the output checkpoint so
    that RNAHybridModel.load_pretrained_geom() can fully initialise Stage A,
    making freeze_stage_a() safe to use.

    MFE supervision is applied when the batch provides an 'mfe' key; for the
    RNAstralign dataset this key is absent and the MFE head is updated only via
    gradients flowing through the bottleneck → pair/SS path (indirect signal).
    """

    def __init__(
        self,
        model_dim:   int           = 128,
        num_layers:  int           = 4,
        reduced_dim: int           = 32,
        ff_dim:      Optional[int] = None,
        dropout:     float         = 0.1,
        max_len:     int           = 256,
    ):
        super().__init__()
        self.encoder   = RNABenderEncoder(
            model_dim   = model_dim,
            num_layers  = num_layers,
            reduced_dim = reduced_dim,
            ff_dim      = ff_dim,
            dropout     = dropout,
            max_len     = max_len,
        )
        self.pair_head = PairMapHead(model_dim)
        self.ss_head   = nn.Linear(model_dim, 3)
        self.mfe_pool  = nn.Linear(model_dim, 1)   # attention weights for MFE pooling
        self.mfe_head  = nn.Linear(model_dim, 1)   # pooled → MFE scalar
        for m in [self.ss_head, self.mfe_pool, self.mfe_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, edge_idx, edge_feat, seq_mask):
        h, _, geom_aux = self.encoder.encode(input_ids, edge_idx, edge_feat, seq_mask)
        pair_logits, _ = self.pair_head(h, seq_mask)
        mfe_pred = self.mfe_head(_pool(h, seq_mask, self.mfe_pool)).squeeze(-1)  # (B,)
        return {
            'pair_logits':   pair_logits,
            'ss_logits':     self.ss_head(h),
            'mfe_pred':      mfe_pred,
            'kappa_list':    geom_aux['kappa_list'],
            'p_bb1_list':    geom_aux['p_bb1_list'],
            'p_struct_list': geom_aux['p_struct_list'],
            'edge_feat':     edge_feat,
        }


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PretrainBenderConfig:
    data:          str   = ''
    data_format:   str   = 'json'       # 'json' | 'csv' | 'bpseq'
    max_len:       int   = 256
    oracle_edges:  bool  = True         # use ground-truth base pairs as graph edges

    model_dim:     int   = 128
    num_layers:    int   = 4
    reduced_dim:   int   = 32
    ff_dim:        Optional[int] = None
    dropout:       float = 0.1

    lambda_pair:   float = 1.0
    lambda_ss:     float = 0.1
    lambda_mfe:    float = 0.01   # MFE regression; only used when batch has 'mfe'
    lambda_curv:   float = 0.01

    epochs:        int   = 60
    batch_size:    int   = 32
    lr:            float = 3e-4
    weight_decay:  float = 1e-2
    clip_grad:     float = 1.0
    patience:      int   = 10
    warmup_steps:  int   = 200
    val_frac:      float = 0.15        # ~15% val; keep families intact

    use_amp:       bool  = True
    eval_every:    int   = 2

    device:        str   = 'auto'
    num_workers:   int   = 0
    output_dir:    str   = 'pretrain_bender_out'
    seed:          int   = 42


# ─── Training helpers ─────────────────────────────────────────────────────────

def pretrain_epoch(
    model:     BenderPretrainModel,
    loader:    DataLoader,
    opt:       torch.optim.Optimizer,
    sched,
    device:    torch.device,
    cfg:       PretrainBenderConfig,
    scaler,
) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc='train', leave=False, dynamic_ncols=True):
        input_ids   = batch['input_ids'].to(device)
        edge_idx    = batch['edge_idx'].to(device)
        edge_feat   = batch['edge_feat'].to(device)
        seq_mask    = batch['seq_mask'].to(device)
        pair_targets = batch['pair_targets'].to(device)
        ss_labels   = batch['ss_labels'].to(device)

        opt.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
            outputs = model(input_ids, edge_idx, edge_feat, seq_mask)
            loss = folding_loss(
                outputs,
                pair_targets = pair_targets,
                ss_labels    = ss_labels,
                seq_mask     = seq_mask,
                lambda_pair  = cfg.lambda_pair,
                lambda_ss    = cfg.lambda_ss,
                lambda_curv  = cfg.lambda_curv,
                lambda_cons  = 0.0,
            )
            # MFE supervision when available (not in rnastralign, but works for
            # any dataset that provides an 'mfe' key in the batch)
            if cfg.lambda_mfe > 0 and 'mfe' in batch:
                import torch.nn.functional as F
                loss = loss + cfg.lambda_mfe * F.mse_loss(
                    outputs['mfe_pred'], batch['mfe'].to(device).float()
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()

        sched.step()
        total += loss.item()

    return total / max(len(loader), 1)


@torch.no_grad()
def pretrain_eval(
    model:  BenderPretrainModel,
    loader: DataLoader,
    device: torch.device,
    cfg:    PretrainBenderConfig,
) -> Dict[str, float]:
    model.eval()
    per_seq = []
    for batch in tqdm(loader, desc='val', leave=False, dynamic_ncols=True):
        input_ids    = batch['input_ids'].to(device)
        edge_idx     = batch['edge_idx'].to(device)
        edge_feat    = batch['edge_feat'].to(device)
        seq_mask     = batch['seq_mask'].to(device)
        pair_targets = batch['pair_targets']   # keep on CPU for metrics
        ss_labels    = batch['ss_labels']

        outputs  = model(input_ids, edge_idx, edge_feat, seq_mask)
        pl       = outputs['pair_logits'].cpu().numpy()    # (B, L, L)
        ssl      = outputs['ss_logits'].cpu().numpy()      # (B, L, 3)
        pt       = pair_targets.numpy()
        ssl_lbl  = ss_labels.numpy()
        sm       = seq_mask.cpu().numpy()

        for b in range(pl.shape[0]):
            seq_len = int(sm[b].sum())
            m = structure_metrics(
                pl[b], pt[b], ssl[b], ssl_lbl[b], seq_len=seq_len
            )
            per_seq.append(m)

    return aggregate_structure_metrics(per_seq)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_pretrain(cfg: PretrainBenderConfig):
    if cfg.device == 'auto':
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_path   = os.path.join(cfg.output_dir, 'best_geom_encoder.pt')
    resume_path = os.path.join(cfg.output_dir, 'pretrain_resume.pt')
    cfg_path    = os.path.join(cfg.output_dir, 'pretrain_cfg.json')
    with open(cfg_path, 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    device = torch.device(cfg.device)
    amp_ok = cfg.use_amp and cfg.device == 'cuda'

    print(f'Pretraining Bender geometry encoder')
    print(f'Data:   {cfg.data} | format={cfg.data_format} | max_len={cfg.max_len}')
    print(f'Device: {cfg.device} | AMP: {amp_ok}')
    print(f'λ_pair={cfg.lambda_pair} λ_ss={cfg.lambda_ss} λ_curv={cfg.lambda_curv}')

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset  = RNAstralignDataset(
        cfg.data,
        data_format            = cfg.data_format,
        max_len                = cfg.max_len,
        use_oracle_struct_edges = cfg.oracle_edges,
    )
    n        = len(dataset)
    families = [dataset[i]['family'] for i in range(n)]
    tr_idx, va_idx = random_family_split(families, val_frac=cfg.val_frac, seed=cfg.seed)

    families_arr = np.array(families)
    tr_fams  = sorted(set(families_arr[tr_idx].tolist()))
    va_fams  = sorted(set(families_arr[va_idx].tolist()))
    n_fam    = len(set(families))
    print(f'Dataset: {n:,} sequences | {n_fam} families | '
          f'{len(tr_idx):,} train ({len(tr_fams)} families) / '
          f'{len(va_idx):,} val ({len(va_fams)} families)')

    # Split metadata — saved in checkpoint so experiments are fully reproducible.
    # train_idx / val_idx are the canonical record: re-running with the same seed
    # produces the same split, but storing indices removes any remaining ambiguity.
    split_meta = {
        'seed':           cfg.seed,
        'val_frac':       cfg.val_frac,
        'n_total':        n,
        'n_train':        int(len(tr_idx)),
        'n_val':          int(len(va_idx)),
        'n_families':     n_fam,
        'train_families': tr_fams,
        'val_families':   va_fams,
        'train_idx':      tr_idx.tolist(),   # exact indices — fully reproducible
        'val_idx':        va_idx.tolist(),
    }

    train_loader = DataLoader(
        Subset(dataset, tr_idx),
        batch_size  = cfg.batch_size,
        shuffle     = True,
        collate_fn  = collate_rnastralign,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device != 'cpu'),
    )
    val_loader = DataLoader(
        Subset(dataset, va_idx),
        batch_size  = cfg.batch_size * 2,
        shuffle     = False,
        collate_fn  = collate_rnastralign,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device != 'cpu'),
    )

    # ── Model & optimiser ──────────────────────────────────────────────────────
    model  = BenderPretrainModel(
        model_dim   = cfg.model_dim,
        num_layers  = cfg.num_layers,
        reduced_dim = cfg.reduced_dim,
        ff_dim      = cfg.ff_dim,
        dropout     = cfg.dropout,
        max_len     = cfg.max_len,
    ).to(device)
    print(f'Model:  dim={cfg.model_dim} layers={cfg.num_layers} r={cfg.reduced_dim} '
          f'| {model.get_num_params():,} params')

    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total  = cfg.epochs * len(train_loader)
    sched  = WarmupCosineScheduler(opt, cfg.warmup_steps, total)
    scaler = torch.amp.GradScaler('cuda') if amp_ok else None

    best_f1      = -np.inf
    best_metrics: Dict = {}
    best_enc_sd  = {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()}
    no_improve   = 0
    start_epoch  = 1

    # ── Resume ─────────────────────────────────────────────────────────────────
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        opt.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        if scaler and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch'] + 1
        best_f1      = ckpt['best_f1']
        best_metrics = ckpt['best_metrics']
        best_enc_sd  = ckpt['best_enc_sd']
        no_improve   = ckpt['no_improve']
        print(f'Resumed from epoch {ckpt["epoch"]} (best_f1={best_f1:.4f})')

    print(f'\nEpochs={cfg.epochs} | batch={cfg.batch_size} | lr={cfg.lr} '
          f'| patience={cfg.patience} | eval_every={cfg.eval_every}\n')

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0         = time.time()
        train_loss = pretrain_epoch(model, train_loader, opt, sched, device, cfg, scaler)
        elapsed    = time.time() - t0

        should_eval = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
        if should_eval:
            metrics  = pretrain_eval(model, val_loader, device, cfg)
            f1       = metrics.get('pair_f1', 0.0)
            improved = f1 > best_f1
            if improved:
                best_f1      = f1
                best_metrics = metrics.copy()
                best_enc_sd  = {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()}
                no_improve   = 0
                torch.save({
                    'geom_encoder_state_dict': best_enc_sd,
                    'pair_head_state_dict': {k: v.cpu().clone()
                                             for k, v in model.pair_head.state_dict().items()},
                    'ss_head_state_dict':   {k: v.cpu().clone()
                                             for k, v in model.ss_head.state_dict().items()},
                    'mfe_head_state_dict':  {k: v.cpu().clone()
                                             for k, v in model.mfe_head.state_dict().items()},
                    'mfe_pool_state_dict':  {k: v.cpu().clone()
                                             for k, v in model.mfe_pool.state_dict().items()},
                    'epoch': epoch,
                    'val_metrics': best_metrics,
                    'cfg': dataclasses.asdict(cfg),
                    'split': split_meta,
                }, best_path)
            else:
                no_improve += 1

            m_str = ' | '.join(f'{k}={v:.4f}' for k, v in metrics.items())
            print(f'  E{epoch:03d} loss={train_loss:.4f} | {m_str} '
                  f'[{elapsed:.1f}s] {"*" if improved else ""}')

            if no_improve >= cfg.patience:
                print(f'  Early stop at epoch {epoch} '
                      f'(no improvement for {cfg.patience} evals)')
                break
        else:
            print(f'  E{epoch:03d} loss={train_loss:.4f} [{elapsed:.1f}s]')

        torch.save({
            'epoch': epoch, 'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'best_f1': best_f1, 'best_metrics': best_metrics,
            'best_enc_sd': best_enc_sd, 'no_improve': no_improve,
        }, resume_path)

    print(f'\nBest pair_f1: {best_f1:.4f}')
    print(f'Metrics: ' + ' | '.join(f'{k}={v:.4f}' for k, v in best_metrics.items()))
    print(f'Geometry encoder: {best_path}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> PretrainBenderConfig:
    p = argparse.ArgumentParser(
        description='Pretrain Bender geometry encoder on RNA structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',          required=True)
    p.add_argument('--data_format',   default='json', choices=['json', 'csv', 'bpseq'])
    p.add_argument('--max_len',       type=int,   default=256)
    p.add_argument('--no_oracle_edges', action='store_true')
    p.add_argument('--model_dim',     type=int,   default=128)
    p.add_argument('--num_layers',    type=int,   default=4)
    p.add_argument('--reduced_dim',   type=int,   default=32)
    p.add_argument('--dropout',       type=float, default=0.1)
    p.add_argument('--lambda_pair',   type=float, default=1.0)
    p.add_argument('--lambda_ss',     type=float, default=0.1)
    p.add_argument('--lambda_mfe',    type=float, default=0.01,
                   help='MFE regression weight (only active when batch has mfe key)')
    p.add_argument('--lambda_curv',   type=float, default=0.01)
    p.add_argument('--epochs',        type=int,   default=60)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--patience',      type=int,   default=10)
    p.add_argument('--val_frac',      type=float, default=0.15)
    p.add_argument('--no_amp',        action='store_true')
    p.add_argument('--eval_every',    type=int,   default=2)
    p.add_argument('--device',        default='auto')
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--output_dir',    default='pretrain_bender_out')
    p.add_argument('--seed',          type=int,   default=42)

    args = p.parse_args()
    return PretrainBenderConfig(
        data          = args.data,
        data_format   = args.data_format,
        max_len       = args.max_len,
        oracle_edges  = not args.no_oracle_edges,
        model_dim     = args.model_dim,
        num_layers    = args.num_layers,
        reduced_dim   = args.reduced_dim,
        dropout       = args.dropout,
        lambda_pair   = args.lambda_pair,
        lambda_ss     = args.lambda_ss,
        lambda_mfe    = args.lambda_mfe,
        lambda_curv   = args.lambda_curv,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        patience      = args.patience,
        val_frac      = args.val_frac,
        use_amp       = not args.no_amp,
        eval_every    = args.eval_every,
        device        = args.device,
        num_workers   = args.num_workers,
        output_dir    = args.output_dir,
        seed          = args.seed,
    )


if __name__ == '__main__':
    cfg = parse_args()
    run_pretrain(cfg)