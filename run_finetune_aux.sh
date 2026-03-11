#!/usr/bin/env bash
# Experiment 2: Pretrained backbone + continued SS/MFE auxiliary supervision.
# Question: does the model still benefit from structure supervision during
# downstream fine-tuning, even after pretraining?
#
# --bpp_backend zero: same zero-edge input as pretraining (isolates aux loss effect).
# --aux_struct: adds ViennaRNA SS/MFE targets as auxiliary supervision signals.

set -euo pipefail

BACKBONE="pretrain_outputs/best_encoder.pt"
MRL="capsule-4214075-data/MRL_Random50Nuc_SynthesisLibrary_Sample"
TE="capsule-4214075-data/TE_REL_Endogenous_Cao"
EXP="capsule-4214075-data/Experimental_Data"

# ── MRL: 8 conditions ────────────────────────────────────────────────────────

echo "=== MRL 4.1 — egfp_unmod_1 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.1_train_data_GSM3130435_egfp_unmod_1.csv" \
  --test_data       "$MRL/4.1_test_data_GSM3130435_egfp_unmod_1.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.1

echo "=== MRL 4.4 — egfp_unmod_2 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.4_train_data_GSM3130436_egfp_unmod_2.csv" \
  --test_data       "$MRL/4.4_test_data_GSM3130436_egfp_unmod_2.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.4

echo "=== MRL 4.7 — egfp_pseudo_1 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.7_train_data_GSM3130437_egfp_pseudo_1.csv" \
  --test_data       "$MRL/4.7_test_data_GSM3130437_egfp_pseudo_1.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.7

echo "=== MRL 4.10 — egfp_pseudo_2 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.10_train_data_GSM3130438_egfp_pseudo_2.csv" \
  --test_data       "$MRL/4.10_test_data_GSM3130438_egfp_pseudo_2.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.10

echo "=== MRL 4.13 — mcherry_1 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.13_train_data_GSM3130441_mcherry_1.csv" \
  --test_data       "$MRL/4.13_test_data_GSM3130441_mcherry_1.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.13

echo "=== MRL 4.16 — mcherry_2 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.16_train_data_GSM3130442_mcherry_2.csv" \
  --test_data       "$MRL/4.16_test_data_GSM3130442_mcherry_2.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.16

echo "=== MRL 4.19 — egfp_m1pseudo_1 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.19_train_data_GSM3130439_egfp_m1pseudo_1.csv" \
  --test_data       "$MRL/4.19_test_data_GSM3130439_egfp_m1pseudo_1.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.19

echo "=== MRL 4.22 — egfp_m1pseudo_2 ==="
python train_utr.py \
  --task            mrl \
  --data            "$MRL/4.22_train_data_GSM3130440_egfp_m1pseudo_2.csv" \
  --test_data       "$MRL/4.22_test_data_GSM3130440_egfp_m1pseudo_2.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/mrl_4.22

# ── TE: 3 cell lines, 5-fold CV ───────────────────────────────────────────────

echo "=== TE HEK ==="
python train_utr.py \
  --task            te \
  --data            "$TE/HEK_sequence.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           5 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/te_HEK

echo "=== TE Muscle ==="
python train_utr.py \
  --task            te \
  --data            "$TE/Muscle_sequence.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           5 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/te_Muscle

echo "=== TE pc3 ==="
python train_utr.py \
  --task            te \
  --data            "$TE/pc3_sequence.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           5 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/te_pc3

# ── Experimental vary-length (RLU) ────────────────────────────────────────────

echo "=== RLU ==="
python train_utr.py \
  --task            rlu \
  --data            "$EXP/Experimental_data_revised_label.csv" \
  --bpp_backend     zero \
  --bpp_cache_dir   ~/bpp_cache \
  --model_dim       128 \
  --num_layers      6 \
  --reduced_dim     16 \
  --dropout         0.1 \
  --pooling         attention \
  --aux_struct \
  --lambda_ss       0.1 \
  --lambda_mfe      0.01 \
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      1 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_aux/rlu
