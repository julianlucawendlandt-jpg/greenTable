#!/usr/bin/env bash
# Experiment 1: Pretrained backbone, NO auxiliary structure supervision during fine-tune.
# Question: did pretraining alone bake in the useful structure bias?

set -euo pipefail

BACKBONE="pretrain_outputs/best_encoder.pt"
MRL="capsule-4214075-data/MRL_Random50Nuc_SynthesisLibrary_Sample"
TE="capsule-4214075-data/TE_REL_Endogenous_Cao"
EXP="capsule-4214075-data/Experimental_Data"

# ── MRL: 8 conditions, each with explicit train/test split ────────────────────
declare -A MRL_PAIRS=(
  ["4.1"]="4.1_train_data_GSM3130435_egfp_unmod_1   4.1_test_data_GSM3130435_egfp_unmod_1"
  ["4.4"]="4.4_train_data_GSM3130436_egfp_unmod_2   4.4_test_data_GSM3130436_egfp_unmod_2"
  ["4.7"]="4.7_train_data_GSM3130437_egfp_pseudo_1  4.7_test_data_GSM3130437_egfp_pseudo_1"
  ["4.10"]="4.10_train_data_GSM3130438_egfp_pseudo_2 4.10_test_data_GSM3130438_egfp_pseudo_2"
  ["4.13"]="4.13_train_data_GSM3130441_mcherry_1     4.13_test_data_GSM3130441_mcherry_1"
  ["4.16"]="4.16_train_data_GSM3130442_mcherry_2     4.16_test_data_GSM3130442_mcherry_2"
  ["4.19"]="4.19_train_data_GSM3130439_egfp_m1pseudo_1 4.19_test_data_GSM3130439_egfp_m1pseudo_1"
  ["4.22"]="4.22_train_data_GSM3130440_egfp_m1pseudo_2 4.22_test_data_GSM3130440_egfp_m1pseudo_2"
)

for KEY in "${!MRL_PAIRS[@]}"; do
  read -r TRAIN TEST <<< "${MRL_PAIRS[$KEY]}"
  echo "=== MRL $KEY ==="
  python train_utr.py \
    --task            mrl \
    --data            "$MRL/${TRAIN}.csv" \
    --test_data       "$MRL/${TEST}.csv" \
    --bpp_backend     zero \
    --bpp_cache_dir   ~/bpp_cache \
    --model_dim       128 \
    --num_layers      6 \
    --reduced_dim     16 \
    --dropout         0.1 \
    --pooling         attention \
    --epochs          60 \
    --batch_size      256 \
    --lr              3e-4 \
    --weight_decay    1e-2 \
    --patience        10 \
    --warmup_steps    200 \
    --folds           1 \
    --val_frac        0.2 \
    --seed            42 \
    --eval_every      5 \
    --num_workers     4 \
    --pretrained_backbone "$BACKBONE" \
    --output_dir      finetune_plain/mrl_${KEY}
done

# ── TE: 3 cell lines, 5-fold CV (no explicit test split) ─────────────────────
for CELL in HEK Muscle pc3; do
  echo "=== TE $CELL ==="
  python train_utr.py \
    --task            te \
    --data            "$TE/${CELL}_sequence.csv" \
    --bpp_backend     zero \
    --bpp_cache_dir   ~/bpp_cache \
    --model_dim       128 \
    --num_layers      6 \
    --reduced_dim     16 \
    --dropout         0.1 \
    --pooling         attention \
    --epochs          60 \
    --batch_size      256 \
    --lr              3e-4 \
    --weight_decay    1e-2 \
    --patience        10 \
    --warmup_steps    200 \
    --folds           5 \
    --val_frac        0.2 \
    --seed            42 \
    --eval_every      5 \
    --num_workers     4 \
    --pretrained_backbone "$BACKBONE" \
    --output_dir      finetune_plain/te_${CELL}
done

# ── Experimental vary-length (RLU), single 80/20 split ───────────────────────
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
  --epochs          60 \
  --batch_size      256 \
  --lr              3e-4 \
  --weight_decay    1e-2 \
  --patience        10 \
  --warmup_steps    200 \
  --folds           1 \
  --val_frac        0.2 \
  --seed            42 \
  --eval_every      5 \
  --num_workers     4 \
  --pretrained_backbone "$BACKBONE" \
  --output_dir      finetune_plain/rlu
