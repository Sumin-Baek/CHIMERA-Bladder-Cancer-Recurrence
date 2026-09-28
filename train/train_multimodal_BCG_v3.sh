#!/usr/bin/env bash
# Training command for the CHIMERA 2025 Task 3 submission (Team NMIL):
# pathway-based RNA features, [mean; max] WSI pooling, AdamW lr 5e-5, weight decay 0.01,
# batch size 16, early stopping (patience 30).
#
#   --gene_method pathway_based -> 8 pathways x 5 statistics = 40 RNA features
#   --wsi_method  concat_based  -> 2048-d slide vector
# which matches MODEL_CONFIG in ../inference.py (clinical 27 / rna 40 / wsi 2048).

set -euo pipefail
cd "$(dirname "$0")"

python BCG_Multimodal_model4.py \
  --task_name 'lr5e5_gm_pathway_wm_concat_T' \
  --lr 5e-5 \
  --gene_method 'pathway_based' \
  --wsi_method 'concat_based' \
  --data_root "${DATA_ROOT:-./data/new_clinic}" \
  --save_dir  "${SAVE_DIR:-./results}"
