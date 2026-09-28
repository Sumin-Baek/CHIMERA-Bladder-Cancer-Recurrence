# CHIMERA 2025 · Task 3 — Bladder cancer recurrence after BCG (Team NMIL)

Source code of Team NMIL's submission to **Task 3 of the CHIMERA challenge (MICCAI 2025)**:
predicting the likelihood of recurrence in BCG-treated non-muscle-invasive bladder cancer
from H&E whole-slide images, bulk RNA-seq and clinical data.
The submission ranked **3rd** on the final leaderboard.

> This repository is published as part of the challenge's open-science policy.
> The code is kept **functionally identical to what was submitted** (including
> its shortcomings, listed under [As-submitted notes](#as-submitted-notes)),
> so that the leaderboard result can be traced back to the exact pipeline.

- Challenge: <https://chimera.grand-challenge.org/>
- Team NMIL: Sumin Baek (DGIST), Jeonghwan Kim, Seungjun Lee

## Method overview

```
H&E WSI ──► 300 patches (level 0, grid stride 1792 px, non-background) ──► UNI ViT-L/16 ──► (300, 1024)
                                                                                │
                                                                                ▼ [mean ; max] over patches
                                                                             WSI vector (2048)
clinical JSON ──► one-hot (11 categorical) + standardised (age, no_instillations) ──► clinical vector (27)
RNA JSON ──► 8 hand-curated gene sets × {mean, std, median, max, q75} ──► standardised ──► RNA vector (40)

                     ┌────────────────────────────────────────────────────┐
[clin, rna, wsi] ──► │ per-modality MLP → 3 tokens (+pos. emb.)          │
                     │ → 4 pre-norm transformer layers (8 heads, d=256)  │ ──► sigmoid risk (submitted as is)
                     │ → multi-head attention → softmax gate → MLP       │
                     └────────────────────────────────────────────────────┘
                              AdvancedBCGTransformerNet (my_survival_model.py)
```

- **Loss:** Cox partial likelihood (`pycox.models.loss.CoxPHLoss`) on the sigmoid output.
  Auxiliary heads (time regression, BRS classification, uncertainty) exist in the model but
  receive no loss and are not used.
- **Training:** 176 patients, single stratified 80/20 split (`random_state=42`); AdamW, lr 5e-5,
  weight decay 0.01, batch size 16, `ReduceLROnPlateau` (factor 0.5, patience 5), early stopping on
  validation C-index (patience 30); the best-epoch checkpoint (`epoch_9`) was submitted.
- **Output:** the sigmoid output is a Cox-trained risk score (higher = earlier recurrence) and is
  written to the output JSON as is.
- **Ablation (validation leaderboard C-index):** Top-genes+attention 0.457, Top-genes+concat 0.652,
  Pathway+attention 0.544, **Pathway+concat 0.783 (submitted)**. The alternative branches remain
  selectable in the code (`--gene_method top_genes`, `--wsi_method attention_based`).
- **Patch encoder:** [UNI](https://github.com/mahmoodlab/UNI) (Chen et al., *Nat. Med.* 2024), frozen.
- **Patch sampling at inference:** the provided tissue masks caused processing failures in our
  pipeline, so patches are sampled on a coarse grid over the slide and kept by a background-intensity
  threshold (up to 300 patches). Training used the pre-extracted 1024-d patch features distributed by the organisers.

## Repository layout

```
.
├── inference.py            Grand-Challenge entry point (reads /input, writes /output)
├── feature_extractor.py    UNI loading + patch sampling/encoding with OpenSlide
├── preprocessor.py         BCGDataPreprocessor (clinical / RNA / WSI aggregation) + pickle loader
├── my_survival_model.py    AdvancedBCGTransformerNet
├── Dockerfile, requirements.in/.txt
├── model/                  (not tracked) weights — see below
└── train/
    ├── BCG_Multimodal_model4.py          training script used for the submission
    ├── train_multimodal_BCG_v3.sh        training command
    └── requirements.txt                  training environment (torch 2.7.1 + cu118)
```

## Model weights

The following files are required under `model/` and are **not** in this repository:

| File | Source |
|---|---|
| `vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin` | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) on Hugging Face (gated; request access) |
| `bcg_best_model_epoch_9.pth` | Not publicly released; contact the corresponding author. |
| `bcg_preprocessor.pkl` | Not publicly released; contact the corresponding author. |

`bcg_preprocessor.pkl` was pickled from the training script running as `__main__`;
`preprocessor.load_preprocessor()` remaps the class reference so it loads from any module.

## Inference (Grand-Challenge container)

```bash
# 1. put the three model files under model/ (see above)
# 2. build
docker build --platform linux/amd64 -t nmil-chimera-task3 .
# 3. run on one case laid out like the challenge interface
docker run --rm --gpus all \
  -v /path/to/case/input:/input:ro \
  -v /path/to/case/output:/output \
  nmil-chimera-task3
# -> /output/likelihood-of-bladder-cancer-recurrence.json
```

Expected input layout (`/input`):

```
images/bladder-cancer-tissue-biopsy-wsi/<id>.tif
images/tissue-mask/<id>.tif
bulk-rna-seq-bladder-cancer.json
chimera-clinical-data-of-bladder-cancer-recurrence-patients.json
inputs.json                       (optional; used to select the interface)
```

## Training

Environment: `pip install -r train/requirements.txt` (CUDA 11.8 wheels).

Data layout expected by `--data_root`:

```
<data_root>/
├── data/<patient_id>/<patient_id>_CD.json      clinical
├── data/<patient_id>/<patient_id>_RNA.json     bulk RNA-seq
└── features/
    ├── features/<patient_id>_HE.pt             patch features (N, 1024) as distributed by the challenge
    └── coordinates/<patient_id>_HE.npy
```

```bash
cd train
DATA_ROOT=/path/to/data_root SAVE_DIR=./results bash train_multimodal_BCG_v3.sh
```

The submitted checkpoint was trained with `--gene_method pathway_based --wsi_method concat_based`
(this is what `MODEL_CONFIG` in `inference.py` encodes:
clinical 27 / RNA 40 / WSI 2048).
Outputs go to `<save_dir>/<task_name>/` (`models/bcg_best_model_epoch_*.pth`, `bcg_preprocessor.pkl`,
`bcg_training.log`, `figures/`).

The published training script keeps the fusion model and the two data-representation options
that were part of the ablation; other model variants explored during the challenge were removed.
The code path executed for the submission is unchanged.

## As-submitted notes

At inference, `StandardScaler.fit_transform` is applied per patient (n = 1), so the RNA features
and the two numerical clinical features are constant at test time. The submitted model therefore
effectively used the one-hot clinical features and the WSI vector.

## Citation

If you use this code, please cite the CHIMERA challenge paper (in preparation; reference will be
added here once available) and, for the patch encoder, Chen et al., "Towards a general-purpose foundation model for
computational pathology", *Nature Medicine*, 2024.

## License

MIT — see [LICENSE](LICENSE). The UNI weights are subject to their own license (CC-BY-NC-ND 4.0).
