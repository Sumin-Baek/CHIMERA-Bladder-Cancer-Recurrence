"""
Grand-Challenge inference entry point for CHIMERA Task 3
(likelihood of recurrence in BCG-treated NMIBC).

Inputs  (mounted at /input):
    images/bladder-cancer-tissue-biopsy-wsi/*.tif                 H&E whole-slide image
    images/tissue-mask/*.tif                                       tissue mask (not used, see README)
    bulk-rna-seq-bladder-cancer.json                               gene -> expression
    chimera-clinical-data-of-bladder-cancer-recurrence-patients.json

Output (written to /output):
    likelihood-of-bladder-cancer-recurrence.json                   single float

Model artefacts expected under MODEL_PATH (not shipped in this repository):
    vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin   UNI patch encoder
    bcg_best_model_epoch_9.pth                                     AdvancedBCGTransformerNet checkpoint
    bcg_preprocessor.pkl                                           fitted BCGDataPreprocessor
"""

from __future__ import annotations

import gc
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch

from feature_extractor import extract_features_with_openslide, get_feature_extractor
from my_survival_model import AdvancedBCGTransformerNet
from preprocessor import BCGDataPreprocessor, load_preprocessor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/app/model")

# Must match the checkpoint: 25 one-hot + 2 numerical clinical features,
# 8 pathways x 5 statistics for RNA, [mean; max] of UNI patch features for WSI (2 x 1024).
MODEL_CONFIG = {
    "clinical_dim": 27,
    "rna_dim": 40,
    "wsi_dim": 2048,
    "hidden_dim": 256,
    "num_transformer_layers": 4,
}

INTERFACE_KEY = (
    "bladder-cancer-tissue-biopsy-whole-slide-image",
    "bulk-rna-seq-bladder-cancer",
    "chimera-clinical-data-of-bladder-cancer-recurrence",
    "tissue-mask",
)


class SafePreprocessor:
    """Wrapper that normalises missing-value tokens and never raises.

    NOTE (as submitted): on any exception the clinical branch falls back to a
    random N(0,1) vector of the expected size and the RNA branch to zeros, so
    that the container always produces an output.
    """

    def __init__(self, original_preprocessor: BCGDataPreprocessor):
        self.preprocessor = original_preprocessor

    def preprocess_clinical_data(self, clinical_data_list):
        try:
            cleaned = []
            for data in clinical_data_list:
                cleaned.append({
                    k: ("Unknown" if v in [None, "NA", "N/A", "", -1, -999] else v)
                    for k, v in data.items()
                })
            return self.preprocessor.preprocess_clinical_data(cleaned)
        except Exception as e:  # noqa: BLE001
            print(f"Clinical preprocessing failed: {e}")
            return np.random.randn(1, MODEL_CONFIG["clinical_dim"]).astype(np.float32), {}

    def preprocess_rna_data(self, rna_data_list, method="pathway_based"):
        try:
            if not rna_data_list or not rna_data_list[0]:
                return np.zeros((1, MODEL_CONFIG["rna_dim"]), dtype=np.float32), {}
            return self.preprocessor.preprocess_rna_data(rna_data_list, method)
        except Exception as e:  # noqa: BLE001
            print(f"RNA preprocessing failed: {e}")
            return np.zeros((1, MODEL_CONFIG["rna_dim"]), dtype=np.float32), {}


def run() -> int:
    interface_key = get_interface_key()
    handler = {INTERFACE_KEY: predict_recurrence}[interface_key]
    return handler()


def predict_recurrence() -> int:
    _show_torch_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load models -------------------------------------------------------
    feature_extractor, transform, _ = get_feature_extractor(
        MODEL_PATH / "vit_large_patch16_224.dinov2.uni_mass100k" / "pytorch_model.bin"
    )
    survival_model = AdvancedBCGTransformerNet(config=MODEL_CONFIG)
    checkpoint = torch.load(MODEL_PATH / "bcg_best_model_epoch_9.pth", map_location=device)
    survival_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    survival_model.to(device).eval()

    original_preprocessor = load_preprocessor(MODEL_PATH / "bcg_preprocessor.pkl")
    preprocessor = SafePreprocessor(original_preprocessor)
    print("Models and preprocessor loaded.")

    # --- load inputs -------------------------------------------------------
    wsi_path = glob(str(INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi/*.tif"))[0]
    mask_path = glob(str(INPUT_PATH / "images/tissue-mask/*.tif"))[0]
    rna_data = load_json_file(INPUT_PATH / "bulk-rna-seq-bladder-cancer.json")
    clinical_data = load_json_file(
        INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json"
    )

    # --- WSI: patch features -----------------------------------------------
    patch_features = extract_features_with_openslide(
        wsi_path, mask_path, feature_extractor, transform, device,
        max_patches=300, batch_size=16,
    )

    # --- clinical / RNA / WSI pooling (same order as submitted) -------------
    clinical_vec, _ = preprocessor.preprocess_clinical_data([clinical_data])
    rna_vec, _ = preprocessor.preprocess_rna_data([rna_data])
    wsi_vec = original_preprocessor.process_wsi_features([patch_features.cpu()], method="concat_based")

    # --- predict -----------------------------------------------------------
    with torch.no_grad():
        outputs = survival_model(
            clinical_data=torch.FloatTensor(clinical_vec).to(device),
            rna_data=torch.FloatTensor(rna_vec).to(device),
            wsi_data=torch.FloatTensor(wsi_vec).to(device),
        )
        # Cox-trained risk score in (0, 1): higher = higher likelihood of recurrence.
        # Submitted directly (see README, "Output").
        risk_score = outputs["progression_risk"].item()

    write_json_file(OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json", risk_score)
    print(f"Prediction saved. Risk score: {risk_score}")

    del patch_features, wsi_vec, clinical_vec, rna_vec, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def get_interface_key():
    try:
        inputs = load_json_file(INPUT_PATH / "inputs.json")
        return tuple(sorted(sv["interface"]["slug"] for sv in inputs))
    except FileNotFoundError:
        print("WARNING: inputs.json not found; assuming the default interface (local testing).")
        return INTERFACE_KEY


def load_json_file(location):
    with open(location, "r") as f:
        return json.load(f)


def write_json_file(location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _show_torch_cuda_info():
    print("=+=" * 10)
    available = torch.cuda.is_available()
    print(f"Torch CUDA is available: {available}")
    if available:
        current = torch.cuda.current_device()
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {current}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
