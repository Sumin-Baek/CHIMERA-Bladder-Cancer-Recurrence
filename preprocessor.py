"""
Data preprocessing for the CHIMERA Task 3 (BCG-treated NMIBC recurrence) pipeline.

This module holds the single definition of ``BCGDataPreprocessor`` used at
inference time.  It is also the class that was pickled after training
(``model/bcg_preprocessor.pkl``), so its attribute layout must stay compatible
with that file:

    clinical_scaler : sklearn.preprocessing.StandardScaler
    rna_scaler      : sklearn.preprocessing.StandardScaler
    label_encoders  : dict[str, sklearn.preprocessing.LabelEncoder]
    categorical_features / numerical_features : list[str]
    data_root       : pathlib.Path

The pickle was written from the training script executed as ``__main__``, so
the stored class reference is ``__main__.BCGDataPreprocessor``.  Use
:func:`load_preprocessor` to load it; it remaps that reference to this module.

NOTE (as submitted): ``preprocess_clinical_data`` and ``preprocess_rna_data``
call ``fit_transform`` on the scalers every time they are invoked.  During
training this was called on the whole cohort; at inference it is called on a
single patient, which standardises every numerical column to exactly 0.  The
behaviour is kept unchanged here so that the published code reproduces the
challenge submission.  See README, section "As-submitted notes".
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AttentionPooling(nn.Module):
    """Attention pooling over a bag of patch features (ablation variant, not used
    by the submitted model).  Note that it is instantiated untrained inside
    ``process_wsi_features``; the submitted model uses ``concat_based`` pooling.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_patches, input_dim) -> (input_dim,)
        a = self.attention_net(x).squeeze(-1)          # (num_patches,)
        att_weights = F.softmax(a, dim=0)               # (num_patches,)
        context_vector = torch.matmul(att_weights.unsqueeze(0), x)  # (1, input_dim)
        return context_vector.squeeze(0)


class BCGDataPreprocessor:
    """Preprocessor for clinical, bulk RNA-seq and WSI features.

    Expected training layout under ``data_root``::

        data/<patient_id>/<patient_id>_CD.json    clinical data
        data/<patient_id>/<patient_id>_RNA.json   bulk RNA-seq (gene -> value)
        features/features/<patient_id>_HE.pt      pre-extracted patch features (N, 1024)
        features/coordinates/<patient_id>_HE.npy  patch coordinates
    """

    def __init__(self, data_root_path: str):
        self.data_root = Path(data_root_path)
        self.clinical_scaler = StandardScaler()
        self.rna_scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}

        self.categorical_features = [
            "sex", "smoking", "tumor", "stage", "substage",
            "grade", "reTUR", "LVI", "variant", "EORTC", "BRS",
        ]
        self.numerical_features = ["age", "no_instillations"]

        print(f"Data root: {self.data_root}")

    # ------------------------------------------------------------------ I/O
    def get_patient_list(self) -> List[str]:
        patient_ids = set()
        data_root = self.data_root / "data"
        for patient_folder in data_root.iterdir():
            if patient_folder.is_dir():
                patient_id = patient_folder.name
                if (patient_folder / f"{patient_id}_CD.json").exists():
                    patient_ids.add(patient_id)
        patient_list = sorted(patient_ids)
        print(f"Found {len(patient_list)} patients")
        return patient_list

    def load_clinical_data(self, patient_id: str) -> Dict:
        cd_file = self.data_root / "data" / patient_id / f"{patient_id}_CD.json"
        if not cd_file.exists():
            raise FileNotFoundError(f"Clinical data not found: {cd_file}")
        with open(cd_file, "r") as f:
            return json.load(f)

    def load_rna_data(self, patient_id: str) -> Dict:
        rna_file = self.data_root / "data" / patient_id / f"{patient_id}_RNA.json"
        if not rna_file.exists():
            print(f"WARNING: RNA data not found for {patient_id}")
            return {}
        with open(rna_file, "r") as f:
            return json.load(f)

    def load_wsi_features(self, patient_id: str) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        feature_file = self.data_root / "features" / "features" / f"{patient_id}_HE.pt"
        coord_file = self.data_root / "features" / "coordinates" / f"{patient_id}_HE.npy"

        features = coordinates = None
        if feature_file.exists():
            try:
                features = torch.load(feature_file, map_location="cpu")
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: could not load features for {patient_id}: {e}")
        if coord_file.exists():
            try:
                coordinates = np.load(coord_file)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: could not load coordinates for {patient_id}: {e}")
        return features, coordinates

    # ------------------------------------------------------------- clinical
    def preprocess_clinical_data(self, clinical_data_list: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """One-hot encode categorical variables and standardise numerical ones.

        Encoders are fitted on first use and reused afterwards (unseen
        categories are mapped to ``'Unknown'``).  The numerical scaler is
        re-fitted on every call (see module docstring).
        """
        print("Preprocessing clinical data ...")
        df = pd.DataFrame(clinical_data_list)

        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f"  missing {col}: {missing}/{len(df)} ({missing / len(df) * 100:.1f}%)")

        processed_categorical_features, categorical_feature_names = [], []
        processed_numerical_features, numerical_feature_names = [], []

        # 1. categorical -> label encode -> one-hot
        for feature in self.categorical_features:
            if feature not in df.columns:
                continue
            df[feature] = df[feature].fillna("Unknown")

            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = df[feature].astype(str)
                self.label_encoders[feature].fit(list(df[feature].unique()))
                encoded = self.label_encoders[feature].transform(df[feature])
            else:
                df[feature] = df[feature].astype(str)
                current_classes = set(self.label_encoders[feature].classes_)
                new_values = set(df[feature].unique()) - current_classes
                if new_values:
                    df.loc[df[feature].isin(new_values), feature] = "Unknown"
                encoded = self.label_encoders[feature].transform(df[feature])

            n_classes = len(self.label_encoders[feature].classes_)
            processed_categorical_features.append(np.eye(n_classes)[encoded])
            categorical_feature_names.extend(
                f"{feature}_{cls}" for cls in self.label_encoders[feature].classes_
            )

        # 2. numerical -> mean imputation -> standardise
        numerical_df = pd.DataFrame()
        for feature in self.numerical_features:
            if feature not in df.columns:
                continue
            values = df[feature].copy()
            if feature == "no_instillations":
                values = values.replace(-1, np.nan)  # -1 encodes missing
            values = values.fillna(values.mean())
            numerical_df[feature] = values
            numerical_feature_names.append(feature)

        if not numerical_df.empty:
            processed_numerical_features = self.clinical_scaler.fit_transform(numerical_df)

        # 3. concatenate
        final_features_list, final_feature_names = [], []
        if processed_categorical_features:
            final_features_list.append(np.concatenate(processed_categorical_features, axis=1))
            final_feature_names.extend(categorical_feature_names)
        if not numerical_df.empty:
            final_features_list.append(processed_numerical_features)
            final_feature_names.extend(numerical_feature_names)

        if final_features_list:
            all_features = np.concatenate(final_features_list, axis=1)
        else:
            all_features = np.zeros((len(df), 1))

        feature_info = {
            "feature_names": final_feature_names,
            "n_features": all_features.shape[1],
            "encoders": self.label_encoders.copy(),
            "scaler": self.clinical_scaler,
        }
        print(f"Clinical features: {all_features.shape}")
        return all_features.astype(np.float32), feature_info

    # ------------------------------------------------------------------ RNA
    def preprocess_rna_data(self, rna_data_list: List[Dict], method: str = "pathway_based") -> Tuple[np.ndarray, Dict]:
        """Summarise bulk RNA-seq into pathway statistics (or top-variance genes)."""
        print("Preprocessing RNA data ...")

        if not rna_data_list or all(not rna_data for rna_data in rna_data_list):
            print("WARNING: no RNA data; generating dummy features")
            dummy = np.random.randn(len(rna_data_list), 50).astype(np.float32)
            return dummy, {"method": "dummy", "n_features": 50}

        all_genes = set()
        for rna_data in rna_data_list:
            if rna_data:
                all_genes.update(rna_data.keys())
        all_genes = sorted(all_genes)
        print(f"Genes available: {len(all_genes)}")

        if method == "pathway_based":
            processed_features, pathway_names = [], []
            for pathway_name, genes in self._get_bcg_relevant_pathways().items():
                available_genes = [g for g in genes if g in all_genes]
                if not available_genes:
                    continue
                pathway_values = []
                for rna_data in rna_data_list:
                    if rna_data:
                        gene_values = [rna_data.get(gene, 0) for gene in available_genes]
                        stats = [
                            np.mean(gene_values),
                            np.std(gene_values),
                            np.median(gene_values),
                            np.max(gene_values),
                            np.percentile(gene_values, 75),
                        ]
                    else:
                        stats = [0.0] * 5
                    pathway_values.append(stats)
                processed_features.append(np.array(pathway_values))
                pathway_names.extend(
                    f"{pathway_name}_{s}" for s in ["mean", "std", "median", "max", "q75"]
                )

            if processed_features:
                rna_features = np.concatenate(processed_features, axis=1)
            else:
                print("WARNING: no pathway genes found; falling back to top genes")
                rna_features = self._process_top_genes(rna_data_list, all_genes)
                pathway_names = [f"gene_{i}" for i in range(rna_features.shape[1])]

        elif method == "top_genes":
            rna_features = self._process_top_genes(rna_data_list, all_genes)
            pathway_names = [f"gene_{i}" for i in range(rna_features.shape[1])]
        else:
            raise ValueError(f"Unknown RNA method: {method}")

        if rna_features.shape[1] > 0:
            rna_features = self.rna_scaler.fit_transform(rna_features)

        rna_info = {
            "method": method,
            "feature_names": pathway_names,
            "n_features": rna_features.shape[1],
            "available_genes": len(all_genes),
        }
        print(f"RNA features: {rna_features.shape}")
        return rna_features.astype(np.float32), rna_info

    @staticmethod
    def _get_bcg_relevant_pathways() -> Dict[str, List[str]]:
        """Hand-curated gene sets used as pathway summaries (8 pathways x 5 stats = 40 features)."""
        return {
            "IMMUNE_RESPONSE": ["IFNG", "TNF", "IL1B", "IL6", "IL10", "IL12A", "IL12B",
                                "CD8A", "CD4", "CD68", "CD3E", "PTPRC"],
            "T_CELL_ACTIVATION": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "GZMA",
                                  "GZMB", "PRF1", "IFNG", "IL2"],
            "INFLAMMATORY_RESPONSE": ["TNF", "IL1B", "IL6", "CXCL8", "CCL2", "CCL5", "CXCL10",
                                      "NOS2", "PTGS2", "NFE2L2"],
            "APOPTOSIS": ["TP53", "BAX", "BCL2", "CASP3", "CASP8", "CASP9", "FAS",
                          "FASLG", "BID", "BAK1", "CYCS"],
            "CELL_CYCLE": ["CCND1", "CDK4", "CDK6", "RB1", "E2F1", "CDKN1A", "CDKN2A",
                           "MYC", "TP53", "CCNE1"],
            "ANGIOGENESIS": ["VEGFA", "VEGFR2", "FGF2", "PDGFB", "ANGPT1", "ANGPT2",
                             "HIF1A", "EPAS1"],
            "INVASION_METASTASIS": ["CDH1", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2",
                                    "VIM", "MMP2", "MMP9", "TIMP1"],
            "DNA_REPAIR": ["BRCA1", "BRCA2", "ATM", "ATR", "PARP1", "RAD51", "XRCC1",
                           "ERCC1", "MSH2", "MLH1"],
        }

    @staticmethod
    def _process_top_genes(rna_data_list: List[Dict], all_genes: List[str], top_k: int = 100) -> np.ndarray:
        """Take the first ``top_k`` genes (alphabetical) and keep the 50 with highest variance."""
        gene_matrix = []
        for gene in all_genes[:top_k]:
            gene_matrix.append([rna_data.get(gene, 0) if rna_data else 0 for rna_data in rna_data_list])
        gene_matrix = np.array(gene_matrix).T  # (samples, genes)
        if gene_matrix.shape[1] > 50:
            top_indices = np.argsort(np.var(gene_matrix, axis=0))[-50:]
            gene_matrix = gene_matrix[:, top_indices]
        return gene_matrix

    # ------------------------------------------------------------------ WSI
    def process_wsi_features(self, features_list: List[torch.Tensor], method: str = "concat_based") -> np.ndarray:
        """Aggregate per-patient patch features (N, 1024) into one slide-level vector.

        ``concat_based``    -> [mean; max] pooling (2048-d)  <- submitted model
        ``attention_based`` -> untrained attention pooling (1024-d), ablation only.
        """
        print("Processing WSI features ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_numpy_features = []

        for features in features_list:
            if features is None or not isinstance(features, torch.Tensor) or len(features) == 0:
                output_dim = 2048 if method == "concat_based" else 1024
                final_numpy_features.append(np.zeros((1, output_dim), dtype=np.float32))
                continue

            with torch.no_grad():
                features = features.to(device)
                if method == "concat_based":
                    feature_mean = torch.mean(features, dim=0, keepdim=True)
                    feature_max = torch.max(features, dim=0, keepdim=True).values
                    pooled = torch.cat([feature_mean, feature_max], dim=1)
                elif method == "attention_based":
                    pooling_layer = AttentionPooling(input_dim=features.shape[1]).to(device)
                    pooled = pooling_layer(features).unsqueeze(0)
                else:
                    raise ValueError(f"Unknown WSI method: {method}")
                final_numpy_features.append(pooled.cpu().numpy())
                del features, pooled

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wsi_features = np.concatenate(final_numpy_features, axis=0)
        print(f"WSI features: {wsi_features.shape}")
        return wsi_features.astype(np.float32)


# ---------------------------------------------------------------- pickle I/O
class _PreprocessorUnpickler(pickle.Unpickler):
    """Remap ``__main__.BCGDataPreprocessor`` / ``__main__.AttentionPooling`` to this module."""

    _REMAP = {"BCGDataPreprocessor": BCGDataPreprocessor, "AttentionPooling": AttentionPooling}

    def find_class(self, module, name):  # noqa: D401
        if name in self._REMAP and module in ("__main__", "inference", "preprocessor"):
            return self._REMAP[name]
        return super().find_class(module, name)


def load_preprocessor(path) -> BCGDataPreprocessor:
    """Load ``bcg_preprocessor.pkl`` regardless of the module it was pickled from."""
    with open(path, "rb") as f:
        return _PreprocessorUnpickler(f).load()
