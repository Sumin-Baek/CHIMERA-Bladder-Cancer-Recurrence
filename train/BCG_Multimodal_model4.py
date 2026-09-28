# ===================================================================
# BCG-treated patient multimodal survival prediction system
# BCG Treatment Patient Multimodal Survival Prediction System

# modified:
# 1.modify risk score sign + -> -
# ===================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torchvision.transforms as transforms
from PIL import Image
import tifffile
from typing import Dict, List, Tuple, Optional, Union
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys
import argparse
from pycox.models.loss import CoxPHLoss

parser = argparse.ArgumentParser(
        description="Run Nested CV for Multimodal Survival Prediction using Dynamic Grid Search"
    )

# Default experiment settings
parser.add_argument("--task_name", type=str, default='Experiment2',
                    help="Task_name.")

# MLP model hyperparameters
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rates.")
parser.add_argument("--gene_method", type=str, default='pathway_based', choices=['pathway_based', 'top_genes'],
                    help="Gene selection method.")
parser.add_argument("--wsi_method", type=str, default='attention_based',
                    choices=['attention_based', 'concat_based'],
                    help="Gene selection method.")
parser.add_argument("--mode", type=str, default='default', choices=['default', 'debug'],
                    help="Debug mode.")
parser.add_argument('--data_root', type=str, default='./data/new_clinic',
                    help='Root folder containing data/<patient_id>/ and features/{features,coordinates}/.')
parser.add_argument('--debug_data_root', type=str, default='./data/debug',
                    help='Data root used when --mode debug.')
parser.add_argument('--save_dir', type=str, default='./results',
                    help='Folder where results/<task_name>/ (checkpoints, log, figures) is written.')

args = parser.parse_args()

warnings.filterwarnings('ignore')
# ===================================================================
# Stage 1: Preprocessor dedicated to BCG patient data
# ===================================================================
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # Network for computing attention scores
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (num_patches, input_dim) - tile features of one patient

        # 1. Compute attention score for each tile
        a = self.attention_net(x)  # shape: (num_patches, 1)

        # 2. Compute weights with softmax
        a = a.squeeze(-1)  # (num_patches,)
        att_weights = F.softmax(a, dim=0)  # (num_patches,)

        # 3. Apply weights to the original features to compute a weighted average
        # (1, num_patches) @ (num_patches, input_dim) -> (1, input_dim)
        context_vector = torch.matmul(att_weights.unsqueeze(0), x)

        return context_vector.squeeze(0)


class BCGDataPreprocessor:
    """
    Preprocessor dedicated to BCG-treated patient data

    Supported formats:
    - Clinical: patient_id_CD.json
    - WSI: patient_id_HE.tif + patient_id_HE_mask.tif
    - Features: patient_id_HE.pt + patient_id_HE.npy
    - RNA: patient_id_HE_RNA.json
    """

    def __init__(self, data_root_path: str):
        self.data_root = Path(data_root_path)
        self.clinical_scaler = StandardScaler()
        self.rna_scaler = StandardScaler()
        self.label_encoders = {}

        # BCG data feature definitions
        self.categorical_features = [
            'sex', 'smoking', 'tumor', 'stage', 'substage',
            'grade', 'reTUR', 'LVI', 'variant', 'EORTC', 'BRS'
        ]
        self.numerical_features = ['age', 'no_instillations']

        print(f"Data root path: {self.data_root}")

    def get_patient_list(self) -> List[str]:
        """Extract patient ID list"""
        patient_ids = set()
        data_root = self.data_root /'data'
        # Extract patient ID from each patient folder
        for patient_folder in data_root.iterdir():
            if patient_folder.is_dir():
                patient_id = patient_folder.name
                # Check that required files exist
                cd_file = patient_folder / f"{patient_id}_CD.json"
                if cd_file.exists():
                    patient_ids.add(patient_id)

        patient_list = sorted(list(patient_ids))
        print(f"Number of patients found: {len(patient_list)}")
        return patient_list

    def load_clinical_data(self, patient_id: str) -> Dict:
        """Load clinical data"""
        cd_file = self.data_root /"data"/ patient_id / f"{patient_id}_CD.json"

        if not cd_file.exists():
            raise FileNotFoundError(f"Clinical data not found: {cd_file}")

        with open(cd_file, 'r') as f:
            clinical_data = json.load(f)

        return clinical_data

    def load_rna_data(self, patient_id: str) -> Dict:
        """Load RNA data"""
        rna_file = self.data_root/"data" / patient_id / f"{patient_id}_RNA.json"

        if not rna_file.exists():
            print(f"WARNING: RNA data not found for {patient_id}")
            return {}

        with open(rna_file, 'r') as f:
            rna_data = json.load(f)

        return rna_data

    def load_wsi_features(self, patient_id: str) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """Load pre-extracted WSI features"""
        feature_file = self.data_root/"features" /"features"/ f"{patient_id}_HE.pt"
        coord_file = self.data_root /"features"/"coordinates" / f"{patient_id}_HE.npy"

        features = None
        coordinates = None

        if feature_file.exists():
            try:
                features = torch.load(feature_file, map_location='cpu')
            except Exception as e:
                print(f"WARNING: Error loading features for {patient_id}: {e}")

        if coord_file.exists():
            try:
                coordinates = np.load(coord_file)
            except Exception as e:
                print(f"WARNING: Error loading coordinates for {patient_id}: {e}")

        return features, coordinates

    # Paste or replace this function inside the existing class (BCGDataPreprocessor).

    def preprocess_clinical_data(self, clinical_data_list: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Clinical data preprocessing (separate scaling for numeric/categorical)"""
        print("Starting clinical data preprocessing (improved method)...")

        # Create DataFrame
        df = pd.DataFrame(clinical_data_list)

        # Print missing value summary (same as before)
        print(f"Missing value summary:")
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count}/{len(df)} ({missing_count / len(df) * 100:.1f}%)")

        # List to hold processed features
        processed_categorical_features = []
        categorical_feature_names = []
        processed_numerical_features = []
        numerical_feature_names = []

        # --- 1. Categorical variable processing ---
        for feature in self.categorical_features:
            if feature in df.columns:
                # Treat missing values as 'Unknown'
                df[feature] = df[feature].fillna('Unknown')

                # Label encoding followed by one-hot encoding
                if feature not in self.label_encoders:
                    # Create and fit a new encoder
                    self.label_encoders[feature] = LabelEncoder()
                    df[feature] = df[feature].astype(str)
                    # Fit all classes in case a class absent from training appears at test time
                    all_possible_values = list(df[feature].unique())
                    self.label_encoders[feature].fit(all_possible_values)
                    encoded = self.label_encoders[feature].transform(df[feature])
                else:
                    # Transform using the existing encoder
                    df[feature] = df[feature].astype(str)
                    # Handle new classes not seen during training
                    current_classes = set(self.label_encoders[feature].classes_)
                    new_values = set(df[feature].unique()) - current_classes
                    if new_values:
                        # Treat the new class as 'Unknown' (safest approach)
                        df.loc[df[feature].isin(new_values), feature] = 'Unknown'
                    encoded = self.label_encoders[feature].transform(df[feature])

                # One-hot encoding
                n_classes = len(self.label_encoders[feature].classes_)
                onehot = np.eye(n_classes)[encoded]

                processed_categorical_features.append(onehot)
                categorical_feature_names.extend([f"{feature}_{cls}" for cls in self.label_encoders[feature].classes_])

        # --- 2. Numeric variable processing ---
        numerical_df = pd.DataFrame()
        for feature in self.numerical_features:
            if feature in df.columns:
                values = df[feature].copy()
                # Special value handling (-1 in no_instillations means missing)
                if feature == 'no_instillations':
                    values = values.replace(-1, np.nan)

                # Impute missing values with the mean
                mean_val = values.mean()
                values = values.fillna(mean_val)

                numerical_df[feature] = values
                numerical_feature_names.append(feature)

        # Scale only when numeric data exists
        if not numerical_df.empty:
            # Apply StandardScaler to numeric data only
            processed_numerical_features = self.clinical_scaler.fit_transform(numerical_df)

        # --- 3. Combine categorical and numeric features ---
        final_features_list = []
        final_feature_names = []

        if processed_categorical_features:
            # Concatenate all categorical features into one array
            categorical_block = np.concatenate(processed_categorical_features, axis=1)
            final_features_list.append(categorical_block)
            final_feature_names.extend(categorical_feature_names)

        if not numerical_df.empty:
            # Add scaled numeric features
            final_features_list.append(processed_numerical_features)
            final_feature_names.extend(numerical_feature_names)

        # Create final feature matrix
        if final_features_list:
            all_features = np.concatenate(final_features_list, axis=1)
        else:
            # No features to process
            all_features = np.zeros((len(df), 1))

        # Create final feature info dictionary
        feature_info = {
            'feature_names': final_feature_names,
            'n_features': all_features.shape[1],
            'encoders': self.label_encoders.copy(),
            'scaler': self.clinical_scaler  # also save the scaler for later use at inference
        }

        print(f"Clinical data processing complete: {all_features.shape}")
        return all_features.astype(np.float32), feature_info

    def preprocess_rna_data(self, rna_data_list: List[Dict], method=args.gene_method) -> Tuple[np.ndarray, Dict]:
        """RNA data preprocessing"""
        print("Starting RNA data preprocessing...")

        if not rna_data_list or all(not rna_data for rna_data in rna_data_list):
            print("WARNING: No RNA data, generating dummy data")
            dummy_features = np.random.randn(len(rna_data_list), 50).astype(np.float32)
            return dummy_features, {'method': 'dummy', 'n_features': 50}

        # Collect list of all genes
        all_genes = set()
        for rna_data in rna_data_list:
            if rna_data:
                all_genes.update(rna_data.keys())

        all_genes = sorted(list(all_genes))
        print(f"Total number of genes: {len(all_genes)}")

        if method == 'pathway_based':
            # Key pathways related to BCG treatment response
            bcg_pathways = self._get_bcg_relevant_pathways()
            processed_features = []
            pathway_names = []

            for pathway_name, genes in bcg_pathways.items():
                available_genes = [g for g in genes if g in all_genes]
                if available_genes:
                    pathway_values = []
                    for rna_data in rna_data_list:
                        if rna_data and available_genes:
                            gene_values = [rna_data.get(gene, 0) for gene in available_genes]
                            # Compute various statistics
                            pathway_stats = [
                                np.mean(gene_values),  # mean
                                np.std(gene_values),  # standard deviation
                                np.median(gene_values),  # median
                                np.max(gene_values),  # maximum
                                np.percentile(gene_values, 75)  # 75th percentile
                            ]
                        else:
                            pathway_stats = [0.0] * 5
                        pathway_values.append(pathway_stats)

                    processed_features.append(np.array(pathway_values))
                    pathway_names.extend([f"{pathway_name}_{stat}"
                                          for stat in ['mean', 'std', 'median', 'max', 'q75']])

            if processed_features:
                rna_features = np.concatenate(processed_features, axis=1)
            else:
                print("WARNING: Pathway-based processing failed, selecting top genes")
                rna_features = self._process_top_genes(rna_data_list, all_genes)
                pathway_names = [f"gene_{i}" for i in range(rna_features.shape[1])]

        elif method == 'top_genes':
            rna_features = self._process_top_genes(rna_data_list, all_genes)
            pathway_names = [f"gene_{i}" for i in range(rna_features.shape[1])]

        # Standardization
        if rna_features.shape[1] > 0:
            rna_features = self.rna_scaler.fit_transform(rna_features)

        rna_info = {
            'method': method,
            'feature_names': pathway_names,
            'n_features': rna_features.shape[1],
            'available_genes': len(all_genes)
        }

        print(f"RNA data processing complete: {rna_features.shape}")
        return rna_features.astype(np.float32), rna_info

    def _get_bcg_relevant_pathways(self) -> Dict[str, List[str]]:
        """Key pathways related to BCG treatment response"""
        return {
            'IMMUNE_RESPONSE': [
                'IFNG', 'TNF', 'IL1B', 'IL6', 'IL10', 'IL12A', 'IL12B',
                'CD8A', 'CD4', 'CD68', 'CD3E', 'PTPRC'
            ],
            'T_CELL_ACTIVATION': [
                'CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B', 'CD4', 'GZMA',
                'GZMB', 'PRF1', 'IFNG', 'IL2'
            ],
            'INFLAMMATORY_RESPONSE': [
                'TNF', 'IL1B', 'IL6', 'CXCL8', 'CCL2', 'CCL5', 'CXCL10',
                'NOS2', 'PTGS2', 'NFE2L2'
            ],
            'APOPTOSIS': [
                'TP53', 'BAX', 'BCL2', 'CASP3', 'CASP8', 'CASP9', 'FAS',
                'FASLG', 'BID', 'BAK1', 'CYCS'
            ],
            'CELL_CYCLE': [
                'CCND1', 'CDK4', 'CDK6', 'RB1', 'E2F1', 'CDKN1A', 'CDKN2A',
                'MYC', 'TP53', 'CCNE1'
            ],
            'ANGIOGENESIS': [
                'VEGFA', 'VEGFR2', 'FGF2', 'PDGFB', 'ANGPT1', 'ANGPT2',
                'HIF1A', 'EPAS1'
            ],
            'INVASION_METASTASIS': [
                'CDH1', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2',
                'VIM', 'MMP2', 'MMP9', 'TIMP1'
            ],
            'DNA_REPAIR': [
                'BRCA1', 'BRCA2', 'ATM', 'ATR', 'PARP1', 'RAD51', 'XRCC1',
                'ERCC1', 'MSH2', 'MLH1'
            ]
        }

    def _process_top_genes(self, rna_data_list: List[Dict], all_genes: List[str], top_k: int = 100) -> np.ndarray:
        """Select top-variance genes"""
        # Build per-gene data matrix
        gene_matrix = []
        for gene in all_genes[:top_k]:  # limited for memory reasons
            gene_values = []
            for rna_data in rna_data_list:
                value = rna_data.get(gene, 0) if rna_data else 0
                gene_values.append(value)
            gene_matrix.append(gene_values)

        gene_matrix = np.array(gene_matrix).T  # [samples, genes]

        # Select top genes by variance
        if gene_matrix.shape[1] > 50:
            variances = np.var(gene_matrix, axis=0)
            top_indices = np.argsort(variances)[-50:]  # top 50
            gene_matrix = gene_matrix[:, top_indices]

        return gene_matrix

    def process_wsi_features(self, features_list: List[torch.Tensor], method=args.wsi_method) -> np.ndarray:
        """
        WSI feature processing (final version that guarantees all feature vectors have the same size)
        """
        print("Starting WSI feature processing...")
        batch_size = 10
        final_features = []

        for i in range(0, len(features_list), batch_size):
            batch_features = features_list[i:i+batch_size]
            batch_results = []
            if method == 'concat_based':
                for feature in batch_features:
                    if feature is not None and isinstance(feature, torch.Tensor) and len(feature) > 0:
                        # Check GPU memory - added None check
                        if hasattr(feature, 'device') and feature.device != torch.device('cpu'):
                            feature = feature.cpu()

                        feature_mean = torch.mean(feature, dim=0, keepdim=True)
                        feature_max = torch.max(feature, dim=0, keepdim=True).values
                        final_feature = torch.cat([feature_mean,feature_max], dim=1)
                    batch_results.append(final_feature)

            elif method == 'attention_based':
                pooling_layer = AttentionPooling(input_dim=1024)
                for feature in batch_features:
                    if feature is not None and isinstance(feature, torch.Tensor) and len(feature) > 0:
                        if hasattr(feature, 'device') and feature.device != torch.device('cpu'):
                            feature = feature.cpu()
                        final_feature = pooling_layer(feature).reshape([1,-1]) # shape: (1024)
                    batch_results.append(final_feature)

            if batch_results:
                batch_tensor = torch.cat(batch_results, dim=0)
                final_features.append(batch_tensor.detach().numpy())

            del batch_results
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        wsi_features = np.concatenate(final_features, axis=0)
        print(f"WSI feature processing complete: {wsi_features.shape}")
        return wsi_features.astype(np.float32)


# ===================================================================
# Stage 2: BCG treatment-specific multimodal architecture
# ===================================================================
# ===================================================================
# 1. Transformer-based high-performance model
# ===================================================================

class MultiHeadCrossAttention(nn.Module):
    """Advanced Cross-Modal Attention"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        B, N, _ = query.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        out = self.out_proj(out)

        return out, attn


class TransformerEncoderLayer(nn.Module):
    """Advanced transformer layer"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class AdvancedBCGTransformerNet(nn.Module):
    """
    Transformer-based high-performance BCG model

    Features:
    - Multi-head cross-modal attention
    - Hierarchical feature extraction
    - Advanced fusion mechanisms
    - Uncertainty quantification
    """

    def __init__(self, config: Dict):
        super(AdvancedBCGTransformerNet, self).__init__()
        self.config = config

        # Dimension settings
        clinical_dim = config['clinical_dim']
        rna_dim = config['rna_dim']
        wsi_dim = config['wsi_dim']
        hidden_dim = config.get('hidden_dim', 256)

        # Per-modality embeddings
        self.clinical_embedding = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.rna_embedding = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.wsi_embedding = nn.Sequential(
            nn.Linear(wsi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Positional embedding for modalities
        self.modality_pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))

        # Transformer encoder layers
        num_layers = config.get('num_transformer_layers', 4)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads=8, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Cross-modal attention
        self.cross_attention = MultiHeadCrossAttention(hidden_dim, num_heads=8)

        # Adaptive fusion
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

        # Final processing
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Prediction heads (same outputs as before)
        self.progression_risk_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.time_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self.brs_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3))

    def forward(self, clinical_data, rna_data, wsi_data, return_attention=False):
        batch_size = clinical_data.size(0)

        # Modality embeddings
        clinical_emb = self.clinical_embedding(clinical_data).unsqueeze(1)  # [B, 1, D]
        rna_emb = self.rna_embedding(rna_data).unsqueeze(1)  # [B, 1, D]
        wsi_emb = self.wsi_embedding(wsi_data).unsqueeze(1)  # [B, 1, D]

        # Combine modalities
        multimodal_tokens = torch.cat([clinical_emb, rna_emb, wsi_emb], dim=1)  # [B, 3, D]

        # Add positional embedding
        multimodal_tokens = multimodal_tokens + self.modality_pos_embedding

        # Transformer processing
        for layer in self.transformer_layers:
            multimodal_tokens = layer(multimodal_tokens)

        # Cross-modal attention
        attended_tokens, attention_weights = self.cross_attention(
            multimodal_tokens, multimodal_tokens, multimodal_tokens
        )

        # Adaptive modality weighting
        fusion_input = attended_tokens.reshape(batch_size, -1)
        modality_gates = self.modality_gate(fusion_input)

        # Weighted fusion
        clinical_weighted = attended_tokens[:, 0] * modality_gates[:, 0:1]
        rna_weighted = attended_tokens[:, 1] * modality_gates[:, 1:2]
        wsi_weighted = attended_tokens[:, 2] * modality_gates[:, 2:3]

        fused_features = clinical_weighted + rna_weighted + wsi_weighted

        # Final feature processing
        final_features = self.feature_processor(fused_features)

        # Predictions
        progression_risk = self.progression_risk_head(final_features)
        time_prediction = self.time_prediction_head(final_features)
        brs_prediction = self.brs_prediction_head(final_features)
        uncertainty = self.uncertainty_head(final_features)

        outputs = {
            'progression_risk': progression_risk,
            'time_prediction': time_prediction,
            'brs_prediction': brs_prediction,
            'uncertainty': uncertainty,
            'fused_features': final_features,
            'modality_weights': F.softmax(self.modality_weights, dim=0)
        }

        if return_attention:
            outputs['attention_weights'] = attention_weights
            outputs['modality_gates'] = modality_gates

        return outputs

# ===================================================================
# Stage 3: BCG treatment-specific loss function
# ===================================================================
class PyCoxLoss(nn.Module):
    """
    Stable and efficient Cox proportional hazards loss using the pycox library.
    This loss directly drives the model to maximize the C-index.
    """

    def __init__(self):
        super().__init__()
        # Create the CoxPHLoss instance provided by pycox
        self.cox_loss_fn = CoxPHLoss()

    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        """
        Compute the loss.

        Args:
            outputs (Dict): model outputs. Must contain the 'progression_risk' key.
            targets (Dict): ground truth. Must contain the 'time' and 'progression' keys.

        Returns:
            Dict: computed loss. Contains the 'total' and 'cox' keys.
        """
        # Model risk predictions (logits)
        # Reshape (batch_size, 1) -> (batch_size,)
        risk_scores = outputs['progression_risk'].squeeze(-1)

        # Target data in the format pycox requires (tuple: (time, event))
        # .squeeze() to get shape (batch_size,)
        durations = targets['time'].squeeze()
        events = targets['progression'].squeeze()
        # Compute Cox loss with pycox
        loss = self.cox_loss_fn(risk_scores, durations,events)

        # Return as a dictionary for compatibility with the training loop
        return {
            'cox': loss,
            'total': loss  # the total loss is now just the Cox loss
        }

# ===================================================================
# Stage 4: BCG patient dataset class
# ===================================================================

class BCGPatientDataset(Dataset):
    """
    BCG-treated patient dataset
    """

    def __init__(self, data_root: str, patient_ids: List[str], preprocessor: BCGDataPreprocessor = None):
        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.preprocessor = preprocessor or BCGDataPreprocessor(data_root)

        # Load and preprocess data
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """Load and preprocess all patient data"""
        print("Starting BCG patient data loading...")

        # Collect raw data
        clinical_data_list = []
        rna_data_list = []
        wsi_features_list = []
        wsi_coords_list = []
        survival_data = []

        for patient_id in tqdm(self.patient_ids, desc="Loading patient data"):
            try:
                # Clinical data
                clinical_data = self.preprocessor.load_clinical_data(patient_id)
                clinical_data_list.append(clinical_data)

                # RNA data
                rna_data = self.preprocessor.load_rna_data(patient_id)
                rna_data_list.append(rna_data)

                # WSI features
                wsi_features, wsi_coords = self.preprocessor.load_wsi_features(patient_id)
                wsi_features_list.append(wsi_features)
                wsi_coords_list.append(wsi_coords)

                # Survival outcomes
                survival_info = {
                    'patient_id': patient_id,
                    'progression': clinical_data.get('progression', 0),
                    'time': clinical_data.get('Time_to_prog_or_FUend', 1.0),
                    'brs': self._parse_brs_label(clinical_data.get('BRS', 'Unknown'))
                }
                survival_data.append(survival_info)

            except Exception as e:
                print(f"WARNING: Error loading data for {patient_id}: {e}")
                # Replace with dummy data
                clinical_data_list.append({})
                rna_data_list.append({})
                wsi_features_list.append(None)
                wsi_coords_list.append(None)
                survival_data.append({
                    'patient_id': patient_id,
                    'progression': 0,
                    'time': 1.0,
                    'brs': -1
                })

        # Data preprocessing
        self.clinical_features, self.clinical_info = self.preprocessor.preprocess_clinical_data(clinical_data_list)
        self.rna_features, self.rna_info = self.preprocessor.preprocess_rna_data(rna_data_list)
        self.wsi_features = self.preprocessor.process_wsi_features(wsi_features_list)

        # Organize survival data
        self.survival_df = pd.DataFrame(survival_data)

        print(f"Data loading complete:")
        print(f"  - Number of patients: {len(self.patient_ids)}")
        print(f"  - Clinical features: {self.clinical_features.shape[1]}")
        print(f"  - RNA features: {self.rna_features.shape[1]}")
        print(f"  - WSI features: {self.wsi_features.shape[1]}")
        print(f"  - Progression events: {self.survival_df['progression'].sum()}/{len(self.survival_df)} ({self.survival_df['progression'].mean():.1%})")
        print(f"  - Mean follow-up: {self.survival_df['time'].mean():.1f} months")

    def _parse_brs_label(self, brs_value: str) -> int:
        """Parse BRS label"""
        if brs_value in ['BRS1', 'BRS2', 'BRS3']:
            return int(brs_value[3]) - 1  # convert to 0, 1, 2
        return -1  # unknown

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return {
            'patient_id': self.patient_ids[idx],
            'clinical': torch.FloatTensor(self.clinical_features[idx]),
            'rna': torch.FloatTensor(self.rna_features[idx]),
            'wsi': torch.FloatTensor(self.wsi_features[idx]),
            'progression': torch.FloatTensor([self.survival_df.iloc[idx]['progression']]),
            'time': torch.FloatTensor([self.survival_df.iloc[idx]['time']]),
            'brs': torch.LongTensor([self.survival_df.iloc[idx]['brs']])
        }

# ===================================================================
# Stage 5: BCG treatment-specific training system
# ===================================================================

class BCGMultimodalTrainer:
    """
    Specialized training system for predicting BCG treatment outcomes
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # BCG-specific loss function
        self.loss_fn = PyCoxLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5)

        # Early stopping
        self.best_score = 0.0
        self.patience_counter = 0
        self.patience = config.get('patience', 15)

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_cindex': [], 'val_cindex': [],
            'train_auc': [], 'val_auc': [], 'learning_rates': []
        }

        self._setup_logging()
        self.debug_first_epoch = False

    def _setup_logging(self):
        """Logging setup"""

        # Direct path specification
        log_file = os.path.join(args.save_dir, args.task_name, 'bcg_training.log')

        # Create folder
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def debug_concordance(self, times, risk_scores, events):
        """C-index debugging"""
        print(f"\nC-index debugging:")
        print(f"Risk scores range: {risk_scores.min():.3f} ~ {risk_scores.max():.3f}")
        print(f"Mean risk of patients with events: {risk_scores[events == 1].mean():.3f}")
        print(f"Mean risk of censored patients: {risk_scores[events == 0].mean():.3f}")

        # Check correlation (negative correlation is expected)
        corr = np.corrcoef(times, risk_scores)[0, 1]
        print(f"Time vs Risk correlation: {corr:.3f} (negative is expected)")

    def train_epoch(self, train_loader):
        """Training epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0, 'cox': 0.0
        }

        all_progression_probs = []
        all_progression_true = []
        all_risk_scores = []
        all_times = []
        all_events = []

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch data
            clinical_data = batch['clinical'].float().to(self.device)
            rna_data = batch['rna'].float().to(self.device)
            wsi_data = batch['wsi'].float().to(self.device)

            targets = {
                'progression': batch['progression'].float().to(self.device).squeeze(),
                'time': batch['time'].float().to(self.device).squeeze(),
                'brs': batch['brs'].long().to(self.device).squeeze()
            }

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(clinical_data, rna_data, wsi_data)

            # Compute loss
            losses = self.loss_fn(outputs, targets)

            # NaN check
            if torch.isnan(losses['total']):
                self.logger.warning(f"NaN loss at batch {batch_idx}, skipping")
                continue

            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.item()

            # Collect predictions (for evaluation)
            progression_probs = torch.sigmoid(outputs['progression_risk']).detach().cpu().numpy()
            risk_scores = outputs['progression_risk'].detach().cpu().numpy()

            all_progression_probs.extend(progression_probs.flatten())
            all_progression_true.extend(targets['progression'].detach().cpu().numpy())
            all_risk_scores.extend(risk_scores.flatten())
            all_times.extend(targets['time'].detach().cpu().numpy())
            all_events.extend(targets['progression'].detach().cpu().numpy())

            # Update progress bar
            # Modified progress bar update section of train_epoch
            # Progress bar update - modified section

            progress_bar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Cox': f"{losses['cox'].item():.4f}",
            })

        # Mean losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Compute evaluation metrics
        train_cindex = self.calculate_concordance_index(
            np.array(all_times), np.array(all_risk_scores), np.array(all_events)
        )

        # Added debugging code (first epoch only)
        if hasattr(self, 'debug_first_epoch') and not self.debug_first_epoch:
            self.debug_concordance(
                np.array(all_times), np.array(all_risk_scores), np.array(all_events)
            )
            self.debug_first_epoch = True

        train_auc = self.calculate_auc(
            np.array(all_progression_true), np.array(all_progression_probs)
        )

        return epoch_losses, train_cindex, train_auc

    def evaluate(self, val_loader, return_predictions=False):
        """Validation evaluation"""
        self.model.eval()
        val_losses = {
            'total': 0.0, 'cox': 0.0
        }

        all_outputs = {
            'progression_probs': [], 'progression_true': [], 'risk_scores': [],
            'time_predictions': [], 'times': [], 'events': [], 'brs_predictions': [],
            'brs_true': [], 'uncertainties': [], 'patient_ids': [],'modality_weights': []
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                clinical_data = batch['clinical'].float().to(self.device)
                rna_data = batch['rna'].float().to(self.device)
                wsi_data = batch['wsi'].float().to(self.device)

                targets = {
                    'progression': batch['progression'].float().to(self.device).squeeze(),
                    'time': batch['time'].float().to(self.device).squeeze(),
                    'brs': batch['brs'].long().to(self.device).squeeze()
                }

                outputs = self.model(clinical_data, rna_data, wsi_data)
                losses = self.loss_fn(outputs, targets)

                # Accumulate losses
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        val_losses[key] += value.item()

                # Collect predictions
                progression_probs = torch.sigmoid(outputs['progression_risk']).cpu().numpy()

                all_outputs['progression_probs'].extend(progression_probs.flatten())
                all_outputs['progression_true'].extend(targets['progression'].cpu().numpy())
                all_outputs['risk_scores'].extend(outputs['progression_risk'].cpu().numpy().flatten())
                all_outputs['time_predictions'].extend(outputs['time_prediction'].cpu().numpy().flatten())
                all_outputs['times'].extend(targets['time'].cpu().numpy())
                all_outputs['events'].extend(targets['progression'].cpu().numpy())
                all_outputs['brs_predictions'].extend(outputs['brs_prediction'].cpu().numpy())
                all_outputs['brs_true'].extend(targets['brs'].cpu().numpy())
                all_outputs['uncertainties'].extend(outputs['uncertainty'].cpu().numpy().flatten())
                all_outputs['patient_ids'].extend(batch['patient_id'])

                if 'modality_weights' in outputs:
                    all_outputs['modality_weights'].extend(outputs['modality_weights'].cpu().numpy())
        # Mean losses
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Comprehensive evaluation metrics
        metrics = self.calculate_comprehensive_metrics(all_outputs)
        metrics.update(val_losses)

        if return_predictions:
            return metrics, all_outputs
        return metrics

    def fix_cindex_calculation(self, val_loader):
        """Fix for C-index computation issue"""

        self.model.eval()
        all_risk_scores = []
        all_times = []
        all_events = []

        with torch.no_grad():
            for batch in val_loader:
                clinical_data = batch['clinical'].float().to(self.device)
                rna_data = batch['rna'].float().to(self.device)
                wsi_data = batch['wsi'].float().to(self.device)

                outputs = self.model(clinical_data, rna_data, wsi_data)

                # Collect data
                risk_scores = outputs['progression_risk'].cpu().numpy().flatten()
                times = batch['time'].numpy().flatten()
                events = batch['progression'].numpy().flatten()

                all_risk_scores.extend(risk_scores)
                all_times.extend(times)
                all_events.extend(events)

        # Convert to NumPy arrays
        risk_scores = np.array(all_risk_scores)
        times = np.array(all_times)
        events = np.array(all_events)

        # Remove NaN
        valid_mask = ~(np.isnan(risk_scores) | np.isnan(times) | np.isnan(events))
        risk_scores = risk_scores[valid_mask]
        times = times[valid_mask]
        events = events[valid_mask]

        print(f"Data check:")
        print(f"  Valid samples: {len(risk_scores)}")
        print(f"  Event ratio: {np.mean(events):.1%}")
        print(f"  Risk range: {np.min(risk_scores):.3f} ~ {np.max(risk_scores):.3f}")

        # Simple C-index computation
        try:
            from lifelines.utils import concordance_index
            c_index = concordance_index(times, risk_scores, events)
            print(f"C-index: {c_index:.4f}")
        except:
            # Manual computation
            c_index = self._manual_cindex(times, risk_scores, events)
            print(f"C-index (manual): {c_index:.4f}")

        return c_index

    def _manual_cindex(self, times, risk_scores, events):
        """Simple manual C-index"""
        concordant = 0
        total = 0

        n = len(times)
        for i in range(n):
            for j in range(i + 1, n):
                # Patients with events should have higher risk
                if events[i] == 1 and events[j] == 0 and times[i] <= times[j]:
                    total += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif events[j] == 1 and events[i] == 0 and times[j] <= times[i]:
                    total += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1

        return concordant / total if total > 0 else 0.5

    def calculate_comprehensive_metrics(self, outputs):
        """BCG treatment-specific evaluation metrics"""
        metrics = {}

        # C-index (survival analysis)
        times = np.array(outputs['times'])
        risk_scores = np.array(outputs['risk_scores'])
        events = np.array(outputs['events'])

        metrics['c_index'] = self.calculate_concordance_index(times, risk_scores, events)

        # AUC (progression prediction)
        progression_probs = np.array(outputs['progression_probs'])
        progression_true = np.array(outputs['progression_true'])

        metrics['auc'] = self.calculate_auc(progression_true, progression_probs)

        # BRS classification accuracy
        brs_preds = np.array(outputs['brs_predictions'])
        brs_true = np.array(outputs['brs_true'])

        # Only when BRS labels exist
        valid_brs_mask = brs_true >= 0
        if valid_brs_mask.sum() > 0:
            brs_pred_classes = np.argmax(brs_preds[valid_brs_mask], axis=1)
            brs_accuracy = np.mean(brs_pred_classes == brs_true[valid_brs_mask])
            metrics['brs_accuracy'] = brs_accuracy
        else:
            metrics['brs_accuracy'] = 0.0

        # Time prediction accuracy (patients with events only)
        event_mask = events == 1
        if event_mask.sum() > 0:
            time_predictions = np.array(outputs['time_predictions'])
            time_mae = np.mean(np.abs(time_predictions[event_mask] - times[event_mask]))
            metrics['time_mae'] = time_mae
        else:
            metrics['time_mae'] = 0.0

        # Uncertainty analysis
        uncertainties = np.array(outputs['uncertainties'])
        metrics['mean_uncertainty'] = np.mean(uncertainties)

        # Performance per risk group
        high_risk_mask = progression_probs > 0.5
        if high_risk_mask.sum() > 0:
            high_risk_progression_rate = np.mean(progression_true[high_risk_mask])
            metrics['high_risk_progression_rate'] = high_risk_progression_rate

        low_risk_mask = progression_probs <= 0.5
        if low_risk_mask.sum() > 0:
            low_risk_progression_rate = np.mean(progression_true[low_risk_mask])
            metrics['low_risk_progression_rate'] = low_risk_progression_rate

        return metrics

    def calculate_concordance_index(self, times, risk_scores, events):
        """Compute C-index"""
        try:
            from lifelines.utils import concordance_index
            return concordance_index(times, -risk_scores, events)
        except ImportError:
            return self._manual_concordance_index(times, -risk_scores, events)

    def _manual_concordance_index(self, times, risk_scores, events):
        """Manual C-index computation"""
        n = len(times)
        concordant = 0
        permissible = 0

        for i in range(n):
            for j in range(i + 1, n):
                if events[i] == 1 and times[i] < times[j]:
                    permissible += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif events[j] == 1 and times[j] < times[i]:
                    permissible += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1

        return concordant / permissible if permissible > 0 else 0.5

    def calculate_auc(self, y_true, y_prob):
        """Compute AUC"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_prob)
        except:
            return 0.5

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""
        self.logger.info("Starting BCG multimodal survival prediction training")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Training patients: {len(train_loader.dataset)}")
        self.logger.info(f"Validation patients: {len(val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            train_losses, train_cindex, train_auc = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_cindex = self.fix_cindex_calculation(val_loader)
            print(f"Epoch {epoch + 1}: Train fixed C-index = {train_cindex:.4f}, Val C-index = {val_cindex:.4f}")
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_metrics.get('total', 0))
            self.history['train_cindex'].append(train_cindex)
            self.history['val_cindex'].append(val_metrics['c_index'])
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Update scheduler
            current_score = val_metrics['c_index']
            self.scheduler.step(current_score)

            # Logging
            self.logger.info(f"Train - Loss: {train_losses['total']:.4f}, C-index: {train_cindex:.4f}, AUC: {train_auc:.4f}")
            self.logger.info(f"Val - C-index: {val_metrics['c_index']:.4f}, AUC: {val_metrics['auc']:.4f}")
            self.logger.info(f"Val - BRS Acc: {val_metrics['brs_accuracy']:.4f}, Time MAE: {val_metrics['time_mae']:.2f}")

            # Early stopping
            if current_score > self.best_score:
                self.best_score = current_score
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                self.logger.info(f"New best model saved! C-index Score: {self.best_score:.4f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping after {epoch+1} epochs")
                break

        self.logger.info(f"Training complete! Best Combined Score: {self.best_score:.4f}")
        return self.history

    def save_checkpoint(self, epoch, metrics):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        # Full file path
        save_path = os.path.join(self.config['save_dir'], 'models', f'bcg_best_model_epoch_{epoch}.pth')

        # Create folder (extract only the folder part from the file path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(checkpoint, save_path)

# ===================================================================
# Stage 6: Experiment settings and main pipeline
# ===================================================================

def create_bcg_config():
    """BCG treatment-specific configuration"""
    if args.mode == 'default':
        data_root_pth = args.data_root
    elif args.mode == 'debug':
        data_root_pth = args.debug_data_root

    return {
        # Data path

        'data_root': data_root_pth,  # change to the actual data path

        # Model hyperparameters (auto-adjusted to the data)
        'clinical_dim': 50,    # determined after preprocessing
        'rna_dim': 50,         # determined after preprocessing
        'wsi_dim': 1000,       # determined after preprocessing

        # Training settings
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'batch_size': 16,     # BCG data usually has few samples
        'num_epochs': 500,
        'patience': 30,        # longer patience due to BCG data characteristics

        # Model settings
        'use_cross_attention': True,

        # BCG-specific loss weights
        'loss_weights': {
            'cox': 1.0,           # Cox proportional hazards (primary)
            'bce': 0.8,           # progression prediction (important)
            'mse': 0.6,           # time prediction
            'brs': 0.5,           # BRS classification (auxiliary)
            'uncertainty': 0.2    # uncertainty regularization
        }
    }

class BCGExperimentRunner:
    """
    BCG-treated patient experiment runner
    """

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.config = create_bcg_config()
        self.config['data_root'] = data_root

    def run_complete_experiment(self):
        """Run the complete BCG experiment"""
        print("Starting BCG-treated patient multimodal survival prediction experiment")
        print("=" * 60)

        # 1. Initialize data preprocessor
        preprocessor = BCGDataPreprocessor(self.data_root)

        # 2. Get patient list
        patient_ids = preprocessor.get_patient_list()

        if len(patient_ids) == 0:
            print("ERROR: No patient data found. Please check the data path.")
            return None

        # 3. Create dataset (check dimensions with full data)
        full_dataset = BCGPatientDataset(self.data_root, patient_ids, preprocessor)

        # 4. Update config (match actual data dimensions)
        self.config['clinical_dim'] = full_dataset.clinical_features.shape[1]
        self.config['rna_dim'] = full_dataset.rna_features.shape[1]
        self.config['wsi_dim'] = full_dataset.wsi_features.shape[1]

        print(f"Data dimension check:")
        print(f"  - Clinical: {self.config['clinical_dim']}")
        print(f"  - RNA: {self.config['rna_dim']}")
        print(f"  - WSI: {self.config['wsi_dim']}")

        # 5. Data split (stratified split)
        train_ids, val_ids = train_test_split(
            patient_ids,
            test_size=0.2,
            stratify=full_dataset.survival_df['progression'],
            random_state=42
        )

        # 6. Training/validation datasets
        train_dataset = BCGPatientDataset(self.data_root, train_ids, preprocessor)
        val_dataset = BCGPatientDataset(self.data_root, val_ids, preprocessor)

        # 7. Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4
        )

        # 8. Create model
        print("Creating BCG-specific multimodal model...")
        model = AdvancedBCGTransformerNet(self.config)

        # 9. Create trainer and train
        trainer = BCGMultimodalTrainer(model, self.config)
        print("Starting training...")
        history = trainer.train(train_loader, val_loader, self.config['num_epochs'])

        # 10. Final evaluation
        print("\nFinal evaluation...")
        final_metrics, predictions = trainer.evaluate(val_loader, return_predictions=True)
        final_cindex = trainer.fix_cindex_calculation(val_loader)
        print(f"Final fixed C-index: {final_cindex:.4f}")

        # 11. save objects
        print("\nSaving preprocessor object for submission...")
        preprocessor_save_path = os.path.join(self.config['save_dir'], 'bcg_preprocessor.pkl')
        with open(preprocessor_save_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"Preprocessor saved to '{preprocessor_save_path}'.")

        # 12. Print results (existing code)
        self.print_final_results(final_metrics, predictions)

        return {
            'model': model,
            'trainer': trainer,
            'history': history,
            'metrics': final_metrics,
            'predictions': predictions,
            'config': self.config
        }

    def print_final_results(self, metrics, predictions):
        """Print final results"""
        print(f"\nBCG-treated patient survival prediction final results")
        print("=" * 50)
        print(f"Survival analysis performance:")
        print(f"  - C-Index: {metrics['c_index']:.4f}")
        print(f"Progression prediction performance:")
        print(f"  - AUC: {metrics['auc']:.4f}")
        print(f"BRS classification performance:")
        print(f"  - Accuracy: {metrics['brs_accuracy']:.4f}")
        print(f"Time prediction performance:")
        print(f"  - MAE: {metrics['time_mae']:.2f} months")
        print(f"Uncertainty analysis:")
        print(f"  - Mean uncertainty: {metrics['mean_uncertainty']:.4f}")

        # Analysis per risk group
        if 'high_risk_progression_rate' in metrics:
            print(f"Progression rate per risk group:")
            print(f"  - High-risk group: {metrics['high_risk_progression_rate']:.1%}")
        if 'low_risk_progression_rate' in metrics:
            print(f"  - Low-risk group: {metrics['low_risk_progression_rate']:.1%}")

        print(f"\nClinical significance:")
        if metrics['c_index'] > 0.7:
            print("  Excellent survival prediction performance (C-index > 0.7)")
        if metrics['auc'] > 0.75:
            print("  Excellent progression prediction performance (AUC > 0.75)")
        if metrics['brs_accuracy'] > 0.6:
            print("  Meaningful BRS classification performance")

        print(f"\nPer-patient prediction samples (top 5):")
        patient_ids = predictions['patient_ids'][:5]
        progression_probs = np.array(predictions['progression_probs'])[:5]
        times = np.array(predictions['times'])[:5]
        uncertainties = np.array(predictions['uncertainties'])[:5]

        for i, (pid, prob, time, unc) in enumerate(zip(patient_ids, progression_probs, times, uncertainties)):
            risk_level = "high" if prob > 0.6 else "medium" if prob > 0.3 else "low"
            print(f"  Patient {pid}: progression risk {prob:.3f} ({risk_level}), follow-up {time:.1f} months, uncertainty {unc:.3f}")


# ===================================================================
# Stage 7: Visualization and analysis tools
# ===================================================================

class BCGVisualizationSuite:
    """
    Visualization tools for analyzing BCG treatment outcomes
    """

    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
    def plot_survival_analysis(self, predictions, save_path=None):
        """Survival analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        times = np.array(predictions['times'])
        events = np.array(predictions['events'])
        risk_scores = np.array(predictions['risk_scores'])
        progression_probs = np.array(predictions['progression_probs'])

        # 1. Kaplan-Meier curves by risk groups
        risk_groups = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_thresholds = [0.33, 0.67]
        colors = ['green', 'orange', 'red']

        for i, (group, color) in enumerate(zip(risk_groups, colors)):
            if i == 0:
                mask = progression_probs <= risk_thresholds[0]
            elif i == 1:
                mask = (progression_probs > risk_thresholds[0]) & (progression_probs <= risk_thresholds[1])
            else:
                mask = progression_probs > risk_thresholds[1]

            if mask.sum() > 0:
                group_times = times[mask]
                group_events = events[mask]

                # Simple survival estimation
                unique_times = np.sort(np.unique(group_times))
                survival_probs = []

                for t in unique_times:
                    at_risk = np.sum(group_times >= t)
                    events_at_t = np.sum((group_times == t) & (group_events == 1))
                    if at_risk > 0:
                        surv_prob = 1 - events_at_t / at_risk
                        survival_probs.append(surv_prob)
                    else:
                        survival_probs.append(1.0)

                cumulative_survival = np.cumprod(survival_probs)
                axes[0, 0].step(unique_times, cumulative_survival,
                                where='post', label=f'{group} (n={mask.sum()})',
                                color=color, linewidth=2)

        axes[0, 0].set_xlabel('Time to Progression (months)')
        axes[0, 0].set_ylabel('Progression-Free Survival')
        axes[0, 0].set_title('Kaplan-Meier Curves by Risk Groups')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Risk score distribution
        axes[0, 1].hist(risk_scores[events == 0], alpha=0.7, label='No Progression',
                        bins=20, color='blue', density=True)
        axes[0, 1].hist(risk_scores[events == 1], alpha=0.7, label='Progression',
                        bins=20, color='red', density=True)
        axes[0, 1].set_xlabel('Risk Score')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Risk Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Time vs Risk scatter
        colors_scatter = ['blue' if e == 0 else 'red' for e in events]
        axes[1, 0].scatter(times, risk_scores, c=colors_scatter, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Follow-up Time (months)')
        axes[1, 0].set_ylabel('Risk Score')
        axes[1, 0].set_title('Risk Score vs Follow-up Time')

        # Add trend line
        z = np.polyfit(times, risk_scores, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(times, p(times), "r--", alpha=0.8, linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Progression probability calibration
        n_bins = 8
        prob_bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2

        observed_freqs = []
        predicted_freqs = []

        for i in range(n_bins):
            bin_mask = (progression_probs >= prob_bins[i]) & (progression_probs < prob_bins[i + 1])
            if bin_mask.sum() > 0:
                observed_freq = events[bin_mask].mean()
                predicted_freq = progression_probs[bin_mask].mean()
                observed_freqs.append(observed_freq)
                predicted_freqs.append(predicted_freq)

        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        if observed_freqs:
            axes[1, 1].scatter(predicted_freqs, observed_freqs,
                               color='red', s=100, alpha=0.8, label='Observed')
            axes[1, 1].plot(predicted_freqs, observed_freqs, 'r-', alpha=0.6)

        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_feature_importance(self, predictions, save_path=None):
        """Feature importance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Modality weights - modified section
        modality_weights_raw = predictions['modality_weights']

        # Handle various forms of modality_weights
        if isinstance(modality_weights_raw, list):
            if len(modality_weights_raw) > 0:
                if isinstance(modality_weights_raw[0], torch.Tensor):
                    # List of torch.Tensor
                    modality_weights = np.array([w.detach().cpu().numpy() if hasattr(w, 'detach') else w.numpy()
                                                 for w in modality_weights_raw])
                elif isinstance(modality_weights_raw[0], np.ndarray):
                    # List of numpy arrays
                    if modality_weights_raw[0].ndim == 2:
                        # List of 2D arrays (from batches)
                        modality_weights = np.concatenate(modality_weights_raw, axis=0)
                    else:
                        # List of 1D arrays
                        modality_weights = np.array(modality_weights_raw)
                else:
                    # Other types
                    modality_weights = np.array(modality_weights_raw)
            else:
                # Empty list: dummy data
                modality_weights = np.array([[0.33, 0.33, 0.34]])
        else:
            # Already a numpy array
            modality_weights = np.array(modality_weights_raw)

        # Check and adjust dimensions
        if modality_weights.ndim == 1:
            modality_weights = modality_weights.reshape(1, -1)

        # Handle the case of not having 3 modalities
        if modality_weights.shape[1] != 3:
            print(f"WARNING: Expected 3 modalities, got {modality_weights.shape[1]}. Using dummy weights.")
            modality_weights = np.array([[0.33, 0.33, 0.34]] * modality_weights.shape[0])

        modality_names = ['Clinical', 'RNA-seq', 'WSI']

        # 1. Average modality importance - modified section
        if len(modality_weights) > 0:
            avg_weights = np.mean(modality_weights, axis=0)
        else:
            avg_weights = np.array([0.33, 0.33, 0.34])  # default

        colors = ['lightblue', 'lightgreen', 'lightcoral']

        bars = axes[0, 0].bar(modality_names, avg_weights, color=colors)
        axes[0, 0].set_title('Average Modality Importance')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].set_ylim(0, max(1, np.max(avg_weights) * 1.1))

        # Add value labels on bars - modified section
        for bar, val in zip(bars, avg_weights):
            height = bar.get_height()
            if not np.isnan(height) and not np.isnan(val):
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                                f'{val:.3f}', ha='center', va='bottom')

        # 2. Modality weight distributions - modified section
        try:
            if len(modality_weights) > 1:
                box_data = [modality_weights[:, i] for i in range(3)]
                axes[0, 1].boxplot(box_data, labels=modality_names)
            else:
                # Fall back to bar chart when data is insufficient
                axes[0, 1].bar(modality_names, avg_weights, color=colors)
            axes[0, 1].set_title('Modality Importance Distribution')
            axes[0, 1].set_ylabel('Weight')
            axes[0, 1].grid(True, alpha=0.3)
        except Exception as e:
            print(f"WARNING: Boxplot error: {e}. Using bar chart instead.")
            axes[0, 1].bar(modality_names, avg_weights, color=colors)
            axes[0, 1].set_title('Modality Importance (Bar Chart)')
            axes[0, 1].set_ylabel('Weight')

        # 3. Correlation between modalities and outcomes - modified section
        progression_true = np.array(predictions['progression_true'])

        correlations = []
        for i in range(3):
            try:
                if len(modality_weights) > 1 and len(progression_true) == len(modality_weights):
                    corr = np.corrcoef(modality_weights[:, i], progression_true)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                correlations.append(corr)
            except Exception as e:
                print(f"WARNING: Correlation calculation error for modality {i}: {e}")
                correlations.append(0.0)

        bars = axes[1, 0].bar(modality_names, correlations,
                              color=['red' if c > 0 else 'blue' for c in correlations])
        axes[1, 0].set_title('Modality-Outcome Correlation')
        axes[1, 0].set_ylabel('Correlation with Progression')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, correlations):
            height = bar.get_height()
            if not np.isnan(height) and not np.isnan(val):
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                                height + (0.01 if val > 0 else -0.03),
                                f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')

        # 4. Modality importance vs uncertainty - modified section
        uncertainties = np.array(predictions['uncertainties'])

        try:
            if len(modality_weights) == len(uncertainties):
                for i, (name, color) in enumerate(zip(modality_names, colors)):
                    axes[1, 1].scatter(modality_weights[:, i], uncertainties,
                                       alpha=0.6, label=name, color=color, s=30)
            else:
                # Dummy scatter plot when lengths do not match
                for i, (name, color) in enumerate(zip(modality_names, colors)):
                    axes[1, 1].scatter([avg_weights[i]], [np.mean(uncertainties)],
                                       alpha=0.6, label=name, color=color, s=50)
        except Exception as e:
            print(f"WARNING: Scatter plot error: {e}. Using simplified visualization.")
            # Fall back to bar chart on error
            axes[1, 1].bar(modality_names, avg_weights, color=colors)

        axes[1, 1].set_xlabel('Modality Weight')
        axes[1, 1].set_ylabel('Prediction Uncertainty')
        axes[1, 1].set_title('Modality Importance vs Uncertainty')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_brs_analysis(self, predictions, save_path=None):
        """BRS subtype analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        brs_predictions = np.array(predictions['brs_predictions'])
        brs_true = np.array(predictions['brs_true'])
        progression_probs = np.array(predictions['progression_probs'])

        # Only when valid BRS labels exist
        valid_mask = brs_true >= 0

        if valid_mask.sum() > 0:
            valid_brs_pred = brs_predictions[valid_mask]
            valid_brs_true = brs_true[valid_mask]
            valid_prog_probs = progression_probs[valid_mask]

            # 1. BRS confusion matrix
            from sklearn.metrics import confusion_matrix
            pred_classes = np.argmax(valid_brs_pred, axis=1)
            cm = confusion_matrix(valid_brs_true, pred_classes)

            im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[0, 0].set_title('BRS Classification Confusion Matrix')
            tick_marks = np.arange(3)
            axes[0, 0].set_xticks(tick_marks)
            axes[0, 0].set_yticks(tick_marks)
            axes[0, 0].set_xticklabels(['BRS1', 'BRS2', 'BRS3'])
            axes[0, 0].set_yticklabels(['BRS1', 'BRS2', 'BRS3'])
            axes[0, 0].set_ylabel('True BRS')
            axes[0, 0].set_xlabel('Predicted BRS')

            # Add text annotations
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")

            # 2. BRS vs Progression Risk
            brs_names = ['BRS1', 'BRS2', 'BRS3']
            brs_prog_rates = []

            for brs_class in range(3):
                brs_mask = valid_brs_true == brs_class
                if brs_mask.sum() > 0:
                    avg_prog_prob = valid_prog_probs[brs_mask].mean()
                    brs_prog_rates.append(avg_prog_prob)
                else:
                    brs_prog_rates.append(0)

            bars = axes[0, 1].bar(brs_names, brs_prog_rates,
                                  color=['lightblue', 'lightgreen', 'lightcoral'])
            axes[0, 1].set_title('Average Progression Risk by BRS')
            axes[0, 1].set_ylabel('Average Progression Probability')
            axes[0, 1].set_ylim(0, 1)

            for bar, val in zip(bars, brs_prog_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                                f'{val:.3f}', ha='center', va='bottom')

        # 3. Progression probability distribution by predicted BRS
        pred_brs_classes = np.argmax(brs_predictions, axis=1)
        brs_colors = ['blue', 'green', 'red']

        for brs_class in range(3):
            brs_mask = pred_brs_classes == brs_class
            if brs_mask.sum() > 0:
                axes[1, 0].hist(progression_probs[brs_mask],
                                alpha=0.6, label=f'Pred BRS{brs_class + 1}',
                                color=brs_colors[brs_class], bins=15, density=True)

        axes[1, 0].set_xlabel('Progression Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Progression Risk Distribution by Predicted BRS')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. BRS prediction confidence
        brs_confidences = np.max(F.softmax(torch.FloatTensor(brs_predictions), dim=-1).numpy(), axis=1)

        axes[1, 1].hist(brs_confidences, bins=20, alpha=0.7, color='purple')
        axes[1, 1].axvline(brs_confidences.mean(), color='red', linestyle='--',
                           label=f'Mean: {brs_confidences.mean():.3f}')
        axes[1, 1].set_xlabel('BRS Prediction Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('BRS Prediction Confidence Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    def save_all_visualizations(self, predictions, save_dir):
        """Save all visualizations at once"""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.plot_survival_analysis(predictions,
                                    save_path=os.path.join(save_dir, 'survival.png'))
        self.plot_feature_importance(predictions,
                                     save_path=os.path.join(save_dir, 'features.png'))
        self.plot_brs_analysis(predictions,
                               save_path=os.path.join(save_dir, 'brs.png'))
        print(f"All visualizations saved: {save_dir}")

# ===================================================================
# Stage 8: Complete execution pipeline
# ===================================================================

def main_bcg_pipeline(data_root: str):
    """
    BCG-treated patient multimodal survival prediction main pipeline
    """

    print("BCG-treated patient multimodal survival prediction system")
    print("=" * 60)
    print("System features:")
    print("  - Multimodal analysis of clinical information, RNA-seq, and WSI")
    print("  - Progression risk and timing prediction")
    print("  - BRS subtype classification")
    print("  - Comprehensive visualization and analysis")
    config = create_bcg_config()
    config['save_dir'] = os.path.join(args.save_dir, args.task_name)
    os.makedirs(config['save_dir'], exist_ok=True)
    # 1. Run experiment
    runner = BCGExperimentRunner(data_root)
    runner.config = config
    try:
        results = runner.run_complete_experiment()

        if results is None:
            print("ERROR: Experiment run failed")
            return None

        # 2. Generate visualizations
        print("\nGenerating result visualizations...")
        visualizer = BCGVisualizationSuite(results['model'], results['trainer'])

        # Save all visualizations at once
        figure_save_dir = os.path.join(args.save_dir, args.task_name, 'figures')
        visualizer.save_all_visualizations(results['predictions'], figure_save_dir)

        # 4. Final recommendations
        print(f"\nBCG treatment optimization recommendations:")
        print(f"=" * 40)

        metrics = results['metrics']
        if metrics['c_index'] > 0.7:
            print("Excellent model performance - suitable for clinical application")
        else:
            print("WARNING: Model performance needs improvement - additional data collection recommended")

        if metrics['auc'] > 0.75:
            print("Excellent progression prediction performance - patient stratification possible")

        if 'high_risk_progression_rate' in metrics and 'low_risk_progression_rate' in metrics:
            risk_separation = metrics['high_risk_progression_rate'] - metrics['low_risk_progression_rate']
            if risk_separation > 0.3:
                print(f"Excellent risk group discrimination (difference: {risk_separation:.1%})")

        print(f"\nResearch directions:")
        print(f"- External validation with multi-center data")
        print(f"- Add treatment response prediction")
        print(f"- Build a real-time monitoring system")
        print(f"- Strengthen explainable AI features")

        return results

    except Exception as e:
        print(f"ERROR: Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ===================================================================
# Main entry point
# ===================================================================

if __name__ == "__main__":
    if args.mode == 'default':
        results = main_bcg_pipeline(args.data_root)
    elif args.mode == 'debug':
        results = main_bcg_pipeline(args.debug_data_root)
