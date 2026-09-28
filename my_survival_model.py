import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MultiHeadCrossAttention(nn.Module):
    """Multi-head self-attention over the three modality tokens (returns attention map)."""

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
    """Pre-norm transformer encoder layer."""

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
    """Multimodal transformer used for the CHIMERA Task 3 submission.

    clinical (27) / RNA (40) / WSI (1024) -> per-modality MLP embedding (256)
    -> 3 modality tokens + learned modality positional embedding
    -> ``num_transformer_layers`` pre-norm transformer layers
    -> one extra multi-head attention block (``cross_attention``)
    -> softmax gate over the three attended tokens -> weighted sum
    -> MLP -> heads.

    Only ``progression_risk`` (sigmoid) is used for the submission; it was trained
    with a Cox partial-likelihood loss.  The ``time``, ``brs`` and
    ``uncertainty`` heads and ``modality_weights`` are unused auxiliary outputs.
    """

    def __init__(self, config: Dict):
        super(AdvancedBCGTransformerNet, self).__init__()
        self.config = config

        # dimensions
        clinical_dim = config['clinical_dim']
        rna_dim = config['rna_dim']
        wsi_dim = config['wsi_dim']
        hidden_dim = config.get('hidden_dim', 256)

        # per-modality embeddings
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

        # Prediction heads
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
            nn.Softplus()
        )

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3))

    def forward(self, clinical_data, rna_data, wsi_data, return_attention=False):
        batch_size = clinical_data.size(0)

        # embed each modality
        clinical_emb = self.clinical_embedding(clinical_data).unsqueeze(1)  # [B, 1, D]
        rna_emb = self.rna_embedding(rna_data).unsqueeze(1)  # [B, 1, D]
        wsi_emb = self.wsi_embedding(wsi_data).unsqueeze(1)  # [B, 1, D]

        # stack as tokens
        multimodal_tokens = torch.cat([clinical_emb, rna_emb, wsi_emb], dim=1)  # [B, 3, D]

        # modality positional embedding
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

