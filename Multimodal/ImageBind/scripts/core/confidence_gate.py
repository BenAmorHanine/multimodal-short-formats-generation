"""
Stage 4: Late Fusion Confidence Gate Module

Combines trimodal ImageBind embeddings (vision, audio, text) using learned
per-modality gates weighted by text transcription confidence (text_trust).

Architecture:
  - Per-modality gate networks: [2 hidden layers] + Sigmoid → soft attention weights
  - Projection heads: project each modality to shared 512-d space
  - Fusion: weighted sum with text_trust as additional multiplier for text modality
  - Output: normalized 512-d unified embedding + per-modality weight dict

Key Design:
  - Text gate receives cross-modal context (vision+audio+text) before weight computation
  - Text trust from AudioClassifier gates the text modality influence
  - Outputs individual weights for interpretability and debugging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceGate(nn.Module):
    """
    Late fusion module learning per-modality confidence weights.
    
    Takes normalized [N, 1024] embeddings from each modality (vision, audio, text)
    and text_trust scores [N,] from preprocessing classification. Returns unified
    [N, 512] embeddings with modality weights.
    
    Args:
        input_dim (int): Input embedding dimension (1024 for ImageBind). Default: 1024
        proj_dim (int): Output projection dimension (512 for downstream tasks). Default: 512
    """
    
    def __init__(self, input_dim=1024, proj_dim=512):
        super().__init__()
        
        # Gate networks: map cross-modal context to per-modality soft weights
        def gate(d):
            return nn.Sequential(
                nn.Linear(d, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()  # soft weight ∈ [0, 1]
            )
        
        # Projection heads: map modality embeddings to shared space
        def proj(d, o):
            return nn.Sequential(
                nn.Linear(d, o),
                nn.GELU(),
                nn.LayerNorm(o)
            )
        
        # Vision gate: sees vision + audio context
        self.w_v = gate(input_dim * 2)  # [v, a] → weight_v
        
        # Audio gate: sees audio + vision context
        self.w_a = gate(input_dim * 2)  # [a, v] → weight_a
        
        # Text gate: sees all three modalities + will be weighted by text_trust
        self.w_t = gate(input_dim * 3)  # [v, a, t] → weight_t (before trust scaling)
        
        # Projection layers
        self.pv = proj(input_dim, proj_dim)  # vision projection
        self.pa = proj(input_dim, proj_dim)  # audio projection
        self.pt = proj(input_dim, proj_dim)  # text projection
    
    def forward(self, v, a, t, trust):
        """
        Fuse trimodal embeddings with learned gates.
        
        Args:
            v (Tensor): Vision embeddings [N, 1024]
            a (Tensor): Audio embeddings [N, 1024]
            t (Tensor): Text embeddings [N, 1024]
            trust (Tensor): Text transcription confidence [N,] ∈ [0, 1]
        
        Returns:
            unified (Tensor): Fused embeddings [N, 512] (L2-normalized)
            weights (dict): Per-modality weight distributions:
                - 'vision': [N,] gate weights for vision modality
                - 'audio': [N,] gate weights for audio modality
                - 'text': [N,] gate weights for text modality (post-trust scaling)
        """
        # Compute per-modality gates (soft attention weights)
        wv = self.w_v(torch.cat([v, a], dim=-1))  # [N, 1]
        wa = self.w_a(torch.cat([a, v], dim=-1))  # [N, 1]
        wt = self.w_t(torch.cat([v, a, t], dim=-1)) * trust.unsqueeze(-1)  # [N, 1] scaled by trust
        
        # Fused embedding: weighted average in projection space
        fused = (wv * self.pv(v) + wa * self.pa(a) + wt * self.pt(t)) / (wv + wa + wt + 1e-8)
        
        return F.normalize(fused, dim=-1), {
            'vision': wv.squeeze(-1),
            'audio': wa.squeeze(-1),
            'text': wt.squeeze(-1)
        }


def create_confidence_gate(input_dim=1024, proj_dim=512, device="cpu"):
    """
    Convenience constructor for ConfidenceGate.
    
    Args:
        input_dim (int): Input embedding dimension
        proj_dim (int): Output projection dimension
        device (str): Device to place model on
    
    Returns:
        ConfidenceGate: Module in eval mode on specified device
    """
    gate = ConfidenceGate(input_dim=input_dim, proj_dim=proj_dim)
    return gate.eval().to(device)
