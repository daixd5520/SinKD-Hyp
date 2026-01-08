"""
Statistical Gating Network for MoG-SKD

Uses interpretable statistical signals to weight different geometry experts.
This makes the gating mechanism transparent and explainable for KDD reviewers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticalGating(nn.Module):
    """
    Statistical Gating Network for Expert Selection.

    KDD Insight:
    The gating is NOT a black box - it uses three interpretable statistical features:
    1. Entropy (Uncertainty)
    2. Top1-Top2 Margin (Sharpness)
    3. Max Probability (Confidence)

    These features allow us to explain WHY certain geometries are preferred.

    Input: Logits [batch_size, num_classes]
    Output: Expert weights [batch_size, 3]
            - weight_fisher (Expert A: Information Geometry)
            - weight_euclid (Expert B: Euclidean)
            - weight_hyper (Expert C: Hyperbolic)
    """
    def __init__(self, hidden_dim=32, num_experts=3):
        super().__init__()

        # Input features: 3 statistical signals
        self.input_dim = 3

        # Output: weights for 3 experts
        self.num_experts = num_experts

        # Simple MLP (not too deep, ensures trainability)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(0.1),  # Light dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # Initialize final layer to produce near-uniform distribution initially
        # This prevents early collapse to a single expert
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def get_features(self, logits):
        """
        Extract statistical features from logits.

        These features capture different aspects of the prediction distribution:
        - Entropy: Overall uncertainty
        - Margin: Sharpness of top prediction
        - Max Prob: Confidence of top prediction

        Args:
            logits: [batch_size, num_classes]

        Returns:
            features: [batch_size, 3] (normalized)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)

        # Feature 1: Entropy (Uncertainty measure)
        # High entropy -> uncertain prediction
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

        # Feature 2: Margin (Sharpness measure)
        # Difference between top-1 and top-2 predictions
        top2_vals, _ = torch.topk(probs, 2, dim=-1)
        margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(-1)

        # Feature 3: Max Probability (Confidence measure)
        max_prob = top2_vals[:, 0].unsqueeze(-1)

        # Normalize features (CRITICAL for training stability!)
        # Without normalization, features have vastly different scales
        # We use min-max normalization per batch

        # Entropy: normalize to [0, 1]
        # Max entropy for n classes is log(n)
        num_classes = logits.size(-1)
        max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
        entropy_norm = entropy / max_entropy

        # Margin: already in [0, 1] since probabilities sum to 1
        margin_norm = margin

        # Max prob: already in [0, 1]
        max_prob_norm = max_prob

        features = torch.cat([entropy_norm, margin_norm, max_prob_norm], dim=-1)

        return features

    def forward(self, logits):
        """
        Compute expert weights based on statistical features.

        Args:
            logits: [batch_size, num_classes]
                    Can use teacher logits (more stable) or student logits

        Returns:
            weights: [batch_size, 3] (sums to 1 for each sample)
        """
        # Extract features
        x = self.get_features(logits)

        # Compute raw scores
        scores = self.net(x)

        # Apply softmax to get weights that sum to 1
        # Temperature 1.0 for standard softmax
        weights = F.softmax(scores, dim=-1)

        return weights

    def get_entropy(self, weights):
        """
        Compute entropy of gating distribution.

        Used for regularization to encourage sparsity/selectivity.

        Args:
            weights: [batch_size, num_experts]

        Returns:
            entropy: [batch_size]
        """
        return -(weights * torch.log(weights + 1e-9)).sum(dim=-1)


class AdaptiveGating(nn.Module):
    """
    Advanced gating with attention mechanism.

    Optional enhancement: uses attention to weight experts
    based on both statistical features and learned embeddings.
    """
    def __init__(self, num_classes, hidden_dim=64, num_experts=3):
        super().__init__()

        self.num_experts = num_experts

        # Statistical feature branch
        self.statistical_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Learnable class embeddings
        self.class_embeddings = nn.Embedding(num_classes, hidden_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, num_experts)

    def get_features(self, logits):
        """Get statistical features (same as StatisticalGating)"""
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)

        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

        top2_vals, _ = torch.topk(probs, 2, dim=-1)
        margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(-1)
        max_prob = top2_vals[:, 0].unsqueeze(-1)

        num_classes = logits.size(-1)
        max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

        features = torch.cat([
            entropy / max_entropy,
            margin,
            max_prob
        ], dim=-1)

        return features

    def forward(self, logits):
        """
        Args:
            logits: [batch_size, num_classes]

        Returns:
            weights: [batch_size, num_experts]
        """
        batch_size = logits.size(0)

        # Statistical features
        stat_features = self.get_features(logits)  # [batch, 3]
        stat_embed = self.statistical_net(stat_features)  # [batch, hidden]

        # Get top class indices
        top_classes = torch.argmax(logits, dim=-1)  # [batch]
        class_embed = self.class_embeddings(top_classes)  # [batch, hidden]

        # Combine features
        combined = stat_embed + class_embed  # [batch, hidden]
        combined = combined.unsqueeze(1)  # [batch, 1, hidden] for attention

        # Self-attention
        attended, _ = self.attention(combined, combined, combined)
        attended = attended.squeeze(1)  # [batch, hidden]

        # Project to expert weights
        scores = self.output_proj(attended)  # [batch, num_experts]
        weights = F.softmax(scores, dim=-1)

        return weights
