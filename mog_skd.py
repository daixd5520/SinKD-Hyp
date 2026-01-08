"""
MoG-SKD: Mixture-of-Geometries Sinkhorn Knowledge Distillation

A unified framework combining multiple geometric experts for knowledge distillation.
This is the main model class used in training scripts.
"""

import torch
import torch.nn as nn
from losses.experts import FisherRaoExpert, EuclideanExpert, HyperbolicExpert
from losses.gating import StatisticalGating


class MoGSKD(nn.Module):
    """
    Mixture-of-Geometries Sinkhorn Knowledge Distillation Framework.

    KDD Core Contribution:
    - Dynamically selects the most appropriate geometry for each sample
    - Combines three experts: Fisher-Rao, Euclidean, Hyperbolic
    - Uses interpretable statistical gating (not a black box)
    - Includes entropy regularization to prevent collapse

    Architecture:
        1. Three Geometry Experts compute per-sample losses
        2. Statistical Gating Network computes expert weights
        3. Weighted combination of expert losses
        4. Entropy regularization encourages selective (sparse) gating

    Args:
        T: Temperature for distillation (default: 1.0)
        lambda_reg: Entropy regularization coefficient (default: 0.1)
        hidden_dim: Hidden dimension for gating network (default: 32)
        use_sinkhorn: Whether EuclideanExpert uses Sinkhorn (default: False)
        learnable_curvature: Whether hyperbolic curvature is learnable (default: True)
    """
    def __init__(
        self,
        T=1.0,
        lambda_reg=0.1,
        hidden_dim=32,
        use_sinkhorn=False,
        learnable_curvature=True,
        hyperbolic_c=1.0
    ):
        super().__init__()

        # Hyperparameters
        self.T = T
        self.lambda_reg = lambda_reg

        # Initialize three geometry experts
        self.expert_fisher = FisherRaoExpert(T=T)
        self.expert_euclid = EuclideanExpert(T=T, use_sinkhorn=use_sinkhorn)
        self.expert_hyper = HyperbolicExpert(T=T, c=hyperbolic_c, learnable_curvature=learnable_curvature)

        # Initialize statistical gating network
        self.gating = StatisticalGating(hidden_dim=hidden_dim, num_experts=3)

        # For logging statistics
        self.register_buffer('step_count', torch.zeros(1))

    def forward(self, student_logits, teacher_logits, return_details=False):
        """
        Compute MoG-SKD loss.

        Args:
            student_logits: Student model outputs [batch_size, num_classes]
            teacher_logits: Teacher model outputs [batch_size, num_classes]
            return_details: If True, return detailed info for visualization

        Returns:
            If return_details=False:
                total_loss: Scalar loss (backward-able)

            If return_details=True:
                total_loss: Scalar loss
                logs: Dictionary with detailed metrics
        """
        batch_size = student_logits.size(0)

        # ========================================
        # 1. Compute losses from each expert
        # All return per-sample losses [batch_size]
        # ========================================
        l_fisher = self.expert_fisher(student_logits, teacher_logits)  # [batch]
        l_euclid = self.expert_euclid(student_logits, teacher_logits)  # [batch]
        l_hyper = self.expert_hyper(student_logits, teacher_logits)    # [batch]

        # Stack losses: [batch_size, 3]
        # Column 0: Fisher-Rao
        # Column 1: Euclidean
        # Column 2: Hyperbolic
        expert_losses = torch.stack([l_fisher, l_euclid, l_hyper], dim=1)

        # ========================================
        # 2. Compute Gating weights
        # ========================================
        # KDD Insight: Use teacher logits for gating
        # This means "select geometry based on knowledge difficulty"
        gate_weights = self.gating(teacher_logits)  # [batch_size, 3]

        # ========================================
        # 3. Weighted combination (MoE)
        # ========================================
        # Element-wise multiply and sum over experts
        weighted_losses = gate_weights * expert_losses  # [batch, 3]
        distill_loss = weighted_losses.sum(dim=1).mean()  # Scalar

        # ========================================
        # 4. Entropy regularization
        # ========================================
        # We want gating to be selective (sparse), not uniform
        # Minimize entropy of gating distribution
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-9)).sum(dim=1).mean()

        # Total loss
        total_loss = distill_loss + self.lambda_reg * gate_entropy

        # Update step counter
        self.step_count += 1

        # ========================================
        # 5. Logging (for KDD "Money Plot")
        # ========================================
        if return_details:
            logs = {
                # Expert losses
                'loss_fisher': l_fisher.mean().item(),
                'loss_euclid': l_euclid.mean().item(),
                'loss_hyper': l_hyper.mean().item(),
                'loss_std_fisher': l_fisher.std().item(),
                'loss_std_euclid': l_euclid.std().item(),
                'loss_std_hyper': l_hyper.std().item(),

                # Gating weights (per batch)
                'weight_fisher': gate_weights[:, 0].mean().item(),
                'weight_euclid': gate_weights[:, 1].mean().item(),
                'weight_hyper': gate_weights[:, 2].mean().item(),
                'weight_std_fisher': gate_weights[:, 0].std().item(),
                'weight_std_euclid': gate_weights[:, 1].std().item(),
                'weight_std_hyper': gate_weights[:, 2].std().item(),

                # Gating statistics
                'gating_entropy': gate_entropy.item(),
                'gating_entropy_per_sample': -(gate_weights * torch.log(gate_weights + 1e-9)).sum(dim=1),

                # Distillation breakdown
                'distill_loss': distill_loss.item(),
                'reg_loss': (self.lambda_reg * gate_entropy).item(),

                # Hyperbolic curvature (if learnable)
                'hyperbolic_curvature': self.expert_hyper.c.item() if hasattr(self.expert_hyper.c, 'item') else self.expert_hyper.c.detach().item(),

                # Individual sample data (for visualization)
                'per_sample_data': {
                    'fisher_loss': l_fisher.detach(),
                    'euclid_loss': l_euclid.detach(),
                    'hyper_loss': l_hyper.detach(),
                    'fisher_weight': gate_weights[:, 0].detach(),
                    'euclid_weight': gate_weights[:, 1].detach(),
                    'hyper_weight': gate_weights[:, 2].detach(),
                    'teacher_entropy': self.gating.get_features(teacher_logits)[:, 0].detach(),  # Entropy feature
                }
            }

            return total_loss, logs
        else:
            return total_loss

    def get_expert_losses(self, student_logits, teacher_logits):
        """
        Get individual expert losses (for analysis/ablation).

        Args:
            student_logits: [batch_size, num_classes]
            teacher_logits: [batch_size, num_classes]

        Returns:
            dict: Dictionary with individual expert losses
        """
        with torch.no_grad():
            l_fisher = self.expert_fisher(student_logits, teacher_logits)
            l_euclid = self.expert_euclid(student_logits, teacher_logits)
            l_hyper = self.expert_hyper(student_logits, teacher_logits)

            return {
                'fisher': l_fisher.mean().item(),
                'euclid': l_euclid.mean().item(),
                'hyper': l_hyper.mean().item()
            }

    def get_gating_weights(self, logits):
        """
        Get gating weights for analysis.

        Args:
            logits: [batch_size, num_classes]

        Returns:
            weights: [batch_size, 3]
        """
        with torch.no_grad():
            return self.gating(logits)


class MoGSKDConfig:
    """
    Configuration class for MoG-SKD.

    Makes it easy to manage hyperparameters and run ablation studies.
    """
    def __init__(
        self,
        T=1.0,
        lambda_reg=0.1,
        hidden_dim=32,
        use_sinkhorn=False,
        learnable_curvature=True,
        hyperbolic_c=1.0,
        # Expert selection (for ablation)
        use_fisher=True,
        use_euclid=True,
        use_hyper=True
    ):
        self.T = T
        self.lambda_reg = lambda_reg
        self.hidden_dim = hidden_dim
        self.use_sinkhorn = use_sinkhorn
        self.learnable_curvature = learnable_curvature
        self.hyperbolic_c = hyperbolic_c
        self.use_fisher = use_fisher
        self.use_euclid = use_euclid
        self.use_hyper = use_hyper

    def create_model(self):
        """Create MoGSKD model from config."""
        return MoGSKD(
            T=self.T,
            lambda_reg=self.lambda_reg,
            hidden_dim=self.hidden_dim,
            use_sinkhorn=self.use_sinkhorn,
            learnable_curvature=self.learnable_curvature,
            hyperbolic_c=self.hyperbolic_c
        )

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'T': self.T,
            'lambda_reg': self.lambda_reg,
            'hidden_dim': self.hidden_dim,
            'use_sinkhorn': self.use_sinkhorn,
            'learnable_curvature': self.learnable_curvature,
            'hyperbolic_c': self.hyperbolic_c,
            'use_fisher': self.use_fisher,
            'use_euclid': self.use_euclid,
            'use_hyper': self.use_hyper
        }
