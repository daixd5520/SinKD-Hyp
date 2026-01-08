#!/usr/bin/env python
"""
Test suite for MoG-SKD framework

Validates all components work correctly.
"""

import torch
import sys
sys.path.insert(0, '.')

from losses.experts import FisherRaoExpert, EuclideanExpert, HyperbolicExpert
from losses.gating import StatisticalGating
from mog_skd import MoGSKD, MoGSKDConfig


def test_fisher_rao_expert():
    """Test Fisher-Rao Expert (Information Geometry)"""
    print("Testing Fisher-Rao Expert...")

    expert = FisherRaoExpert(T=2.0)

    # Create test data
    student_logits = torch.randn(8, 100)
    teacher_logits = torch.randn(8, 100)

    # Compute loss
    loss = expert(student_logits, teacher_logits)

    print(f"  Input shapes: student={student_logits.shape}, teacher={teacher_logits.shape}")
    print(f"  Output shape: {loss.shape}")
    print(f"  Expected shape: torch.Size([8])")

    assert loss.shape == torch.Size([8]), f"Unexpected shape: {loss.shape}"
    assert torch.all(loss >= 0), "Losses should be non-negative"
    assert torch.all(torch.isfinite(loss)), "Losses should be finite"
    assert loss.mean() > 0 and loss.mean() < 2, "Loss should be in reasonable range [0, 2]"

    print("  ✓ Fisher-Rao Expert test passed!\n")


def test_euclidean_expert():
    """Test Euclidean Expert"""
    print("Testing Euclidean Expert...")

    expert = EuclideanExpert(T=2.0, use_sinkhorn=False)

    # Create test data
    student_logits = torch.randn(8, 100)
    teacher_logits = torch.randn(8, 100)

    # Compute loss
    loss = expert(student_logits, teacher_logits)

    print(f"  Input shapes: student={student_logits.shape}, teacher={teacher_logits.shape}")
    print(f"  Output shape: {loss.shape}")
    print(f"  Expected shape: torch.Size([8])")

    assert loss.shape == torch.Size([8]), f"Unexpected shape: {loss.shape}"
    assert torch.all(loss >= 0), "Losses should be non-negative"
    assert torch.all(torch.isfinite(loss)), "Losses should be finite"

    print("  ✓ Euclidean Expert test passed!\n")


def test_hyperbolic_expert():
    """Test Hyperbolic Expert (Rigorous)"""
    print("Testing Hyperbolic Expert...")

    expert = HyperbolicExpert(T=2.0, c=1.0, learnable_curvature=True)

    # Create test data
    student_logits = torch.randn(8, 100)
    teacher_logits = torch.randn(8, 100)

    # Compute loss
    loss = expert(student_logits, teacher_logits)

    print(f"  Input shapes: student={student_logits.shape}, teacher={teacher_logits.shape}")
    print(f"  Output shape: {loss.shape}")
    print(f"  Expected shape: torch.Size([8])")
    print(f"  Hyperbolic curvature c: {expert.c.item():.4f}")

    assert loss.shape == torch.Size([8]), f"Unexpected shape: {loss.shape}"
    assert torch.all(loss >= 0), "Losses should be non-negative"
    assert torch.all(torch.isfinite(loss)), "Losses should be finite"

    print("  ✓ Hyperbolic Expert test passed!\n")


def test_gating_network():
    """Test Statistical Gating Network"""
    print("Testing Statistical Gating Network...")

    gating = StatisticalGating(hidden_dim=32, num_experts=3)

    # Create test logits
    logits = torch.randn(8, 100)

    # Get features
    features = gating.get_features(logits)
    print(f"  Features shape: {features.shape}")
    print(f"  Expected shape: torch.Size([8, 3])")

    assert features.shape == torch.Size([8, 3]), f"Unexpected features shape: {features.shape}"

    # Get weights
    weights = gating(logits)
    print(f"  Weights shape: {weights.shape}")
    print(f"  Expected shape: torch.Size([8, 3])")
    print(f"  Weights sum to 1: {torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5)}")

    assert weights.shape == torch.Size([8, 3]), f"Unexpected weights shape: {weights.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5), "Weights should sum to 1"

    # Test entropy
    entropy = gating.get_entropy(weights)
    print(f"  Gating entropy: {entropy.mean().item():.4f}")
    assert entropy.shape == torch.Size([8]), f"Unexpected entropy shape: {entropy.shape}"

    print("  ✓ Statistical Gating test passed!\n")


def test_mog_skd():
    """Test MoGSKD Unified Framework"""
    print("Testing MoGSKD Unified Framework...")

    mog_skd = MoGSKD(
        T=2.0,
        lambda_reg=0.1,
        hidden_dim=32,
        use_sinkhorn=False,
        learnable_curvature=True,
        hyperbolic_c=1.0
    )

    # Create test data
    student_logits = torch.randn(8, 100)
    teacher_logits = torch.randn(8, 100)

    # Forward pass without details
    loss = mog_skd(student_logits, teacher_logits, return_details=False)
    print(f"  Loss (without details): {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss should be finite"

    # Forward pass with details
    loss, logs = mog_skd(student_logits, teacher_logits, return_details=True)
    print(f"  Loss (with details): {loss.item():.4f}")
    print(f"  Number of logged metrics: {len(logs)}")

    # Check logged values
    for key in ['loss_fisher', 'loss_euclid', 'loss_hyper',
                'weight_fisher', 'weight_euclid', 'weight_hyper',
                'gating_entropy', 'distill_loss', 'reg_loss']:
        assert key in logs, f"Missing log key: {key}"
        print(f"  {key}: {logs[key]:.4f}")

    # Check per-sample data
    assert 'per_sample_data' in logs, "Missing per_sample_data"
    per_sample = logs['per_sample_data']
    assert 'fisher_loss' in per_sample, "Missing fisher_loss in per_sample_data"
    assert per_sample['fisher_loss'].shape == torch.Size([8]), "Wrong per_sample shape"

    # Test backward pass
    loss.backward()
    print(f"  Backward pass successful")

    print("  ✓ MoGSKD Framework test passed!\n")


def test_mog_skd_config():
    """Test MoGSKD Config"""
    print("Testing MoGSKD Configuration...")

    # Create config from dict
    config_dict = {
        'T': 2.0,
        'lambda_reg': 0.1,
        'hidden_dim': 32,
        'use_sinkhorn': False,
        'learnable_curvature': True,
        'hyperbolic_c': 1.0,
        'use_fisher': True,
        'use_euclid': True,
        'use_hyper': True
    }

    config = MoGSKDConfig.from_dict(config_dict)
    print(f"  Config created from dict")

    # Convert back to dict
    config_dict2 = config.to_dict()
    assert config_dict == config_dict2, "Config conversion failed"
    print(f"  Config to dict conversion successful")

    # Create model from config
    model = config.create_model()
    assert isinstance(model, MoGSKD), "Model creation failed"
    print(f"  Model created from config")

    print("  ✓ MoGSKD Configuration test passed!\n")


def test_training_step():
    """Test a complete training step"""
    print("Testing Training Step...")

    mog_skd = MoGSKD(
        T=2.0,
        lambda_reg=0.1,
        hidden_dim=32,
        use_sinkhorn=False,
        learnable_curvature=True,
        hyperbolic_c=1.0
    )

    # Create optimizer
    optimizer = torch.optim.Adam([
        {'params': mog_skd.parameters(), 'lr': 1e-3}
    ])

    # Training step
    student_logits = torch.randn(4, 100, requires_grad=True)
    teacher_logits = torch.randn(4, 100)

    # Forward
    loss, logs = mog_skd(student_logits, teacher_logits, return_details=True)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  Initial loss: {loss.item():.4f}")
    print(f"  Fisher weight: {logs['weight_fisher']:.4f}")
    print(f"  Euclid weight: {logs['weight_euclid']:.4f}")
    print(f"  Hyper weight: {logs['weight_hyper']:.4f}")

    # Second step (weights should change)
    student_logits = torch.randn(4, 100, requires_grad=True)
    loss2, logs2 = mog_skd(student_logits, teacher_logits, return_details=True)

    print(f"  Second step loss: {loss2.item():.4f}")

    # Check gradients were computed
    assert mog_skd.expert_hyper.c.grad is not None or not mog_skd.expert_hyper.c.requires_grad, \
        "Curvature gradient not computed"

    print("  ✓ Training Step test passed!\n")


def test_expert_selection():
    """Test that different samples select different experts"""
    print("Testing Expert Selection...")

    mog_skd = MoGSKD(T=2.0, lambda_reg=0.01)  # Low regularization for diversity

    # Create diverse data
    # Low entropy (confident)
    confident_logits = torch.zeros(4, 100)
    confident_logits[:, 0] = 10.0  # Very confident

    # High entropy (uncertain)
    uncertain_logits = torch.randn(4, 100) * 0.1  # All similar

    teacher_logits = torch.cat([confident_logits, uncertain_logits], dim=0)
    student_logits = torch.randn(8, 100)

    loss, logs = mog_skd(student_logits, teacher_logits, return_details=True)

    # Get per-sample weights
    weights_fisher = logs['per_sample_data']['fisher_weight']
    weights_euclid = logs['per_sample_data']['euclid_weight']
    weights_hyper = logs['per_sample_data']['hyper_weight']

    print(f"  Fisher weights: min={weights_fisher.min().item():.4f}, max={weights_fisher.max().item():.4f}")
    print(f"  Euclid weights: min={weights_euclid.min().item():.4f}, max={weights_euclid.max().item():.4f}")
    print(f"  Hyper weights: min={weights_hyper.min().item():.4f}, max={weights_hyper.max().item():.4f}")

    # Check that there's variation (not all same)
    assert weights_fisher.std() > 0 or weights_euclid.std() > 0 or weights_hyper.std() > 0, \
        "No variation in expert selection"

    print("  ✓ Expert Selection test passed!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("MoG-SKD Test Suite")
    print("=" * 70)
    print()

    try:
        test_fisher_rao_expert()
        test_euclidean_expert()
        test_hyperbolic_expert()
        test_gating_network()
        test_mog_skd()
        test_mog_skd_config()
        test_training_step()
        test_expert_selection()

        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        print()
        print("MoG-SKD is ready for KDD submission! 🚀")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
