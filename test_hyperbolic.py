#!/usr/bin/env python
"""
Test script to verify hyperbolic distance implementation.
"""

import torch
import sys
sys.path.insert(0, 'T0')

from lossd import poincare_distance, compute_hyperbolic_knn_temperature, KL, Sinkhorn

def test_poincare_distance():
    """Test Poincaré distance computation."""
    print("Testing Poincaré distance...")

    # Create simple test data
    x = torch.randn(4, 10) * 0.1  # Small values to stay within unit disk
    y = torch.randn(6, 10) * 0.1

    # Compute distance
    dist = poincare_distance(x, y, c=1.0)

    print(f"  Input shapes: x={x.shape}, y={y.shape}")
    print(f"  Output shape: {dist.shape}")
    print(f"  Expected shape: torch.Size([4, 6])")

    assert dist.shape == torch.Size([4, 6]), f"Unexpected shape: {dist.shape}"
    assert torch.all(dist >= 0), "Distances should be non-negative"
    assert torch.all(torch.isfinite(dist)), "Distances should be finite"

    print("  ✓ Poincaré distance test passed!\n")


def test_knn_temperature():
    """Test KNN temperature computation."""
    print("Testing KNN temperature computation...")

    # Create test logits
    teacher_logits = torch.randn(8, 100)
    student_logits = torch.randn(8, 100)

    # Compute temperature
    temp_t, temp_s = compute_hyperbolic_knn_temperature(
        teacher_logits,
        student_logits,
        k=3,
        c=1.0
    )

    print(f"  Teacher temperature: {temp_t.item():.4f}")
    print(f"  Student temperature: {temp_s.item():.4f}")

    assert torch.isfinite(temp_t), "Teacher temperature should be finite"
    assert torch.isfinite(temp_s), "Student temperature should be finite"
    assert temp_t > 0, "Teacher temperature should be positive"
    assert temp_s > 0, "Student temperature should be positive"

    print("  ✓ KNN temperature test passed!\n")


def test_kl_loss():
    """Test KL loss with temperature parameter."""
    print("Testing KL loss...")

    loss_kl = KL()

    # Create test logits
    student_logits = torch.randn(4, 32128)
    teacher_logits = torch.randn(4, 32128)

    # Test with default temperature
    loss_default = loss_kl(student_logits, teacher_logits)
    print(f"  Loss (default temp): {loss_default.item():.4f}")

    # Test with custom temperature
    loss_custom = loss_kl(student_logits, teacher_logits, temperature=3.0)
    print(f"  Loss (custom temp=3.0): {loss_custom.item():.4f}")

    assert torch.isfinite(loss_default), "Loss should be finite"
    assert torch.isfinite(loss_custom), "Loss should be finite"
    assert loss_default != loss_custom, "Different temperatures should give different losses"

    print("  ✓ KL loss test passed!\n")


def test_sinkhorn_loss():
    """Test Sinkhorn loss with hyperbolic distance."""
    print("Testing Sinkhorn loss...")

    loss_sk = Sinkhorn()

    # Create test logits
    student_logits = torch.randn(4, 32128)
    teacher_logits = torch.randn(4, 32128)

    # Test with hyperbolic distance (default)
    loss_hyperbolic = loss_sk(student_logits, teacher_logits)
    print(f"  Loss (hyperbolic distance): {loss_hyperbolic.item():.6f}")

    assert torch.isfinite(loss_hyperbolic), "Loss should be finite"
    assert loss_hyperbolic >= 0, "Loss should be non-negative"

    print("  ✓ Sinkhorn loss test passed!\n")


def test_integration():
    """Test full integration with temperature."""
    print("Testing full integration...")

    loss_kl = KL()
    loss_sk = Sinkhorn()

    # Simulate training step
    student_logits = torch.randn(4, 32128)
    teacher_logits = torch.randn(4, 32128)

    # Compute hyperbolic temperature
    temp_t, temp_s = compute_hyperbolic_knn_temperature(
        teacher_logits,
        student_logits,
        k=5,
        c=1.0
    )
    hyperbolic_temp = (temp_t + temp_s) / 2.0

    print(f"  Hyperbolic temperature: {hyperbolic_temp.item():.4f}")

    # Compute losses with hyperbolic temperature
    kl_loss = loss_kl(student_logits, teacher_logits, temperature=hyperbolic_temp)
    sk_loss = loss_sk(student_logits, teacher_logits, temperature=hyperbolic_temp)

    print(f"  KL loss: {kl_loss.item():.4f}")
    print(f"  Sinkhorn loss: {sk_loss.item():.6f}")

    # Combined loss (as in training loop)
    total_loss = 0.1 * kl_loss + 1.0 * sk_loss
    print(f"  Total distillation loss: {total_loss.item():.4f}")

    assert torch.isfinite(total_loss), "Total loss should be finite"

    print("  ✓ Integration test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Hyperbolic Distance Implementation Test Suite")
    print("=" * 60)
    print()

    try:
        test_poincare_distance()
        test_knn_temperature()
        test_kl_loss()
        test_sinkhorn_loss()
        test_integration()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
