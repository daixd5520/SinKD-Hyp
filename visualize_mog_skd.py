"""
Visualization and Analysis Tools for MoG-SKD

Generates the "Money Plot" for KDD paper:
Figure X: Adaptive Geometry Selection based on Prediction Uncertainty
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict


def plot_money_plot(entropy_data, weight_data, save_path="money_plot.png"):
    """
    Generate the KDD "Money Plot": Expert weights vs. Prediction uncertainty.

    This plot shows HOW the gating network selects different geometries
    based on the teacher's prediction uncertainty (entropy).

    Expected trend:
    - Low entropy (easy samples) -> Euclidean or Fisher-Rao
    - High entropy (hard samples) -> Hyperbolic

    Args:
        entropy_data: List of entropy values [num_samples]
        weight_data: Dict of expert weights
            {
                'fisher': [num_samples],
                'euclid': [num_samples],
                'hyper': [num_samples]
            }
        save_path: Path to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by entropy for smooth curves
    sorted_indices = np.argsort(entropy_data)
    entropy_sorted = np.array(entropy_data)[sorted_indices]

    # Plot each expert's weight
    colors = {
        'fisher': '#2ecc71',  # Green
        'euclid': '#3498db',  # Blue
        'hyper': '#e74c3c'    # Red
    }

    labels = {
        'fisher': 'Fisher-Rao (Information Geometry)',
        'euclid': 'Euclidean (Baseline)',
        'hyper': 'Hyperbolic (Rigorous)'
    }

    for expert, weights in weight_data.items():
        weights_sorted = np.array(weights)[sorted_indices]

        # Smooth with moving average
        window = max(1, len(weights_sorted) // 20)
        if window > 1:
            weights_smooth = np.convolve(weights_sorted, np.ones(window)/window, mode='same')
        else:
            weights_smooth = weights_sorted

        ax.plot(entropy_sorted, weights_smooth,
               label=labels[expert],
               color=colors[expert],
               linewidth=2.5,
               alpha=0.8)

    ax.set_xlabel('Teacher Prediction Entropy (Uncertainty)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Expert Weight', fontsize=14, fontweight='bold')
    ax.set_title('Adaptive Geometry Selection in MoG-SKD', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Add shaded regions to indicate easy/hard samples
    ax.axvspan(0, 0.3, alpha=0.1, color='green', label='Easy Samples')
    ax.axvspan(0.3, 0.7, alpha=0.1, color='yellow', label='Medium Samples')
    ax.axvspan(0.7, 1.0, alpha=0.1, color='red', label='Hard Samples')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Money plot saved to {save_path}")
    plt.close()


def plot_expert_losses(losses_data, save_path="expert_losses.png"):
    """
    Plot expert losses across training.

    Args:
        losses_data: Dict of lists
            {
                'fisher': [epoch1, epoch2, ...],
                'euclid': [epoch1, epoch2, ...],
                'hyper': [epoch1, epoch2, ...]
            }
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(losses_data['fisher']) + 1)

    colors = {
        'fisher': '#2ecc71',
        'euclid': '#3498db',
        'hyper': '#e74c3c'
    }

    labels = {
        'fisher': 'Fisher-Rao Expert',
        'euclid': 'Euclidean Expert',
        'hyper': 'Hyperbolic Expert'
    }

    for expert, losses in losses_data.items():
        ax.plot(epochs, losses, label=labels[expert],
               color=colors[expert], linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Expert Losses During Training', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Expert losses plot saved to {save_path}")
    plt.close()


def plot_gating_entropy(gating_entropy_data, save_path="gating_entropy.png"):
    """
    Plot gating entropy over training (shows specialization).

    Lower entropy = more selective (better).
    Higher entropy = uniform selection (worse).

    Args:
        gating_entropy_data: List of entropy values per epoch
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(gating_entropy_data) + 1)

    ax.plot(epochs, gating_entropy_data, linewidth=2.5, color='#9b59b6', marker='o', markersize=4)

    # Add reference line for uniform distribution
    # Max entropy for 3 experts = log(3)
    max_entropy = np.log(3)
    ax.axhline(y=max_entropy, color='red', linestyle='--', linewidth=2,
              label=f'Uniform (Max Entropy = {max_entropy:.2f})')

    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Gating Entropy', fontsize=14, fontweight='bold')
    ax.set_title('Gating Network Specialization', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gating entropy plot saved to {save_path}")
    plt.close()


def plot_hyperbolic_curvature(curvature_data, save_path="hyperbolic_curvature.png"):
    """
    Plot hyperbolic curvature over time (if learnable).

    Args:
        curvature_data: List of curvature values per epoch
        save_path: Path to save the plot
    """
    if not curvature_data or len(curvature_data) < 2:
        print("Skipping curvature plot (not enough data or not learnable)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(curvature_data) + 1)

    ax.plot(epochs, curvature_data, linewidth=2.5, color='#e67e22', marker='o', markersize=4)

    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hyperbolic Curvature (c)', fontsize=14, fontweight='bold')
    ax.set_title('Learned Hyperbolic Curvature', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at c=1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5,
              label='Initial value (c=1.0)')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Hyperbolic curvature plot saved to {save_path}")
    plt.close()


def generate_all_plots(logs_path, output_dir):
    """
    Generate all visualization plots from training logs.

    Args:
        logs_path: Path to mog_skd_logs.json
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load logs
    with open(logs_path, 'r') as f:
        logs = json.load(f)

    # Extract data
    expert_losses = defaultdict(list)
    expert_weights = defaultdict(list)
    gating_entropies = []
    hyperbolic_curvatures = []

    for epoch_log in logs:
        expert_losses['fisher'].append(epoch_log['loss_fisher'])
        expert_losses['euclid'].append(epoch_log['loss_euclid'])
        expert_losses['hyper'].append(epoch_log['loss_hyper'])

        expert_weights['fisher'].append(epoch_log['weight_fisher'])
        expert_weights['euclid'].append(epoch_log['weight_euclid'])
        expert_weights['hyper'].append(epoch_log['weight_hyper'])

        gating_entropies.append(epoch_log['gating_entropy'])
        hyperbolic_curvatures.append(epoch_log['hyperbolic_curvature'])

    # Generate plots
    print("Generating MoG-SKD visualization plots...")

    plot_expert_losses(expert_losses, save_path=output_path / "expert_losses.png")

    plot_gating_entropy(gating_entropies, save_path=output_path / "gating_entropy.png")

    plot_hyperbolic_curvature(hyperbolic_curvatures, save_path=output_path / "hyperbolic_curvature.png")

    print("All plots generated successfully!")


def generate_money_plot_from_model(model, dataloader, save_path="money_plot.png"):
    """
    Generate Money Plot directly from model and data.

    This collects per-sample data to create the entropy vs. weights plot.

    Args:
        model: MoGSKD model
        dataloader: Validation dataloader
        save_path: Path to save plot
    """
    model.eval()
    entropy_data = []
    weight_data = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            # Get logits (simplified)
            teacher_logits = batch.get('teacher_logits')
            if teacher_logits is None:
                continue

            # Get gating weights
            gate_weights = model.get_gating_weights(teacher_logits)

            # Get entropy feature
            entropy_features = model.gating.get_features(teacher_logits)
            entropy = entropy_features[:, 0].cpu().numpy()  # First feature is entropy

            # Store data
            entropy_data.extend(entropy.tolist())
            weight_data['fisher'].extend(gate_weights[:, 0].cpu().numpy().tolist())
            weight_data['euclid'].extend(gate_weights[:, 1].cpu().numpy().tolist())
            weight_data['hyper'].extend(gate_weights[:, 2].cpu().numpy().tolist())

    # Generate plot
    plot_money_plot(entropy_data, weight_data, save_path=save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MoG-SKD visualizations")
    parser.add_argument("--logs_path", type=str, required=True,
                       help="Path to mog_skd_logs.json")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Output directory for plots")

    args = parser.parse_args()

    generate_all_plots(args.logs_path, args.output_dir)
