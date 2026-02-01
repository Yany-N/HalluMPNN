#!/usr/bin/env python3
"""
Training Charts Generator for HalluMPNN
Generates training progress plots with English titles.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_charts(
    metrics_history: list,
    output_dir: str,
    reward_history: list = None
):
    """
    Generate training progress charts.
    
    Args:
        metrics_history: List of step metrics dicts
        output_dir: Directory to save charts
        reward_history: Optional list of reward values
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not metrics_history:
        print("No metrics to plot")
        return
    
    # Extract data
    steps = [m.get('step', i) for i, m in enumerate(metrics_history)]
    rewards = [m.get('best_reward', 0) for m in metrics_history]
    mean_rewards = [m.get('mean_reward', 0) for m in metrics_history]
    
    # Extract detailed metrics
    iptm_vals = []
    ptm_vals = []
    pae_vals = []
    
    for m in metrics_history:
        best_metrics = m.get('best_metrics', {})
        iptm_vals.append(best_metrics.get('iptm', 0))
        ptm_vals.append(best_metrics.get('ptm', 0))
        pae_vals.append(best_metrics.get('mean_pae', 31.75))
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_dpi = 150
    
    # ====================================
    # Plot 1: Reward Progress
    # ====================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=fig_dpi)
    
    ax.plot(steps, rewards, 'b-', linewidth=2, label='Best Reward', marker='o', markersize=3)
    ax.plot(steps, mean_rewards, 'g--', linewidth=1.5, label='Mean Reward', alpha=0.7)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('HalluMPNN Training Progress - Reward', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(steps) > 5:
        z = np.polyfit(steps, rewards, 1)
        p = np.poly1d(z)
        ax.plot(steps, p(steps), 'r:', alpha=0.5, label='Trend')
    
    fig.tight_layout()
    fig.savefig(output_path / 'reward_progress.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path / 'reward_progress.png'}")
    
    # ====================================
    # Plot 2: Confidence Metrics
    # ====================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=fig_dpi)
    
    # iPTM
    ax = axes[0, 0]
    ax.plot(steps, iptm_vals, 'b-', linewidth=2, marker='o', markersize=3)
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    ax.set_xlabel('Step')
    ax.set_ylabel('iPTM')
    ax.set_title('Interface pTM Score')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # pTM
    ax = axes[0, 1]
    ax.plot(steps, ptm_vals, 'g-', linewidth=2, marker='s', markersize=3)
    ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Threshold (0.6)')
    ax.set_xlabel('Step')
    ax.set_ylabel('pTM')
    ax.set_title('Predicted TM Score')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PAE
    ax = axes[1, 0]
    ax.plot(steps, pae_vals, 'orange', linewidth=2, marker='^', markersize=3)
    ax.axhline(y=10.0, color='r', linestyle='--', alpha=0.5, label='Threshold (10.0)')
    ax.set_xlabel('Step')
    ax.set_ylabel('PAE (Angstroms)')
    ax.set_title('Predicted Alignment Error')
    ax.set_ylim(0, 35)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Combined Reward
    ax = axes[1, 1]
    ax.plot(steps, rewards, 'purple', linewidth=2, marker='D', markersize=3)
    ax.fill_between(steps, rewards, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Combined Reward Score')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('HalluMPNN Training Metrics', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'metrics_dashboard.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path / 'metrics_dashboard.png'}")
    
    # ====================================
    # Plot 3: Training Summary
    # ====================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=fig_dpi)
    
    x = np.arange(len(steps))
    width = 0.25
    
    if len(steps) <= 20:  # Bar chart for few steps
        ax.bar(x - width, iptm_vals, width, label='iPTM', color='blue', alpha=0.7)
        ax.bar(x, ptm_vals, width, label='pTM', color='green', alpha=0.7)
        ax.bar(x + width, [1 - p/31.75 for p in pae_vals], width, label='1-PAE/max', color='orange', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in steps])
    else:  # Line chart for many steps
        ax.plot(steps, iptm_vals, 'b-', linewidth=2, label='iPTM')
        ax.plot(steps, ptm_vals, 'g-', linewidth=2, label='pTM')
        ax.plot(steps, [1 - p/31.75 for p in pae_vals], 'orange', linewidth=2, label='1-PAE/max')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('HalluMPNN Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path / 'metrics_comparison.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path / 'metrics_comparison.png'}")
    
    print(f"All training charts saved to {output_path}")


def main():
    """Test chart generation with sample data."""
    sample_metrics = [
        {'step': 0, 'best_reward': 0.3, 'mean_reward': 0.25, 'best_metrics': {'iptm': 0.4, 'ptm': 0.5, 'mean_pae': 20.0}},
        {'step': 1, 'best_reward': 0.35, 'mean_reward': 0.28, 'best_metrics': {'iptm': 0.45, 'ptm': 0.52, 'mean_pae': 18.5}},
        {'step': 2, 'best_reward': 0.42, 'mean_reward': 0.35, 'best_metrics': {'iptm': 0.52, 'ptm': 0.55, 'mean_pae': 16.2}},
        {'step': 3, 'best_reward': 0.48, 'mean_reward': 0.40, 'best_metrics': {'iptm': 0.58, 'ptm': 0.60, 'mean_pae': 14.0}},
        {'step': 4, 'best_reward': 0.55, 'mean_reward': 0.45, 'best_metrics': {'iptm': 0.65, 'ptm': 0.62, 'mean_pae': 12.5}},
    ]
    
    plot_training_charts(sample_metrics, './test_charts')
    print("Test charts generated!")


if __name__ == '__main__':
    main()
