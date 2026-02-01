#!/usr/bin/env python3
"""
Training Logger for HalluMPNN

Features:
- Per-step CSV logging
- Excel export with all sequences and metrics
- Training curve plots (reward, iptm, RMSD, loss)
- Resume-aware: appends to existing logs
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("HalluMPNN.Logger")


class TrainingLogger:
    """Logger for HalluMPNN training metrics and sequences."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.csv_path = self.output_dir / "training_log.csv"
        self.sequences_csv = self.output_dir / "sequences.csv"
        
        # Initialize CSV headers if files don't exist
        self._init_csv_files()
        
        # In-memory history for plotting
        self.metrics_history: List[Dict] = []
        self.sequence_history: List[Dict] = []
        
        logger.info(f"TrainingLogger initialized: {self.output_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist."""
        # Main metrics CSV
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'timestamp', 'best_reward', 'mean_reward',
                    'best_iptm', 'best_ptm', 'best_pae', 'best_conf_rmsd',
                    'loss', 'kl_div', 'gradient_norm'
                ])
        
        # Sequences CSV
        if not self.sequences_csv.exists():
            with open(self.sequences_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'variant_idx', 'sequence', 'reward',
                    'iptm', 'ptm', 'pae', 'conf_rmsd', 'is_best'
                ])
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        loss: float = 0.0,
        kl_div: float = 0.0,
        gradient_norm: float = 0.0
    ):
        """Log metrics for a training step."""
        timestamp = datetime.now().isoformat()
        
        best_metrics = metrics.get('best_metrics', {})
        
        row = {
            'step': step,
            'timestamp': timestamp,
            'best_reward': metrics.get('best_reward', 0),
            'mean_reward': metrics.get('mean_reward', 0),
            'best_iptm': best_metrics.get('iptm', 0),
            'best_ptm': best_metrics.get('ptm', 0),
            'best_pae': best_metrics.get('mean_pae', 0),
            'best_conf_rmsd': best_metrics.get('conf_rmsd', 0),
            'loss': loss,
            'kl_div': kl_div,
            'gradient_norm': gradient_norm,
        }
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        
        # Store in memory
        self.metrics_history.append(row)
    
    def log_sequences(
        self,
        step: int,
        sequences: List[str],
        rewards: List[float],
        reward_infos: List[Dict],
        best_idx: int
    ):
        """Log all generated sequences for a step."""
        with open(self.sequences_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, (seq, reward, info) in enumerate(zip(sequences, rewards, reward_infos)):
                writer.writerow([
                    step, i, seq, reward,
                    info.get('iptm', 0),
                    info.get('ptm', 0),
                    info.get('mean_pae', 0),
                    info.get('conf_rmsd', 0),
                    1 if i == best_idx else 0
                ])
                
                self.sequence_history.append({
                    'step': step,
                    'variant_idx': i,
                    'sequence': seq,
                    'reward': reward,
                    'is_best': i == best_idx
                })
    
    def export_excel(self, filename: str = "training_results.xlsx"):
        """Export all data to Excel with multiple sheets."""
        try:
            import pandas as pd
            
            excel_path = self.output_dir / filename
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Metrics sheet
                if self.metrics_history:
                    df_metrics = pd.DataFrame(self.metrics_history)
                    df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Sequences sheet
                if self.sequence_history:
                    df_seqs = pd.DataFrame(self.sequence_history)
                    df_seqs.to_excel(writer, sheet_name='Sequences', index=False)
                
                # Read from CSV files if in-memory is empty
                if not self.metrics_history and self.csv_path.exists():
                    df_metrics = pd.read_csv(self.csv_path)
                    df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                
                if not self.sequence_history and self.sequences_csv.exists():
                    df_seqs = pd.read_csv(self.sequences_csv)
                    df_seqs.to_excel(writer, sheet_name='Sequences', index=False)
            
            logger.info(f"Excel exported: {excel_path}")
            return str(excel_path)
            
        except ImportError:
            logger.warning("pandas/openpyxl not available, skipping Excel export")
            return None
    
    def plot_training_curves(self, filename: str = "training_curves.png"):
        """Generate training curve plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Load data
            if self.metrics_history:
                df = pd.DataFrame(self.metrics_history)
            elif self.csv_path.exists():
                df = pd.read_csv(self.csv_path)
            else:
                logger.warning("No metrics data available for plotting")
                return None
            
            if len(df) < 2:
                logger.warning("Not enough data points for plotting")
                return None
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('HalluMPNN Training Progress', fontsize=14, fontweight='bold')
            
            # 1. Reward curves
            ax1 = axes[0, 0]
            ax1.plot(df['step'], df['best_reward'], 'b-', label='Best Reward', linewidth=2)
            ax1.plot(df['step'], df['mean_reward'], 'g--', label='Mean Reward', alpha=0.7)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Reward')
            ax1.set_title('Reward Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. iPTM/pTM curves
            ax2 = axes[0, 1]
            ax2.plot(df['step'], df['best_iptm'], 'r-', label='Best iPTM', linewidth=2)
            if 'best_ptm' in df.columns:
                ax2.plot(df['step'], df['best_ptm'], 'm--', label='Best pTM', alpha=0.7)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Score')
            ax2.set_title('AF3 Confidence Metrics')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # 3. Conformational RMSD
            ax3 = axes[1, 0]
            if 'best_conf_rmsd' in df.columns:
                ax3.plot(df['step'], df['best_conf_rmsd'], 'c-', label='Conf RMSD', linewidth=2)
                ax3.axhline(y=5.0, color='r', linestyle='--', label='Target (5Å)', alpha=0.5)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('RMSD (Å)')
            ax3.set_title('Venus Flytrap RMSD (bound vs unbound)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Loss/KL curves
            ax4 = axes[1, 1]
            ax4.plot(df['step'], df['loss'], 'k-', label='Loss', linewidth=2)
            ax4.plot(df['step'], df['kl_div'], 'orange', label='KL Divergence', linewidth=1.5)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Value')
            ax4.set_title('Training Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.output_dir / filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved: {plot_path}")
            return str(plot_path)
            
        except ImportError as e:
            logger.warning(f"matplotlib/pandas not available: {e}, skipping plot generation")
            return None
    
    def plot_realtime(self, step: int = None):
        """Generate real-time training progress chart (called every step).
        
        Matches NG-5-5 chart style exactly:
        - Simple 2x2 subplot layout
        - No legends (clean look)
        - English titles
        - Grid enabled
        - Single color per subplot
        
        Args:
            step: Current step number (for filename)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create figures subdirectory
            figures_dir = self.output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Load data
            if self.metrics_history:
                data = self.metrics_history
            else:
                return None
            
            if len(data) < 1:
                return None
            
            # Extract history lists
            losses = [m.get('loss', 0) for m in data]
            rewards = [m.get('mean_reward', 0) for m in data]
            kls = [m.get('kl_div', 0) for m in data]
            best_rewards = [m.get('best_reward', 0) for m in data]
            
            # Create figure (same size as NG-5-5)
            plt.figure(figsize=(15, 12))
            
            # Subplot 1: Total Loss Curve
            plt.subplot(2, 2, 1)
            plt.plot(losses)
            plt.title('Total Loss Curve')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Subplot 2: Average Reward Curve
            plt.subplot(2, 2, 2)
            plt.plot(rewards, color='g')
            plt.title('Average Reward Curve (AF3 Score)')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.grid(True)
            
            # Subplot 3: KL Divergence Curve  
            plt.subplot(2, 2, 3)
            plt.plot(kls, color='r')
            plt.title('KL Divergence Curve')
            plt.xlabel('Step')
            plt.ylabel('KL Divergence')
            plt.grid(True)
            
            # Subplot 4: Best Reward Curve
            plt.subplot(2, 2, 4)
            plt.plot(best_rewards, color='orange')
            plt.title('Best Reward Curve')
            plt.xlabel('Step')
            plt.ylabel('Best Reward')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save with step number in filename (like NG-5-5)
            current_step = step if step is not None else len(data)
            save_path = figures_dir / f"grpo_progress_step_{current_step}.png"
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Training chart saved: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
            return None
    
    def load_from_csv(self):
        """Load existing data from CSV files (for resume)."""
        if self.csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.csv_path)
                self.metrics_history = df.to_dict('records')
                logger.info(f"Loaded {len(self.metrics_history)} metrics from CSV")
            except Exception as e:
                logger.warning(f"Failed to load metrics CSV: {e}")
        
        if self.sequences_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.sequences_csv)
                self.sequence_history = df.to_dict('records')
                logger.info(f"Loaded {len(self.sequence_history)} sequences from CSV")
            except Exception as e:
                logger.warning(f"Failed to load sequences CSV: {e}")
    
    def finalize(self):
        """Generate final exports (call at end of training)."""
        self.export_excel()
        self.plot_training_curves()
        logger.info("Training logs finalized")
