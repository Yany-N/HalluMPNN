# ============================================================
# CRITICAL: JAX 0.9.0 compatibility patch
# This patch must run BEFORE any module imports jax/alphafold3
# ============================================================
import types
try:
    import jax
    if not hasattr(jax, 'util'):
        # JAX 0.9.0 removed jax.util; restore it for legacy code
        try:
            import jax._src.util as _src_util
            jax.util = _src_util
        except ImportError:
            # Create minimal stub with commonly needed functions
            jax.util = types.ModuleType('jax.util')
            if hasattr(jax, 'tree_util'):
                jax.util.safe_zip = lambda *args: list(zip(*args))
                jax.util.safe_map = lambda f, *args: list(map(f, *args))
except ImportError:
    pass
# ============================================================

import os
import sys
import copy
import json
import time
import random
import shutil
import csv
import logging
import argparse
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# ============================================
# Path setup
# ============================================
PROJECT_DIR = Path(__file__).parent.parent.resolve()
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add LigandMPNN to path
LIGANDMPNN_PATH = "/data/home/scvi041/run/LigandMPNN"
if os.path.exists(LIGANDMPNN_PATH) and LIGANDMPNN_PATH not in sys.path:
    sys.path.insert(0, LIGANDMPNN_PATH)

# ============================================
# Local imports
# ============================================
from af3_utils import (
    create_af3_input_json,
    generate_slurm_script,
    submit_af3_job,
    parse_af3_output,
    run_af3_prediction,
    run_af3_msa_only, # NEW: Import MSA generator
    LDOPA_SMILES,
)

from ligandmpnn_utils import (
    load_ligandmpnn_model,
    generate_sequences_with_ligandmpnn,
    get_per_token_log_probs,
    extract_designed_sequence,
    RESTYPE_INT_TO_STR,
)

from hallu_utils import (
    find_pocket_residues,
    hallu_design_phase,
    get_sequence_from_pdb,
)

from reward_utils import (
    calculate_rmsd,
    calculate_rmsd_reward,
    calculate_comprehensive_reward,
)

from training_logger import TrainingLogger
from chart_utils import plot_training_charts


# ============================================
# Logging
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HalluMPNN")

# ============================================
# Default Configuration
# ============================================
DEFAULT_CONFIG = {
    # Paths
    "ligandmpnn_weights": "/data/home/scvi041/run/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
    "af3_sif": "/data/home/scvi041/run/af3/3.0.1_run/alphafold3.sif",
    "af3_model_dir": "/data/home/scvi041/run/af3/model",
    "af3_db_dir": "/data/public/alphafold3/dataset",
    
    # Training
    "training_steps": 1000,
    "num_generations": 8,
    "learning_rate": 1e-5,
    "beta": 0.01,  # KL weight
    "temperature": 0.3,
    "chain_to_design": "A",
    
    # Reward weights (ligand_pocket preset with RMSD)
    "reward_weights": {
        "iptm": 0.35,
        "ptm": 0.25,
        "pae": 0.25,
        "rmsd": 0.15,       # Conformational stability reward
        "clash_penalty": 0.10,
        "specificity": 0.5, # Weight for specificity bonus
        "spec_gap": 0.1,    # Minimum gap for specificity bonus
    },
    
    # Decoys for Negative Design (Tyr, Trp)
    "decoy_smiles": [
        "N[C@@H](Cc1ccc(O)cc1)C(=O)O",      # Tyrosine
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"  # Tryptophan
    ],
    
    # RMSD configuration
    "rmsd_cutoff": 3.0,  # Angstroms - beyond AF3 resolution limit
    
    # HalluDesign trigger thresholds
    "hallu_trigger": {
        "iptm_min": 0.7,
        "ptm_min": 0.6,
        "pae_max": 10.0,
    },
    "hallu_cycles": 5,
    
    # Structural References
    "open_pdb": str(PROJECT_DIR / "inputs/3lft-open.pdb"),
    
    # Checkpoint
    "save_every": 10,
}

PAE_MAX = 31.75  # Maximum PAE for normalization


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================
# RMSD Calculation Functions
# ============================================

def extract_ca_coords_from_cif(cif_path: str, chain_id: str = "A") -> Optional[np.ndarray]:
    """
    Extract CA (alpha carbon) coordinates from a CIF file.
    
    Args:
        cif_path: Path to CIF file
        chain_id: Chain to extract (default "A")
    
    Returns:
        coords: (N, 3) array of CA coordinates or None if failed
    """
    try:
        coords = []
        with open(cif_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    parts = line.split()
                    if len(parts) >= 15:
                        # CIF format: try to find CA atoms for target chain
                        atom_name = parts[3] if len(parts) > 3 else ""
                        chain = parts[6] if len(parts) > 6 else ""
                        if atom_name == 'CA' and chain == chain_id:
                            try:
                                x = float(parts[10])
                                y = float(parts[11])
                                z = float(parts[12])
                                coords.append([x, y, z])
                            except (ValueError, IndexError):
                                continue
        
        if coords:
            return np.array(coords)
        return None
    except Exception as e:
        logger.warning(f"Failed to extract coords from CIF {cif_path}: {e}")
        return None


def extract_ca_coords_from_pdb(pdb_path: str, chain_id: str = "A") -> Optional[np.ndarray]:
    """
    Extract CA (alpha carbon) coordinates from a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain to extract (default "A")
    
    Returns:
        coords: (N, 3) array of CA coordinates or None if failed
    """
    try:
        coords = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    chain = line[21:22].strip()
                    if atom_name == 'CA' and (chain == chain_id or chain_id == ""):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            continue
        
        if coords:
            return np.array(coords)
        return None
    except Exception as e:
        logger.warning(f"Failed to extract coords from PDB {pdb_path}: {e}")
        return None


def calculate_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    align: bool = True
) -> float:
    """
    Calculate RMSD between two coordinate sets using Kabsch alignment.
    
    Args:
        coords1: Reference coordinate set (N, 3)
        coords2: Predicted coordinate set (N, 3)
        align: Whether to align structures first (Kabsch algorithm)
    
    Returns:
        rmsd: Root mean square deviation in Angstroms
    """
    if len(coords1) != len(coords2):
        # Use minimum length for comparison
        min_len = min(len(coords1), len(coords2))
        logger.warning(f"Coordinate length mismatch: {len(coords1)} vs {len(coords2)}, using first {min_len}")
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]
    
    if len(coords1) == 0:
        return float('inf')
    
    if align:
        # Kabsch alignment
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)
        
        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        c2_aligned = c2 @ R
        diff = c1 - c2_aligned
    else:
        diff = coords1 - coords2
    
    return np.sqrt((diff ** 2).sum() / len(coords1))


def calculate_rmsd_reward(
    rmsd: float,
    cutoff: float = 3.0
) -> float:
    """
    Calculate RMSD-based reward with cutoff.
    
    Beyond the cutoff (approximately AF3's resolution limit ~3Å),
    RMSD is considered unreliable for evaluating structural quality.
    
    Args:
        rmsd: Calculated RMSD in Angstroms
        cutoff: Cutoff threshold (default 3.0 Å)
    
    Returns:
        reward: [0, 1] reward value (1.0 = perfect match, 0.0 = beyond cutoff)
    """
    # Smooth exponential decay across the entire range
    # Avoids hard cutoff at 0.0 and discontinuity
    # Maps: 0A -> 1.0, 3.0A -> ~0.22, 6.0A -> ~0.05
    return float(np.exp(-0.5 * rmsd))


# ============================================
# Reward Calculation (adapted for AF3 with RMSD)
# ============================================

def calculate_af3_reward(
    af3_result: Dict[str, Any],
    reward_weights: Dict[str, float] = None,
    reference_pdb: str = None,
    chain_id: str = "A",
    rmsd_cutoff: float = 3.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate reward from AlphaFold3 prediction output with RMSD.
    
    Args:
        af3_result: Output from parse_af3_output()
        reward_weights: Weight dict for reward components
        reference_pdb: Path to reference PDB for RMSD calculation
        chain_id: Chain ID for coordinate extraction
        rmsd_cutoff: RMSD cutoff in Angstroms (default 3.0)
    
    Returns:
        total_reward: Combined reward score [0, 1]
        reward_info: Detailed breakdown
    """
    if reward_weights is None:
        reward_weights = DEFAULT_CONFIG["reward_weights"]
    
    if not af3_result.get('success', False):
        return 0.0, {'error': 'AF3 prediction failed', 'total_reward': 0.0}
    
    # Extract metrics
    iptm = af3_result.get('iptm', 0.0)
    ptm = af3_result.get('ptm', 0.0)
    pae = af3_result.get('pae', PAE_MAX)
    has_clash = af3_result.get('has_clash', True)
    ranking_score = af3_result.get('ranking_score', 0.0)
    
    # Calculate component rewards
    iptm_reward = iptm
    ptm_reward = ptm
    pae_reward = 1.0 - (pae / PAE_MAX)
    pae_reward = max(0.0, min(1.0, pae_reward))
    clash_penalty = -1.0 if has_clash else 0.0
    
    # ========================================
    # RMSD calculation (with cutoff)
    # ========================================
    rmsd = None
    rmsd_reward = 0.5  # Neutral default if RMSD cannot be calculated
    
    if reference_pdb and os.path.exists(reference_pdb):
        # Extract reference coordinates
        if reference_pdb.endswith('.cif'):
            ref_coords = extract_ca_coords_from_cif(reference_pdb, chain_id)
        else:
            ref_coords = extract_ca_coords_from_pdb(reference_pdb, chain_id)
        
        # Extract predicted coordinates
        pred_path = af3_result.get('cif_path') or af3_result.get('pdb_path')
        if pred_path and os.path.exists(pred_path):
            if pred_path.endswith('.cif'):
                pred_coords = extract_ca_coords_from_cif(pred_path, chain_id)
            else:
                pred_coords = extract_ca_coords_from_pdb(pred_path, chain_id)
        else:
            logger.warning(f"No predicted structure path for coordinates extraction")

        # Extract Open Reference Coords (if available) - for Compaction Calculation
        # NOTE: In run_hallumpnn.py, 'ref_coords' is technically the SCAFFOLD coords (used for RMSD)
        # We need the specific OPEN reference for compaction comparison.
        # We can extract it from the Open PDB if present.
        open_ref_coords = None
        # How to get open_pdb path here? It's not passed to calculate_af3_reward currently.
        # FIXME: We should rely on calculate_comprehensive_reward to do this if we pass the array?
        # But we need to load it. For efficiency, maybe load once outside?
        # For now, let's assume we can't easily get it here without passing it into calculate_af3_reward.
        # BUT: calculate_af3_reward is a helper. Let's update its signature to accept open_ref_coords.
        pass

    # ========================================
    # Combine rewards (Delegate to comprehensive reward)
    # ========================================
    # We use calculate_comprehensive_reward which now handles everything including Rg and RMSD
    from reward_utils import calculate_comprehensive_reward
    
    total_reward, reward_info = calculate_comprehensive_reward(
        summary_data=af3_result,
        rmsd=rmsd, # Still pass the basic scaffold RMSD if available
        open_rmsd=None, # Not calculating Open RMSD here explicitly yet
        unbound_ptm=None, # Not calculating unbound here
        decoy_stats=None,
        reward_weights=reward_weights,
        bound_coords=pred_coords,
        open_ref_coords=open_ref_coords # We need to pass this!
    )
    
    return total_reward, reward_info
    
    return total_reward, reward_info


def reshape_rewards(rewards: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Apply non-linear transformation to rewards to increase differentiation.
    
    From NG-5-5 reference implementation: sign(r) * |r|^alpha
    This amplifies differences in rewards, making gradient updates more effective.
    
    Args:
        rewards: Raw reward tensor
        alpha: Power exponent (default 0.7 from NG-5-5)
    
    Returns:
        Transformed rewards
    """
    return torch.sign(rewards) * torch.pow(torch.abs(rewards), alpha)


def compute_group_relative_advantages(
    rewards: torch.Tensor, 
    scale_rewards: bool = True,
    scale_factor: float = 10.0  # Increased from 5.0 for stronger gradient signal
) -> torch.Tensor:
    """
    Compute group-relative advantages for GRPO.
    
    Matches NG-5-5 reference implementation exactly.
    
    Args:
        rewards: Tensor of rewards [batch_size]
        scale_rewards: Whether to scale by std (True) or just center
        scale_factor: Fallback scale if std is too small
    
    Returns:
        advantages: Normalized advantages
    """
    if len(rewards) <= 1:
        return torch.zeros_like(rewards)
    
    # Apply non-linear reward shaping first (from NG-5-5)
    reshaped_rewards = reshape_rewards(rewards)
    mean_reward = reshaped_rewards.mean()
    
    if scale_rewards:
        std_reward = reshaped_rewards.std()
        if std_reward > 1e-8:
            advantages = (reshaped_rewards - mean_reward) / std_reward
        else:
            # Fallback: multiply by scale_factor when std is too small
            advantages = (reshaped_rewards - mean_reward) * scale_factor
    else:
        advantages = reshaped_rewards - mean_reward
    
    return advantages


def compute_grpo_loss(
    current_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss.
    
    Args:
        current_logps: Current policy log probs [batch, length]
        ref_logps: Reference policy log probs [batch, length]
        advantages: Per-sample advantages [batch]
        mask: Loss mask [batch, length]
        beta: KL divergence weight
    
    Returns:
        loss: Total loss
        policy_loss: Policy gradient loss
        mean_kl: Mean KL divergence
    """
    # KL divergence (unbiased estimate)
    per_token_kl = (
        torch.exp(ref_logps - current_logps)
        - (ref_logps - current_logps)
        - 1
    )
    
    # Policy gradient term
    policy_term = (
        torch.exp(current_logps - current_logps.detach())
        * advantages.unsqueeze(1)
    )
    
    # Combined loss
    per_token_loss = -(policy_term - beta * per_token_kl)
    
    # Apply mask and average
    masked_loss = per_token_loss * mask
    num_tokens = mask.sum(dim=1).clamp(min=1.0)
    loss_per_seq = masked_loss.sum(dim=1) / num_tokens
    loss = loss_per_seq.mean()
    
    # Compute monitoring metrics
    masked_kl = per_token_kl * mask
    mean_kl = (masked_kl.sum(dim=1) / num_tokens).mean()
    
    masked_policy = policy_term * mask
    policy_loss = -(masked_policy.sum(dim=1) / num_tokens).mean()
    
    return loss, policy_loss, mean_kl


# ============================================
# Utility Functions
# ============================================

def convert_cif_to_pdb(cif_path: str, pdb_path: str) -> bool:
    """Convert CIF to PDB format."""
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_path)
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)
        
        logger.info(f"Converted {cif_path} to {pdb_path}")
        return True
    except Exception as e:
        logger.error(f"CIF to PDB conversion failed: {e}")
        return False


# ============================================
# Main Trainer Class
# ============================================

class HalluMPNNTrainer:
    """Combined BetterMPNN + HalluDesign trainer using AlphaFold3."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup directories
        paths = config.get('paths', {})
        self.output_dir = Path(paths.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoints and figures inside output_dir (like NG-5-5)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.figures_dir = self.output_dir / "figures"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Get AF3 config
        self.af3_config = {
            'sif_path': paths.get('af3_sif', DEFAULT_CONFIG['af3_sif']),
            'model_dir': paths.get('af3_model_dir', DEFAULT_CONFIG['af3_model_dir']),
            'db_dir': paths.get('af3_db_dir', DEFAULT_CONFIG['af3_db_dir']),
        }
        
        # Training parameters
        grpo_config = config.get('grpo', {})
        self.training_steps = grpo_config.get('training_steps', DEFAULT_CONFIG['training_steps'])
        self.num_generations = grpo_config.get('num_generations', DEFAULT_CONFIG['num_generations'])
        self.learning_rate = grpo_config.get('learning_rate', DEFAULT_CONFIG['learning_rate'])
        self.beta = grpo_config.get('beta', DEFAULT_CONFIG['beta'])
        self.temperature = grpo_config.get('temperature', DEFAULT_CONFIG['temperature'])
        self.chain_to_design = config.get('chain_to_design', DEFAULT_CONFIG['chain_to_design'])
        
        
        # New: MSA Template Path
        self.template_json_path = self.config.get('template_json', PROJECT_DIR / "configs/template.json")
        self.msa_hydrated = False

        # Reward weights
        self.reward_weights = grpo_config.get('reward_weights', DEFAULT_CONFIG['reward_weights'])
        
        # Ligand
        ligand_config = config.get('ligand', {})
        self.ligand_smiles = ligand_config.get('smiles', LDOPA_SMILES)
        
        # HalluDesign settings
        # Support both keys for compatibility
        hallu_config = config.get('halludesign') or config.get('hallu_design', {})
        self.hallu_enabled = hallu_config.get('enabled', False)
        self.hallu_trigger = hallu_config.get('trigger_threshold', DEFAULT_CONFIG['hallu_trigger'])
        self.hallu_cycles = hallu_config.get('num_cycles', DEFAULT_CONFIG['hallu_cycles'])
        
        # Negative Design Decoys
        self.decoy_smiles = config.get('decoy_smiles', DEFAULT_CONFIG['decoy_smiles'])
        if not isinstance(self.decoy_smiles, list):
             self.decoy_smiles = [self.decoy_smiles] if self.decoy_smiles else []
        logger.info(f"Initialized with {len(self.decoy_smiles)} decoy ligands for negative design")
        
        # Residue-specific design (NEW)
        # If specified, only these residues will be redesigned; others remain fixed
        # Format: ["A15", "A16", "A90", ...] (chain + residue number)
        design_config = config.get('design', {})
        self.redesign_residues = design_config.get('redesign_residues', None)
        if self.redesign_residues:
            logger.info(f"Restricting design to {len(self.redesign_residues)} specific residues: {self.redesign_residues}")
        else:
            logger.info("Designing entire chain (no residue restriction)")
        
        # Open State Reference
        self.open_pdb = None
        open_pdb_path = config.get('grpo', {}).get('open_rmsd', {}).get('path')
        if open_pdb_path and os.path.exists(open_pdb_path):
            self.open_pdb = Path(open_pdb_path)
            logger.info(f"Using Open State Reference: {self.open_pdb}")
            
        # Closed State Reference (NEW)
        self.closed_pdb = None
        closed_pdb_path = config.get('grpo', {}).get('references', {}).get('closed_pdb')
        if closed_pdb_path and os.path.exists(closed_pdb_path):
            self.closed_pdb = Path(closed_pdb_path)
            logger.info(f"Using Closed State Reference: {self.closed_pdb}")
        else:
            # Fallback to scaffold if not specified
            logger.info("No separate Closed Reference provided, will use Scaffold as Closed Reference")
        
        # Training state
        self.step = 0
        self.best_reward = 0.0
        self.best_sequence = ""
        # IMPORTANT: scaffold_pdb is CONSTANT - used for LigandMPNN input every step
        self.scaffold_pdb = None  # Set in train() from scaffold_pdb argument
        # current_pdb tracks best structure for HalluDesign reference (not for LigandMPNN)
        self.current_pdb = None
        self.reward_history = []
        self.metrics_history = []
        
        # HalluDesign periodic trigger tracking
        self.last_hallu_step = -float('inf')  # Last step HalluDesign was run
        self.force_hallu_done = False         # Force trigger at step N done
        self.manual_hallu_done = False        # Manual CLI trigger done
        self.threshold_reached = False        # Metric threshold reached
        self.threshold_reached_step = -1      # Step when threshold was reached
        self.manual_hallu_trigger = False     # Set from CLI --hallu_trigger
        
        # Initialize Logger
        self.training_logger = TrainingLogger(str(self.output_dir))
        
        # Load LigandMPNN model
        self._load_ligandmpnn_model()
        
        # Check/Hydrate MSA Template - DEFERRED to train()
        # self._check_and_hydrate_template()
        
        logger.info(f"HalluMPNN Trainer initialized")
    
    def _load_ligandmpnn_model(self):
        """Load LigandMPNN model for sequence generation."""
        checkpoint_path = self.config.get('paths', {}).get(
            'ligandmpnn_weights',
            DEFAULT_CONFIG['ligandmpnn_weights']
        )
        
        logger.info(f"Loading LigandMPNN from: {checkpoint_path}")
        self.model, self.checkpoint_info = load_ligandmpnn_model(
            checkpoint_path,
            model_type='ligand_mpnn',
            device=self.device
        )
        self.model.train()

    def _check_and_hydrate_template(self, scaffold_pdb_path: str):
        """Check if template.json needs MSA hydration."""
        template_path = Path(self.template_json_path)
        if not template_path.exists():
            logger.warning(f"Template JSON not found at {template_path}, skipping MSA features.")
            return

        try:
            with open(template_path, 'r') as f:
                data = json.load(f)
            
            # Check if unpairedMsa is empty for protein chain A (or first protein)
            needs_hydration = False
            for seq in data.get('sequences', []):
                if 'protein' in seq:
                    prot = seq['protein']
                    msa = prot.get('unpairedMsa', '')
                    # User Request: Only hydrate if strictly empty
                    if not msa:
                        needs_hydration = True
                        break
            
            if needs_hydration:
                logger.info(f"MSA Template needs hydration! Generating MSA for scaffold: {scaffold_pdb_path}")
                # Use scaffold sequence
                scaffold_seq = self.load_scaffold(scaffold_pdb_path)
                if not scaffold_seq:
                    logger.error("Could not load scaffold sequence for MSA generation!")
                    return
                
                success = run_af3_msa_only(
                    sequence=scaffold_seq,
                    output_json_path=str(template_path),
                    af3_config=self.af3_config
                )
                
                if success:
                    logger.info("MSA Template successfully hydrated!")
                    self.msa_hydrated = True
                else:
                    logger.error("Failed to hydrate MSA template.")
            else:
                logger.info("MSA Template already populated.")
                self.msa_hydrated = True
                
        except Exception as e:
            logger.error(f"Error checking template JSON: {e}")
        
        # Create reference model (frozen)
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info("LigandMPNN model loaded successfully")
    
    def load_scaffold(self, pdb_path: str) -> str:
        """Load scaffold sequence from config or PDB.
        
        Priority:
        1. Config 'scaffold.sequence' (explicit)
        2. Extract from PDB file
        3. Raise error (no silent fallback)
        """
        # First try config
        scaffold_config = self.config.get('scaffold', {})
        if 'sequence' in scaffold_config:
            seq = scaffold_config['sequence']
            logger.info(f"Using scaffold sequence from config: {len(seq)} residues")
            return seq
        
        # Try to extract from PDB
        if pdb_path and os.path.exists(pdb_path):
            try:
                seq = get_sequence_from_pdb(pdb_path)
                if seq:
                    logger.info(f"Extracted sequence from PDB: {len(seq)} residues")
                    return seq
            except Exception as e:
                logger.error(f"Failed to extract sequence from PDB: {e}")
        
        # No fallback - require explicit sequence
        raise ValueError(
            f"Could not load scaffold sequence. Please either:\n"
            f"  1. Set 'scaffold.sequence' in config YAML, or\n"
            f"  2. Provide a valid PDB file with extractable sequence\n"
            f"  Current PDB path: {pdb_path}"
        )
    
    def grpo_step(
        self,
        current_sequence: str,
        step: int,
        current_pdb_path: Optional[Path] = None,
        precomputed_features: Optional[Dict[str, Any]] = None, # NEW: support pre-loaded features
    ) -> Dict[str, Any]:
        """
        Execute one GRPO training step.
        
        Args:
            current_sequence: Current best sequence
            step: Step number
            current_pdb_path: Path to current best PDB structure (refined by HalluDesign)
            precomputed_features: Pre-calculated LigandMPNN features (for multi-backbone mode)
        
        Returns:
            step_results: Results including reward and metrics
        """
        step_dir = self.output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # ====================
        # 1. Generate sequence variants with LigandMPNN
        # ====================
        # Priority:
        # 1. Precomputed features (Multi-Backbone Mode) -> Passed directly
        # 2. current_pdb_path (HalluDesign refined) -> Parse this
        # 3. self.scaffold_pdb (Standard Mode) -> Parse this
        
        # Determine input source
        input_pdb = None
        if not precomputed_features:
            input_pdb = str(current_pdb_path) if current_pdb_path and current_pdb_path.exists() else str(self.scaffold_pdb)
        
        if precomputed_features or (input_pdb and os.path.exists(input_pdb)):
            logger.info(f"Generating {self.num_generations} variants with LigandMPNN (Source: {'Precomputed' if precomputed_features else input_pdb})...")
            try:
                result = generate_sequences_with_ligandmpnn(
                    model=self.model,
                    pdb_path=input_pdb if input_pdb else "dummy", # pdb_path ignored if features provided
                    chain_to_design=self.chain_to_design,
                    num_variants=self.num_generations,
                    temperature=self.temperature,
                    redesigned_residues=self.redesign_residues,  # NEW: restrict to specific pocket residues
                    use_ligand_context=True,
                    device=self.device,
                    feature_dict=precomputed_features # Pass precomputed features
                )
                variants = result["sequences"]
                feature_dict = result["feature_dict"]
                S_sample = result["S_sample"]
                output_dict = result["output_dict"]
                chain_mask = result["chain_mask"]
                
                # Extract designed chain sequences
                chain_letters = list(result.get("chain_letters", [self.chain_to_design] * len(variants[0])))
                designed_sequences = []
                for var_seq in variants:
                    des_seq = extract_designed_sequence(
                        var_seq, chain_mask, chain_letters, self.chain_to_design
                    )
                    designed_sequences.append(des_seq if des_seq else var_seq)
                
            except Exception as e:
                logger.warning(f"LigandMPNN generation failed: {e}")
                logger.info("Falling back to random mutations")
                designed_sequences = self._random_mutations(current_sequence, self.num_generations)
                result = None
        else:
            logger.info("No PDB available, using random mutations")
            designed_sequences = self._random_mutations(current_sequence, self.num_generations)
            result = None
        
        logger.info(f"Generated {len(designed_sequences)} sequence variants")
        
        # ====================
        # 2. Evaluate each variant with AF3
        # ====================
        rewards_list = []
        all_reward_info = []
        best_structures = []
        
        for i, seq in enumerate(designed_sequences):
            variant_dir = step_dir / f"variant_{i}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Step {step}, Variant {i}: Running dual AF3 (with/without ligand)")
            
            # ====================
            # Venus Flytrap: Dual AF3 Prediction + Negative Design
            # ====================
            
            # 1. AF3 WITH ligand (Bound State)
            with_ligand_dir = variant_dir / "with_ligand"
            with_ligand_dir.mkdir(parents=True, exist_ok=True)
            
            af3_with_ligand = run_af3_prediction(
                sequence=seq,
                output_dir=str(with_ligand_dir),
                ligand_smiles=self.ligand_smiles,
                name=f"step{step}_var{i}_bound",
                wait=True,
                af3_config=self.af3_config,
                template_json_path=str(self.template_json_path) if self.msa_hydrated else None # Use Template for Bound
            )
            
            # 2. AF3 WITHOUT ligand (Unbound/Open State)
            without_ligand_dir = variant_dir / "without_ligand"
            without_ligand_dir.mkdir(parents=True, exist_ok=True)
            
            af3_without_ligand = run_af3_prediction(
                sequence=seq,
                output_dir=str(without_ligand_dir),
                ligand_smiles=None,
                name=f"step{step}_var{i}_unbound",
                wait=True,
                af3_config=self.af3_config
                # NO TEMPLATE for Unbound (Single Sequence Mode)
            )
            
            # ====================
            # Extract coordinates FIRST (needed for decoy conf_rmsd)
            # ====================
            bound_path = af3_with_ligand.get('cif_path') or af3_with_ligand.get('pdb_path')
            unbound_path = af3_without_ligand.get('cif_path') or af3_without_ligand.get('pdb_path')
            
            bound_coords = None
            unbound_coords = None
            
            if bound_path and os.path.exists(bound_path):
                bound_coords = extract_ca_coords_from_cif(bound_path, self.chain_to_design) if bound_path.endswith('.cif') else extract_ca_coords_from_pdb(bound_path, self.chain_to_design)
            
            if unbound_path and os.path.exists(unbound_path):
                unbound_coords = extract_ca_coords_from_cif(unbound_path, self.chain_to_design) if unbound_path.endswith('.cif') else extract_ca_coords_from_pdb(unbound_path, self.chain_to_design)

            # 3. AF3 WITH Decoys (Negative Design + Conformational Specificity)
            # Goal: Decoys should have LOW iPTM (weak binding) and LOW conf_rmsd (no closing)
            decoy_iptms = []
            decoy_conf_rmsds = []  # Track if decoys cause closing (vs Unbound)
            decoy_closed_rmsds = [] # Track if decoys match Closed Ref
            decoy_contrast_rmsds = [] # NEW: Track if decoy matches Target Bound (Contrastive)
            
            # Pre-load Closed Ref coords for efficiency
            closed_ref_coords = None
            if self.closed_pdb and os.path.exists(str(self.closed_pdb)):
                 ref_p = str(self.closed_pdb)
                 closed_ref_coords = extract_ca_coords_from_cif(ref_p, self.chain_to_design) if ref_p.endswith('.cif') else extract_ca_coords_from_pdb(ref_p, self.chain_to_design)
            elif self.scaffold_pdb:
                 ref_p = str(self.scaffold_pdb)
                 closed_ref_coords = extract_ca_coords_from_cif(ref_p, self.chain_to_design) if ref_p.endswith('.cif') else extract_ca_coords_from_pdb(ref_p, self.chain_to_design)
            
            for d_idx, d_smiles in enumerate(self.decoy_smiles):
                decoy_dir = variant_dir / f"decoy_{d_idx}"
                decoy_dir.mkdir(parents=True, exist_ok=True)
                
                # AF3 with decoy ligand
                af3_decoy = run_af3_prediction(
                    sequence=seq,
                    output_dir=str(decoy_dir),
                    ligand_smiles=d_smiles,
                    name=f"step{step}_var{i}_decoy{d_idx}",
                    wait=True,
                    af3_config=self.af3_config,
                    template_json_path=str(self.template_json_path) if self.msa_hydrated else None # Use Template for Decoys too
                )
                
                if af3_decoy.get('success', False):
                    decoy_iptms.append(af3_decoy.get('iptm', 0.0))
                    
                    # Compute conformational metrics
                    decoy_path = af3_decoy.get('cif_path') or af3_decoy.get('pdb_path')
                    if decoy_path and os.path.exists(decoy_path):
                        decoy_coords = extract_ca_coords_from_cif(decoy_path, self.chain_to_design) if decoy_path.endswith('.cif') else extract_ca_coords_from_pdb(decoy_path, self.chain_to_design)
                        
                        if decoy_coords is not None:
                            # 1. Conf RMSD (vs Unbound) [Should be LOW]
                            if unbound_coords is not None:
                                d_conf = calculate_rmsd(decoy_coords, unbound_coords, align=True)
                                decoy_conf_rmsds.append(d_conf)
                            else:
                                decoy_conf_rmsds.append(0.0)
                                
                            # 2. Closed Ref RMSD (vs Scaffold) [Should be HIGH]
                            if closed_ref_coords is not None:
                                d_closed = calculate_rmsd(decoy_coords, closed_ref_coords, align=True)
                                decoy_closed_rmsds.append(d_closed)
                            else:
                                decoy_closed_rmsds.append(99.9)
                                
                            # 3. Contrast RMSD (vs Target Bound) [Should be HIGH]
                            if bound_coords is not None:
                                d_contrast = calculate_rmsd(decoy_coords, bound_coords, align=True)
                                decoy_contrast_rmsds.append(d_contrast)
                                logger.info(f"    Decoy {d_idx}: Conf={decoy_conf_rmsds[-1]:.2f}Å, Contrast={d_contrast:.2f}Å")
                            else:
                                decoy_contrast_rmsds.append(99.9)
                        else:
                            decoy_conf_rmsds.append(0.0)
                            decoy_closed_rmsds.append(99.9)
                            decoy_contrast_rmsds.append(99.9)
                    else:
                        decoy_conf_rmsds.append(0.0)
                        decoy_closed_rmsds.append(99.9)
                        decoy_contrast_rmsds.append(99.9)
                else:
                    decoy_iptms.append(0.0)
                    decoy_conf_rmsds.append(0.0)
                    decoy_closed_rmsds.append(99.9)
                    decoy_contrast_rmsds.append(99.9)

            # ====================
            # NEW: Contrastive RMSD (L-DOPA Bound vs Decoy Bound)
            # ====================
            # This block is now handled within the decoy loop above.
            # decoy_contrast_rmsds = []
            # if bound_coords is not None:
            #     # We need to re-iterate or store decoy coords. 
            #     # Since we didn't store decoy coords in the loop above (only RMSDs), 
            #     # we technically should have. But for minimal invasion, let's re-extract or move extraction.
            #     # BETTER: Modify the loop above to store contrast RMSD on the fly.
            #     pass 
                
            # Actually, let's rewrite the loop above to include contrast calc

            
            decoy_stats = {
                'iptms': decoy_iptms,
                'conf_rmsds': decoy_conf_rmsds,
                'closed_rmsds': decoy_closed_rmsds, # NEW
            }
            
            # ====================
            # Calculate RMSDs
            # ====================
            
            # A. Conformational RMSD (Bound vs Unbound)
            conformational_rmsd = None
            if bound_coords is not None and unbound_coords is not None:
                conformational_rmsd = calculate_rmsd(bound_coords, unbound_coords, align=True)
                logger.info(f"  Conformational RMSD (bound vs unbound): {conformational_rmsd:.2f}Å")
                
            # B. Reference RMSD (Bound vs Closed Ref) - Logic 1: L-DOPA should match Closed Ref
            ref_rmsd = None
            # Prioritize self.closed_pdb
            ref_source = self.closed_pdb if self.closed_pdb else self.scaffold_pdb
            
            if ref_source and os.path.exists(str(ref_source)) and bound_coords is not None:
                # Use scaffold/closed coords
                ref_pdb_path = str(ref_source)
                ref_coords = extract_ca_coords_from_cif(ref_pdb_path, self.chain_to_design) if ref_pdb_path.endswith('.cif') else extract_ca_coords_from_pdb(ref_pdb_path, self.chain_to_design)
                
                if ref_coords is not None:
                    ref_rmsd = calculate_rmsd(ref_coords, bound_coords, align=True)
                    logger.info(f"  Reference RMSD (bound vs scaffold): {ref_rmsd:.2f}Å")
            
            # C. Open State RMSD (Unbound vs Open Reference)
            open_rmsd = None
            open_coords = None  # CRITICAL: Initialize to avoid UnboundLocalError
            if self.open_pdb and os.path.exists(str(self.open_pdb)) and unbound_coords is not None:
                open_pdb_path = str(self.open_pdb)
                open_coords = extract_ca_coords_from_cif(open_pdb_path, self.chain_to_design) if open_pdb_path.endswith('.cif') else extract_ca_coords_from_pdb(open_pdb_path, self.chain_to_design)
                
                if open_coords is not None:
                    open_rmsd = calculate_rmsd(open_coords, unbound_coords, align=True)
                    logger.info(f"  Open State RMSD (unbound vs open ref): {open_rmsd:.2f}Å")

            # ====================
            # Calculate Comprehensive Reward (NEW: Curriculum-based)
            # ====================
            unbound_ptm = af3_without_ligand.get('ptm', 0.0)
            
            # Build curriculum config from YAML
            curriculum_config = {
                'early_steps': self.config.get('grpo', {}).get('curriculum', {}).get('early_steps', 40),
                'late_steps': self.config.get('grpo', {}).get('curriculum', {}).get('late_steps', 80),
                'early_threshold': self.config.get('grpo', {}).get('curriculum', {}).get('early_threshold', 1.0),
                'late_threshold': self.config.get('grpo', {}).get('curriculum', {}).get('late_threshold', 2.5),
                'target_conf_rmsd': self.config.get('grpo', {}).get('target_conformational_rmsd', 5.0),
                'sbp_max_deviation': self.config.get('grpo', {}).get('sbp_max_deviation', 8.0),
                'decoy_max_conf_rmsd': self.config.get('decoy_max_conf_rmsd', 2.0),  # NEW: max rmsd for decoys
                'quality_weight': 0.50,
                'conf_weight': 0.50,
            }
            
            reward, reward_info = calculate_comprehensive_reward(
                summary_data=af3_with_ligand,
                rmsd=ref_rmsd,
                conf_rmsd=conformational_rmsd,
                open_rmsd=open_rmsd,
                unbound_ptm=unbound_ptm,
                decoy_stats=decoy_stats,
                reward_weights=self.reward_weights,
                step=step,
                curriculum_config=curriculum_config,
                bound_coords=bound_coords,   # NEW: Pass bound coords for Rg calc
                open_ref_coords=open_coords,  # NEW: Pass open ref coords for Rg calc
                unbound_coords=unbound_coords # NEW: Pass unbound coords for Hyper-Open check
            )
            
            logger.info(f"  Variant {i}: Total Reward={reward:.4f}")
            logger.info(f"    Breakdown: Quality={reward_info.get('quality_reward',0):.3f}, ConfReward={reward_info.get('conf_reward',0):.3f}, SBP={reward_info.get('sbp_factor',1.0):.2f}")
            
            rewards_list.append(reward)
            all_reward_info.append(reward_info)
            
            # Store structure path (use bound structure)
            cif_path = af3_with_ligand.get('cif_path')
            if cif_path:
                pdb_path = cif_path.replace('.cif', '.pdb')
                if not os.path.exists(pdb_path):
                    convert_cif_to_pdb(cif_path, pdb_path)
                best_structures.append(pdb_path if os.path.exists(pdb_path) else cif_path)
            else:
                best_structures.append(str(self.scaffold_pdb) if self.scaffold_pdb else None)
            
            # Track best
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_sequence = seq
                logger.info(f"  New best! reward={reward:.4f}")
        
        rewards = torch.tensor(rewards_list, device=self.device)
        mean_reward = rewards.mean().item()
        
        # ====================
        # 3. Track best structure (for HalluDesign reference only, NOT for LigandMPNN input)
        # ====================
        # NOTE: BetterMPNN uses SAME scaffold PDB every step
        # Only the model PARAMETERS change via GRPO, not the input structure
        best_idx = rewards.argmax().item()
        # Store best structure path for potential HalluDesign use, but don't change scaffold
        self.best_structure_path = best_structures[best_idx] if best_structures[best_idx] else None
        
        # ====================
        # 4. GRPO update (if LigandMPNN was used)
        # ====================
        # CRITICAL: Clear JAX/AF3 GPU memory before PyTorch GRPO step
        try:
            import gc
            import jax
            # Clear JAX compilation cache and release memory
            jax.clear_caches()
            gc.collect()
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")
        
        if result is not None:
            # Compute advantages
            advantages = compute_group_relative_advantages(rewards)
            
            # Get log probabilities
            with torch.set_grad_enabled(True):
                current_logps, mask = get_per_token_log_probs(
                    self.model, feature_dict, S_sample, output_dict
                )
            
            with torch.no_grad():
                ref_logps, _ = get_per_token_log_probs(
                    self.ref_model, feature_dict, S_sample, output_dict
                )
            
            # Compute GRPO loss
            loss, policy_loss, kl_div = compute_grpo_loss(
                current_logps, ref_logps, advantages, mask, beta=self.beta
            )
            
            # Backward and update
            self.optimizer.zero_grad()
            loss.backward()
            
            # Verify gradients are flowing (critical for debugging)
            total_grad_norm = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_grad_norm < 1e-10:
                logger.warning(f"Step {step}: ZERO GRADIENTS! Check gradient propagation.")
            else:
                logger.info(f"Step {step}: gradient_norm={total_grad_norm:.6f}")
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            logger.info(f"Step {step}: loss={loss.item():.4f}, policy_loss={policy_loss.item():.6f}, kl={kl_div.item():.6f}")
        else:
            loss = torch.tensor(0.0)
            kl_div = torch.tensor(0.0)
        
        logger.info(f"Step {step}: best_reward={self.best_reward:.4f}, mean={mean_reward:.4f}")
        
        # Store metrics
        step_metrics = {
            'step': step,
            'best_reward': self.best_reward,
            'mean_reward': mean_reward,
            'best_sequence': self.best_sequence,
            'best_metrics': all_reward_info[best_idx] if all_reward_info else {},
        }
        self.metrics_history.append(step_metrics)
        self.reward_history.append(mean_reward)
        
        # Log to CSV/Excel
        self.training_logger.log_step(
            step=step,
            metrics=step_metrics,
            loss=loss.item() if hasattr(loss, 'item') else 0.0,
            kl_div=kl_div.item() if hasattr(kl_div, 'item') else 0.0,
            gradient_norm=total_grad_norm if 'total_grad_norm' in locals() else 0.0
        )
        
        # Log sequences
        self.training_logger.log_sequences(
            step=step,
            sequences=designed_sequences,
            rewards=rewards_list,
            reward_infos=all_reward_info,
            best_idx=best_idx
        )
        
        # Generate real-time chart (updated every step for live monitoring)
        self.training_logger.plot_realtime(step=step)
        
        return step_metrics
    
    def _random_mutations(self, sequence: str, num_variants: int, mutation_rate: float = 0.05) -> List[str]:
        """Generate variants with random mutations."""
        AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
        variants = []
        for _ in range(num_variants):
            mutated = list(sequence)
            num_mutations = max(1, int(len(sequence) * mutation_rate * random.random()))
            positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
            for pos in positions:
                mutated[pos] = random.choice([aa for aa in AMINO_ACIDS if aa != mutated[pos]])
            variants.append(''.join(mutated))
        return variants
    
    def check_hallu_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Check if HalluDesign should be triggered.
        
        New logic:
        1. Force trigger at `force_trigger_step` (default 10)
        2. After threshold met, trigger every `trigger_every_n_steps` (default 20)
        3. Manual trigger from CLI still supported
        
        Returns:
            True if HalluDesign should be triggered this step
        """
        step = metrics.get('step', -1)
        
        # Get config values
        force_trigger_step = self.hallu_trigger.get('force_trigger_step', 10)
        trigger_every_n_steps = self.hallu_trigger.get('trigger_every_n_steps', 20)
        
        # 1. Force trigger at specific step (once only)
        if step == force_trigger_step and not self.force_hallu_done:
            logger.info(f"HalluDesign force triggered at step {step}")
            self.force_hallu_done = True
            return True
        
        # 2. Manual trigger from CLI (once only)
        if self.manual_hallu_trigger and step == 0 and not self.manual_hallu_done:
            logger.info(f"HalluDesign manually triggered at step 0")
            self.manual_hallu_done = True
            return True
        
        # 3. Periodic trigger after threshold met
        best_metrics = metrics.get('best_metrics', {})
        iptm = best_metrics.get('iptm', 0)
        ptm = best_metrics.get('ptm', 0)
        pae = best_metrics.get('mean_pae', PAE_MAX)
        
        # Check if threshold is met
        threshold_met = (
            iptm > 0 and
            iptm >= self.hallu_trigger.get('iptm_min', 0.7) and
            ptm >= self.hallu_trigger.get('ptm_min', 0.6) and
            pae <= self.hallu_trigger.get('pae_max', 10.0)
        )
        
        if threshold_met:
            # Mark that threshold has been reached
            if not self.threshold_reached:
                self.threshold_reached = True
                self.threshold_reached_step = step
                logger.info(f"HalluDesign threshold reached at step {step}: iptm={iptm:.3f}, ptm={ptm:.3f}, pae={pae:.2f}")
        
        # Check if we should trigger periodically
        # Critical: Only trigger periodically AFTER the force trigger step
        # This prevents step 0 triggering if the scaffold already meets criteria
        if self.threshold_reached and step > force_trigger_step:
            steps_since_last = step - self.last_hallu_step
            if steps_since_last >= trigger_every_n_steps:
                logger.info(f"HalluDesign periodic trigger at step {step} ({steps_since_last} steps since last)")
                return True
        
        return False
    
    def hallu_design_step(self, current_pdb: str, current_sequence: str) -> Tuple[Dict, str]:
        """Execute HalluDesign pocket refinement.
        
        STRICT: Fails if HalluDesign execution encounters errors.
        """
        logger.info("=" * 60)
        logger.info("Starting HalluDesign pocket refinement")
        logger.info("=" * 60)
        
        hallu_dir = self.output_dir / f"hallu_design_step_{self.step}"
        hallu_dir.mkdir(parents=True, exist_ok=True)
        
        # Get HalluDesign configs
        hallu_config = self.config.get('halludesign', {})
        use_protenix = hallu_config.get('use_protenix', True)
        ref_time_steps = hallu_config.get('ref_time_steps', 50)
        
        # Setup Protenix config
        protenix_config = hallu_config.get('protenix_config', {})
        if 'static_configs' not in protenix_config:
             protenix_config['static_configs'] = {
                'use_deepspeed_evo_attention': False,
                # Default model dir if not provided
                'model_dir': hallu_config.get('protenix_model_dir', '/data/home/scvi041/run/HalluDesign-main/model')
            }

        # Use the complete hallu_design_phase function
        # Exceptions will propagate up to abort the task as requested
        best_result, best_pdb = hallu_design_phase(
            input_pdb=current_pdb,
            output_dir=str(hallu_dir),
            ligand_smiles=self.ligand_smiles,
            num_cycles=self.hallu_cycles,
            num_sequences_per_cycle=hallu_config.get('num_sequences_per_cycle', 4),
            ref_time_steps=ref_time_steps, 
            af3_config=self.af3_config,
            protenix_config=protenix_config,
            use_protenix=use_protenix,
            checkpoint_path=self.config.get('paths', {}).get('ligandmpnn_weights', DEFAULT_CONFIG['ligandmpnn_weights']),
            closed_pdb=str(self.closed_pdb) if self.closed_pdb else None,
            open_pdb=str(self.open_pdb) if self.open_pdb else None,
            template_json_path=str(self.template_json_path) if self.msa_hydrated else None # Pass MSA Template
        )
        
        # Track last HalluDesign step for periodic triggering
        self.last_hallu_step = self.step
        
        best_score = best_result.get('reward', 0) if best_result else 0
        logger.info(f"HalluDesign complete. Best reward: {best_score:.4f}")
        
        # Update best if HalluDesign found better
        if best_score > self.best_reward:
            self.best_reward = best_score
            self.best_sequence = get_sequence_from_pdb(best_pdb)
            logger.info(f"HalluDesign improved reward: {best_score:.4f}")
        
        return best_result or {}, best_pdb or current_pdb
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint."""
        state = {
            'step': step,
            'best_reward': self.best_reward,
            'best_sequence': self.best_sequence,
            'current_pdb': str(self.current_pdb) if self.current_pdb else None,
            'reward_history': self.reward_history,
            'hallu_trigger': self.hallu_trigger,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(state, checkpoint_path)
        
        # Also save simple state.json for resume
        state_json = {
            'step': step,
            'best_reward': self.best_reward,
            'best_sequence': self.best_sequence,
            'current_pdb': str(self.current_pdb) if self.current_pdb else None,
        }
        with open(self.checkpoint_dir / "state.json", 'w') as f:
            json.dump(state_json, f, indent=2)
        
        logger.info(f"Checkpoint saved: step {step}")
    
    def load_checkpoint(self, checkpoint_path: str = None, resume_mode: str = 'continue'):
        """
        Load from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file (None = use state.json)
            resume_mode: 'continue' = keep step/history, 'fresh' = reset step to 0
        """
        if checkpoint_path is None:
            state_path = self.checkpoint_dir / "state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                
                if resume_mode == 'continue':
                    self.step = state.get('step', 0)
                    self.best_reward = state.get('best_reward', 0)
                    # Try to load history from CSV if available
                    self.training_logger.load_from_csv()
                else:  # fresh mode
                    self.step = 0
                    self.best_reward = 0
                    logger.info("  Fresh mode: resetting step to 0, not loading history")
                
                self.best_sequence = state.get('best_sequence', '')
                current_pdb_str = state.get('current_pdb')
                if current_pdb_str:
                    self.current_pdb = Path(current_pdb_str)
                
                logger.info(f"Resumed from state.json: step {self.step} (mode={resume_mode})")
                return True
        else:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't support weights_only
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if resume_mode == 'continue':
                self.step = checkpoint.get('step', 0)
                self.best_reward = checkpoint.get('best_reward', 0)
                
                # Load reward history from checkpoint (authoritative source for GRPO)
                if 'reward_history' in checkpoint:
                    self.reward_history = checkpoint['reward_history']
                    logger.info(f"  Loaded {len(self.reward_history)} reward history entries from checkpoint")
                
                # Load CSV history for plotting, but TRUNCATE to match checkpoint step
                # This ensures consistency: CSV may have extra steps beyond checkpoint
                self.training_logger.load_from_csv()
                
                # Truncate metrics_history to checkpoint step
                if self.training_logger.metrics_history:
                    original_len = len(self.training_logger.metrics_history)
                    # Keep only entries where step <= checkpoint step
                    self.training_logger.metrics_history = [
                        m for m in self.training_logger.metrics_history 
                        if m.get('step', 0) <= self.step
                    ]
                    truncated_len = len(self.training_logger.metrics_history)
                    if original_len != truncated_len:
                        logger.info(f"  Truncated metrics history: {original_len} -> {truncated_len} (to match checkpoint step {self.step})")
                
                # Truncate sequence_history similarly
                if self.training_logger.sequence_history:
                    self.training_logger.sequence_history = [
                        s for s in self.training_logger.sequence_history 
                        if s.get('step', 0) <= self.step
                    ]
            else:  # fresh mode
                self.step = 0
                self.best_reward = 0
                logger.info("  Fresh mode: resetting step to 0, not loading history")
            
            self.best_sequence = checkpoint.get('best_sequence', '')
            current_pdb_str = checkpoint.get('current_pdb')
            if current_pdb_str:
                self.current_pdb = Path(current_pdb_str)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            # Note: optimizer_state_dict ignored - no optimizer used in this training loop
            
            logger.info(f"Loaded checkpoint: step {self.step} (mode={resume_mode})")
            return True
        
        return False
    
    def cleanup_step_files(self, step_dir: Path):
        """
        Clean up intermediate files to save space.
        
        Policy:
        1. Move *_model.cif to output_dir/all_cif/
        2. Move *_summary_confidences.json to output_dir/all_json/
        3. DELETE everything else in step_dir (including subdirs)
        """
        if not step_dir.exists():
            return
            
        logger.info(f"Processing results and cleaning up: {step_dir}")
        
        # Create destination directories
        all_cif_dir = self.output_dir / "all_cif"
        all_json_dir = self.output_dir / "all_json"
        all_cif_dir.mkdir(exist_ok=True)
        all_json_dir.mkdir(exist_ok=True)
        
        cif_count = 0
        json_count = 0
        deleted_count = 0
        
        # Walk through all files
        # Use topdown=False to remove files before directories
        for root, dirs, files in os.walk(step_dir, topdown=False):
            for file in files:
                file_path = Path(root) / file
                
                # 1. CIF Models -> all_cif
                if file.endswith('_model.cif'):
                    dest = all_cif_dir / file
                    if not dest.exists():
                        try:
                            shutil.copy2(file_path, dest)
                            cif_count += 1
                        except OSError as e:
                            logger.warning(f"Failed to copy CIF {file}: {e}")
                
                # 2. JSON Stats -> all_json
                elif file.endswith('_summary_confidences.json'):
                    dest = all_json_dir / file
                    if not dest.exists():
                        try:
                            shutil.copy2(file_path, dest)
                            json_count += 1
                        except OSError as e:
                            logger.warning(f"Failed to copy JSON {file}: {e}")
                
                # 3. DELETE EVERYTHING (original CIF/JSON included)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
            
            # Remove empty directories
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass # Directory might not be empty
                    
        # Try to remove the step directory itself
        try:
            os.rmdir(step_dir)
        except OSError:
            pass # Keep if not empty
            
        logger.info(f"Cleaned up {step_dir}: Moved {cif_count} CIFs, {json_count} JSONs. Deleted {deleted_count} files.")

    def train(
        self, 
        scaffold_pdb: str, 
        resume: bool = True,
        resume_from: str = None,
        resume_mode: str = 'continue',
        hallu_trigger: bool = False,
        hallu_cycles: int = None
    ):
        """
        Main training loop.
        
        Args:
            scaffold_pdb: Path to scaffold PDB
            resume: Resume from checkpoint
            resume_from: Specific checkpoint path
            resume_mode: 'continue' (keep history) or 'fresh' (reset metrics)
            hallu_trigger: Manually trigger HalluDesign
            hallu_cycles: Override HalluDesign cycles
        """
        # Load checkpoint if resuming
        if resume and not resume_from:
            self.load_checkpoint(resume_mode=resume_mode)
        elif resume_from:
            self.load_checkpoint(resume_from, resume_mode=resume_mode)
        
        # Set manual trigger from CLI
        self.manual_hallu_trigger = hallu_trigger
        
        # Load scaffold sequence
        current_sequence = self.load_scaffold(scaffold_pdb)
        
        # CRITICAL: Set scaffold_pdb (CONSTANT) - used for LigandMPNN input every step
        # This NEVER changes during BetterMPNN training
        self.scaffold_pdb = Path(scaffold_pdb)
        
        self.scaffold_pdb = Path(scaffold_pdb)
        
        # Hydrate Template NOW (using scaffold)
        self._check_and_hydrate_template(scaffold_pdb)
        
        if self.current_pdb is None:
            self.current_pdb = Path(scaffold_pdb)
            
        # Helper: Get open PDB path from config
        open_pdb_path = self.config.get('grpo', {}).get('open_rmsd', {}).get('path')
        if open_pdb_path and os.path.exists(open_pdb_path):
            self.open_pdb = Path(open_pdb_path)
            logger.info(f"Using Open State Reference: {self.open_pdb}")
        else:
            logger.warning("No Open State Reference PDB found/configured. Open State RMSD will be skipped.")
        
        # Initialize best_sequence if empty
        if not self.best_sequence:
            self.best_sequence = current_sequence
        
        if hallu_cycles:
            self.hallu_cycles = hallu_cycles
        
        logger.info(f"Starting training: steps={self.training_steps}, hallu={hallu_trigger}")
        logger.info(f"Scaffold sequence: {len(current_sequence)} residues")
        
        # Support for Multi-Backbone / Directory Input
        scaffold_path_obj = Path(scaffold_pdb)
        self.backbone_mode = "single"
        self.backbone_features_list = []
        
        if scaffold_path_obj.is_dir():
            logger.info(f"Scaffold path is a directory: {scaffold_pdb}")
            self.backbone_mode = "multi"
            # Find all pdb files
            pdb_files = sorted(list(scaffold_path_obj.glob("*.pdb")))
            if not pdb_files:
                raise ValueError(f"No .pdb files found in {scaffold_pdb}")
            
            logger.info(f"Found {len(pdb_files)} PDB scaffolds.")
            logger.info("Pre-calculating LigandMPNN features for all backbones... (this may take a moment)")
            
            from ligandmpnn_utils import prepare_ligandmpnn_features
            
            for p in pdb_files:
                try:
                    features = prepare_ligandmpnn_features(
                        pdb_path=str(p),
                        chain_to_design=self.chain_to_design,
                        fixed_residues=None, # Assuming restrictions handled elsewhere or uniform
                        redesigned_residues=self.redesign_residues, 
                        use_ligand_context=True,
                        device=self.device
                    )
                    # Store path for logging reference if needed
                    features["_source_pdb"] = str(p)
                    self.backbone_features_list.append(features)
                except Exception as e:
                    logger.warning(f"Failed to process backbone {p}: {e}")
            
            if not self.backbone_features_list:
                raise RuntimeError("Failed to process any backbones from directory!")
                
            logger.info(f"Successfully loaded {len(self.backbone_features_list)} backbone feature sets.")
            
        else:
            logger.info(f"Scaffold PDB (single): {self.scaffold_pdb}")
            
        # Main training loop
        for step in range(self.step, self.training_steps):
            self.step = step
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step}/{self.training_steps}")
            logger.info(f"{'='*60}")
            
            # Select random backbone if in multi mode
            current_features = None
            pdb_source = str(self.scaffold_pdb)
            
            if self.backbone_mode == "multi":
                # NEW: If current_pdb is a file (refined by HalluDesign), prioritize it!
                if self.current_pdb.is_file():
                    current_features = None
                    pdb_source = str(self.current_pdb)
                    logger.info(f"Using refined structure from HalluDesign: {self.current_pdb.name}")
                else:
                    import random
                    current_features = random.choice(self.backbone_features_list)
                    pdb_source = current_features.get("_source_pdb", "unknown")
                    logger.info(f"Multi-Backbone: Selected scaffold {os.path.basename(pdb_source)}")
            
            # GRPO step
            # Pass initialized features to avoid re-parsing
            metrics = self.grpo_step(
                current_sequence, 
                step, 
                current_pdb_path=self.current_pdb, # Always pass current_pdb (will handle file vs dir inside)
                precomputed_features=current_features # Pass features directly
            )
            
            # Check HalluDesign trigger (periodic, force, or manual)
            should_trigger = self.hallu_enabled and self.check_hallu_trigger(metrics)
            
            if should_trigger:
                logger.info("=" * 60)
                logger.info(f"Triggering HalluDesign at step {step}")
                logger.info("=" * 60)
                
                # Run HalluDesign with current best structure
                # NEW PRORITY: Use self.best_structure_path (best AF3 result) if available
                # Fallback to self.current_pdb (refined) or initial scaffold.
                hallu_input = str(self.current_pdb)
                if hasattr(self, 'best_structure_path') and self.best_structure_path and os.path.exists(str(self.best_structure_path)):
                    hallu_input = str(self.best_structure_path)
                    logger.info(f"HalluDesign starting from current best generated structure: {os.path.basename(hallu_input)}")
                elif self.current_pdb.is_dir():
                    # Safeguard for initial multi-backbone state if no best_structure_path yet
                    logger.warning("HalluDesign triggered but no best structure found yet, fallback to directory (likely will fail)")
                
                results, refined_pdb = self.hallu_design_step(
                    hallu_input, 
                    self.best_sequence
                )
                
                # Update current PDB to refined structure
                self.current_pdb = Path(refined_pdb)
                
                logger.info(f"Resuming GRPO with refined structure: {refined_pdb}")
            
            # Update current sequence for next iteration
            current_sequence = self.best_sequence
            
            # Save checkpoint
            save_every = self.config.get('checkpoint', {}).get('save_every', 10)
            if (step + 1) % save_every == 0:
                self.save_checkpoint(step)
            
            # CLEANUP POLICY: Every 3 steps
            if (step + 1) % 3 == 0:
                step_dir = self.output_dir / f"step_{step}"
                self.cleanup_step_files(step_dir)
                
                # Also try to clean previous 2 steps if they exist and skipped
                for prev_step in range(step - 2, step):
                    if prev_step >= 0:
                        self.cleanup_step_files(self.output_dir / f"step_{prev_step}")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final save
        self.save_checkpoint(self.training_steps)
        self.save_results()
        
        # Generate final plots
        plot_training_charts(
            self.metrics_history, 
            str(self.output_dir), 
            self.reward_history
        )
        self.training_logger.finalize()
        
        logger.info("Training complete!")
    
    def save_results(self):
        """Save final results."""
        results = {
            'best_reward': self.best_reward,
            'best_sequence': self.best_sequence,
            'best_pdb': str(self.current_pdb) if self.current_pdb else None,
            'total_steps': self.step,
            'hallu_trigger': self.hallu_trigger,
            'reward_history': self.reward_history,
        }
        
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics to Excel if available
        try:
            import pandas as pd
            df = pd.DataFrame(self.metrics_history)
            df.to_excel(self.output_dir / "training_metrics.xlsx", index=False)
            logger.info("Metrics exported to training_metrics.xlsx")
        except ImportError:
            logger.warning("pandas not available, skipping Excel export")
        
        logger.info(f"Results saved to {self.output_dir / 'results.json'}")


# ============================================
# CLI
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='HalluMPNN: BetterMPNN + HalluDesign with AlphaFold3'
    )
    
    parser.add_argument('--scaffold', type=str, required=True,
                       help='Path to scaffold PDB file')
    parser.add_argument('--config', type=str,
                       default=str(PROJECT_DIR / 'configs' / 'default.yaml'),
                       help='Path to config YAML')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--smiles', type=str, default=LDOPA_SMILES,
                       help='Ligand SMILES string')
    parser.add_argument('--hallu_trigger', action='store_true',
                       help='Manually trigger HalluDesign at start')
    parser.add_argument('--hallu_cycles', type=int, default=None,
                       help='Override HalluDesign cycles')
    parser.add_argument('--steps', type=int, default=None,
                       help='Override training steps')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume from latest checkpoint in output_dir')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from specific checkpoint path')
    parser.add_argument('--resume_mode', type=str, default='continue',
                       choices=['continue', 'fresh'],
                       help='Resume mode: continue=keep history, fresh=reset metrics')
    # --no_resume removed, now --resume is opt-in
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    # Apply CLI overrides
    if args.output_dir:
        config.setdefault('paths', {})['output_dir'] = args.output_dir
    if args.smiles:
        config.setdefault('ligand', {})['smiles'] = args.smiles
    if args.steps:
        config.setdefault('grpo', {})['training_steps'] = args.steps
    
    # Create timestamped output directory
    if 'paths' not in config or 'output_dir' not in config['paths']:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        config.setdefault('paths', {})['output_dir'] = f'./outputs/{timestamp}'
    
    # Initialize trainer
    trainer = HalluMPNNTrainer(config)
    
    # Run training
    trainer.train(
        scaffold_pdb=args.scaffold,
        resume=(args.resume or args.resume_from is not None),
        resume_from=args.resume_from,
        resume_mode=args.resume_mode,
        hallu_trigger=args.hallu_trigger,
        hallu_cycles=args.hallu_cycles
    )


if __name__ == '__main__':
    main()
