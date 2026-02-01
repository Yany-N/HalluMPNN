# -*- coding: utf-8 -*-
"""
Reward Utilities for HalluMPNN
Extended reward functions for GRPO training with RMSD support.
Based on BetterMPNN reward_utils.py with additions for structure validation.
"""

import os
import json
import logging
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

# ============================================
# Reward Weights
# ============================================
DEFAULT_REWARD_WEIGHTS = {
    'iptm': 0.35,           # Interface pTM (protein-ligand)
    'ptm': 0.25,            # Overall structure quality
    'pae': 0.25,            # Predicted alignment error
    'rmsd': 0.15,           # Conformational stability
    'clash_penalty': 0.10,  # Atomic clash penalty
}

PAE_MAX = 31.75  # Maximum PAE value for normalization


def calculate_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    align: bool = True
) -> float:
    """
    Calculate RMSD between two coordinate sets.
    
    Args:
        coords1: First coordinate set (N, 3)
        coords2: Second coordinate set (N, 3)
        align: Whether to align structures first (Kabsch algorithm)
    
    Returns:
        rmsd: Root mean square deviation in Angstroms
    """
    if len(coords1) != len(coords2):
        logger.warning(f"Coordinate length mismatch: {len(coords1)} vs {len(coords2)}")
        return float('inf')
    
    if align:
        # Kabsch alignment
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)
        
        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
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
    
    Beyond the cutoff (approximately AF3 resolution), RMSD is considered
    unreliable for evaluating structural quality.
    
    Args:
        rmsd: Calculated RMSD in Angstroms
        cutoff: Cutoff threshold (default 3.0 Å)
    
    Returns:
        reward: [0, 1] reward value
    """
    # Smooth exponential decay across the entire range
    # Avoids hard cutoff at 0.0 and discontinuity
    # Maps: 0A -> 1.0, 3.0A -> ~0.22, 6.0A -> ~0.05
    return float(np.exp(-0.5 * rmsd))


def calculate_pae_reward(pae: float, pae_max: float = PAE_MAX) -> float:
    """Convert PAE to positive reward (lower PAE = higher reward)."""
    reward = 1.0 - (pae / pae_max)
    return max(0.0, min(1.0, reward))


def calculate_specificity_reward(
    target_iptm: float,
    decoy_iptms: List[float],
    min_gap: float = 0.1
) -> Tuple[float, float]:
    """
    Calculate specificity bonus based on gap between target and decoys.
    
    Args:
        target_iptm: iPTM of the target ligand (e.g., L-DOPA)
        decoy_iptms: List of iPTMs for decoy ligands (e.g., Tyr, Trp)
        min_gap: Minimum required gap for bonus
        
    Returns:
        bonus: Specificity bonus value
        max_decoy_iptm: The highest decoy iPTM found
    """
    if not decoy_iptms:
        return 0.0, 0.0
        
    max_decoy_iptm = max(decoy_iptms)
    gap = target_iptm - max_decoy_iptm
    
    # Sigmoid-like activation for gap
    # Optimization: Lower slope from 10 to 5 for smoother gradient
    slope = 5.0
    if gap > min_gap:
        bonus = torch.sigmoid(torch.tensor((gap - min_gap) * slope)).item()
    else:
        # Penalize if gap is too small or negative
        # Smooth penalty: -0.5 * (min_gap - gap)
        bonus = -0.5 * (min_gap - gap)
        
    return bonus, max_decoy_iptm


def compute_conformational_specificity_reward(
    target_conf_rmsd: Optional[float],
    decoy_conf_rmsds: List[float],
    target_threshold: float = 3.0,
    decoy_max_rmsd: float = 2.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate conformational specificity reward.
    
    GOAL: Protein should CLOSE with target (L-DOPA) but STAY OPEN with decoys (Tyr)
    
    - Target (L-DOPA): HIGH conf_rmsd = closing = GOOD
    - Decoys (Tyr): LOW conf_rmsd = staying open = GOOD
    
    Args:
        target_conf_rmsd: Conformational RMSD with target ligand (bound vs unbound)
        decoy_conf_rmsds: List of conf_rmsds with decoy ligands
        target_threshold: Min conf_rmsd required for target (default 3.0 Å)
        decoy_max_rmsd: Max conf_rmsd allowed for decoys (default 2.0 Å)
    
    Returns:
        reward: Conformational specificity reward in [0, 1]
        info: Detailed breakdown
    """
    info = {
        'target_conf_rmsd': target_conf_rmsd,
        'decoy_conf_rmsds': decoy_conf_rmsds,
        'target_closes': False,
        'decoys_stay_open': False,
    }
    
    # 1. Check if target causes closing
    target_score = 0.0
    if target_conf_rmsd is not None:
        if target_conf_rmsd >= target_threshold:
            target_score = 1.0
            info['target_closes'] = True
        else:
            # Soft ramp: partial credit for approaching threshold
            target_score = target_conf_rmsd / target_threshold
    
    # 2. Check if decoys stay open (low conf_rmsd)
    decoy_score = 1.0  # Default: pass if no decoys
    if decoy_conf_rmsds:
        max_decoy_rmsd = max(decoy_conf_rmsds)
        info['max_decoy_conf_rmsd'] = max_decoy_rmsd
        
        if max_decoy_rmsd <= decoy_max_rmsd:
            # All decoys stay open
            decoy_score = 1.0
            info['decoys_stay_open'] = True
        else:
            # Penalty: some decoy caused closing
            # Strict linear decay: fully penalized (zero score) at +6Å excess (was 3Å)
            excess = max_decoy_rmsd - decoy_max_rmsd
            decoy_score = max(0.0, 1.0 - excess / 6.0)
    
    # 3. Combine: Multiplicative
    # If either fails (target doesn't close OR decoy closes), reward suffers
    reward = target_score * decoy_score
    
    info['target_score'] = target_score
    info['decoy_score'] = decoy_score
    info['total_conf_specificity'] = reward
    
    return reward, info


def calculate_switch_penalty(
    conf_rmsd: float,
    unbound_ptm: float,
    open_rmsd: Optional[float] = None,
    rmsd_threshold: float = 2.0,
    ptm_threshold: float = 0.5,
    open_rmsd_cutoff: float = 4.0
) -> float:
    """
    Calculate penalty for "constitutive closing" AND "unfolding stability".
    
    1. Stability Logic: If Unbound confidence is low (low pTM), it's unfolded.
    2. Switch Logic:
       - If Open Reference provided: Reward LOW open_rmsd (matches specific open state).
       - If No Reference: Reward HIGH conf_rmsd (matches any open state).
    
    Args:
        conf_rmsd: RMSD between Bound and Unbound structures
        unbound_ptm: pTM score of the Unbound structure
        open_rmsd: RMSD between Unbound and Open Reference (optional)
        rmsd_threshold: Min RMSD requirement (if no open ref)
        ptm_threshold: Min pTM requirement
        open_rmsd_cutoff: Max RMSD to match Open Reference
        
    Returns:
        penalty_factor: Multiplier for total reward (0.1 to 1.0)
    """
    penalty = 1.0
    
    # 1. Stability Check (Must be folded without ligand)
    # Stricter Sigmoid: center at 0.65 instead of 0.5
    # If unbound_ptm < 0.65, penalty kicks in hard.
    # pTM 0.50 -> sigmoid(-1.5) ~0.18 -> factor ~0.26
    # pTM 0.65 -> sigmoid(0) = 0.5 -> factor 0.55
    # pTM 0.80 -> sigmoid(1.5) ~0.82 -> factor ~0.83
    
    # Use explicitly passed threshold or default strict 0.65
    eff_threshold = max(ptm_threshold, 0.65)
    
    stability_score = torch.sigmoid(torch.tensor((unbound_ptm - eff_threshold) * 10.0)).item()
    
    # Scale stability to range [0.1, 1.0]
    stability_factor = 0.1 + 0.9 * stability_score
    
    if stability_factor < 0.6:
        logger.info(f"  Stability Penalty: Unbound pTM {unbound_ptm:.2f} (thresh {eff_threshold}) -> factor={stability_factor:.2f}")
        return stability_factor
        
    # 2. Switch Check
    switch_factor = 1.0
    
    if open_rmsd is not None:
        # GUIDED MODE: Gaussian-like Match to Open Reference
        # Reward = exp(-(rms - cutoff)^2 / sigma) if rms > cutoff?
        # Actually we want low open_rmsd. 
        # Match score = exp(-open_rmsd / 2.0) ? 
        # Let's use a soft threshold.
        
        # If open_rmsd < open_rmsd_cutoff: factor ~ 1.0
        # If open_rmsd > open_rmsd_cutoff: decay
        
        # Shifted sigmoid decay: center at cutoff + 1.0
        # If cutoff=4.0, center=5.0. 
        # RMSD 4.0 -> good. RMSD 6.0 -> bad.
        
        decay_center = open_rmsd_cutoff + 1.0
        switch_factor = 1.0 - torch.sigmoid(torch.tensor((open_rmsd - decay_center) * 3.0)).item()
        
        # Scale to [0.1, 1.0]
        switch_factor = 0.1 + 0.9 * switch_factor
        
        if switch_factor < 0.9:
             logger.info(f"  Guidance Penalty: Open Ref Match RMSD {open_rmsd:.2f} -> factor={switch_factor:.2f}")

    else:
        # BLIND MODE: High Conf RMSD
        # Sigmoid ramp up
        # Center at rmsd_threshold, slope 3
        ramp_score = torch.sigmoid(torch.tensor((conf_rmsd - rmsd_threshold) * 3.0)).item()
        switch_factor = 0.1 + 0.9 * ramp_score
        
        if switch_factor < 0.9:
            logger.info(f"  Switch Penalty: Conf RMSD {conf_rmsd:.2f} -> factor={switch_factor:.2f}")
            
    # Combine factors
    total_penalty = stability_factor * switch_factor
    return total_penalty


# ============================================
# Curriculum-based Conformational Reward (NEW)
# ============================================

def compute_curriculum_threshold(
    step: int,
    early_steps: int = 40,
    late_steps: int = 80,
    early_threshold: float = 1.0,
    late_threshold: float = 2.5
) -> float:
    """
    Compute conformational RMSD threshold based on training progress.
    
    Curriculum Learning:
    - Early (step < early_steps): easy threshold (1.0 Å)
    - Middle (early_steps <= step < late_steps): linear ramp
    - Late (step >= late_steps): strict threshold (2.5 Å)
    
    Args:
        step: Current training step
        early_steps: End of early training phase
        late_steps: Start of strict training phase
        early_threshold: RMSD threshold during early phase
        late_threshold: RMSD threshold during late phase
    
    Returns:
        threshold: Current conformational RMSD threshold
    """
    if step < early_steps:
        return early_threshold
    elif step >= late_steps:
        return late_threshold
    else:
        # Linear interpolation
        progress = (step - early_steps) / (late_steps - early_steps)
        return early_threshold + progress * (late_threshold - early_threshold)


def compute_curriculum_conf_reward(
    conf_rmsd: Optional[float],
    step: int,
    target_conf_rmsd: float = 5.0,
    early_steps: int = 40,
    late_steps: int = 80,
    early_threshold: float = 1.0,
    late_threshold: float = 2.5
) -> float:
    """
    Compute conformational change reward with curriculum learning.
    
    The reward encourages bound-unbound structural difference (conf_rmsd),
    with requirements that increase as training progresses.
    
    Args:
        conf_rmsd: RMSD between bound and unbound structures
        step: Current training step
        target_conf_rmsd: Target RMSD for maximum reward (default 5.0 Å)
        early_steps: End of early phase (default 40)
        late_steps: Start of late phase (default 80)
        early_threshold: Min RMSD in early phase (default 1.0 Å)
        late_threshold: Min RMSD in late phase (default 2.5 Å)
    
    Returns:
        conf_reward: Reward value in [0, 1]
    """
    if conf_rmsd is None:
        return 0.0
    
    # Get current threshold
    threshold = compute_curriculum_threshold(
        step, early_steps, late_steps, early_threshold, late_threshold
    )
    
    if conf_rmsd < threshold:
        # Below threshold: small but non-zero reward (soft penalty)
        # Allows "almost correct" sequences to contribute gradient signal
        return 0.1 * (conf_rmsd / threshold)
    elif conf_rmsd >= target_conf_rmsd:
        # At or above target: maximum reward
        return 1.0
    else:
        # Between threshold and target: linear ramp
        progress = (conf_rmsd - threshold) / (target_conf_rmsd - threshold)
        return 0.1 + 0.9 * progress


def compute_sbp_preservation_factor(
    reference_rmsd: Optional[float],
    sbp_max_deviation: float = 8.0,
    soft_penalty: float = 0.1
) -> float:
    """
    Compute SBP (Substrate Binding Protein) structure preservation factor.
    
    CRITICAL: High conf_rmsd can mean either:
    1. Good: Correct open/closed conformational change (desired)
    2. Bad: Structure has deviated from SBP fold (undesired)
    
    This function distinguishes between them by checking if the bound 
    structure still matches the original SBP scaffold.
    
    Args:
        reference_rmsd: RMSD between predicted bound structure and scaffold
        sbp_max_deviation: Maximum allowed deviation from SBP scaffold (Å)
        soft_penalty: Penalty factor when structure deviates (default 0.1)
    
    Returns:
        sbp_factor: Multiplier in [soft_penalty, 1.0]
    """
    if reference_rmsd is None:
        # Cannot verify, assume okay
        return 1.0
    
    # STRICT Check:
    # If rmsd > sbp_max_deviation, we kill the reward heavily.
    # We use a steep sigmoid to avoid hard cutoffs but punish effectively.
    
    if reference_rmsd <= sbp_max_deviation:
        return 1.0
    else:
        # Exceeded max deviation
        # Calculate how much it exceeded
        excess = reference_rmsd - sbp_max_deviation
        
        # Exponential decay penalty
        # If excess is 1.0A -> factor ~0.36
        # If excess is 2.0A -> factor ~0.13
        penalty = np.exp(-excess)
        
        # Floor at soft_penalty
        sbp_factor = max(soft_penalty, penalty)
        
        if sbp_factor < 0.9:
            logger.warning(f"SBP structure deviation: ref_rmsd={reference_rmsd:.2f}Å > {sbp_max_deviation}Å -> factor={sbp_factor:.2f}")
            
        return sbp_factor


def calculate_comprehensive_reward(
    summary_data: Dict[str, Any],
    rmsd: Optional[float] = None,
    conf_rmsd: Optional[float] = None,
    open_rmsd: Optional[float] = None,
    unbound_ptm: Optional[float] = None,
    decoy_stats: Optional[Dict[str, List[float]]] = None,
    reward_weights: Dict[str, float] = None,
    # NEW: Curriculum parameters
    step: int = 0,
    curriculum_config: Dict[str, Any] = None,
    # Legacy parameters (for backwards compatibility with run_hallumpnn.py)
    # These are now UNUSED since we removed Rg logic, but kept to avoid TypeError
    bound_coords: Optional[np.ndarray] = None,
    open_ref_coords: Optional[np.ndarray] = None,
    unbound_coords: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate comprehensive reward with Curriculum Learning and SBP Validation.
    
    NEW REWARD FORMULA (v2):
    1. quality_reward = iptm * w_iptm + ptm * w_ptm + pae * w_pae (additive)
    2. conf_reward = curriculum_conf_reward(conf_rmsd, step) (additive)
    3. sbp_factor = sbp_preservation_factor(rmsd) (multiplicative gate)
    4. total = (quality_weight * quality + conf_weight * conf) * sbp_factor
    
    Args:
        summary_data: AF3/Protenix confidence summary (Target)
        rmsd: Reference RMSD (Bound vs Scaffold) - for SBP preservation check
        conf_rmsd: Conformational RMSD (Bound vs Unbound) - for dynamics reward
        open_rmsd: Open Reference RMSD (Unbound vs Open Ref) - optional
        unbound_ptm: pTM of Unbound structure (for stability check)
        decoy_stats: Dictionary with 'iptms' list for decoys
        reward_weights: Custom reward weights
        step: Current training step (for curriculum)
        curriculum_config: Curriculum learning parameters
    
    Returns:
        total_reward: Final reward value
        reward_info: Detailed breakdown
    """
    if reward_weights is None:
        reward_weights = DEFAULT_REWARD_WEIGHTS
    
    # Default curriculum config
    if curriculum_config is None:
        curriculum_config = {
            'early_steps': 40,
            'late_steps': 80,
            'early_threshold': 1.0,
            'late_threshold': 2.5,
            'target_conf_rmsd': 5.0,
            'sbp_max_deviation': 8.0,
            'quality_weight': 0.50,
            'conf_weight': 0.50,
        }
    
    # Extract metrics
    iptm = summary_data.get('iptm', 0.0)
    ptm = summary_data.get('ptm', 0.0)
    has_clash = summary_data.get('has_clash', True)
    ranking_score = summary_data.get('ranking_score', 0.0)
    
    # Extract PAE
    chain_pair_pae = summary_data.get('chain_pair_pae_min', [])
    if chain_pair_pae and len(chain_pair_pae) >= 2:
        try:
            a_to_b = chain_pair_pae[0][1] if len(chain_pair_pae[0]) > 1 else PAE_MAX
            b_to_a = chain_pair_pae[1][0] if len(chain_pair_pae) > 1 else PAE_MAX
            mean_pae = (a_to_b + b_to_a) / 2.0
        except (IndexError, TypeError):
            mean_pae = PAE_MAX
    else:
        mean_pae = summary_data.get('pae', PAE_MAX)
    
    # ========================================
    # 1. Quality Reward (Binding & Structure)
    # ========================================
    iptm_reward = iptm
    ptm_reward = ptm
    pae_reward = calculate_pae_reward(mean_pae)
    clash_penalty = -0.5 if has_clash else 0.0  # Reduced from -1.0
    
    # Weighted quality score
    quality_reward = (
        reward_weights.get('iptm', 0.50) * iptm_reward +
        reward_weights.get('ptm', 0.30) * ptm_reward +
        reward_weights.get('pae', 0.20) * pae_reward +
        0.10 * clash_penalty
    )
    quality_reward = max(0.0, quality_reward)
    
    # ========================================
    # 2. Specificity Bonus (Negative Design)
    # ========================================
    spec_bonus = 0.0
    max_decoy_iptm = 0.0
    if decoy_stats and 'iptms' in decoy_stats:
        spec_bonus, max_decoy_iptm = calculate_specificity_reward(
            iptm, 
            decoy_stats['iptms'],
            min_gap=reward_weights.get('spec_gap', 0.1)
        )
        # Add to quality reward
        spec_weight = reward_weights.get('specificity', 0.3)
        quality_reward += (spec_weight * spec_bonus)
    
    # ========================================
    # 3. Structure Reward (RMSD Dominant - No Rg)
    # ========================================
    # User Request: "RMSD Dominant" version, NO Radius of Gyration.
    # Logic: 
    # - 70% Reward for matching Closed Crystal Structure (rmsd)
    # - 30% Reward for matching Open Reference (open_rmsd)
    
    # A. Closed Structure Match (Dominant)
    # rmsd = RMSD(Bound, Closed_Ref/Scaffold)
    closed_rmsd_reward = 0.0
    if rmsd is not None:
        # Use simple exponential decay: 1.0 at 0A
        # calculate_rmsd_reward uses exp(-0.5 * rmsd)
        closed_rmsd_reward = calculate_rmsd_reward(rmsd)
        
    # B. Open Structure Match (Secondary)
    # open_rmsd = RMSD(Unbound, Open_Ref)
    open_rmsd_reward = 0.0
    if open_rmsd is not None:
        open_rmsd_reward = calculate_rmsd_reward(open_rmsd)
    
    # Combined Structure Score
    if open_rmsd is not None:
        structure_score = (0.7 * closed_rmsd_reward) + (0.3 * open_rmsd_reward)
    else:
        # If no open ref provided, rely 100% on closed match
        structure_score = closed_rmsd_reward

    # Log the contributions
    rmsd_str = f"{rmsd:.2f}" if rmsd is not None else "None"
    open_rmsd_str = f"{open_rmsd:.2f}" if open_rmsd is not None else "None"
    logger.debug(f"  Structure: ClosedRMSD={rmsd_str}({closed_rmsd_reward:.2f}) OpenRMSD={open_rmsd_str}({open_rmsd_reward:.2f}) -> Score={structure_score:.2f}")

    # ========================================
    # 4. Stability Check (Unbound must be folded)
    # ========================================
    stability_factor = 1.0
    strict_ptm_threshold = curriculum_config.get('unbound_ptm_threshold', 0.65)
    
    if unbound_ptm is not None:
        if unbound_ptm < strict_ptm_threshold:
            stability_factor = 0.1 + 0.9 * (unbound_ptm / strict_ptm_threshold)
    
    # ========================================
    # 5. SBP Structure Preservation Gate
    # ========================================
    # We keep this as a secondary check, but since we are directly rewarding 'rmsd',
    # this might be redundant? 
    # Actually, rmsd is now part of the POSITIVE reward, so we don't need a separate negative gate 
    # unless deviation is extreme.
    sbp_factor = 1.0
    if rmsd is not None and rmsd > curriculum_config.get('sbp_max_deviation', 8.0):
        # Only penalize if REALLY bad (broken fold)
        sbp_factor = 0.1

    # ========================================
    # 6. Conformational Specificity
    # ========================================
    # L-DOPA should cause closing, but decoys (Tyr) should NOT.
    # Since we are using RMSD-dominant logic, "closing" means low RMSD to Closed Ref.
    # Decoys should have HIGH RMSD to Closed Ref (stay open).
    conf_spec_reward = 1.0
    conf_spec_info = {}
    
    if decoy_stats and 'closed_rmsds' in decoy_stats:
        # Using the NEW logic: Decoys must not look like Closed Ref
        decoy_closed_rmsds = decoy_stats['closed_rmsds']
        target_threshold = 2.5 # Good closing
        decoy_safe_distance = 4.0 # Decoys should be at least this far from closed
        
        # 1. Target Score (Passed if closed_rmsd_reward is high? Already in structure_score)
        # We focus on DECOY PENALTY here.
        
        failed_decoys = [d for d in decoy_closed_rmsds if d < decoy_safe_distance]
        if failed_decoys:
            # Penalty: At least one decoy is too close to Closed Ref
            # Worst offender
            min_decoy_rmsd = min(failed_decoys)
            # if min_decoy_rmsd = 2.0 (bad) -> penalty
            # if min_decoy_rmsd = 4.0 (safe) -> no penalty
            # 1.0 - (1.0 - 2.0/4.0) ? 
            penalty_strength = max(0.0, 1.0 - (min_decoy_rmsd / decoy_safe_distance))
            # If 2.0/4.0 = 0.5 -> penalty 0.5 -> score 0.5
            conf_spec_reward = 1.0 - penalty_strength
            
            conf_spec_info['decoys_stay_open'] = False
            conf_spec_info['min_decoy_closed_rmsd'] = min_decoy_rmsd
            logger.warning(f"  ✗ Decoy too close to Closed Ref! min_rmsd={min_decoy_rmsd:.2f}")
        else:
            conf_spec_info['decoys_stay_open'] = True


    # ========================================
    # 7. Final Combination
    # ========================================
    quality_weight = curriculum_config.get('quality_weight', 0.50)
    structure_weight = curriculum_config.get('conf_weight', 0.50)
    
    # Main Equation
    # Total = (Quality * 0.5 + Structure * 0.5) * Gates
    base_reward = (quality_weight * quality_reward) + (structure_weight * structure_score)
    
    total_reward = base_reward * stability_factor * sbp_factor * conf_spec_reward
    
    # Clamp
    total_reward = max(0.0, min(1.0, total_reward))
    

    reward_info = {
        'total_reward': total_reward,
        'quality_reward': quality_reward,
        'structure_score': structure_score,       # CHANGED: was conf_reward
        'closed_rmsd_reward': closed_rmsd_reward, # NEW: detailed
        'open_rmsd_reward': open_rmsd_reward,     # NEW: detailed
        'conf_spec_reward': conf_spec_reward,
        'iptm': iptm,
        'iptm_reward': iptm_reward,
        'ptm': ptm,
        'ptm_reward': ptm_reward,
        'mean_pae': mean_pae,
        'pae_reward': pae_reward,
        'rmsd': rmsd,
        'conf_rmsd': conf_rmsd,
        'open_rmsd': open_rmsd,
        'sbp_factor': sbp_factor,
        'stability_factor': stability_factor,
        'conf_spec_info': conf_spec_info,
        'spec_bonus': spec_bonus,
        'max_decoy_iptm': max_decoy_iptm,
        'has_clash': has_clash,
        'clash_penalty': clash_penalty,
        'ranking_score': ranking_score,
        'step': step,
    }
    
    return total_reward, reward_info


def compute_group_relative_advantages(
    rewards: np.ndarray,
    scale: bool = True,
    scale_factor: float = 5.0
) -> np.ndarray:
    """
    Compute GRPO advantages (group relative).
    
    Args:
        rewards: Array of rewards [batch_size]
        scale: Whether to scale by std
        scale_factor: Fallback scale factor
    
    Returns:
        advantages: Normalized advantages
    """
    if len(rewards) <= 1:
        return np.zeros_like(rewards)
    
    mean_reward = rewards.mean()
    
    if scale:
        std_reward = rewards.std()
        if std_reward > 1e-8:
            advantages = (rewards - mean_reward) / std_reward
        else:
            advantages = (rewards - mean_reward) * scale_factor
    else:
        advantages = rewards - mean_reward
    
    return advantages


def check_hallu_trigger(
    reward_info: Dict[str, Any],
    thresholds: Dict[str, float]
) -> bool:
    """
    Check if HalluDesign should be triggered based on thresholds.
    
    Args:
        reward_info: Current reward information
        thresholds: Trigger thresholds (iptm_min, ptm_min, pae_max)
    
    Returns:
        should_trigger: Whether to trigger HalluDesign
    """
    iptm = reward_info.get('iptm', 0.0)
    ptm = reward_info.get('ptm', 0.0)
    pae = reward_info.get('mean_pae', 31.75)
    
    iptm_ok = iptm >= thresholds.get('iptm_min', 0.7)
    ptm_ok = ptm >= thresholds.get('ptm_min', 0.6)
    pae_ok = pae <= thresholds.get('pae_max', 10.0)
    
    if iptm_ok and ptm_ok and pae_ok:
        logger.info(f"HalluDesign trigger: iptm={iptm:.3f}, ptm={ptm:.3f}, pae={pae:.2f}")
        return True
    
    return False


# ============================================
# Preset Configurations
# ============================================
REWARD_PRESETS = {
    "default": DEFAULT_REWARD_WEIGHTS,
    
    "ligand_focused": {
        'iptm': 0.45,
        'ptm': 0.20,
        'pae': 0.20,
        'rmsd': 0.15,
        'clash_penalty': 0.10,
    },
    
    "structure_focused": {
        'iptm': 0.25,
        'ptm': 0.35,
        'pae': 0.25,
        'rmsd': 0.15,
        'clash_penalty': 0.10,
    },
    
    "strict_rmsd": {
        'iptm': 0.30,
        'ptm': 0.20,
        'pae': 0.20,
        'rmsd': 0.30,
        'clash_penalty': 0.10,
    },
}


if __name__ == "__main__":
    # Test Suite for Stricter Reward Logic
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Stricter Reward Logic ===")
    
    # 1. Test Base Case (Perfect)
    base_data = {
        'iptm': 0.90, 'ptm': 0.85, 'mean_pae': 5.0,
        'has_clash': False, 'ranking_score': 0.9
    }
    
    # Configuration mimicking strict mode
    config = {
        'sbp_max_deviation': 3.5, 
        'unbound_ptm_threshold': 0.65,
        'early_steps': 0, 'late_steps': 10, # Force late phase
        'early_threshold': 1.0, 'late_threshold': 4.0,
        'target_conf_rmsd': 6.0
    }
    
    # Case A: Perfect closing, perfect structure
    r_perfect, i_perfect = calculate_comprehensive_reward(
        base_data, rmsd=1.0, conf_rmsd=6.0, unbound_ptm=0.8,
        curriculum_config=config, step=100
    )
    print(f"\n[Case A] Perfect: Reward={r_perfect:.4f} (Exp: ~1.5-2.0)")
    
    # Case B: High iPTM but SBP Broken (RMSD 8.0A)
    # Old logic: factor ~0.7. New logic: factor ~exp(-4.5) ~0.01 -> min 0.1
    r_broken, i_broken = calculate_comprehensive_reward(
        base_data, rmsd=8.0, conf_rmsd=6.0, unbound_ptm=0.8,
        curriculum_config=config, step=100
    )
    print(f"\n[Case B] Broken SBP (8.0A): Reward={r_broken:.4f} (Exp: Low due to SBP factor)")
    print(f"  -> SBP Factor: {i_broken['sbp_factor']:.4f}")
    
    # Case C: Valid Binding but Unstable Unbound (pTM 0.4)
    # New logic: ptm < 0.65 -> penalty
    r_unstable, i_unstable = calculate_comprehensive_reward(
        base_data, rmsd=1.0, conf_rmsd=6.0, unbound_ptm=0.4,
        curriculum_config=config, step=100
    )
    print(f"\n[Case C] Unstable Unbound (pTM 0.4): Reward={r_unstable:.4f} (Exp: Low due to Stability)")
    print(f"  -> Stability Factor: {i_unstable['stability_factor']:.4f}")

    # Case D: Decoy Caused Closing (Specificity Fail)
    decoy_stats = {'iptms': [0.1], 'conf_rmsds': [5.0]} # Decoy closed 5.0A!
    # limit is 2.0A
    r_fail, i_fail = calculate_comprehensive_reward(
        base_data, rmsd=1.0, conf_rmsd=6.0, unbound_ptm=0.8,
        decoy_stats=decoy_stats,
        curriculum_config=config, step=100
    )
    print(f"\n[Case D] Decoy Closed (5.0A): Reward={r_fail:.4f} (Exp: Low due to Conf Spec)")
    print(f"  -> Conf Spec Reward: {i_fail['conf_spec_reward']:.4f}")