# -*- coding: utf-8 -*-
"""
HalluDesign Utilities for HalluMPNN - FIXED VERSION v2
====================================================

Fixed issues:
1. AttributeError when best_pdb_path is None
2. Residue ID parsing with insertion codes (e.g., "A123A", "B456R")
3. Better error handling for PDB parsing edge cases
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import types

# Monkey patch for JAX 0.9.0 compatibility (jax.util removed)
try:
    import jax
    if not hasattr(jax, 'util'):
        # Try to use jax._src.util if available, otherwise create a dummy module
        # linking essential functions to their new locations
        try:
            import jax._src.util as _src_util
            jax.util = _src_util
        except ImportError:
            # Create a dummy jax.util module
            jax.util = types.ModuleType('jax.util')
            if hasattr(jax, 'tree_util'):
                 jax.util.register_pytree_node = jax.tree_util.register_pytree_node
except ImportError:
    pass

import numpy as np
import torch

# Local imports
from af3_utils import (
    run_af3_prediction,
    parse_af3_output,
    LDOPA_SMILES,
    extract_ca_coords_from_cif,
    extract_ca_coords_from_pdb,
)

from ligandmpnn_utils import (
    load_ligandmpnn_model,
    generate_sequences_with_ligandmpnn,
    parse_pdb_for_ligandmpnn,
)

from reward_utils import calculate_comprehensive_reward, calculate_rmsd

logger = logging.getLogger(__name__)

# ============================================
# Protenix Integration for Structure-Guided Diffusion
# ============================================

# Add HalluDesign-main to Python path for AF3 import
HALLUDESIGN_PATH = "/data/home/scvi041/run/HalluDesign-main"
# NOTE: Do NOT add another conda environment's site-packages here!
# That causes cross-environment pollution and JAX plugin conflicts.

# Add HalluDesign source directories only
if HALLUDESIGN_PATH not in sys.path:
    sys.path.insert(0, os.path.join(HALLUDESIGN_PATH, "src"))
    sys.path.insert(0, HALLUDESIGN_PATH)

try:
    from af3_model import AF3DesignerPack
    AF3_AVAILABLE = True
    logger.info("AF3DesignerPack imported successfully from af3_model.py")
except ImportError as e:
    AF3_AVAILABLE = False
    logger.warning(f"Could not import AF3DesignerPack: {e}")

# Import process_single_file for converting PDB to PKL (needed for guided diffusion)
try:
    from local_scripts.input_pkl_preprocess import process_single_file
    logger.info("Successfully imported process_single_file")
except ImportError:
    # Fallback: add path manually if not already added
    if HALLUDESIGN_PATH not in sys.path:
        sys.path.insert(0, HALLUDESIGN_PATH)
    try:
        from local_scripts.input_pkl_preprocess import process_single_file
        logger.info("Successfully imported process_single_file after path update")
    except ImportError as e:
        logger.error(f"Could not import process_single_file: {e}")
        process_single_file = None

PROTENIX_AVAILABLE = False # Disabled per user request


# ============================================
# Pocket Detection - FIXED VERSION
# ============================================

def parse_residue_id(res_id_tuple):
    """
    Parse BioPython residue ID tuple safely.
    
    BioPython residue.id format: (hetfield, resseq, icode)
    - hetfield: ' ' for standard residues, 'H_' for hetero atoms
    - resseq: residue sequence number (integer)
    - icode: insertion code (string, usually ' ')
    
    Args:
        res_id_tuple: BioPython residue.id tuple
    
    Returns:
        str: Formatted residue identifier (e.g., "123", "123A")
    """
    hetfield, resseq, icode = res_id_tuple
    
    # Format: residue_number + insertion_code (if present)
    res_id_str = str(resseq)
    if icode.strip():  # If insertion code exists
        res_id_str += icode.strip()
    
    return res_id_str


def find_pocket_residues(
    pdb_path: str,
    ligand_chain: str = "B",
    cutoff: float = 8.0
) -> List[str]:
    """
    Find protein residues within cutoff distance of ligand.
    
    FIXED: Properly handles residue IDs with insertion codes.
    
    Args:
        pdb_path: Path to PDB file
        ligand_chain: Chain ID of ligand (default "B")
        cutoff: Distance cutoff in Angstroms
    
    Returns:
        pocket_residues: List of residue identifiers (e.g. ["A10", "A15A", ...])
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser
        import numpy as np
        
        # Choose parser based on file extension
        if str(pdb_path).endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            # Try CIF parser first if PDB parser might fail
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('protein', pdb_path)
            except (ValueError, Exception) as e:
                # Fallback: try to read the CIF file if a corresponding one exists
                cif_path = str(pdb_path).replace('.pdb', '.cif')
                if os.path.exists(cif_path):
                    logger.info(f"PDB parsing failed, trying CIF: {cif_path}")
                    parser = MMCIFParser(QUIET=True)
                    structure = parser.get_structure('protein', cif_path)
                else:
                    raise e
        
        if 'structure' not in locals():
            structure = parser.get_structure('protein', pdb_path)
        
        # Get ligand atoms
        ligand_atoms = []
        for model in structure:
            for chain in model:
                if chain.id == ligand_chain:
                    for residue in chain:
                        for atom in residue:
                            ligand_atoms.append(atom.coord)
        
        if not ligand_atoms:
            logger.warning(f"No ligand found in chain {ligand_chain}")
            return []
        
        ligand_coords = np.array(ligand_atoms)
        
        # Find protein residues within cutoff
        pocket_residues = []
        for model in structure:
            for chain in model:
                if chain.id == ligand_chain:
                    continue  # Skip ligand chain
                
                for residue in chain:
                    # Skip hetero atoms and water
                    if residue.id[0] != ' ':
                        continue
                    
                    # Use CA atom for distance calculation
                    if 'CA' not in residue:
                        continue
                    
                    ca_coord = residue['CA'].coord
                    
                    # Calculate minimum distance to any ligand atom
                    distances = np.linalg.norm(ligand_coords - ca_coord, axis=1)
                    min_dist = distances.min()
                    
                    if min_dist <= cutoff:
                        # FIXED: Use parse_residue_id to handle insertion codes
                        res_id_str = parse_residue_id(residue.id)
                        res_code = f"{chain.id}{res_id_str}"
                        pocket_residues.append(res_code)
        
        logger.info(f"Found {len(pocket_residues)} pocket residues within {cutoff}Å")
        return pocket_residues
        
    except Exception as e:
        logger.error(f"Error finding pocket residues: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_sequence_from_pdb(pdb_path: str, chain_id: str = "A") -> str:
    """
    Extract protein sequence from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain to extract
    
    Returns:
        sequence: Single-letter amino acid sequence
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.SeqUtils import seq1
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    residues = []
                    for residue in chain:
                        if residue.id[0] == ' ':  # Standard residue
                            try:
                                residues.append(seq1(residue.resname))
                            except KeyError:
                                residues.append('X')
                    return ''.join(residues)
        
        logger.warning(f"Chain {chain_id} not found in {pdb_path}")
        return ""
        
    except Exception as e:
        logger.error(f"Error extracting sequence: {e}")
        return ""


def convert_cif_to_pdb(cif_path: str, pdb_path: str) -> bool:
    """Convert CIF to PDB format with STRICT column formatting for ProDy compatibility."""
    try:
        from Bio.PDB import MMCIFParser
        
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_path)
        
        # Manual PDB writing with strict formatting
        atom_serial = 1
        lines = []
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()[:1] if chain.get_id() else 'A'
                for residue in chain:
                    res_name = residue.get_resname()[:3]
                    res_seq = residue.get_id()[1] % 10000
                    
                    is_hetatm = residue.get_id()[0] != ' '
                    record_type = "HETATM" if is_hetatm else "ATOM  "
                    
                    for atom in residue:
                        x, y, z = atom.get_coord()
                        occupancy = atom.get_occupancy() if atom.get_occupancy() else 1.0
                        bfactor = atom.get_bfactor() if atom.get_bfactor() else 0.0
                        atom_name = atom.get_name()[:4]
                        element = atom.element.upper().strip() if atom.element else atom_name[0]
                        
                        # Format atom name (4 chars, special alignment)
                        if len(atom_name) < 4:
                            formatted_name = f" {atom_name:<3s}"
                        else:
                            formatted_name = atom_name[:4]
                        
                        # STRICT PDB format with proper column widths
                        # This prevents coordinate column merging
                        serial = atom_serial % 100000
                        line = (
                            f"{record_type:6s}"
                            f"{serial:5d} "
                            f"{formatted_name:4s}"
                            f" "  # AltLoc
                            f"{res_name:3s} "
                            f"{chain_id:1s}"
                            f"{res_seq:4d}"
                            f"    "  # iCode + padding
                            f"{x:8.3f}"
                            f"{y:8.3f}"
                            f"{z:8.3f}"
                            f"{occupancy:6.2f}"
                            f"{bfactor:6.2f}"
                            f"          "
                            f"{element:>2s}"
                        )
                        lines.append(line)
                        atom_serial += 1
                        
        lines.append("END")
        
        with open(pdb_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Converted {cif_path} to {pdb_path} (strict format)")
        return True
    except Exception as e:
        logger.error(f"CIF to PDB conversion failed: {e}")
        return False


# ============================================
# HalluDesign Runner
# ============================================

class HalluDesignRunner:
    """
    Iterative structure hallucination for pocket refinement.
    
    Workflow:
    1. Identify pocket residues (distance-based)
    2. Generate sequence variants with LigandMPNN (pocket only)
    3. Predict structures with AF3 ref-guided diffusion
    4. Select best structure as reference for next cycle
    5. Repeat for N cycles
    """
    
    def __init__(
        self,
        output_dir: str,
        ligand_smiles: str = LDOPA_SMILES,
        num_cycles: int = 5,
        num_sequences_per_cycle: int = 8,
        pocket_cutoff: float = 8.0,
        ref_time_steps: int = 50,
        mpnn_temperature: float = 0.3,
        chain_to_design: str = "A",
        ligand_chain: str = "B",
        af3_config: Dict[str, str] = None,
        protenix_config: Dict[str, Any] = None,
        use_protenix: bool = True,
        checkpoint_path: str = "/data/home/scvi041/run/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        closed_pdb: Optional[str] = None, # NEW: Closed Reference
        open_pdb: Optional[str] = None,   # NEW: Open Reference
        template_json_path: Optional[str] = None, # NEW: MSA Template
    ):
        """Initialize HalluDesign runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        
        self.closed_pdb = closed_pdb
        self.open_pdb = open_pdb
        self.template_json_path = template_json_path
        
        if self.closed_pdb: logger.info(f"HalluDesign using Closed Ref: {self.closed_pdb}")
        if self.open_pdb: logger.info(f"HalluDesign using Open Ref: {self.open_pdb}")
        
        self.closed_pdb = closed_pdb
        self.open_pdb = open_pdb
        
        if self.closed_pdb: logger.info(f"HalluDesign using Closed Ref: {self.closed_pdb}")
        if self.open_pdb: logger.info(f"HalluDesign using Open Ref: {self.open_pdb}")
        
        self.ligand_smiles = ligand_smiles
        self.num_cycles = num_cycles
        self.num_sequences_per_cycle = num_sequences_per_cycle
        self.pocket_cutoff = pocket_cutoff
        self.ref_time_steps = ref_time_steps
        self.mpnn_temperature = mpnn_temperature
        self.chain_to_design = chain_to_design
        self.ligand_chain = ligand_chain
        self.af3_config = af3_config or {}
        self.protenix_config = protenix_config or {}
        # We reuse 'use_protenix' flag to indicating "use structure guidance" but via AF3 JAX
        self.use_guided = use_protenix
        
        # JAX AF3 Model
        self.af3_model = None
        self._load_af3_model()
        
        # LigandMPNN
        # LigandMPNN
        self._load_mpnn_model()
        logger.info("LigandMPNN loaded successfully")
        
        logger.info(f"HalluDesignRunner initialized: {num_cycles} cycles, using JAX AF3 (Guided={self.use_guided})")
    
    def _load_mpnn_model(self):
        """Load LigandMPNN model."""
        try:
            logger.info(f"Loading LigandMPNN from: {self.checkpoint_path}")
            self.ligandmpnn_model, _ = load_ligandmpnn_model(
                self.checkpoint_path,
                device=self.device
            )
            self.ligandmpnn_model.eval()
        except Exception as e:
            logger.error(f"Failed to load LigandMPNN: {e}")
            raise
    
    def _load_af3_model(self):
        """Load AF3 JAX model from af3_model.py"""
        if not AF3_AVAILABLE:
             raise RuntimeError("AF3DesignerPack is required but could not be imported.")
        
        try:
            jax_cache = os.path.expanduser("~/.cache/jax")
            # AF3DesignerPack hardcodes model dir to ~/model or env var AF3_MODEL_DIR
            self.af3_model = AF3DesignerPack(jax_compilation_dir=jax_cache)
            logger.info("AF3DesignerPack loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load AF3DesignerPack: {e}") from e
            
    # _create_af3_input_json REMOVED - using af3_utils.create_af3_input_json instead
    
    def _run_af3_prediction(self, sequence: str, output_dir: str, name: str, ref_pdb_path: Optional[str] = None) -> Dict[str, Any]:
        """Run AF3 prediction (Pure or Guided).
        
        Args:
            sequence: Protein sequence
            output_dir: Output directory
            name: Job name
            ref_pdb_path: Reference file path (PDB or PKL) for guided diffusion
            
        Returns:
            Dictionary with success, pdb_path, cif_path, pkl_path
        """
        try:
            json_path = os.path.join(output_dir, f"{name}.json")
            
            # Use shared utility with template support
            from af3_utils import create_af3_input_json
            create_af3_input_json(
                sequence=sequence,
                ligand_smiles=self.ligand_smiles,
                name=name,
                chain_id=self.chain_to_design,
                ligand_id=self.ligand_chain,
                num_seeds=1,
                output_path=json_path,
                template_json_path=self.template_json_path # Use cached MSA
            )
            
            # CRITICAL CORRECTION:
            # If ref_pdb_path is present: use partial diffusion (e.g. 50 steps) from that structure.
            # If NO ref_pdb_path (Cycle 0): use FULL diffusion (200 steps) from noise.
            
            # Handle PDB -> PKL conversion for guided diffusion
            # AF3 guided diffusion requires tensors in a specific format (pkl)
            # If we only have a PDB, we must convert it using process_single_file
            if ref_pdb_path and ref_pdb_path.endswith('.pdb') and process_single_file:
                try:
                    ref_dir = os.path.dirname(ref_pdb_path)
                    ref_name = os.path.basename(ref_pdb_path)
                    
                    # Convert to PKL in the same directory
                    pkl_name = ref_name.replace('.pdb', '.pkl')
                    expected_pkl_path = os.path.join(ref_dir, pkl_name)
                    
                    if not os.path.exists(expected_pkl_path):
                        logger.info(f"Converting reference PDB to PKL: {ref_pdb_path}")
                        # process_single_file args: (pdb_file, input_dir, output_dir)
                        # insert=None (no PTM insertion)
                        success, _, error = process_single_file((ref_name, ref_dir, ref_dir), None)
                        if success:
                            ref_pdb_path = expected_pkl_path # Use the generated PKL
                            logger.info(f"Using converted reference PKL: {ref_pdb_path}")
                        else:
                            logger.error(f"PDB->PKL conversion failed: {error}")
                    else:
                        ref_pdb_path = expected_pkl_path # Use existing PKL
                        logger.info(f"Using existing reference PKL: {ref_pdb_path}")
                        
                except Exception as e:
                    logger.error(f"Error during PDB->PKL conversion: {e}")
            
            steps = self.ref_time_steps if ref_pdb_path else 200
            samples = 1 
            
            # PKL dump path for saving atom positions in correct tensor format
            # This pkl can be used as reference in future cycles
            pkl_dump_path = os.path.join(output_dir, f"{name}_ref.pkl")
            
            logger.info(f"Running JAX AF3 for {name} (Ref steps: {steps})")
            
            # If conversion failed and we still have a PDB, AF3 might fail with shape mismatch
            # But process_single_file should handle it now.
            
            self.af3_model.single_file_process(
                json_path=json_path,
                out_dir=output_dir,
                ref_pdb_path=ref_pdb_path,
                ref_time_steps=steps,
                num_samples=samples,
                ref_pkl_dump_path=pkl_dump_path  # Save pkl for future reference!
            )
            
            # Locate output PDB (convert from CIF if needed)
            cif_files = list(Path(output_dir).glob("**/*.cif"))
            final_pkl = pkl_dump_path if os.path.exists(pkl_dump_path) else None
            
            if not cif_files:
                 pdb_files = list(Path(output_dir).glob("**/*.pdb"))
                 if pdb_files:
                     final_pdb = str(pdb_files[0])
                     final_cif = None
                 else:
                     raise FileNotFoundError("No output CIF/PDB found from AF3 prediction")
            else:
                 final_cif = str(cif_files[0])
                 final_pdb = final_cif.replace('.cif', '.pdb')
                 
                 if not os.path.exists(final_pdb):
                     success = convert_cif_to_pdb(final_cif, final_pdb)
                     if not success:
                         logger.warning("Fell back to CIF as PDB conversion failed")
                         final_pdb = final_cif 
            
            return {
                'success': True,
                'pdb_path': final_pdb,
                'cif_path': final_cif,
                'pkl_path': final_pkl,  # NEW: Return pkl path for guided diffusion
                'iptm': 0.5, 'ptm': 0.5, 'mean_pae': 10.0
            }
            
        except Exception as e:
            logger.error(f"AF3 prediction failed: {e}")
            raise RuntimeError(f"AF3 prediction failed for {name}: {e}") from e
    
    def run_cycle(
        self,
        input_pdb: str,
        cycle_num: int,
        reference_ref: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
        """
        Run one HalluDesign cycle.
        
        Args:
            input_pdb: Input PDB structure (for pocket definition)
            cycle_num: Cycle number
            reference_ref: Reference file (PDB or PKL) for guided diffusion
        
        Returns:
            best_result: Best result from this cycle
            best_pdb: Path to best PDB structure (for next cycle input)
            best_ref: Path to best reference (PKL or PDB) for next cycle guidance
        """
        cycle_dir = self.output_dir / f"cycle_{cycle_num}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"HalluDesign Cycle {cycle_num}/{self.num_cycles}")
        
        # 1. Find pocket residues (Always use PDB!)
        pocket_residues = find_pocket_residues(
            input_pdb,
            ligand_chain=self.ligand_chain,
            cutoff=self.pocket_cutoff
        )
        
        if not pocket_residues:
            logger.warning(f"No pocket residues found in cycle {cycle_num}")
            return {}, None, None
            
        logger.info(f"Found {len(pocket_residues)} pocket residues within {self.pocket_cutoff}Å")
        
        # 2. Generate sequences with LigandMPNN
        sequences = {}
        try:
            # Generate sequences using LigandMPNN
            mpnn_dir = cycle_dir / "mpnn"
            mpnn_dir.mkdir(exist_ok=True)
            
            logger.info(f"Designing {len(pocket_residues)} pocket residues")
            
            mpnn_result = generate_sequences_with_ligandmpnn(
                model=self.ligandmpnn_model,
                pdb_path=input_pdb,
                chain_to_design=self.chain_to_design,
                fixed_residues=None, 
                redesigned_residues=pocket_residues,
                num_variants=self.num_sequences_per_cycle,
                temperature=self.mpnn_temperature,
                use_ligand_context=True,
                device=self.device
            )
            
            sequences = mpnn_result.get('sequences', {})
            logger.info(f"Generated {len(sequences)} sequences from LigandMPNN")
            
        except Exception as e:
            logger.error(f"LigandMPNN generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}, None, None
        
        # 3. Evaluate each sequence with structure prediction
        results = []
        for seq_idx, seq in enumerate(sequences):
            seq_dir = cycle_dir / f"seq_{seq_idx}"
            seq_dir.mkdir(parents=True, exist_ok=True)
            
            name = f"cycle{cycle_num}_seq{seq_idx}"
            
            # Determine structure prediction strategy
            # Use guidance if reference is provided and cycle > 0
            use_guidance = (reference_ref is not None and cycle_num > 0 and self.use_guided)
            
            logger.info(f"  Cycle {cycle_num}, Seq {seq_idx}: JAX AF3 (Guided={use_guidance})...")
            
            pred_result = self._run_af3_prediction(
                sequence=seq,
                output_dir=str(seq_dir),
                name=name,
                ref_pdb_path=reference_ref if use_guidance else None
            )
            
            af3_result = pred_result
            
            # Helper to extract coords (uses module-level imports from af3_utils)
            def get_coords(res_dict):
                p = res_dict.get('cif_path') or res_dict.get('pdb_path')
                if not p or not os.path.exists(p): return None
                if p.endswith('.cif'): return extract_ca_coords_from_cif(p, self.chain_to_design)
                return extract_ca_coords_from_pdb(p, self.chain_to_design)
                
            from reward_utils import calculate_rmsd # Import here to avoid circular
            
            # Calculate RMSDs for Reference-Guided Logic
            rmsd = None # Bound vs Closed Ref
            open_rmsd = None # Unbound vs Open Ref (We don't have Unbound simulation here though... assuming Bound approx?)
            
            # NOTE: HalluDesign currently only runs ONE simulation per sequence (Bound State)
            # To properly calculate Open/Decoy rewards, we would need to run 3 simulations per seq!
            # That is too expensive for inner loop.
            # COMPROMISE: We only check "Bound vs Closed Ref". 
            # If it matches Closed Ref, it's a good candidate backbone.
            # We assume Specificity is handled by the outer GRPO loop.
            
            # 1. Calculate Bound vs Closed Ref
            if af3_result.get('success', False):
                bound_coords = get_coords(af3_result)
                
                # Load Closed Ref Coords (Cache this?)
                # For simplicity, load if needed
                if self.closed_pdb and os.path.exists(self.closed_pdb) and bound_coords is not None:
                     p = self.closed_pdb
                     if p.endswith('.cif'): ref_c = extract_ca_coords_from_cif(p, self.chain_to_design)
                     else: ref_c = extract_ca_coords_from_pdb(p, self.chain_to_design)
                     
                     if ref_c is not None:
                         rmsd = calculate_rmsd(ref_c, bound_coords, align=True)
                         logger.info(f"    RMSD to Closed Ref: {rmsd:.2f}Å")

            # Load Open Ref Coords (Cache this potentially)
            open_ref_coords = None
            if self.open_pdb and os.path.exists(self.open_pdb):
                 p = self.open_pdb
                 if p.endswith('.cif'): open_ref_coords = extract_ca_coords_from_cif(p, self.chain_to_design)
                 else: open_ref_coords = extract_ca_coords_from_pdb(p, self.chain_to_design)

            # Calculate reward
            if af3_result.get('success', False):
                reward, reward_info = calculate_comprehensive_reward(
                    summary_data=af3_result,
                    rmsd=rmsd,
                    open_rmsd=None,
                    unbound_ptm=None,
                    decoy_stats=None,
                    bound_coords=bound_coords,   # NEW: Pass bound coords
                    open_ref_coords=open_ref_coords # NEW: Pass open ref coords
                )
                
                # NOTE: Since open_rmsd is None, open_ref_reward will be 0.0 or 1.0 depending on logic.
                # In strict logic, it might be 0. We should fix reward_utils to handle partial data?
                # Or we accept that HalluDesign only optimizes for Positive Design (Closing).
                # User Goal: "Select best start". A good closer is a good start.
                pass
                
                result_entry = {
                    'cycle': cycle_num,
                    'seq_idx': seq_idx,
                    'sequence': seq,
                    'reward': reward,
                    'iptm': reward_info.get('iptm', 0),
                    'ptm': reward_info.get('ptm', 0),
                    'pae': reward_info.get('mean_pae', 31.75),
                    'has_clash': reward_info.get('has_clash', True),
                    'cif_path': af3_result.get('cif_path'),
                    'pdb_path': af3_result.get('pdb_path'),
                    'pkl_path': af3_result.get('pkl_path'),
                }
                
                results.append(result_entry)
                
                logger.info(f"    Reward: {reward:.4f}, iPTM: {reward_info.get('iptm', 0):.3f}")
            else:
                logger.warning(f"    AF3 prediction failed for seq {seq_idx}")
        
        # 4. Select best result
        if not results:
            logger.warning(f"No valid results in cycle {cycle_num}")
            return {}, None, None
        
        best_result = max(results, key=lambda x: x['reward'])
        best_cif = best_result.get('cif_path')
        
        # Convert CIF to PDB for user visualization/output AND next cycle input
        best_pdb_path = best_result.get('pdb_path')
        if not best_pdb_path and best_cif and os.path.exists(best_cif):
            best_pdb_path = best_cif.replace('.cif', '.pdb')
            if not os.path.exists(best_pdb_path):
                convert_cif_to_pdb(best_cif, best_pdb_path)
        
        # Determine best reference for next cycle's GUIDANCE
        # Priority: PKL > PDB (PKL has correct tensors for AF3 guided diffusion)
        best_ref_path = best_result.get('pkl_path') or best_pdb_path
        
        logger.info(f"Cycle {cycle_num} best: reward={best_result['reward']:.4f}")
        logger.info(f"Selected reference for next cycle guidance: {best_ref_path}")
        logger.info(f"Selected input for next cycle pocket finding: {best_pdb_path}")
        
        # Save cycle results
        cycle_results_path = cycle_dir / "cycle_results.json"
        with open(cycle_results_path, 'w') as f:
            json.dump({
                'cycle': cycle_num,
                'results': results,
                'best': best_result,
                'best_ref': best_ref_path,
                'best_pdb': best_pdb_path
            }, f, indent=2, default=str)
        
        return best_result, best_pdb_path, best_ref_path
    
    def run(self, initial_pdb: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Run complete HalluDesign workflow.
        
        Args:
            initial_pdb: Starting PDB structure
        
        Returns:
            overall_best: Best result across all cycles
            overall_best_pdb: Path to overall best structure
        """
        logger.info("=" * 60)
        logger.info("Starting HalluDesign Workflow")
        logger.info("=" * 60)
        
        current_pdb = initial_pdb
        current_ref = None # No reference for Cycle 0
        
        overall_best = None
        overall_best_pdb = None
        overall_best_reward = -1.0
        
        all_cycle_results = []
        
        for cycle in range(self.num_cycles):
            # Run cycle
            cycle_result, cycle_pdb, cycle_ref = self.run_cycle(
                input_pdb=current_pdb,
                cycle_num=cycle,
                reference_ref=current_ref
            )
            
            all_cycle_results.append(cycle_result)
            
            if not cycle_result or 'reward' not in cycle_result:
                logger.warning(f"Cycle {cycle} produced no valid results, stopping early")
                break
            
            # Update overall best
            cycle_reward = cycle_result.get('reward', 0)
            if cycle_reward > overall_best_reward:
                overall_best = cycle_result
                overall_best_reward = cycle_reward
                overall_best_pdb = cycle_pdb
            
            # Prepare for next cycle
            if cycle_pdb and os.path.exists(cycle_pdb):
                current_pdb = cycle_pdb  # PDB for pocket finding
                current_ref = cycle_ref  # PKL (or PDB) for guided diffusion
            else:
                logger.warning(f"Cycle {cycle} did not produce a valid PDB, stopping.")
                break
                
        # Save final results
        final_results = {
            'num_cycles': self.num_cycles,
            'all_cycles': all_cycle_results,
            'overall_best': overall_best,
            'overall_best_pdb': str(overall_best_pdb) if overall_best_pdb else None,
            'overall_best_reward': overall_best_reward,
        }
        
        results_path = self.output_dir / "hallu_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info(f"HalluDesign Complete")
        logger.info(f"Overall Best Reward: {overall_best_reward:.4f}")
        logger.info(f"Best PDB: {overall_best_pdb}")
        logger.info("=" * 60)
        
        return overall_best or {}, overall_best_pdb


# ============================================
# Convenience Function
# ============================================

def hallu_design_phase(
    input_pdb: str,
    output_dir: str,
    ligand_smiles: str = LDOPA_SMILES,
    num_cycles: int = 5,
    num_sequences_per_cycle: int = 8,
    ref_time_steps: int = 50,
    pocket_cutoff: float = 8.0,
    af3_config: Dict[str, str] = None,
    protenix_config: Dict[str, Any] = None,
    use_protenix: bool = True,
    checkpoint_path: str = None,
    # New: Template Support
    template_json_path: Optional[str] = None,
    # New: References
    closed_pdb: Optional[str] = None,
    open_pdb: Optional[str] = None
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Run HalluDesign pocket refinement phase.
    
    This is a convenience wrapper for the HalluDesignRunner class.
    
    Args:
        input_pdb: Input PDB structure
        output_dir: Output directory
        ligand_smiles: Ligand SMILES
        num_cycles: Number of refinement cycles
        num_sequences_per_cycle: Sequences per cycle
        ref_time_steps: AF3/Protenix ref-guided diffusion steps
        pocket_cutoff: Pocket distance cutoff (A)
        af3_config: AF3 configuration
        protenix_config: Protenix configuration
        use_protenix: Whether to use Protenix for HalluDesign (default: True)
        checkpoint_path: Path to LigandMPNN weights
    
    Returns:
        best_result: Best result dict
        best_pdb: Path to best PDB
    """
    runner = HalluDesignRunner(
        output_dir=output_dir,
        ligand_smiles=ligand_smiles,
        num_cycles=num_cycles,
        num_sequences_per_cycle=num_sequences_per_cycle,
        pocket_cutoff=pocket_cutoff,
        ref_time_steps=ref_time_steps,
        af3_config=af3_config,
        protenix_config=protenix_config,
        use_protenix=use_protenix,
        checkpoint_path=checkpoint_path,
        template_json_path=template_json_path,
        closed_pdb=closed_pdb,
        open_pdb=open_pdb
    )
    
    return runner.run(input_pdb)


# ============================================
# Main (for testing)
# ============================================

if __name__ == "__main__":
    # Test pocket detection
    test_pdb = "/data/home/scvi041/run/HalluMPNN/inputs/3lft.pdb"
    
    if os.path.exists(test_pdb):
        pocket = find_pocket_residues(test_pdb, cutoff=8.0)
        print(f"Found {len(pocket)} pocket residues")
        print(f"First 10: {pocket[:10]}")
        
        seq = get_sequence_from_pdb(test_pdb)
        print(f"Sequence length: {len(seq)}")
    else:
        print(f"Test PDB not found: {test_pdb}")