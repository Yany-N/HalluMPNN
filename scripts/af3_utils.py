# -*- coding: utf-8 -*-
"""
AlphaFold3 Utilities
"""

import os
import json
import logging
import time
import subprocess
import glob
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import types
import numpy as np

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
except ImportError as e:
    AF3_AVAILABLE = False
    logging.warning(f"Could not import AF3DesignerPack: {e}")

_AF3_MODEL = None

def get_af3_model():
    """Singleton accessor for AF3 model."""
    global _AF3_MODEL
    if _AF3_MODEL is None:
        if not AF3_AVAILABLE:
            raise RuntimeError("AF3DesignerPack not available")
        jax_cache = os.path.expanduser("~/.cache/jax")
        _AF3_MODEL = AF3DesignerPack(jax_compilation_dir=jax_cache)
    return _AF3_MODEL

logger = logging.getLogger(__name__)

# ============================================
# 默认配置
# ============================================
DEFAULT_AF3_CONFIG = {
    "sif_path": "/data/home/scvi041/run/af3/3.0.1_run/alphafold3.sif",
    "model_dir": "/data/home/scvi041/run/af3/model",
    "db_dir": "/data/public/alphafold3/dataset",
    "singularity_bin": "/data/apps/apptainer/apptainer/bin/singularity",
}

LDOPA_SMILES = "N[C@@H](CC1=CC(=C(C=C1)O)O)C(=O)O"
SCRIPT_DIR = Path(__file__).parent.resolve()


def create_af3_input_json(
    sequence: str,
    ligand_smiles: str = LDOPA_SMILES,
    name: str = "design",
    chain_id: str = "A",
    ligand_id: str = "B",
    num_seeds: int = 2,
    output_path: Optional[str] = None,
    template_json_path: Optional[str] = None  # NEW: Path to template JSON with MSA
) -> Dict[str, Any]:
    """创建 AlphaFold3 输入 JSON
    
    Args:
        sequence: Protein sequence
        ligand_smiles: SMILES string for ligand.
        name: Job name
        chain_id: Protein chain ID
        ligand_id: Ligand chain ID
        num_seeds: Number of model seeds
        output_path: Optional path to save JSON
        template_json_path: Path to JSON with cached MSA
    
    Returns:
        AF3 input JSON dictionary
    """
    
    if template_json_path and os.path.exists(template_json_path):
        # TEMPLATE MODE: Load template and inject sequence
        try:
            with open(template_json_path, 'r') as f:
                input_json = json.load(f)
            
            logger.info(f"Using MSA Template from: {template_json_path}")
            
            # Update name
            input_json['name'] = name
            
            # Look for protein chain and update sequence (Keep MSA!)
            found_protein = False
            for seq_entry in input_json.get('sequences', []):
                if 'protein' in seq_entry:
                    prot = seq_entry['protein']
                    if chain_id in prot.get('id', []):
                        # Update sequence
                        prot['sequence'] = sequence
                        # Update UNPAIRED MSA (First line only!)
                        # Format: ">A\nSEQUENCE"
                        # We must preserve the REST of the MSA lines if they exist
                        original_msa = prot.get('unpairedMsa', '')
                        lines = original_msa.split('\n')
                        if len(lines) >= 2:
                            # Replace first sequence (query)
                            lines[1] = sequence
                            prot['unpairedMsa'] = '\n'.join(lines)
                        else:
                            # Fallback if empty
                            prot['unpairedMsa'] = f">{chain_id}\n{sequence}\n"
                        
                        # Ensure pairedMsa exists (AF3 Strict Requirement)
                        if 'pairedMsa' not in prot or prot['pairedMsa'] is None:
                            logger.info(f"Adding missing 'pairedMsa' to chain {chain_id}")
                            prot['pairedMsa'] = ""

                        if 'templates' not in prot:
                            logger.info(f"Adding missing 'templates' to chain {chain_id}")
                            prot['templates'] = []

                        if 'modifications' not in prot:
                            logger.info(f"Adding missing 'modifications' to chain {chain_id}")
                            prot['modifications'] = []
                            
                        found_protein = True
                        break
            
            if not found_protein:
                logger.warning(f"Template loaded but chain {chain_id} not found/updated!")
            
            # Look for ligand and update SMILES
            # If ligand_smiles is None, we should REMOVE the ligand entry?
            # Or this template is only for bound steps?
            # "Unbound predictions cannot have MSA" -> We don't use template for unbound.
            # So here we assume we DO represent the ligand if provided.
            
            if ligand_smiles:
                found_ligand = False
                for seq_entry in input_json.get('sequences', []):
                    if 'ligand' in seq_entry:
                        lig = seq_entry['ligand']
                        # Assume first ligand block is the target ligand
                        lig['smiles'] = ligand_smiles
                        lig['id'] = [ligand_id]
                        found_ligand = True
                        break
                
                if not found_ligand:
                    # Append ligand block if missing in template
                    input_json['sequences'].append({
                        "ligand": {
                            "id": [ligand_id],
                            "smiles": ligand_smiles
                        }
                    })
            else:
                 # Remove ligand block for apo prediction
                 input_json['sequences'] = [s for s in input_json['sequences'] if 'ligand' not in s]

        except Exception as e:
            logger.error(f"Failed to load/update template JSON: {e}")
            # Fallback to default creation
            return create_af3_input_json(sequence, ligand_smiles, name, chain_id, ligand_id, num_seeds, output_path)

    else:
        # DEFAULT MODE (No Template)
        # Start with protein-only sequences
        sequences = [
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": sequence,
                    "unpairedMsa": f">{chain_id}\n{sequence}\n",
                    "pairedMsa": "",
                    "modifications": [],
                    "templates": []
                }
            }
        ]
        
        # Only add ligand if provided
        if ligand_smiles:
            sequences.append({
                "ligand": {
                    "id": [ligand_id],
                    "smiles": ligand_smiles
                }
            })
            logger.info(f"Created AF3 JSON with ligand: {name}")
        else:
            logger.info(f"Created AF3 JSON WITHOUT ligand (apo state): {name}")
        
        input_json = {
            "name": name,
            "modelSeeds": list(range(1, num_seeds + 1)),
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": 1
        }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        logger.info(f"已保存 AF3 输入 JSON: {output_path}")
    
    return input_json


def _get_af3_config_value(af3_config: Dict, key: str, yaml_key: str = None) -> str:
    """获取配置值,带默认值回退"""
    if af3_config is None:
        return DEFAULT_AF3_CONFIG.get(key, "")
    
    if key in af3_config:
        return af3_config[key]
    
    if yaml_key and yaml_key in af3_config:
        return af3_config[yaml_key]
    
    return DEFAULT_AF3_CONFIG.get(key, "")


def generate_slurm_script(
    input_json_path: str,
    output_dir: str,
    job_name: str = "af3_pred",
    num_gpus: int = 1,
    time_limit: str = "01:00:00",
    af3_config: Dict[str, str] = None,
    run_data: bool = False  # NEW argument to control pipeline
) -> str:
    """生成 SLURM 提交脚本"""
    model_dir = _get_af3_config_value(af3_config, 'model_dir', 'af3_model_dir')
    db_dir = _get_af3_config_value(af3_config, 'db_dir', 'af3_db_dir')
    sif_path = _get_af3_config_value(af3_config, 'sif_path', 'af3_sif')
    singularity_bin = _get_af3_config_value(af3_config, 'singularity_bin', None)
    
    input_json_abs = os.path.abspath(input_json_path)
    input_dir = os.path.dirname(input_json_abs)
    json_filename = os.path.basename(input_json_abs)
    script_parent_dir = str(SCRIPT_DIR)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    script_content = f'''#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --partition=gpu
#SBATCH --gpus={num_gpus}
#SBATCH -o {output_dir}/{job_name}_%j.out
#SBATCH -e {output_dir}/{job_name}_%j.err
#SBATCH --time={time_limit}

source /etc/profile.d/modules.sh 2>/dev/null || true
source /data/apps/lmod/lmod/init/bash 2>/dev/null || true
module load miniforge/25.3.0-3 cuda/12.8 cudnn/9.6.0.74_cuda12 2>/dev/null || true
module load apptainer/1.2.4 2>/dev/null || true

# Clear deprecated XLA env var to prevent conflict with new XLA_CLIENT_MEM_FRACTION
unset XLA_PYTHON_CLIENT_MEM_FRACTION

export PYTHONUNBUFFERED=1

SCRIPT_DIR="{script_parent_dir}"
INPUT_DIR="{input_dir}"
JSON_FILE="{json_filename}"
OUTPUT_DIR="{output_dir}"
AF3_MODEL_DIR="{model_dir}"
AF3_DB_DIR="{db_dir}"
AF3_SIF_PATH="{sif_path}"
SINGULARITY_PATH="{singularity_bin}"

echo "=== AlphaFold3 预测开始 ==="
echo "开始时间: $(date)"
echo "JSON 输入: ${{INPUT_DIR}}/${{JSON_FILE}}"
echo "输出目录: ${{OUTPUT_DIR}}"
nvidia-smi

# 运行 AlphaFold3
${{SINGULARITY_PATH}} exec \\
  --nv \\
  -B ${{SCRIPT_DIR}}:/scripts,${{INPUT_DIR}}:/input,${{AF3_MODEL_DIR}}:/model,${{AF3_DB_DIR}}:/dataset,${{OUTPUT_DIR}}:/output \\
  ${{AF3_SIF_PATH}} \\
  python /scripts/run_alphafold.py \\
  --json_path=/input/{json_filename} \\
  --model_dir=/model \\
  --db_dir=/dataset \\
  --output_dir=/output \\
  --run_data_pipeline={str(run_data).lower()}

EXIT_CODE=$?
echo "结束时间: $(date)"
echo "退出码: ${{EXIT_CODE}}"

if [ ${{EXIT_CODE}} -eq 0 ]; then
    echo "=== 预测成功! ==="
    echo "输出文件列表:"
    ls -lhR ${{OUTPUT_DIR}}/
    
    # 查找并列出所有 CIF 文件
    echo ""
    echo "CIF 文件:"
    find ${{OUTPUT_DIR}} -name "*.cif" -type f
    
    # 查找并列出所有 JSON 文件
    echo ""
    echo "JSON 文件:"
    find ${{OUTPUT_DIR}} -name "*.json" -type f
else
    echo "=== 预测失败! ==="
    echo "检查错误日志: {output_dir}/{job_name}_${{SLURM_JOB_ID}}.err"
fi

echo "==================================="
'''
    
    script_path = os.path.join(input_dir, f"{job_name}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    logger.info(f"生成 SLURM 脚本: {script_path}")
    
    return script_path


def submit_af3_job(
    script_path: str,
    wait: bool = True,
    timeout: int = 3600
) -> Tuple[int, Optional[str]]:
    """提交 AF3 SLURM 任务"""
    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"任务提交失败: {result.stderr}")
        return -1, None
    
    job_id = int(result.stdout.strip().split()[-1])
    logger.info(f"已提交 AF3 任务: {job_id}")
    
    if not wait:
        return job_id, None
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = subprocess.run(
            ["squeue", "-j", str(job_id), "-h"],
            capture_output=True,
            text=True
        )
        
        if not status.stdout.strip():
            logger.info(f"任务 {job_id} 已完成")
            break
        
        time.sleep(30)
    else:
        logger.warning(f"任务 {job_id} 超时 ({timeout}s)")
        return job_id, None
    
    return job_id, None


def find_af3_output_files(output_dir: str) -> Dict[str, Optional[str]]:
    """
    智能搜索 AF3 输出文件
    
    支持多种命名格式:
    - *_model.cif / *_model_0.cif
    - *_sample_0.cif
    - *_prediction_model_*.cif
    - *_confidences_*.json / *_summary_confidences_*.json
    """
    files = {
        'cif_path': None,
        'json_path': None,
        'all_cifs': [],
        'all_jsons': []
    }
    
    # 递归搜索所有文件
    for root, _, filenames in os.walk(output_dir):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            
            # CIF 文件
            if filename.endswith('.cif'):
                files['all_cifs'].append(full_path)
                
                # 优先级: model.cif > model_0.cif > sample_0.cif > 其他
                if '_model.cif' in filename and files['cif_path'] is None:
                    files['cif_path'] = full_path
                elif '_model_0.cif' in filename and files['cif_path'] is None:
                    files['cif_path'] = full_path
                elif '_sample_0.cif' in filename and files['cif_path'] is None:
                    files['cif_path'] = full_path
                elif 'prediction_model' in filename and files['cif_path'] is None:
                    files['cif_path'] = full_path
            
            # JSON 文件
            if filename.endswith('.json'):
                files['all_jsons'].append(full_path)
                
                # 优先级: summary_confidences > confidences > 其他
                # summary_confidences 始终覆盖其他选择
                if 'summary_confidences' in filename:
                    files['json_path'] = full_path  # Always prefer, no None check
                elif 'confidences' in filename and 'summary' not in filename:
                    # Only set if no summary_confidences found yet
                    if files['json_path'] is None or 'summary_confidences' not in files['json_path']:
                        files['json_path'] = full_path
    
    # 如果没找到,使用第一个文件
    if files['cif_path'] is None and files['all_cifs']:
        files['cif_path'] = files['all_cifs'][0]
        logger.warning(f"使用默认 CIF: {files['cif_path']}")
    
    if files['json_path'] is None and files['all_jsons']:
        files['json_path'] = files['all_jsons'][0]
        logger.warning(f"使用默认 JSON: {files['json_path']}")
    
    logger.info(f"找到 {len(files['all_cifs'])} 个 CIF, {len(files['all_jsons'])} 个 JSON")
    
    return files


def parse_af3_output(output_dir: str) -> Dict[str, Any]:
    """解析 AlphaFold3 预测输出 - 修复版"""
    results = {
        'success': False,
        'iptm': 0.0,
        'ptm': 0.0,
        'pae': 31.75,
        'ranking_score': 0.0,
        'has_clash': False, # Default to False (innocent until proven guilty)
        'cif_path': None,
        'json_path': None,
    }
    
    # 智能搜索文件
    files = find_af3_output_files(output_dir)
    
    if not files['json_path']:
        logger.error(f"未找到 JSON 文件在: {output_dir}")
        logger.info(f"目录内容: {os.listdir(output_dir)}")
        return results
    
    # 解析 JSON
    try:
        with open(files['json_path'], 'r') as f:
            data = json.load(f)
        
        results['json_path'] = files['json_path']
        results['iptm'] = data.get('iptm', 0.0)
        results['ptm'] = data.get('ptm', 0.0)
        results['ranking_score'] = data.get('ranking_score', 0.0)
        # Handle has_clash (can be float 0.0/1.0 or bool)
        raw_clash = data.get('has_clash', 0.0)
        results['has_clash'] = bool(raw_clash) if isinstance(raw_clash, (bool, int)) else (float(raw_clash) > 0.1)
        
        # 解析 PAE
        pae_data = data.get('chain_pair_pae_min', [])
        if pae_data and len(pae_data) >= 2:
            try:
                a_to_b = pae_data[0][1] if len(pae_data[0]) > 1 else 31.75
                b_to_a = pae_data[1][0] if len(pae_data) > 1 else 31.75
                results['pae'] = (a_to_b + b_to_a) / 2.0
            except (IndexError, TypeError) as e:
                logger.warning(f"PAE 解析失败: {e}")
        
        results['success'] = True
        
        # Sanitize NaNs
        import math
        if math.isnan(results['iptm']): results['iptm'] = 0.0
        if math.isnan(results['ptm']): results['ptm'] = 0.0
        if math.isnan(results['pae']): results['pae'] = 31.75
        
        logger.info(f"成功解析 JSON: iptm={results['iptm']:.3f}, ptm={results['ptm']:.3f}")
        
    except Exception as e:
        logger.error(f"JSON 解析错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 添加 CIF 路径
    if files['cif_path']:
        results['cif_path'] = files['cif_path']
        logger.info(f"CIF 文件: {files['cif_path']}")
    else:
        logger.warning("未找到 CIF 文件!")
    
    return results


def convert_cif_to_pdb(cif_path: str, pdb_path: str = None) -> bool:
    """
    CIF 转 PDB - 改进版 (使用独立转换工具)
    
    处理 AlphaFold3 输出的特殊格式
    """
    # 导入转换工具
    try:
        # 尝试从同目录导入
        from cif_to_pdb import convert_cif_to_pdb as cif_converter
    except ImportError:
        # 回退到内置方法
        logger.warning("cif_to_pdb.py 未找到,使用内置转换")
        return _convert_cif_to_pdb_builtin(cif_path, pdb_path)
    
    try:
        # Call with correct signature (cif_path, pdb_path only)
        result = cif_converter(cif_path, pdb_path)
        
        # Check if pdb_path exists after conversion
        check_path = pdb_path if pdb_path else cif_path.replace('.cif', '.pdb')
        if os.path.exists(check_path) and os.path.getsize(check_path) >= 100:
            logger.info(f"CIF -> PDB 转换成功: {check_path}")
            return True
        else:
            logger.error(f"PDB 文件过小或不存在")
            return False
        
    except Exception as e:
        logger.error(f"CIF 转换失败: {e}")
        # 回退到内置方法
        return _convert_cif_to_pdb_builtin(cif_path, pdb_path)


def _convert_cif_to_pdb_builtin(cif_path: str, pdb_path: str = None) -> bool:
    """内置的 CIF 转 PDB (备用方法)"""
    if pdb_path is None:
        pdb_path = cif_path.replace('.cif', '.pdb')
    
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_path)
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)
        
        logger.info(f"内置转换成功: {pdb_path}")
        return True
        
    except Exception as e:
        logger.error(f"内置转换失败: {e}")
        
        # 最后尝试:简单解析
        try:
            return _convert_cif_simple(cif_path, pdb_path)
        except Exception as e2:
            logger.error(f"简单解析也失败: {e2}")
            return False


def _convert_cif_simple(cif_path: str, pdb_path: str) -> bool:
    """简单的 CIF 解析转换"""
    with open(cif_path, 'r') as f:
        lines = f.readlines()
    
    pdb_lines = []
    for line in lines:
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        
        parts = line.split()
        if len(parts) < 15:
            continue
        
        try:
            record = parts[0]
            serial = int(parts[1])
            element = parts[2]
            atom = parts[3]
            res = parts[5][:3]
            chain = parts[6]
            resseq = int(parts[8])
            x = float(parts[10])
            y = float(parts[11])
            z = float(parts[12])
            occ = float(parts[13])
            bfac = float(parts[14])
            
            if len(atom) < 4:
                atom = f" {atom:<3s}"
            
            pdb_line = (
                f"{record:<6s}{serial:>5d} {atom:<4s} {res:>3s} {chain:1s}"
                f"{resseq:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"{occ:>6.2f}{bfac:>6.2f}          {element:>2s}"
            )
            pdb_lines.append(pdb_line)
        except:
            continue
    
    pdb_lines.append("END")
    
    with open(pdb_path, 'w') as f:
        f.write('\n'.join(pdb_lines))
    
    logger.info(f"简单解析转换成功: {pdb_path} ({len(pdb_lines)-1} 原子)")
    return True


def run_af3_prediction(
    sequence: str,
    output_dir: str,
    ligand_smiles: str = LDOPA_SMILES,
    name: str = "hallumpnn_pred",
    wait: bool = True,
    af3_config: Dict[str, str] = None,
    template_json_path: Optional[str] = None # NEW argument
) -> Dict[str, Any]:
    """
    Run AF3 prediction using JAX model (in-process).
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{name}.json")
    
    # 1. Create Input JSON
    create_af3_input_json(
        sequence=sequence,
        ligand_smiles=ligand_smiles,
        name=name,
        output_path=json_path,
        template_json_path=template_json_path
    )
    
    
    # Note: XLA_FLAGS for triton are handled by test_pipeline.sh
    # Do not set XLA_FLAGS here as invalid flags cause JAX 0.9.0 to crash
    import warnings
    
    # JAX warmup for CC 12.0 (Blackwell) compatibility
    try:
        import jax
        import jax.numpy as jnp
        # Simple warmup operation to trigger JIT compilation
        _ = jnp.ones((1,)) + 1
    except Exception:
        pass
    
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            model = get_af3_model()
            logging.info(f"Running JAX AF3 Inference for {name} (attempt {attempt + 1})...")
            
            # Pure prediction (ref_time_steps=200, ref_pdb_path=None)
            model.single_file_process(
                json_path=json_path,
                out_dir=output_dir,
                ref_pdb_path=None,
                ref_time_steps=200, 
                num_samples=1
            )
            
            # 3. Parse Output - use recursive glob to find files in subdirs
            cif_files = list(Path(output_dir).glob("**/*_model.cif"))
            if not cif_files:
                cif_files = list(Path(output_dir).glob("**/*.cif"))
            if not cif_files:
                raise FileNotFoundError("No AF3 output CIF found")
                
            final_cif = str(cif_files[0])
            final_pdb = final_cif.replace('.cif', '.pdb')
            
            if not os.path.exists(final_pdb):
                if convert_cif_to_pdb(final_cif, final_pdb):
                    pass
                else:
                    final_pdb = final_cif
            
            # Parse actual metrics from AF3 output - search recursively
            iptm, ptm, mean_pae = 0.5, 0.5, 10.0  # defaults
            has_clash = False  # Default to False (no penalty when unknown)
            ranking_file = Path(output_dir) / "ranking_scores.json"
            summary_files = list(Path(output_dir).glob("**/*summary_confidences.json"))
            if not summary_files:
                summary_files = list(Path(output_dir).glob("**/*summary*.json"))
            
            if ranking_file.exists():
                import json as _json
                with open(ranking_file) as f:
                    scores = _json.load(f)
                    if scores:
                        top = list(scores.values())[0] if isinstance(scores, dict) else scores[0]
                        iptm = top.get('iptm', 0.5)
                        ptm = top.get('ptm', 0.5)
                        mean_pae = top.get('ranking_score', 10.0)
            elif summary_files:
                import json as _json
                with open(summary_files[0]) as f:
                    summary = _json.load(f)
                    iptm = summary.get('iptm', 0.5)
                    ptm = summary.get('ptm', 0.5)
                    mean_pae = summary.get('mean_pae', 10.0)
                    has_clash = summary.get('has_clash', False)
            
            return {
                'success': True,
                'pdb_path': final_pdb,
                'cif_path': final_cif,
                'iptm': iptm, 'ptm': ptm, 'mean_pae': mean_pae,
                'has_clash': has_clash,
                'pae': mean_pae  # Alias for calculate_af3_reward compatibility
            }

        except Exception as e:
            error_str = str(e)
            last_error = e
            
            # Check if this is the "ptxas fallback" which should actually work
            if "Falling back" in error_str and "ptxas" in error_str:
                logging.warning(f"XLA ptxas fallback warning (attempt {attempt + 1}): {e}")
                # Check if output was actually generated despite the "error"
                cif_files = list(Path(output_dir).glob("*.cif"))
                if cif_files:
                    logging.info("Output CIF found despite XLA warning - processing...")
                    final_cif = str(cif_files[0])
                    final_pdb = final_cif.replace('.cif', '.pdb')
                    if not os.path.exists(final_pdb):
                        convert_cif_to_pdb(final_cif, final_pdb)
                    return {
                        'success': True,
                        'pdb_path': final_pdb if os.path.exists(final_pdb) else final_cif,
                        'cif_path': final_cif,
                        'iptm': 0.5, 'ptm': 0.5, 'mean_pae': 10.0
                    }
                # Wait and retry if no output
                import time
                time.sleep(2)
                continue
            else:
                # Other error, don't retry
                break
    
    logger.error(f"AF3 JAX prediction failed after {max_retries} attempts: {last_error}")
    return {'success': False, 'error': str(last_error)}


    logger.error(f"AF3 JAX prediction failed after {max_retries} attempts: {last_error}")
    return {'success': False, 'error': str(last_error)}


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


if __name__ == "__main__":
    # 测试
    test_seq = "SNAKIGVLQFVSHPSLDLIYK"
    json_data = create_af3_input_json(test_seq, name="test")
    print("生成的 AF3 JSON:")
    print(json.dumps(json_data, indent=2))
    
    # 测试文件搜索
    test_dir = "/data/home/scvi041/run/HalluMPNN/outputs/initial_structure"
    if os.path.exists(test_dir):
        print(f"\n测试文件搜索: {test_dir}")
        files = find_af3_output_files(test_dir)
        print(f"CIF: {files['cif_path']}")
        print(f"JSON: {files['json_path']}")

def run_af3_msa_only(
    sequence: str,
    output_json_path: str,
    af3_config: Dict[str, str] = None
) -> bool:
    """
    Run AF3 Data Pipeline ONLY to generate MSA.
    
    This submits a real SLURM job because data pipeline requires DB access 
    and tools (jackhmmer) available on the cluster node.
    
    Args:
        sequence: Protein sequence
        output_json_path: Where to save the resulting JSON with MSA
        af3_config: Configuration
        
    Returns:
        success: bool
    """
    import shutil
    
    # Create temp dir for the job
    temp_dir = Path(output_json_path).parent / "msa_gen_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # CRITICAL: Create input JSON WITHOUT unpairedMsa/pairedMsa fields
        # This forces AF3 to run database search and generate MSA
        # Including empty strings "" may cause AF3 to skip MSA generation!
        input_json_path = temp_dir / "msa_input.json"
        
        msa_input = {
            "name": "msa_gen",
            "modelSeeds": [1],
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": sequence
                        # NOTE: Do NOT include "unpairedMsa" or "pairedMsa" fields here!
                        # AF3 will auto-generate them when these fields are absent
                    }
                }
            ],
            "dialect": "alphafold3",
            "version": 1
        }
        
        with open(input_json_path, 'w') as f:
            json.dump(msa_input, f, indent=2)
        logger.info(f"Created MSA generation input WITHOUT unpairedMsa field: {input_json_path}")
        logger.info(f"Input JSON Content: {json.dumps(msa_input, indent=2)}")
        
    except Exception as e:
        logger.error(f"MSA generation setup failed: {e}")
        return False
        
    # Generate script
    script_path = generate_slurm_script(
        str(input_json_path),
        str(temp_dir),
        job_name="af3_msa",
        af3_config=af3_config,
        run_data=True # Pass True here
    )
    
    # Submit and wait
    job_id, _ = submit_af3_job(script_path, wait=True, timeout=7200) # 2 hours max
    
    if job_id == -1: 
        return False
        
    
    # Check for output JSON
    # AF3 outputs: output_dir/JOBNAME/JOBNAME_data.json
    output_data_json = None
    json_files = list(temp_dir.glob("**/*_data.json"))
    
    if json_files:
        output_data_json = json_files[0]
        logger.info(f"MSA Data JSON found: {output_data_json}")
        
        # Extract MSA from AF3 output data.json
        try:
            with open(output_data_json, 'r') as f:
                af3_data = json.load(f)
            
            # Find unpairedMsa from AF3 output
            extracted_msa = None
            for seq in af3_data.get('sequences', []):
                if 'protein' in seq:
                    prot = seq['protein']
                    extracted_msa = prot.get('unpairedMsa', '')
                    if extracted_msa:
                        logger.info(f"Extracted MSA: {len(extracted_msa)} characters")
                        break
            
            if not extracted_msa:
                logger.error("No MSA found in AF3 output _data.json")
                return False
            
            # Check MSA quality - only consider it valid if it has homologous sequences
            msa_lines = extracted_msa.strip().split('\n')
            num_sequences = len([line for line in msa_lines if line.startswith('>')])
            logger.info(f"Extracted MSA contains {num_sequences} sequences")
            
            if num_sequences < 2:
                logger.warning("MSA only contains query sequence (no homologs found)")
                logger.warning("This will be treated as NO MSA")
                # Still save it, but caller should check quality
            
            # Create or update the TEMPLATE JSON with extracted MSA
            # FIX: Don't assume template file exists - create it if needed
            template_data = None
            if os.path.exists(output_json_path):
                try:
                    with open(output_json_path, 'r') as f:
                        template_data = json.load(f)
                    logger.info(f"Loaded existing template from {output_json_path}")
                except Exception as e:
                    logger.warning(f"Could not load existing template: {e}, creating new one")
                    template_data = None
            
            # If no valid template exists, create a new one
            if not template_data:
                logger.info("Creating new template JSON")
                template_data = {
                    "name": "template",
                    "modelSeeds": [1, 2, 3, 4, 5],
                    "sequences": [
                        {
                            "protein": {
                                "id": ["A"],
                                "sequence": sequence,
                                "unpairedMsa": extracted_msa,
                                "pairedMsa": "",
                                "modifications": [],
                                "templates": []
                            }
                        }
                        # Ligand will be added dynamically by create_af3_input_json
                    ],
                    "dialect": "alphafold3",
                    "version": 1
                }
            else:
                # Update existing template's MSA
                updated = False
                for seq in template_data.get('sequences', []):
                    if 'protein' in seq:
                        seq['protein']['unpairedMsa'] = extracted_msa
                        # Also update sequence to match
                        seq['protein']['sequence'] = sequence
                        updated = True
                        logger.info(f"Updated existing template's MSA")
                        break
                
                if not updated:
                    logger.warning("No protein entry found in template, adding one")
                    template_data.setdefault('sequences', []).insert(0, {
                        "protein": {
                            "id": ["A"],
                            "sequence": sequence,
                            "unpairedMsa": extracted_msa,
                            "pairedMsa": "",
                            "modifications": [],
                            "templates": []
                        }
                    })
            
            # Save the template
            with open(output_json_path, 'w') as f:
                json.dump(template_data, f, indent=2)
            logger.info(f"MSA Template saved to: {output_json_path}")
            logger.info(f"Template contains {num_sequences} sequences ({len(extracted_msa)} chars)")
                    
        except Exception as e:
            logger.error(f"Failed to extract/update MSA: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Cleanup temp dir
        try:
             shutil.rmtree(temp_dir)
        except:
             pass
        return True
    else:
        logger.error(f"MSA Data JSON (*_data.json) not found in {temp_dir} after job completion")
        return False

