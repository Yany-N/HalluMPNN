#!/bin/bash
#SBATCH --job-name=hallumpnn


# Configuration
HALLUMPNN_DIR="/data/home/scvi041/run/HalluMPNN"
CONFIG_FILE="${HALLUMPNN_DIR}/configs/default.yaml"
SCAFFOLD_PDB="${HALLUMPNN_DIR}/inputs/3lft-ldopa.pdb"
OUTPUT_DIR="${HALLUMPNN_DIR}/outputs/$(date +%Y%m%d_%H%M%S)"

# ====================
# CHECKPOINT RESUME (Edit this to resume from a specific checkpoint)
# ====================
# Set to empty string "" to start fresh, or specify path to resume:
# CHECKPOINT_PATH="${HALLUMPNN_DIR}/configs/checkpoint_step_50.pt"
CHECKPOINT_PATH=""

# RESUME_MODE: How to handle training history when resuming
#   "CONTINUE" = Copy old CSV logs to new folder, continue plotting from saved step
#   "FRESH"    = Only load model weights, start metrics/plots from step 0
RESUME_MODE="CONTINUE"

# Parse additional arguments
EXTRA_ARGS="$@"

# ============================================
# Environment Setup
# ============================================
echo "============================================"
echo "HalluMPNN Training"
echo "============================================"
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS}"
echo ""

# Load modules (source module init first)
source /etc/profile.d/modules.sh 2>/dev/null || true
source /data/apps/lmod/lmod/init/bash 2>/dev/null || true
module load miniforge/25.3.0-3 cuda/12.8 2>/dev/null || true
# Note: DO NOT load cudnn module - JAX 0.9.0 bundles cuDNN 9.8
module load apptainer/1.2.4 2>/dev/null || true

# Activate conda environment
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate HalluMPNN

# ============================================
# JAX ENVIRONMENT SETUP
# ============================================
echo "Setting up JAX environment..."

# 1. Clear JAX cache to prevent stale compilations
rm -rf ~/.cache/jax/*

# 2. Force JAX to use local NVIDIA libs (for GPU support)
export SITE_PACKAGES=$CONDA_PREFIX/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. Disable Triton optimizations (Fixes LLVM crash)
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_graph_level=0"

# 5. Prevent cross-environment plugin loading
unset JAX_PLUGINS_DIR
# Add HalluDesign-main to PYTHONPATH so we can import its utility scripts
export PYTHONPATH="/data/home/scvi041/run/HalluDesign-main:$PYTHONPATH"

echo "JAX environment repaired."
# ============================================

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS-1)))

# JAX memory management - prevent OOM when using JAX+PyTorch together
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5  # Limit JAX to 50% GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Disable full preallocation

echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Check GPU
nvidia-smi

# ============================================
# Create Output Directory
# ============================================
mkdir -p ${OUTPUT_DIR}
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# ============================================
# Run HalluMPNN
# ============================================
cd ${HALLUMPNN_DIR}

# Build resume argument if checkpoint path is specified
RESUME_ARG=""
MODE_ARG=""
if [ -n "${CHECKPOINT_PATH}" ] && [ -f "${CHECKPOINT_PATH}" ]; then
    echo "Resuming from checkpoint: ${CHECKPOINT_PATH}"
    echo "Resume mode: ${RESUME_MODE}"
    RESUME_ARG="--resume_from ${CHECKPOINT_PATH}"
    
    # Pass resume mode to Python
    if [ "${RESUME_MODE}" = "CONTINUE" ]; then
        MODE_ARG="--resume_mode continue"
        
        # Try to copy old CSV files to new output dir for continuous plotting
        CKPT_DIR=$(dirname "${CHECKPOINT_PATH}")
        PARENT_DIR=$(dirname "${CKPT_DIR}")
        
        # Strategy 1: Look in same dir as checkpoint (User's custom case: configs/)
        if [ -f "${CKPT_DIR}/training_log.csv" ]; then
            echo "Copying history from checkpoint dir: ${CKPT_DIR}"
            cp "${CKPT_DIR}/training_log.csv" "${OUTPUT_DIR}/" 2>/dev/null || true
            cp "${CKPT_DIR}/sequences.csv" "${OUTPUT_DIR}/" 2>/dev/null || true
            
        # Strategy 2: Look in parent dir (Standard structure: outputs/run/checkpoints/ -> outputs/run/)
        elif [ -f "${PARENT_DIR}/training_log.csv" ]; then
            echo "Copying history from parent dir: ${PARENT_DIR}"
            cp "${PARENT_DIR}/training_log.csv" "${OUTPUT_DIR}/" 2>/dev/null || true
            cp "${PARENT_DIR}/sequences.csv" "${OUTPUT_DIR}/" 2>/dev/null || true
            
        else
            echo "WARNING: RESUME=CONTINUE but no training_log.csv found near checkpoint."
            echo "Checked: ${CKPT_DIR} and ${PARENT_DIR}"
        fi
    else
        MODE_ARG="--resume_mode fresh"
    fi
elif [ -n "${CHECKPOINT_PATH}" ]; then
    echo "WARNING: Checkpoint file not found: ${CHECKPOINT_PATH}"
    echo "Starting fresh training..."
fi

python scripts/run_hallumpnn.py \
    --config ${CONFIG_FILE} \
    --scaffold ${SCAFFOLD_PDB} \
    --output_dir ${OUTPUT_DIR} \
    ${RESUME_ARG} \
    ${MODE_ARG} \
    ${EXTRA_ARGS}
# 
# Checkpoint Control Options:
#   CHECKPOINT_PATH    : Set at top of script to resume from specific .pt file
#   --resume           : Resume from latest checkpoint in output_dir
#   --resume_from PATH : Resume from specific checkpoint file (CLI override)
#
# HalluDesign Options:
#   --hallu_trigger    : Manually trigger HalluDesign at start
#   --hallu_cycles N   : Override number of HalluDesign cycles
#   --steps N          : Override training steps

EXIT_CODE=$?

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "Training Complete"
echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Success! Results saved in: ${OUTPUT_DIR}"
    ls -la ${OUTPUT_DIR}/
else
    echo "Training failed. Check logs for details."
fi

echo "============================================"
