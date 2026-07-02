#!/bin/bash
# Continued-pretraining (CPT) of Llama-3.1-8B on the waste+general DBL mix, with
# structural mask-learning DISABLED (pure weight CPT). Uses the pruning repo's
# pruning_llama3_pretrain.py via the --extra_domains hook that adds our small
# `waste` domain to the RedPajama mix.
#
# Mask-off recipe: target_sparsity 0.0 + mask_lr 0 + reg_weight 0 + high
# logit_init (masks frozen ≈1.0) + enable_layer_drop false.
#
# Env knobs (override at submit):
#   RUN_NAME       (default cpt_waste_300M)
#   TOTAL_TOKENS   (default 300000000; set ~2000000 for a smoke test)
#
# Usage:  sbatch slurm_cpt_waste.sh        # full run
#         TOTAL_TOKENS=2000000 RUN_NAME=cpt_smoke sbatch slurm_cpt_waste.sh
#
#SBATCH --job-name=cpt_waste
#SBATCH --account=IscrC_FICHE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0

RUN_NAME=${RUN_NAME:-cpt_waste_300M}
TOTAL_TOKENS=${TOTAL_TOKENS:-300000000}

PRUNING=/leonardo/home/userexternal/adiecidu/scripts/pruning
WASTEVLM=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
DATA_ROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/cpt_mix
MODEL_PATH=/leonardo_scratch/large/userexternal/adiecidu/pruning/results/llama/basemodel/llama-3.1-8b
RESULT_ROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/llm
REF_LOSSES=${WASTEVLM}/src/corpus/reference_losses_waste.json

NUM_TASKS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
cd "$PRUNING"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} run=$RUN_NAME tokens=$TOTAL_TOKENS start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun --export=ALL --ntasks=${NUM_TASKS} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
    python pruning_llama3_pretrain.py \
        --run_name ${RUN_NAME} \
        --model_path ${MODEL_PATH} \
        --data_root ${DATA_ROOT} \
        --output_dir ${RESULT_ROOT} \
        --extra_domains waste \
        --extra_domain_weights 0.3 \
        --target_sparsity 0.0 \
        --enable_layer_drop false \
        --mask_lr 0.0 \
        --reg_weight 0.0 \
        --logit_init 10.0 \
        --total_tokens ${TOTAL_TOKENS} \
        --seq_len 2048 \
        --batch_size 4 \
        --grad_accum 8 \
        --weight_lr 5e-5 \
        --warmup_steps 100 \
        --checkpoint_every_steps 300 \
        --eval_every_steps 300 \
        --log_every_steps 20 \
        --dbl_enabled true \
        --dbl_update_every 200 \
        --dbl_warmup 100 \
        --dbl_reference_json ${REF_LOSSES} \
        --gradient_checkpointing true \
        --num_workers 2

echo "[slurm] done run=${RUN_NAME} at $(date -Is)"
