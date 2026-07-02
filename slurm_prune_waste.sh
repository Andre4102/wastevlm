#!/bin/bash
# ── Learned-mask structural pruning + LoRA-FT of Llama-3.1-8B (base) ─────────
# Calibrated on the MIXED waste-mostly calib (build_prune_calib.py). Sweeps
# target sparsity {0.5, 0.9}. uniformity_weight>0 is REQUIRED to materialize a
# dense save_pretrained model (materialized_thr*/), which our src.eval harness
# then scores on the waste benchmark.
#
# Submits one job per target sparsity. Tune per-target rw_max from the
# achieved-sparsity printed in the log if a run under/overshoots.
#
# Env knobs:  EPOCHS (default 15)
# Usage:      bash slurm_prune_waste.sh

set -euo pipefail

# Override the sweep for a smoke, e.g.  TS_LIST=0.5 RW_LIST=7.0 EPOCHS=1 bash slurm_prune_waste.sh
read -ra TARGET_SPARSITIES <<< "${TS_LIST:-0.5 0.9}"
read -ra RW_MAXES         <<< "${RW_LIST:-7.0 80.0}"   # controller cap, index-matched

PRUNING="/leonardo/home/userexternal/adiecidu/scripts/pruning"
WASTEVLM="/leonardo/home/userexternal/adiecidu/scripts/wastevlm"
RESULT_ROOT="/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/llm"
MODEL_PATH="/leonardo_scratch/large/userexternal/adiecidu/pruning/results/llama/basemodel/llama-3.1-8b"
CALIB_DATASET="/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/prune_calib_waste_mostly_seq2048"

N_CALIB=1384                  # size of the mixed calib (waste 1038 + general 346)
EPOCHS=${EPOCHS:-15}
BATCH=4
NUM_GPU=3
GRAD_ACCUM=4
SEARCH_PERCENT=50
MASK_LR="0.1"
TAU_START="1.0"
TAU_MIN="0.01"
TAU_DECAY="0.5"

CONTROLLER_KP="5.0"
CONTROLLER_KI="0.25"
CONTROLLER_WARMUP=50
REG_WEIGHT="0.001"
UNIFORMITY_WEIGHT="0.01"       # >0 → enables materialization of a dense model
MATERIALIZE_THRESHOLDS="0.3,0.4,0.5,0.6,0.7"

PLATEAU_REL_THRESHOLD="1e-4"
LOGIT_INIT="2.0"

JOINT_USE_LORA="true"
JOINT_LORA_RANK=32
JOINT_LORA_ALPHA="64.0"
JOINT_MERGE_LORA="true"
JOINT_WEIGHT_LR="5e-5"

TOTAL_STEPS=$(( (N_CALIB / (BATCH * NUM_GPU * GRAD_ACCUM)) * EPOCHS ))
SEARCH_STEPS=$(( TOTAL_STEPS * SEARCH_PERCENT / 100 ))
NUM_DECAYS=$(python -c "import math; print(math.ceil(math.log($TAU_MIN/$TAU_START)/math.log($TAU_DECAY)))")
TAU_DECAY_STEPS=$(( SEARCH_STEPS / NUM_DECAYS ))
PLATEAU_MIN_STEPS=$(( SEARCH_STEPS * 8 / 10 ))

cd "$PRUNING"
mkdir -p "${WASTEVLM}/logs"

for i in "${!TARGET_SPARSITIES[@]}"; do
    TS=${TARGET_SPARSITIES[$i]}
    RW_MAX=${RW_MAXES[$i]}
    PORT=$(shuf -i 10000-65000 -n 1)
    RUN_NAME="prune_waste_ts${TS}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=prune_waste${TS}
#SBATCH --account=IscrC_FICHE
#SBATCH --output=${WASTEVLM}/logs/%x_%j.out
#SBATCH --error=${WASTEVLM}/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPU}
#SBATCH --gres=gpu:${NUM_GPU}
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${PORT}
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0

echo "== ${RUN_NAME}  ts=${TS} rw_max=${RW_MAX} calib=${CALIB_DATASET} steps=${TOTAL_STEPS} =="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
cd ${PRUNING}

srun --export=ALL --ntasks=${NUM_GPU} --ntasks-per-node=${NUM_GPU} python pruning_vicuna_masked.py \\
    --model_family llama3 \\
    --model_name_or_path ${MODEL_PATH} \\
    --run_name ${RUN_NAME} \\
    --training_set_override ${CALIB_DATASET} \\
    --num_train_samples ${N_CALIB} \\
    --mask_epochs ${EPOCHS} \\
    --batch_size ${BATCH} \\
    --grad_accum ${GRAD_ACCUM} \\
    --mask_lr ${MASK_LR} \\
    --tau_start ${TAU_START} \\
    --tau_min ${TAU_MIN} \\
    --tau_decay ${TAU_DECAY} \\
    --tau_decay_every_steps ${TAU_DECAY_STEPS} \\
    --target_sparsity ${TS} \\
    --controller_kp ${CONTROLLER_KP} \\
    --controller_ki ${CONTROLLER_KI} \\
    --controller_warmup ${CONTROLLER_WARMUP} \\
    --rw_max ${RW_MAX} \\
    --reg_weight ${REG_WEIGHT} \\
    --uniformity_weight ${UNIFORMITY_WEIGHT} \\
    --materialize_thresholds ${MATERIALIZE_THRESHOLDS} \\
    --plateau_min_steps ${PLATEAU_MIN_STEPS} \\
    --plateau_rel_threshold ${PLATEAU_REL_THRESHOLD} \\
    --logit_init ${LOGIT_INIT} \\
    --joint_train_weight_lr ${JOINT_WEIGHT_LR} \\
    --joint_use_lora ${JOINT_USE_LORA} \\
    --joint_lora_rank ${JOINT_LORA_RANK} \\
    --joint_lora_alpha ${JOINT_LORA_ALPHA} \\
    --joint_merge_lora_before_eval ${JOINT_MERGE_LORA} \\
    --skip_initial_eval true \\
    --eval_masked true \\
    --gradient_checkpointing true \\
    --output_dir ${RESULT_ROOT}

echo "Done: ${RUN_NAME}"
EOF
    echo "Submitted: ${RUN_NAME} (ts=${TS}, rw_max=${RW_MAX}, port ${PORT})"
done
