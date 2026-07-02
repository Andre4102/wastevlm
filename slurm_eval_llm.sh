#!/bin/bash
# Score one or more causal LMs on the waste knowledge-retention benchmark
# (src/eval): MC acc_norm on low_qa + concept_qa, and PPL on appearance +
# held-out corpus_eval. Model paths/ids are passed as positional args, so the
# same script scores base / CPT / materialized-pruned models.
#
# Usage:
#   sbatch slurm_eval_llm.sh <model1> [<model2> ...]
# e.g. baseline:
#   sbatch slurm_eval_llm.sh /leonardo_scratch/.../basemodel/llama-3.1-8b
#
#SBATCH --job-name=eval_llm
#SBATCH --account=IscrC_FICHE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=FAIL,END

set -euo pipefail
if [ "$#" -lt 1 ]; then echo "usage: sbatch slurm_eval_llm.sh <model> [<model> ...]"; exit 2; fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gausdino

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

WASTEVLM=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
DATA=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data
LOW_QA=${DATA}/waste_eval/low_qa.jsonl
CONCEPT_QA=${DATA}/waste_eval/concept_qa.jsonl
APPEARANCE=${WASTEVLM}/src/eval/data/appearance.jsonl
CORPUS_EVAL=${DATA}/waste_corpus_web/corpus_eval.jsonl
cd "$WASTEVLM"

echo "[slurm] job=${SLURM_JOB_ID:-?} models: $* start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "===== MC: low_qa (memorisation) ====="
python -m src.eval.mc_score --bench "$LOW_QA"     --models "$@" --out "${LOW_QA}.results.json"
echo "===== MC: concept_qa (paraphrased concept) ====="
python -m src.eval.mc_score --bench "$CONCEPT_QA" --models "$@" --out "${CONCEPT_QA}.results.json"
echo "===== PPL: appearance (visual world-knowledge) ====="
python -m src.eval.ppl_eval --eval "$APPEARANCE"  --models "$@"
echo "===== PPL: corpus_eval (held-out waste prose) ====="
python -m src.eval.ppl_eval --eval "$CORPUS_EVAL" --models "$@"

echo "[slurm] done=$(date -Is)"
