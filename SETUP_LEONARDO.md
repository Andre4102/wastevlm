# Waste-VLM on Leonardo (CINECA) — setup & VLM visual stage

*Ported from the original IDS cluster. Set up 2026-06-30.*

This repo was developed on the IDS cluster (`/home/ids/diecidue/...`, SLURM
partition `mm`, H100). It now runs on **Leonardo (CINECA)**: A100 64GB GPUs,
SLURM partition `boost_usr_prod`, account `iscrc_fiche`. All original hardcoded
paths have been repointed (with env-var overrides).

## Locations

| What | Path |
|---|---|
| Repo | `/leonardo/home/userexternal/adiecidu/scripts/wastevlm` |
| Conda env | `waste_vlm` (`miniconda3/envs/waste_vlm`, Python 3.11) |
| Weights | `/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights` |
| DINOv3 hub repo | `/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/dinov3_repo` |
| Results | `/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results` |
| HF cache | `/leonardo_scratch/large/userexternal/adiecidu/hf_cache` |

Weights live on **scratch** (home has a 50 GB quota). Downloaded:
`Qwen2.5-7B-Instruct` (the LLM, *not* Qwen2.5-VL), `RADIO-L` (nvidia/RADIO-L,
the RADIOv2.5-L pick), `dinov3-vitb16-pretrain-lvd1689m`.

## Environment

Built fresh from the actual imports (no wastevlm-specific requirements file
existed). Reproduce with `requirements.txt`:

```bash
conda create -y -n waste_vlm python=3.11 && conda activate waste_vlm
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Key versions: torch 2.5.1+cu121, transformers 4.49.0, timm 1.0.27, peft 0.15.2,
accelerate 1.4.0.

Env vars the code reads (set by `slurm_vlm_train.sh`):
`WASTE_VLM_WEIGHTS`, `DINOV3_REPO`, `HF_HOME`, `HF_HUB_OFFLINE=1`.

## The VLM visual stage (CLIP → DINO/RADIO swap)

LLaVA-style: frozen DINO/RADIO encoder → 2-layer MLP projector → Qwen2.5-7B
(LoRA). The encoder replaces the CLIP ViT a stock VLM would use. Projector input
is the encoder **patch** dim (radio-l → 1024, dinov3-b → 768; RADIO's 3072 cls
dim is unused). Image tokens are spliced in via LLaVA's `IMAGE_TOKEN_INDEX=-200`
marker.

New files:
- `src/vlm_model.py` — `WasteVLM` assembly + smoke (`--smoke`).
- `src/vlm_data.py` — VQA dataset (LLaVA + flat schemas), collator
  (assistant-only loss masking), synthetic samples.
- `src/vlm_train.py` — Trainer; two-group optimizer (projector 2e-4, LoRA 2e-5),
  bf16, gradient checkpointing.
- `slurm_vlm_train.sh` — Leonardo SLURM launcher.

### Two-stage LLaVA-1.5 recipe

| Stage | What trains | Data | Launch |
|---|---|---|---|
| `pretrain` | projector only (LLM frozen, no LoRA) | LLaVA-Pretrain LCS-558K | `sbatch slurm_vlm_train.sh pretrain radio-l` |
| `finetune` | projector + LoRA on Qwen | LLaVA-Instruct-150K + COCO | `sbatch slurm_vlm_train.sh finetune radio-l` |

`finetune` warm-starts the projector from the `pretrain` output
(`projector.pt`). Swap `radio-l` → `dinov3-b` to use the other encoder.

Smoke test (synthetic data, no dataset needed, ~1 min on one A100):
```bash
sbatch --qos=boost_qos_dbg --time=00:20:00 slurm_vlm_train.sh smoke radio-l
```

### Datasets (on scratch, `$WROOT/data`)

| Dataset | Path | Role |
|---|---|---|
| LLaVA-Pretrain LCS-558K | `data/llava_pretrain/{blip_laion_cc_sbu_558k.json, images/}` | connector |
| LLaVA-Instruct-150K | `data/llava_instruct/llava_instruct_150k.json` | instruction |
| COCO train2017 | `data/coco/train2017/` | instruction images |

The collator (`src/vlm_data.py`) handles both JSON-array and JSON-lines files
and multi-turn conversations, masking the loss to assistant turns only.

## Status / TODO

- ✅ env, weights, encoders, full-pipeline smoke (radio-l + dinov3-b, both stages).
- ✅ LLaVA-1.5 datasets downloading to scratch (connector + instruction).
- ▶️ Next: launch `pretrain` then `finetune` once the downloads finish.
- The project's *own* aerial-waste VQA (project.md plan) can later replace
  LLaVA-Instruct-150K as the stage-2 data — same schema, just point `--train` at it.
- Eval (`src/vlm_eval.py`) still uses IDS paths for its zero-shot baselines
  (CLIP/Qwen2.5-VL/InternVL3) — repoint when those baselines are re-run here.
