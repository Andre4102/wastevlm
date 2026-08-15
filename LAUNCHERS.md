# Launchers — how to run each stage yourself

Everything runs from `/leonardo/home/userexternal/adiecidu/scripts/wastevlm`.
`sbatch` writes logs to `logs/<jobname>_<jobid>.{out,err}`.

Two conda envs, and the choice matters:

| env | use for | why |
|---|---|---|
| `waste_vlm` | training (`slurm_vlm_modern.sh`), eval of stock checkpoints | transformers 4.49; has the `anthropic` SDK |
| `myenv` | pruned/materialized checkpoints, notebooks, analysis scripts | transformers 4.57 + pruning stack + `ipykernel` |

Common gotchas, all of which have bitten this project:

- **`IMG_SIZE`/`PSHUF` must match how the checkpoint was trained.** The projector's
  input dim is `patch_dim x PSHUF^2`, so a mismatch fails on `load_state_dict`
  (loudly — good) or silently trains a differently-shaped connector.
- **Compute nodes have no internet.** Anything hitting the network (HF download,
  Anthropic API) runs on a login node. `HF_HUB_OFFLINE=1` is set in the launchers.
- **The debug QOS (`--qos=boost_qos_dbg`) allows only 2 queued/running jobs.**
  Submit smokes in pairs.
- **A `MAXSTEPS=N` probe shares its output dir with the real run.** `pretrain`
  writes the final `projector.pt` to `<TAG>_pretrain/` whenever `train()` returns
  — including after a 20-step memory gate. So `projector.pt` existing does **not**
  mean stage 1 finished, and a stage 2 launched against it silently warm-starts
  from a 20-step connector. Confirm completion from the log's `[cost]` line (only
  printed after the final step), not from the file. Give probes their own
  `TAG`, or move the stub aside before the real run lands.

---

## 1. Training — `slurm_vlm_modern.sh`

```
sbatch [--qos=... --time=... --gres=...] slurm_vlm_modern.sh <MODE> [ENCODER]
```

`MODE` = `smoke` | `pretrain` | `finetune` | `finetune_next`; `ENCODER` defaults to
`cradiov4-so`.

| env var | default | meaning |
|---|---|---|
| `IMG_SIZE` | 768 | encoder input, must be a multiple of 16 |
| `PSHUF` | 2 | pixel-shuffle; visual tokens = `(IMG_SIZE/(16*PSHUF))^2` |
| `BS` | 4 | per-GPU micro-batch |
| `ACCUM` | 16 (pretrain) / 8 (finetune) | grad accumulation |
| `MAXSTEPS` | 2000 | stage-1 only |
| `RESUME` | unset | `auto` restores optimizer+scheduler+step from newest ckpt |

Global batch = `BS * ACCUM * NGPU`; the recipe is **256 for stage 1, 128 for stage 2**.
If you lower `BS`, raise `ACCUM` by the same factor or you have changed the recipe,
not just the memory footprint. The job prints the resulting global batch on startup.

Output goes to `$WROOT/results/vlm/<ENCODER>_r<IMG_SIZE>ps<PSHUF>_<mode>/`, so
arms never collide.

The four modes:

- **`smoke`** — 4 steps on synthetic images, 1 GPU, ~2 min. Validates the forward
  pass and now prints `s/step` and `peak=<GB>` so you can size a real run.
- **`pretrain`** (stage 1) — projector only, LLM frozen, dense-caption alignment
  mix. Step-capped by `MAXSTEPS`. ps2@768 took **12h25m for 2000 steps** on 4 GPUs.
- **`finetune`** (stage 2, small) — projector + LoRA on LLaVA-Instruct-150K, 1 epoch
  = 1232 steps. Warm-starts from `<TAG>_pretrain/projector.pt`. ps2@768: **5h31m**.
- **`finetune_next`** (stage 2, scaled) — same but on the ~819K NeXT+Vision-Flan mix,
  6081 steps. Exceeds the 16h wall, so run it as the resumable chain below.

```bash
# smoke first — always
sbatch --qos=boost_qos_dbg --time=00:25:00 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
       --export=ALL,IMG_SIZE=768,PSHUF=2,BS=4 slurm_vlm_modern.sh smoke

# stage 1, then stage 2
sbatch --export=ALL,IMG_SIZE=768,PSHUF=2 slurm_vlm_modern.sh pretrain
sbatch --export=ALL,IMG_SIZE=768,PSHUF=2 slurm_vlm_modern.sh finetune
```

### Measured cost per configuration

Smoke measurements (1 A100-64GB, bs 4, accum 8 = 32 images per optimizer step —
the same per-GPU work as a real stage-2 step), from the `[cost]` line each run
now prints:

| config | visual tokens | s/step | peak VRAM | stage 1 (2000) | stage 2 150K (1232) | stage 2 819K (6081) |
|---|---:|---:|---:|---:|---:|---:|
| ps2@768 (current) | 576 | 12.2 | 22.5 GB | 12h (measured) | 5.5h (measured) | 31h (measured) |
| ps2@1024 | 1024 | 21.8 | 26.4 GB | ~24h | ~10h | ~37h |
| ps1@768 | 2304 | 32.5 | 37.6 GB | ~36h | ~15h | ~55h |

`BS=4` fits every configuration, so no recipe change is needed — but the smoke
uses *synthetic* short prompts, and stage-1 dense captions are much longer. If a
ps1 stage-1 OOMs, drop to `BS=2 ACCUM=32` (same global batch 256).

**Stage 1 only has to be rerun when `PSHUF` changes.** The projector is
`patch_dim * PSHUF^2 -> hidden`, applied per token, so it is independent of
`IMG_SIZE`: a 1024px arm at ps2 reuses an existing ps2 `projector.pt`, and a
768/ps2 checkpoint can even be *evaluated* at 1024 untouched (C-RADIO's CPE
handles the larger grid). Changing `PSHUF` changes the projector's input width
and invalidates the checkpoint.

**Chaining `finetune_next`** (one epoch > one wall): `./submit_finetune_next_chain.sh`
submits job1 fresh, job2 `afterany:job1` with `RESUME=auto`, job3 `afternotok:job2`
as a safety net. `afterany`, not `afterok`, because job 1 *times out* (non-zero) by
design. Each job restores optimizer+scheduler+step so the cosine LR is one curve.
The script does not take IMG_SIZE/PSHUF — edit it or export them first.

### The two resolution/shuffle arms (launched 2026-08-11)

```bash
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm

# Arm B — pixel-shuffle 1 @768 (2304 tokens). PSHUF changed => new projector
# shape => stage 1 from scratch, ~36h, so a 3-job chain.
sbatch --time=00:40:00 --export=ALL,IMG_SIZE=768,PSHUF=1,BS=4,MAXSTEPS=20 \
       slurm_vlm_modern.sh pretrain                      # real-data memory gate first
sbatch --dependency=afterok:<probe> --export=ALL,IMG_SIZE=768,PSHUF=1,BS=4 slurm_vlm_modern.sh pretrain
sbatch --dependency=afterany:<j1>  --export=ALL,IMG_SIZE=768,PSHUF=1,BS=4,RESUME=auto slurm_vlm_modern.sh pretrain
sbatch --dependency=afterany:<j2>  --export=ALL,IMG_SIZE=768,PSHUF=1,BS=4,RESUME=auto slurm_vlm_modern.sh pretrain
# then stage 2 (~15h):
sbatch --export=ALL,IMG_SIZE=768,PSHUF=1,BS=4 slurm_vlm_modern.sh finetune

# Arm C — 1024 @ps2 (1024 tokens). PSHUF unchanged => projector transfers from
# the 768 stage-1 => NO stage 1, straight to stage 2 (~10h).
sbatch --export=ALL,IMG_SIZE=1024,PSHUF=2,BS=4,\
PROJ_INIT=$WROOT/results/vlm/cradiov4-so_r768ps2_pretrain/projector.pt \
       slurm_vlm_modern.sh finetune
```

Both use stage-2 = **LLaVA-Instruct-150K**, so they compare against the existing
`cradiov4-so_r768ps2_finetune` arm with only the token geometry changed. That arm
also holds AerialWaste's best score to date (aw_m2 0.289), which is the number the
resolution branch has to beat.

## 2. Eval — `slurm_vlm_eval_trained.sh`

```
sbatch slurm_vlm_eval_trained.sh <ENCODER> <CKPT_DIR> [DATASET] [LIMIT] [PROMPT_STYLE]
```

- `CKPT_DIR` — a finetune output dir containing `llm_merged/` + `projector.pt`
- `DATASET` — `dw_paper10` (1504 imgs) | `aw_m2` (581) | `aw_m4` (581)
- `LIMIT` — `0` = full split; `>0` = first N images (smoke)
- `PROMPT_STYLE` — `closed_vocab` (default in the harness) | `open_cot` | `open_cot_confident`
- env: `IMG_SIZE`, `PSHUF` **must match the checkpoint**; `ENV=myenv` for pruned decoders

Output: `$WROOT/results/vlm_eval/vlm_<ENCODER>_r<IMG>ps<PS>_<stage>_<dataset>_<style>/`
with `test_eval.json` + `raw_responses.jsonl`. The `<stage>` component comes from the
ckpt dir basename — without it a second stage silently overwrites the first.

Cost: closed_vocab ~0.8 s/img, open_cot (two-turn) ~2.8 s/img, 1 GPU. Full dw
closed_vocab ≈ 20 min; all six cells ≈ 2.5 GPU-hours and they run in parallel.

```bash
CK=$WROOT/results/vlm/cradiov4-so_r768ps2_finetune_next
for DS in dw_paper10 aw_m2 aw_m4; do
  for PS in closed_vocab open_cot; do
    sbatch --export=ALL,IMG_SIZE=768,PSHUF=2 slurm_vlm_eval_trained.sh cradiov4-so "$CK" $DS 0 $PS
  done
done
```

## 3. Analysis (no SLURM — run on the login node with `myenv`)

```bash
M=/leonardo/home/userexternal/adiecidu/miniconda3/envs/myenv/bin/python
export WASTE_DATA_ROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data
A=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/vlm_eval/_analysis

$M scripts/output_stats.py --csv $A/output_stats.csv        # length / refusal / hedge / parse-fail per cell
$M scripts/input_geometry.py --out-dir $A                   # native size, GSD, object area in tokens, resolution headroom
$M scripts/build_agreement_sample.py --out-dir $A --n 150   # stratified hand-labelling sample
$M scripts/inspect_outputs.py --run <eval_dir> --out page.html [--compare <dir>]

# why is an AW number bad? split micro-F1 into detection / naming / prior
$M scripts/aw_diagnose.py --gt --noise           # split structure + label-noise audit
$M scripts/aw_diagnose.py --runs <eval_dir_name> ...

# significance of a micro-F1 delta between two arms (paired over images)
$M scripts/paired_bootstrap.py --a <baseline_dir> --b <candidate_dir> --n 2000
$M scripts/paired_bootstrap.py --grid 'vlm_cradiov4-so_r1024ps2_finetune_{ds}_{ps}'
```

`inspect_vlm_eval.ipynb` is the interactive version (kernel: **myenv**) — edit the
config cell, re-run the viewer.

## 4. Other launchers in the repo

| script | what it does |
|---|---|
| `slurm_vlm_align_arm.sh <arm> <enc> [steps]` | alignment-budget probe, connector-only, one data arm (`density`/`diversity`/`spatial`/`combined`) |
| `slurm_vlm_train.sh` | legacy 512/no-shuffle trainer, superseded by `slurm_vlm_modern.sh` |
| `slurm_vlm_prune*.sh`, `slurm_prune_waste.sh` | pruning tracks (LLM + Qwen-VL) |
| `slurm_cpt_waste.sh` | continued pretraining / recovery CPT |
| `slurm_eval_llm.sh` | waste-benchmark eval for text-only LLMs |
| `slurm_seg_*.sh`, `slurm_aw_classify_probe.sh` | segmentation + frozen-feature probes |
