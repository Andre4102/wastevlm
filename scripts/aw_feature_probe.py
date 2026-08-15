"""Is AerialWaste's waste/no-waste signal in the frozen encoder, or lost downstream?

The VLM's binding constraint on AW is the binary decision "does this image contain
waste" (see EXPERIMENTS.md "Why AerialWaste is bad"). Its best operating point over
13 eval runs is Youden J = 0.234; on DroneWaste the same machinery reaches 0.673.

This asks whether that ceiling is set by the encoder or by everything after it.
Same frozen encoder, same resolution and preprocessing as the VLM, same train/test
images, same target -- but a linear probe instead of the projector -> LLM ->
sampled-token path. If the probe separates well, the information is present in the
features and the failure is downstream (=> instruction tuning is the lever). If the
probe is also weak, the encoder does not represent it (=> the alignment stage, or
the encoder, is the thing to redesign).

Three pooling variants, because they answer different questions:
  cls   -- the summary token, i.e. what a global "what is this image of" head sees
  mean  -- mean over patch tokens
  max   -- max over patch tokens; a small waste region survives max-pooling but is
           diluted by mean and may be absent from the summary altogether. AW's
           median annotated object is 0.92 tokens, so this gap is the whole point.

The projector consumes *patch* tokens, so a large max-over-cls gap would say the
signal reaches the projector's input even when the global summary discards it.

    python scripts/aw_feature_probe.py --version m4 --image-size 768
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import load_aerialwaste_mcml  # noqa: E402
from src.vision_encoder import VisionEncoder  # noqa: E402

DATA = pathlib.Path(
    os.environ.get(
        "WASTE_DATA_ROOT",
        "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data",
    )
)


def encode_split(enc: VisionEncoder, samples, batch_size: int) -> dict[str, np.ndarray]:
    """-> {'cls': [N, D], 'mean': [N, Dp], 'max': [N, Dp]}"""
    feats: dict[str, list[np.ndarray]] = {"cls": [], "mean": [], "max": []}
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        imgs = [Image.open(s.image_path).convert("RGB") for s in chunk]
        out = enc.encode(imgs)
        feats["cls"].append(out.cls.cpu().numpy())
        feats["mean"].append(out.patches.mean(dim=1).cpu().numpy())
        feats["max"].append(out.patches.max(dim=1).values.cpu().numpy())
        if i % (batch_size * 20) == 0:
            print(f"  encoded {i + len(chunk)}/{len(samples)}", flush=True)
    return {k: np.concatenate(v) for k, v in feats.items()}


def fit_and_score(X_tr, y_tr, X_te, y_te) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import normalize

    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(normalize(X_tr), y_tr)
    score = clf.predict_proba(normalize(X_te))[:, 1]
    auc = roc_auc_score(y_te, score)
    fpr, tpr, _thr = roc_curve(y_te, score)
    j = float(np.max(tpr - fpr))
    k = int(np.argmax(tpr - fpr))
    return {
        "auc": float(auc),
        "youden_j": j,
        "tpr_at_j": float(tpr[k]),
        "fpr_at_j": float(fpr[k]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--version", default="m4", choices=["m2", "m4"])
    ap.add_argument("--image-size", type=int, default=768,
                    help="match the VLM arm being compared against")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    aw = str(DATA / "aerialwaste")
    _cats, train = load_aerialwaste_mcml(aw, split="train", version=args.version)
    _cats, test = load_aerialwaste_mcml(aw, split="test", version=args.version)
    # same on-disk filter as the eval: AW ships a PNEO subset whose files are absent
    train = [s for s in train if s.image_path.exists()]
    test = [s for s in test if s.image_path.exists()]
    y_tr = np.array([1 if s.extra["gt_categories"] else 0 for s in train])
    y_te = np.array([1 if s.extra["gt_categories"] else 0 for s in test])
    print(f"[data] train {len(train)} ({y_tr.mean():.1%} positive)  "
          f"test {len(test)} ({y_te.mean():.1%} positive)", flush=True)

    enc = VisionEncoder(args.encoder, device="cuda", image_size=args.image_size)
    print(f"[model] {args.encoder} @ {args.image_size}px, patch_dim={enc.patch_dim}",
          flush=True)
    print("[encode] train", flush=True)
    F_tr = encode_split(enc, train, args.batch_size)
    print("[encode] test", flush=True)
    F_te = encode_split(enc, test, args.batch_size)

    rep = {
        "encoder": args.encoder,
        "version": args.version,
        "image_size": args.image_size,
        "n_train": len(train),
        "n_test": len(test),
        "base_rate": float(y_te.mean()),
        "pooling": {},
    }
    print(f"\n=== AW {args.version} binary waste/no-waste, frozen {args.encoder} "
          f"@ {args.image_size}px, linear probe")
    print(f"  {'pooling':8s} {'AUC':>7s} {'J':>7s} {'TPR@J':>7s} {'FPR@J':>7s}")
    for pool in ("cls", "mean", "max"):
        r = fit_and_score(F_tr[pool], y_tr, F_te[pool], y_te)
        rep["pooling"][pool] = r
        print(f"  {pool:8s} {r['auc']:7.4f} {r['youden_j']:7.3f} "
              f"{r['tpr_at_j']:7.3f} {r['fpr_at_j']:7.3f}")
    print("\n  VLM reference on the same decision: best of 13 eval runs is J=0.234 "
          "(aw_m2), J=0.139 (aw_m4); DroneWaste reaches J=0.673.")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rep, indent=2))
        print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
