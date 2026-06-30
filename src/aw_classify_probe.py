"""Supervised classification probe on AerialWaste MCML — extended backbones.

`src.dino_probe` already runs the DINOv2-B probe (AW m2 micro F1 0.60, §3.5.1).
This script extends to DINOv3-B and RADIOv2.5-L using the same protocol:
    encode AW train+test with the chosen frozen backbone (CLS / summary token)
    per-class logistic regression on training embeddings (class-balanced)
    per-class F1-tuned threshold on the train scores
    eval on test → micro / macro / per-class F1 via dinotxt's `ml_metrics`

JSON output shape matches `dinotxt_zeroshot_aw_<ver>.json` so the new numbers
plug into the §3.5.1 supervised-probe comparison directly.

Usage:
    python -m src.aw_classify_probe --backbone-type dinov3 --version m2
    python -m src.aw_classify_probe --backbone-type radio  --version m4 \\
        --backbone-id /home/ids/diecidue/results/waste_vlm/weights/RADIO-L
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import load_aerialwaste, load_aerialwaste_mcml  # noqa: E402
from src.dinotxt_zeroshot import ml_metrics  # noqa: E402
from src.dinov3_backbone import load_dinov3_vitb16, load_dinov3_vitl16, load_dinov3_vitl16_lvd
import torchvision.models as tvm
from collections import OrderedDict as _OD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_backbone(backbone_type: str, backbone_id: str, device: str,
                   multi_block: list[int] | None = None):
    """Returns (model_eval, embed_fn, image_size_recommend, embed_dim).

    When multi_block is given, the embed_fn returns CLS/summary tokens from
    each specified block concatenated along the feature dim.
    """
    if backbone_type == "dinov2":
        m = AutoModel.from_pretrained(backbone_id, torch_dtype=torch.float32).to(device).eval()
        d = int(m.config.hidden_size)
        if multi_block:
            @torch.inference_mode()
            def embed(pix):
                out = m(pixel_values=pix, output_hidden_states=True)
                # hidden_states[0] = embedding, [i+1] = after block i
                parts = [out.hidden_states[i + 1][:, 0] for i in multi_block]
                return torch.cat(parts, dim=-1).float().cpu().numpy()
            return m, embed, 518, d * len(multi_block)
        else:
            @torch.inference_mode()
            def embed(pix):
                out = m(pixel_values=pix)
                return out.pooler_output.float().cpu().numpy()
            return m, embed, 518, d

    if backbone_type == "dinov3":
        if backbone_id and "vitl_lvd" in backbone_id:
            m = load_dinov3_vitl16_lvd(device=device)
            d = 1024
        elif backbone_id and "vitl" in backbone_id:
            m = load_dinov3_vitl16(device=device)
            d = 1024
        else:
            m = load_dinov3_vitb16(device=device)
            d = 768
        if multi_block:
            @torch.inference_mode()
            def embed(pix):
                parts = m.get_intermediate_layers(
                    pix, n=multi_block,
                    reshape=False, return_class_token=True, norm=True,
                )
                # each element is (patch_tokens, cls_token); cls_token: [B, D]
                cls_tokens = [cls for _patches, cls in parts]
                return torch.cat(cls_tokens, dim=-1).float().cpu().numpy()
            return m, embed, 512, d * len(multi_block)
        else:
            @torch.inference_mode()                 
            def embed(pix):
                out = m.forward_features(pix)
                return out["x_norm_clstoken"].float().cpu().numpy()
            return m, embed, 512, d

    if backbone_type == "radio":
        m = AutoModel.from_pretrained(backbone_id, trust_remote_code=True).to(device).eval()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            summary, _ = m(dummy)
            d = int(summary.shape[-1])

        if multi_block:
            # Hook intermediate ViT blocks; first token = CLS/summary.
            _hook_feats: dict[int, torch.Tensor] = {}
            _blocks = None
            for attr in ["model.blocks", "radio_model.model.blocks", "blocks"]:
                obj = m
                try:
                    for part in attr.split("."): obj = getattr(obj, part)
                    _blocks = obj; break
                except AttributeError:
                    continue
            if _blocks is None:
                raise RuntimeError("Cannot locate ViT blocks in RADIO for multi-block probe.")
            for idx in multi_block:
                def _hook(mod, inp, out, _i=idx):
                    _hook_feats[_i] = out[:, 0]  # CLS token
                _blocks[idx].register_forward_hook(_hook)

            @torch.inference_mode()
            def embed(pix):
                x01 = pix * std + mean
                m(x01)
                parts = [_hook_feats[i].float() for i in multi_block]
                return torch.cat(parts, dim=-1).cpu().numpy()
            return m, embed, 512, d * len(multi_block)
        else:
            @torch.inference_mode()
            def embed(pix):
                x01 = pix * std + mean
                summary, _spatial = m(x01)
                return summary.float().cpu().numpy()
            return m, embed, 512, d

    if backbone_type == "resnet":
        ckpt_path = backbone_id
        m = tvm.resnet50(weights=None)
        _ckpt = torch.load(ckpt_path, map_location="cpu")
        _n_out = _ckpt["fc.linear1.weight"].shape[0]
        m.fc = torch.nn.Sequential(_OD([("linear1", torch.nn.Linear(2048, _n_out))]))
        m.load_state_dict(_ckpt, strict=False)  # layer0.* and prototype are extras
        m = m.to(device).eval()
        d = 2048

        @torch.inference_mode()
        def embed(pix):
            x = m.conv1(pix); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1).float().cpu().numpy()  # [B, 2048]

        return m, embed, 224, d

    if backbone_type == "swint":
        # Checkpoint is a flat torchvision SwinTransformer state dict with a custom
        # Sequential head (key: head.linear1) instead of the default bare Linear.
        import torchvision.models as tvm
        from collections import OrderedDict as _OD
        ckpt_path = backbone_id
        m = tvm.swin_t(weights=None)
        # Replace head to match checkpoint key scheme (head.linear1, not head)
        _n_out = torch.load(ckpt_path, map_location="cpu")["head.linear1.weight"].shape[0]
        m.head = torch.nn.Sequential(_OD([("linear1", torch.nn.Linear(768, _n_out))]))
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        m = m.to(device).eval()
        d = 768

        @torch.inference_mode()
        def embed(pix):
            # avg-pooled final-stage features [B, 768], head skipped
            x = m.features(pix)
            x = m.norm(x)
            x = m.permute(x)
            x = m.avgpool(x)
            x = m.flatten(x)
            return x.float().cpu().numpy()

        return m, embed, 224, d

    raise ValueError(f"unknown backbone_type {backbone_type!r}")


def encode_samples(samples, embed_fn, image_size: int, batch_size: int, device: str) -> np.ndarray:
    tfm = _build_transform(image_size)
    out: list[np.ndarray] = []
    buf: list[torch.Tensor] = []

    def flush():
        if not buf:
            return
        batch = torch.stack(buf).to(device)
        out.append(embed_fn(batch))
        buf.clear()

    for s in tqdm(samples, desc=f"embed @ {image_size}²"):
        try:
            img = Image.open(s.image_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {s.image_path}: {e}", file=sys.stderr); continue
        buf.append(tfm(img))
        if len(buf) >= batch_size:
            flush()
    flush()
    return np.concatenate(out, axis=0)


def per_class_threshold(y_train: np.ndarray, scores_train: np.ndarray) -> np.ndarray:
    """F1-tuned per-class threshold on train scores (same recipe as dinotxt)."""
    from sklearn.metrics import f1_score
    C = y_train.shape[1]
    th = np.zeros(C, dtype=np.float32)
    for c in range(C):
        s = scores_train[:, c]
        cands = np.unique(np.percentile(s, np.linspace(1, 99, 50)))
        best, best_t = -1.0, 0.5
        for t in cands:
            f1 = f1_score(y_train[:, c], (s >= t).astype(int), zero_division=0)
            if f1 > best:
                best, best_t = f1, float(t)
        th[c] = best_t
    return th


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-type", choices=["dinov2", "dinov3", "radio", "swint", "resnet"], required=True)
    p.add_argument("--backbone-id", default=None,
                   help="HF id or local path; defaults: dinov2=facebook/dinov2-base, "
                        "dinov3=local, radio=results/.../weights/RADIO-L")
    p.add_argument("--version", choices=["m2", "m4"], default=None,
                   help="MCML split version; required when --task mcml")
    p.add_argument("--task", choices=["mcml", "binary"], default="mcml")
    p.add_argument("--image-size", type=int, default=None,
                   help="defaults: dinov2=518, dinov3=512, radio=512")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--multi-block", type=str, default=None,
                   help="Comma-separated block indices for CLS-token concat, e.g. '3,7,11'.")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    multi_block = (
        [int(x) for x in args.multi_block.split(",") if x.strip()]
        if args.multi_block else None
    )

    if args.task == "mcml" and args.version is None:
        p.error("--version is required when --task mcml")

    # Resolve backbone id defaults.
    if args.backbone_id is None:
        args.backbone_id = {
            "dinov2": "facebook/dinov2-base",
            "dinov3": "local",
            "radio":  "/home/ids/diecidue/results/waste_vlm/weights/RADIO-L",
        }[args.backbone_type]

    device = "cuda"
    print(f"[load] backbone_type={args.backbone_type} id={args.backbone_id} "
          f"task={args.task}  multi_block={multi_block}", flush=True)
    _model, embed_fn, default_size, embed_dim = _load_backbone(
        args.backbone_type, args.backbone_id, device, multi_block=multi_block,
    )
    image_size = args.image_size if args.image_size is not None else default_size
    print(f"[model] embed_dim={embed_dim}  image_size={image_size}", flush=True)

    AW_ROOT = "/home/ids/diecidue/data/aerialwaste"

    if args.task == "binary":
        from sklearn.metrics import f1_score, precision_recall_fscore_support
        train_raw = load_aerialwaste(AW_ROOT, split="training")
        test_raw = load_aerialwaste(AW_ROOT, split="testing")
        train_raw = [s for s in train_raw if s.image_path.exists()]
        test_raw = [s for s in test_raw if s.image_path.exists()]
        print(f"[data] AW binary  train={len(train_raw)}  test={len(test_raw)}", flush=True)

        X_train = encode_samples(train_raw, embed_fn, image_size, args.batch_size, device)
        X_test = encode_samples(test_raw, embed_fn, image_size, args.batch_size, device)
        y_train = np.array([s.label for s in train_raw], dtype=np.int32)
        y_test = np.array([s.label for s in test_raw], dtype=np.int32)

        X_tr_n = normalize(X_train)
        X_te_n = normalize(X_test)
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", n_jobs=-1)
        clf.fit(X_tr_n, y_train)
        proba = clf.predict_proba(X_te_n)[:, 1]
        pred = (proba >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0,
        )
        rep = {
            "task": "binary",
            "backbone_type": args.backbone_type,
            "backbone_id": args.backbone_id,
            "image_size": image_size,
            "embed_dim": embed_dim,
            "multi_block": multi_block,
            "n_test": int(len(y_test)),
            "base_rate": float(y_test.mean()),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        print()
        print(f"=== AW binary — {args.backbone_type} ({image_size}²) ===")
        print(f"  F1={f1:.4f}  precision={prec:.4f}  recall={rec:.4f}")

    else:  # mcml
        cats_tr, train = load_aerialwaste_mcml(AW_ROOT, split="train", version=args.version)
        cats_te, test = load_aerialwaste_mcml(AW_ROOT, split="test", version=args.version)
        assert cats_tr == cats_te
        cats = cats_tr
        bt, bte = len(train), len(test)
        train = [s for s in train if s.image_path.exists()]
        test = [s for s in test if s.image_path.exists()]
        print(f"[data] AW {args.version}  train={len(train)} (-{bt-len(train)}) "
              f"test={len(test)} (-{bte-len(test)})  classes={len(cats)}", flush=True)

        X_train = encode_samples(train, embed_fn, image_size, args.batch_size, device)
        X_test = encode_samples(test, embed_fn, image_size, args.batch_size, device)

        Y_train = np.zeros((len(train), len(cats)), dtype=np.int32)
        for r, s in enumerate(train):
            for c in s.extra["gt_categories"]:
                if c in cats:
                    Y_train[r, cats.index(c)] = 1
        Y_test = np.zeros((len(test), len(cats)), dtype=np.int32)
        for r, s in enumerate(test):
            for c in s.extra["gt_categories"]:
                if c in cats:
                    Y_test[r, cats.index(c)] = 1

        X_tr_n = normalize(X_train)
        X_te_n = normalize(X_test)
        scores_train = np.zeros_like(Y_train, dtype=np.float32)
        scores_test = np.zeros_like(Y_test, dtype=np.float32)
        for c in range(len(cats)):
            clf = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", n_jobs=-1,
            )
            clf.fit(X_tr_n, Y_train[:, c])
            scores_train[:, c] = clf.predict_proba(X_tr_n)[:, 1]
            scores_test[:, c] = clf.predict_proba(X_te_n)[:, 1]

        thr = per_class_threshold(Y_train, scores_train)
        P_test = (scores_test >= thr).astype(int)
        rep = ml_metrics(cats, Y_test, P_test, scores_test)
        rep["backbone_type"] = args.backbone_type
        rep["backbone_id"] = args.backbone_id
        rep["version"] = args.version
        rep["image_size"] = image_size
        rep["embed_dim"] = embed_dim
        rep["multi_block"] = multi_block
        rep["thresholding"] = "per-class F1-tuned on train scores"
        rep["per_class_threshold"] = {cats[c]: float(thr[c]) for c in range(len(cats))}

        print()
        print(f"=== AW {args.version} — {args.backbone_type} ({image_size}²) "
              f"supervised probe ===")
        print(f"  micro F1 = {rep['micro']['f1']:.4f}   "
              f"macro F1 = {rep['macro']['f1']:.4f}")
        print(f"per-class F1:")
        for name in cats:
            d = rep["per_class"].get(name, {})
            f1 = d.get("f1")
            sup = d.get("support", 0)
            f1_str = "n/a" if f1 is None else f"{f1:.3f}"
            print(f"  {name[:42]:42s}  F1={f1_str}  support={sup}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(rep, f, indent=2)
    print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
