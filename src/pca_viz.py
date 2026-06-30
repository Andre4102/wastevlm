"""Patch-feature PCA/NMF visualisation across vision backbones (DINOv2-paper style).

Replicates the Oquab et al. 2023 Figure-1 diagnostic on DroneWaste:
 - Extract per-patch features from a frozen backbone.
 - Joint 3-component PCA over all patches across selected images -> RGB.
   Same-coloured regions across images = same semantic part.

Each ``--panel`` is one column: ``<backbone>:<image_size>``.  Supported backbones:

  dinov2   HF facebook/dinov2-base, patch 14.  dinov2:518 -> 37×37 grid.
  dinov3   local DINOv3 ViT-B/16, patch 16.    dinov3:512 -> 32×32.
  radio    local RADIOv2.5-L, patch 16.        radio:512  -> 32×32.
  clip     HF openai/clip-vit-base-patch32.    clip:224   -> 7×7  (resized internally).
  vit      timm ViT-B/16 ImageNet-21k.         vit:224    -> 14×14 (resized internally).
  frcnn    torchvision Faster R-CNN FPN P4.    frcnn:512  -> 32×32 (stride 16).
  yolo     YOLOv8x backbone (P4, stride 16).   yolo:640   -> 40×40.
           Requires ultralytics + --yolo-weights path/to/best.pt.

All PCA maps are bilinearly upsampled to ``--display-grid`` (default 32) before
plotting so columns with different native grids are visually comparable.

Output format is inferred from the ``--out`` extension (.png / .pdf).

Usage:
    python -m src.pca_viz --panels dinov2:518,dinov3:512,radio:512
    python -m src.pca_viz --panels dinov2:518,clip:224,vit:224,frcnn:512 \\
        --out figures/encoder_comparison.pdf
    python -m src.pca_viz --panels yolo:640 --yolo-weights /path/to/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seg_dataset import DroneWasteSegmentation  # noqa: E402

AW_ROOT = Path("/home/ids/diecidue/data/aerialwaste")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
FIGURES_DIR = ROOT / "figures"
DISP_SIZE = 518  # resolution for the original-image + GT-mask columns
RADIO_L_PATH = "/home/ids/diecidue/results/waste_vlm/weights/RADIO-L"
RADIO_B_PATH = "/home/ids/diecidue/results/waste_vlm/weights/RADIO-B"
SWINT_PATH   = "/home/ids/diecidue/results/waste_vlm/weights/swint/checkpoint.pth"
RESNET_PATH  = "/home/ids/diecidue/results/waste_vlm/weights/resnet/checkpoint.pth"
YOLO_PATH    = "/home/ids/diecidue/results/waste_vlm/weights/yolov8_2025-07-12_exp_4/best.pt"


def _minmax(x: np.ndarray, axis=None) -> np.ndarray:
    lo = x.min(axis=axis, keepdims=True)
    hi = x.max(axis=axis, keepdims=True)
    return (x - lo) / (hi - lo + 1e-8)


# --- backbone wrappers -----------------------------------------------------

_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Weights path for YOLOv8; override via --yolo-weights CLI flag.
_YOLO_WEIGHTS: str | None = None


def _to_01(pix: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation → [0, 1] tensor."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
    return pix * std + mean


def build_backbone(kind: str):
    """Return (model, patch_stride, extract_fn, override_grid).

    extract_fn : [1,3,H,W] ImageNet-normalised tensor -> [P, D] ndarray.
    override_grid : int | None — when set, use this as the grid side length
        instead of deriving it from (image_size // patch_stride). Used for
        CLIP and ViT whose native resolution differs from the dataset crop.
    """
    if kind == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "facebook/dinov2-base", torch_dtype=torch.float32
        ).eval()

        @torch.no_grad()
        def extract(pix):
            out = model(pixel_values=pix)
            return out.last_hidden_state[0, 1:].numpy()  # drop CLS

        return model, model.config.patch_size, extract, None

    if kind == "dinov3":
        from src.dinov3_backbone import load_dinov3_vitb16, patch_tokens
        model = load_dinov3_vitb16(device="cpu")

        @torch.no_grad()
        def extract(pix):
            return patch_tokens(model, pix)[0].numpy()

        return model, 16, extract, None

    if kind == "radio":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(RADIO_L_PATH, trust_remote_code=True).eval()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

        @torch.no_grad()
        def extract(pix):
            x01 = pix * std + mean  # RADIO expects [0,1], normalises internally
            _summary, features = model(x01)
            return features[0].numpy()

        return model, 16, extract, None

    if kind == "dinov3l":
        from src.dinov3_backbone import load_dinov3_vitl16, patch_tokens
        model = load_dinov3_vitl16(device="cpu")

        @torch.no_grad()
        def extract(pix):
            return patch_tokens(model, pix)[0].numpy()

        return model, 16, extract, None

    if kind == "radio_b":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(RADIO_B_PATH, trust_remote_code=True).eval()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

        @torch.no_grad()
        def extract(pix):
            x01 = pix * std + mean
            _summary, features = model(x01)
            return features[0].numpy()

        return model, 16, extract, None

    if kind == "clip":
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=True
        ).eval()
        # patch 32, native 224² → 7×7 grid
        clip_mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        clip_std  = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)

        @torch.no_grad()
        def extract(pix):
            x01  = _to_01(pix)
            x224 = torch.nn.functional.interpolate(x01, size=(224, 224),
                                                   mode="bicubic", align_corners=False)
            x_clip = (x224 - clip_mean) / clip_std
            out = model(pixel_values=x_clip)
            return out.last_hidden_state[0, 1:].numpy()  # [49, 768]

        return model, 32, extract, 7  # native grid = 224//32 = 7

    if kind == "vit":
        import timm
        model = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True
        ).eval()

        @torch.no_grad()
        def extract(pix):
            # timm ViT expects ImageNet-normed 224² input
            x224 = torch.nn.functional.interpolate(pix, size=(224, 224),
                                                   mode="bicubic", align_corners=False)
            out = model.forward_features(x224)  # [1, 197, 768]
            return out[0, 1:].numpy()  # drop CLS → [196, 768]

        return model, 16, extract, 14  # native grid = 224//16 = 14

    if kind == "swint":
        import torchvision.models as tvm
        from collections import OrderedDict as _OD
        _ckpt = torch.load(SWINT_PATH, map_location="cpu")
        _n_out = _ckpt["head.linear1.weight"].shape[0]
        model = tvm.swin_t(weights=None)
        model.head = torch.nn.Sequential(_OD([("linear1", torch.nn.Linear(768, _n_out))]))
        model.load_state_dict(_ckpt, strict=True)
        model = model.eval()

        @torch.no_grad()
        def extract(pix):
            x = torch.nn.functional.interpolate(pix, size=(224, 224),
                                                mode="bicubic", align_corners=False)
            x = model.features(x)   # [1, 7, 7, 768]  (H,W,C layout)
            x = model.norm(x)
            return x[0].reshape(-1, 768).numpy()  # [49, 768]

        return model, 32, extract, 7  # effective stride=32 → 7×7 native grid

    if kind == "resnet":
        import torchvision.models as tvm
        from collections import OrderedDict as _OD
        _ckpt = torch.load(RESNET_PATH, map_location="cpu")
        _n_out = _ckpt["fc.linear1.weight"].shape[0]
        model = tvm.resnet50(weights=None)
        model.fc = torch.nn.Sequential(_OD([("linear1", torch.nn.Linear(2048, _n_out))]))
        model.load_state_dict(_ckpt, strict=False)
        model = model.eval()

        @torch.no_grad()
        def extract(pix):
            x = torch.nn.functional.interpolate(pix, size=(224, 224),
                                                mode="bicubic", align_corners=False)
            x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
            x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
            # [1, 2048, 7, 7] -> [49, 2048]
            return x[0].flatten(1).T.numpy()

        return model, 32, extract, 7  # stride-32 → 7×7 native grid at 224px

    if kind == "frcnn":
        import torchvision
        det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        backbone = det.backbone.eval()

        @torch.no_grad()
        def extract(pix):
            # Backbone was trained with [0,1] → ImageNet norm inside transform.
            # We pass already ImageNet-normed tensors directly to the backbone
            # (matching internal state after GeneralizedRCNNTransform).
            feats = backbone(pix)       # OrderedDict; keys 0,1,2,3,pool
            f = feats["2"]              # stride-16 → H/16 × W/16 at input size
            return f[0].flatten(1).T.numpy()  # [h*w, 256]

        # stride=16; at 512² input → grid=32. Let main() compute from size.
        return backbone, 16, extract, None

    if kind == "yolo":
        from ultralytics import YOLO as _YOLO
        weights = _YOLO_WEIGHTS or YOLO_PATH
        yolo = _YOLO(weights)
        det_model = yolo.model.eval()   # DetectionModel — routes multi-input layers correctly

        _yolo_feat: list = []

        def _hook(m, inp, out):
            _yolo_feat.clear()
            _yolo_feat.append(out.detach())

        # model.model[18] = final P4 neck C2f output (stride 16, 640-d).
        det_model.model[18].register_forward_hook(_hook)

        @torch.no_grad()
        def extract(pix):
            x01 = _to_01(pix)
            _ = det_model(x01)
            f = _yolo_feat[0]      # [1, C, H, W] at stride 16
            return f[0].flatten(1).T.numpy()

        return det_model, 16, extract, None  # grid from size//16

    raise ValueError(f"unknown backbone {kind!r}. Choices: dinov2, dinov3, dinov3l, radio, radio_b, clip, vit, swint, resnet, frcnn, yolo")


def select_partial_coverage(ds, n: int, cov_min: float, cov_max: float, seed: int) -> list[int]:
    cand = []
    for i, s in enumerate(ds.samples):
        area = sum(float(a.get("area", 0.0)) for a in s["annotations"])
        frac = area / (s["width"] * s["height"] + 1e-9)
        if not (cov_min < frac < cov_max):
            continue
        biggest = max(s["annotations"], key=lambda a: float(a.get("area", 0.0)))
        cand.append((i, biggest["category_id"]))
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    picked, seen = [], set()
    for i, dom in cand:
        if dom in seen:
            continue
        picked.append(i)
        seen.add(dom)
        if len(picked) >= n:
            break
    for i, dom in cand:
        if len(picked) >= n:
            break
        if i not in picked:
            picked.append(i)
    return picked


def select_aw_images(root: Path, n: int, seed: int,
                     split: str = "testing", label: int = 1) -> list[Path]:
    """Pick n waste-positive (label=1) AerialWaste images at random."""
    import json
    data = json.load((root / f"{split}.json").open())
    cands = [
        root / "images" / img["file_name"]
        for img in data["images"]
        if img["is_candidate_location"] == label
    ]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[i] for i in idx]


def load_aw_tensor(path: Path, size: int) -> torch.Tensor:
    """Load an AerialWaste PNG → ImageNet-normalised [3,H,W] tensor."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    t = TF.to_tensor(img)                                    # [3,H,W] in [0,1]
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    return (t - mean) / std                                  # ImageNet-normalised


def pca_rgb_to_edges(rgb: np.ndarray) -> np.ndarray:
    """[H,W,3] normalized PCA RGB map → [H,W] edge magnitude in [0,1].

    Takes the per-channel gradient magnitude and returns the max across
    channels, so any semantic boundary (regardless of which PC encodes it)
    shows up as a bright edge.
    """
    edges = np.zeros(rgb.shape[:2], dtype=np.float32)
    for c in range(3):
        gy, gx = np.gradient(rgb[:, :, c].astype(np.float32))
        edges = np.maximum(edges, np.sqrt(gx ** 2 + gy ** 2))
    return _minmax(edges)


def _upsample_map(arr: np.ndarray, target: int) -> np.ndarray:
    """Bilinearly upsample a [g, g, C] spatial map to [target, target, C]."""
    if arr.shape[0] == target:
        return arr
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # [1,C,g,g]
    t = torch.nn.functional.interpolate(t, size=(target, target),
                                        mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()


def decompose_rasters(extract_fn, ds, picked: list[int], grid: int, seed: int,
                      method: str = "pca", display_grid: int | None = None,
                      precomputed_feats: np.ndarray | None = None):
    """Per-image 3-component decomposition rasters + per-component weights.

    method="pca": signed PCA. comp channels symmetric-normed to [-1,1].
    method="nmf": Non-negative Matrix Factorisation — activations >= 0.

    display_grid : if set, bilinearly upsample all rasters to this grid size
        so panels with different native grids can be displayed at the same
        pixel size (e.g. CLIP 7×7 upsampled to 32×32).
    precomputed_feats : if provided, skip feature extraction and use these directly.

    Returns:
        rgb    : list of [D, D, 3] where D = display_grid or native grid.
        comp   : list of [D, D, 3] — per-component normalised maps.
        weight : 3 per-component importance ratios.
    """
    if precomputed_feats is not None:
        all_feats = precomputed_feats
    else:
        feats = [extract_fn(ds[idx][0].unsqueeze(0)) for idx in picked]
        all_feats = np.concatenate(feats, axis=0)
    P = grid * grid

    if method == "pca":
        model = PCA(n_components=3, random_state=seed)
        T = model.fit_transform(all_feats)  # [N*P, 3]
        weight = [float(v) for v in model.explained_variance_ratio_]
        comp_all = T / (np.abs(T).max(axis=0, keepdims=True) + 1e-8)
    elif method == "nmf":
        from sklearn.decomposition import NMF
        X_nn = all_feats - all_feats.min()
        model = NMF(n_components=3, init="nndsvda", random_state=seed, max_iter=500)
        T = model.fit_transform(X_nn)
        total = T.sum() + 1e-8
        weight = [float(T[:, k].sum() / total) for k in range(3)]
        comp_all = T / (T.max(axis=0, keepdims=True) + 1e-8)
    else:
        raise ValueError(f"unknown method {method!r}")

    rgb_all = _minmax(T, axis=0)
    dg = display_grid or grid
    n_imgs = len(all_feats) // (grid * grid)
    rgb, comp = [], []
    for k in range(n_imgs):
        r = rgb_all[k * P:(k + 1) * P].reshape(grid, grid, 3)
        c = comp_all[k * P:(k + 1) * P].reshape(grid, grid, 3)
        rgb.append(_upsample_map(r, dg))
        comp.append(_upsample_map(c, dg))
    return rgb, comp, weight


def explained_variance_curves(panels_feats: list[tuple[str, np.ndarray]],
                              n_components: int, seed: int) -> dict[str, np.ndarray]:
    """Fit PCA with n_components for each backbone and return cumulative EVR curves."""
    curves = {}
    for label, feats in panels_feats:
        n = min(n_components, feats.shape[0], feats.shape[1])
        pca = PCA(n_components=n, random_state=seed)
        pca.fit(feats)
        curves[label] = np.cumsum(pca.explained_variance_ratio_)
    return curves


def plot_ev_curves(curves: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, cumev) in enumerate(curves.items()):
        xs = np.arange(1, len(cumev) + 1)
        c = colours[i % len(colours)]
        name = label.split("\n")[0]  # strip grid annotation if present
        axes[0].plot(xs, cumev * 100, label=name, color=c, lw=2)
        axes[1].plot(xs, np.diff(cumev, prepend=0) * 100, label=name, color=c, lw=2)

    for ax, title, ylabel in zip(
        axes,
        ["Cumulative explained variance", "Per-component explained variance"],
        ["Cumulative EVR (%)", "EVR per PC (%)"],
    ):
        ax.set_xlabel("Number of PCA components")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("PCA explained variance ratio by backbone", fontsize=15)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    dpi = 300 if out.suffix == ".png" else 200
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Patch-feature PCA/NMF comparison across vision backbones."
    )
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument(
        "--panels", type=str,
        default="dinov2:518,dinov3:512,radio:512",
        help="comma-separated <backbone>:<dataset_crop_size>; one column each. "
             "Choices: dinov2, dinov3, radio, clip, vit, frcnn, yolo.",
    )
    p.add_argument("--split", default="test")
    p.add_argument(
        "--dataset", choices=["dronewaste", "aerialwaste"], default="dronewaste",
        help="Source dataset. aerialwaste: no GT column; images loaded from --aw-root.",
    )
    p.add_argument(
        "--aw-root", type=Path, default=AW_ROOT,
        help="AerialWaste root directory (default: %(default)s).",
    )
    p.add_argument("--channels", action="store_true",
                   help="show PC1/PC2/PC3 as separate maps plus the RGB mixture")
    p.add_argument("--method", choices=["pca", "nmf"], default="pca")
    p.add_argument("--cov-min", type=float, default=0.05)
    p.add_argument("--cov-max", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--display-grid", type=int, default=32,
        help="upsample all PCA maps to this grid size for visual comparison "
             "(default 32; set 0 to keep each panel at its native grid).",
    )
    p.add_argument(
        "--yolo-weights", type=str, default=None,
        help="Path to YOLOv8 best.pt weights (required for yolo panel).",
    )
    p.add_argument("--out", type=Path, default=FIGURES_DIR / "pca_comparison.png",
                   help="Output path. Extension determines format: .png or .pdf.")
    p.add_argument(
        "--bw-edges", action="store_true",
        help="Show PCA maps as grayscale border/edge images instead of RGB colour maps. "
             "Highlights semantic region boundaries without distracting hue variation.",
    )
    p.add_argument(
        "--ev-plot", action="store_true",
        help="Also generate a cumulative explained-variance plot per backbone.",
    )
    p.add_argument(
        "--ev-components", type=int, default=32,
        help="Number of PCA components for the EV curve plot (default 32).",
    )
    args = p.parse_args()

    global _YOLO_WEIGHTS
    _YOLO_WEIGHTS = args.yolo_weights

    dg = args.display_grid if args.display_grid > 0 else None

    panels = []
    for tok in args.panels.split(","):
        kind, size = tok.split(":")
        panels.append((kind.strip(), int(size)))

    use_aw = args.dataset == "aerialwaste"

    if use_aw:
        aw_paths = select_aw_images(args.aw_root, args.n_images, args.seed)
        print(f"[data] aerialwaste: picked {len(aw_paths)} images")
        imgs_disp = []
        for p_ in aw_paths:
            t = load_aw_tensor(p_, DISP_SIZE)
            disp = t.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
            imgs_disp.append(np.clip(disp, 0, 1))
        masks_disp = None
        ds_disp = None
    else:
        ds_disp = DroneWasteSegmentation(split=args.split, image_size=DISP_SIZE, seed=args.seed)
        picked = select_partial_coverage(
            ds_disp, args.n_images, args.cov_min, args.cov_max, args.seed
        )
        print(f"[data] dronewaste: picked {len(picked)} images ({args.split}): idx={picked}")
        imgs_disp, masks_disp = [], []
        for idx in picked:
            pix, tgt = ds_disp[idx]
            disp = pix.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
            imgs_disp.append(np.clip(disp, 0, 1))
            masks_disp.append(tgt["mask"].numpy())

    # Cache backbones so the same kind loaded for multiple panels shares the instance.
    cache: dict[str, tuple] = {}
    panel_out = []
    panels_feats: list[tuple[str, np.ndarray]] = []  # for EV curve plot
    for kind, size in panels:
        if kind not in cache:
            print(f"[model] loading {kind} ...")
            cache[kind] = build_backbone(kind)
        model, patch, extract, override_grid = cache[kind]
        if override_grid is not None:
            grid = override_grid
        else:
            if size % patch != 0:
                raise ValueError(f"{kind}: size {size} not divisible by stride {patch}")
            grid = size // patch
        disp_g = dg or grid
        print(f"[{args.method}] {kind} @ {size}² → native {grid}×{grid}, displayed {disp_g}×{disp_g}")
        if use_aw:
            feats_list = [extract(load_aw_tensor(p_, size).unsqueeze(0)) for p_ in aw_paths]
        else:
            ds = DroneWasteSegmentation(split=args.split, image_size=size, seed=args.seed)
            feats_list = [extract(ds[idx][0].unsqueeze(0)) for idx in picked]
        all_feats = np.concatenate(feats_list, axis=0)
        panels_feats.append((kind, all_feats))
        rgb, comp, ev = decompose_rasters(extract, None, None, grid, args.seed,
                                          args.method, display_grid=dg,
                                          precomputed_feats=all_feats)
        panel_out.append((f"{kind}\n{grid}×{grid} grid", rgb, comp, ev))
        print(f"  3-comp weights = {[round(v, 3) for v in ev]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n = len(imgs_disp)
    cols_per_panel = 4 if args.channels else 1
    n_fixed = 1 if use_aw else 2   # AW has no GT column
    ncols = n_fixed + cols_per_panel * len(panel_out)
    fig, axes = plt.subplots(n, ncols, figsize=(3.0 * ncols, 3.0 * n))
    if n == 1:
        axes = axes[None, :]

    if not use_aw:
        base = cm.get_cmap("tab20", ds_disp.num_classes)
        colours = np.vstack([[1, 1, 1, 1], base(np.arange(ds_disp.num_classes))])
        seg_cmap = matplotlib.colors.ListedColormap(colours)

    comp_name = "PC" if args.method == "pca" else "NMF"
    ch_cmap = "RdBu_r" if args.method == "pca" else "gray"
    ch_vmin = -1 if args.method == "pca" else 0

    for k in range(n):
        axes[k, 0].imshow(imgs_disp[k])
        if not use_aw:
            axes[k, 1].imshow(masks_disp[k], cmap=seg_cmap,
                              vmin=0, vmax=ds_disp.num_classes, interpolation="nearest")
        col = n_fixed
        for label, rgb, comp, ev in panel_out:
            if args.channels:
                for ch in range(3):
                    axes[k, col].imshow(comp[k][:, :, ch], cmap=ch_cmap,
                                        vmin=ch_vmin, vmax=1, interpolation="nearest")
                    if k == 0:
                        axes[k, col].set_title(
                            f"{label}\n{comp_name}{ch+1} ({ev[ch]*100:.0f}%)", fontsize=11)
                    col += 1
            if args.bw_edges:
                edge_map = pca_rgb_to_edges(rgb[k])
                axes[k, col].imshow(edge_map, cmap="gray_r", vmin=0, vmax=1,
                                    interpolation="nearest")
            else:
                axes[k, col].imshow(rgb[k], interpolation="nearest")
            if k == 0:
                axes[k, col].set_title(
                    f"{label}\n{comp_name}1-3 → RGB" if args.channels else label,
                    fontsize=12)
            col += 1
        if k == 0:
            axes[k, 0].set_title("Image", fontsize=12)
            if not use_aw:
                axes[k, 1].set_title("GT seg", fontsize=12)
        for j in range(ncols):
            axes[k, j].axis("off")

    if args.method == "nmf":
        mode = "NMF components + RGB" if args.channels else "3-component NMF → RGB"
    elif args.bw_edges:
        mode = "PCA region boundaries (B&W)"
    else:
        mode = "PC1/2/3 (RdBu) + RGB" if args.channels else "3-comp PCA → RGB"
    fig.suptitle(f"Patch-feature decomposition — {mode}", y=1.01, fontsize=14)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # PDF is vector so dpi only affects rasterised imshow cells; 200 keeps them sharp.
    dpi = 300 if args.out.suffix == ".png" else 200
    fig.savefig(args.out, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {args.out}")

    if args.ev_plot:
        ev_out = args.out.with_name(args.out.stem + "_ev" + args.out.suffix)
        print(f"[ev] fitting PCA({args.ev_components}) per backbone ...")
        curves = explained_variance_curves(panels_feats, args.ev_components, args.seed)
        plot_ev_curves(curves, ev_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
