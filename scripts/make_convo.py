"""Record a real user<->model conversation over one image, for inspection and figures.

Split in two on purpose. `--generate` needs a GPU and produces a transcript JSON;
rendering reads that JSON and needs nothing. Figure typography gets iterated on a
laptop far more often than the model gets re-run, and a renderer that drags a
7B decoder behind it does not get iterated at all.

The transcript is the RECOMMENDED PRODUCT STACK, in order, and every line of it is
a real model output:

  1. the calibrated gate  -- Yes/No margin at the first assistant token, compared
     against a threshold fitted on the train split (`--calib`)
  2. the description      -- open turn, the answer the gate decided was worth asking

Turn 2 of the eval's `open_cot` (the closed-vocabulary commit) is deliberately NOT
part of this: it answers `none` on images it has just described as containing
debris, and a thesis figure should show the stack that works rather than the one
being retired. It can still be recorded with --with-commit for a failure figure.

    # GPU: record transcripts for three images
    python scripts/make_convo.py --generate --ckpt <dir> --encoder cradiov4-so \
        --image-size 768 --pixel-shuffle 2 --dataset aw_m2 \
        --calib <calibration.json> --ids 25,113,207 --out convos.json

    # CPU: render one of them
    python scripts/make_convo.py --render convos.json --index 0 --out fig.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def describe_prompt() -> str:
    """The eval's own turn-1 prompt, imported rather than paraphrased.

    A figure has to be generated with the prompt the reported numbers came from,
    or it illustrates a model nobody measured. Checked empirically: a shorter
    hand-written variant of this dropped image 193 from "visible piles of solid
    waste in the bottom-left" to a description that never mentions waste at all.
    Prompt wording is part of the result, not packaging around it.
    """
    from src.vlm_eval import PROMPT_DESCRIBE
    return PROMPT_DESCRIBE


COMMIT = ("Based on your description, list the waste categories present. "
          "Answer `none` if there are none.")


def load_samples(dataset: str):
    """Test-split samples plus the category list, mirroring the eval's loaders."""
    from src.vlm_eval import WASTE_DATA_ROOT, DATASETS

    if dataset in ("aw_m2", "aw_m4"):
        from src.datasets import load_aerialwaste_mcml
        return load_aerialwaste_mcml(
            str(WASTE_DATA_ROOT / "aerialwaste"), split="test",
            version="m2" if dataset == "aw_m2" else "m4")
    if dataset == "dw_paper10":
        from src.datasets import load_dronewaste_multilabel
        return load_dronewaste_multilabel(
            str(WASTE_DATA_ROOT / "dronewaste"),
            categories_filter=DATASETS[dataset]["cats"])
    raise SystemExit(f"unknown dataset {dataset}")


def generate(args) -> None:
    from PIL import Image

    from src import vlm_calib
    from src.vlm_eval import WasteVLMAdapter

    cats, samples = load_samples(args.dataset)
    by_id = {s.image_id: s for s in samples}
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",")]
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"image ids not in {args.dataset} test split: {missing}")
        chosen = [by_id[i] for i in wanted]
    else:
        chosen = [s for s in samples if s.extra["gt_categories"]][:args.limit]

    thr, calib_meta = vlm_calib.load_threshold(args.calib) if args.calib else (None, None)
    describe = describe_prompt()

    adapter = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                              image_size=args.image_size,
                              pixel_shuffle=args.pixel_shuffle,
                              max_new_tokens=args.max_new_tokens)
    adapter.load()

    out = []
    for s in chosen:
        img = Image.open(s.image_path).convert("RGB")
        margin = adapter.decision_margin(img, vlm_calib.QUESTION)
        turns = [{
            "role": "user", "text": vlm_calib.QUESTION, "with_image": True,
        }, {
            "role": "model", "text": "Yes." if (thr is None or margin >= thr) else "No.",
            "kind": "gate", "margin": margin, "threshold": thr,
        }, {
            "role": "user", "text": describe,
        }, {
            "role": "model", "text": adapter.generate(img, describe), "kind": "describe",
        }]
        if args.with_commit:
            turns += [{"role": "user", "text": COMMIT},
                      {"role": "model", "text": adapter.generate(img, COMMIT),
                       "kind": "commit"}]
        out.append({
            "dataset": args.dataset, "image_id": s.image_id,
            "image_path": str(s.image_path),
            "gt_categories": s.extra["gt_categories"],
            "ckpt": str(args.ckpt), "calibration": calib_meta, "turns": turns,
        })
        print(f"[convo] {s.image_id}  margin={margin:+.2f}  gt={s.extra['gt_categories']}",
              flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out}  ({len(out)} conversations)")


def wrap(text: str, width: int) -> list[str]:
    import textwrap
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width) or [""])
    return lines


def render(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    convos = json.loads(pathlib.Path(args.render).read_text())
    c = convos[args.index]
    turns = c["turns"]
    if args.drop_commit:
        # Drop the commit exchange (user turn + its answer) so the figure shows the
        # stack being recommended rather than the component being retired.
        keep, skip = [], False
        for t in turns:
            if t.get("kind") == "commit":
                keep.pop()          # its preceding user turn goes with it
                continue
            keep.append(t)
        turns = keep

    # Print-safe palette: distinguishable in greyscale, since a thesis may be
    # printed in black and white and two pastels would merge into one.
    USER_BG, MODEL_BG = "#e8eaf0", "#f7f3e8"
    USER_FG, MODEL_FG = "#2b3040", "#3d3526"
    CHARS = 58          # chars per bubble line

    # Layout is normalised, not guessed. Everything below is measured in "line
    # units" and then scaled so the conversation exactly fills the axes; the
    # figure grows in INCHES with the same count, so physical text size stays
    # constant while the content is always guaranteed to fit. Fixing the axes
    # fractions instead (and growing only the canvas) cannot work: bubble heights
    # are axes-relative, so a long exchange overflows at any figsize.
    PAD_U, GAP_U = 0.45, 1.9        # bubble padding / inter-bubble gap, in lines
    INCHES_PER_LINE = 0.175

    wrapped = [wrap(t["text"].strip() + ("\n[margin]" if t.get("kind") == "gate"
                                         and t.get("threshold") is not None else ""),
                    CHARS) for t in turns]
    units = sum(len(w) + 2 * PAD_U for w in wrapped) + GAP_U * (len(turns) - 1)
    LH = 0.985 / units
    PAD, GAP = PAD_U * LH, GAP_U * LH

    height = args.height or max(3.2, INCHES_PER_LINE * units)
    # With auto-height LH is constant in inches, so the given fontsize always fits;
    # a forced --height shrinks the lines and the text has to come with them.
    fontsize = args.fontsize
    if args.height:
        fontsize = min(args.fontsize, args.fontsize * (height / (INCHES_PER_LINE * units)))
    fig = plt.figure(figsize=(args.width, height), dpi=args.dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.04,
                          left=0.03, right=0.97, top=0.94, bottom=0.05)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(Image.open(c["image_path"]).convert("RGB"))
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_anchor("N")   # top-align with the first bubble
    for sp in ax_img.spines.values():
        sp.set_edgecolor("#c8c8c8")
    gt = ", ".join(c["gt_categories"]) or "no waste"
    ax_img.set_title(f"{c['dataset']}  ·  image {c['image_id']}", fontsize=9,
                     color="#555", pad=6)
    ax_img.set_xlabel(f"ground truth: {gt}", fontsize=8, color="#555", labelpad=6)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.985
    for t, lines in zip(turns, wrapped):
        is_user = t["role"] == "user"
        if t.get("kind") == "gate" and t.get("threshold") is not None:
            # `wrapped` reserved a line for this; fill in the real numbers.
            lines = lines[:-1] + [f"[margin {t['margin']:+.2f} vs threshold "
                                  f"{t['threshold']:+.2f}]"]
        h = len(lines) * LH + 2 * PAD
        # User turns sit right, model turns left, each inset from the other side
        # so the two speakers stay separable without needing colour.
        x0, x1 = (0.14, 1.0) if is_user else (0.0, 0.86)
        ax.add_patch(FancyBboxPatch(
            (x0, y - h), x1 - x0, h,
            boxstyle="round,pad=0.006,rounding_size=0.02",
            linewidth=0, facecolor=USER_BG if is_user else MODEL_BG,
            transform=ax.transAxes, clip_on=False))
        ax.text(x0 + 0.022 if not is_user else x1 - 0.022, y - PAD,
                "\n".join(lines), transform=ax.transAxes,
                ha="left" if not is_user else "right", va="top",
                fontsize=fontsize, linespacing=1.5,
                family="DejaVu Sans", color=USER_FG if is_user else MODEL_FG)
        tag = "user" if is_user else "model"
        ax.text(x0 + 0.022 if not is_user else x1 - 0.022, y + 0.004, tag,
                transform=ax.transAxes, ha="left" if not is_user else "right",
                va="bottom", fontsize=6.5, color="#8a8a8a")
        y -= h + GAP

    # y has one trailing GAP subtracted past the last bubble; that is not overflow.
    if y + GAP < -0.02:
        print(f"[warn] conversation overflows the axes (y={y + GAP:.2f}); "
              f"raise --height or lower --fontsize", flush=True)

    out = pathlib.Path(args.out)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    if args.also_pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"[write] {out}" + (f" and {out.with_suffix('.pdf')}" if args.also_pdf else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="GPU: record transcripts")
    ap.add_argument("--render", metavar="TRANSCRIPT_JSON", help="CPU: draw a figure")
    ap.add_argument("--ckpt", type=pathlib.Path)
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--calib", help="threshold or calibration.json for the gate")
    ap.add_argument("--ids", help="comma-separated image ids; default = first N positives")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--with-commit", action="store_true",
                    help="also record the closed-vocabulary commit turn")
    ap.add_argument("--index", type=int, default=0, help="which conversation to draw")
    ap.add_argument("--width", type=float, default=9.5)
    ap.add_argument("--height", type=float, default=None,
                    help="inches; default scales with the conversation length")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--fontsize", type=float, default=8.0)
    ap.add_argument("--drop-commit", action="store_true",
                    help="omit the closed-vocabulary commit exchange")
    ap.add_argument("--also-pdf", action="store_true", help="vector copy for LaTeX")
    ap.add_argument("--out", default="convo.png")
    args = ap.parse_args()

    if args.generate:
        if not args.ckpt:
            raise SystemExit("--generate needs --ckpt")
        generate(args)
    elif args.render:
        render(args)
    else:
        raise SystemExit("need --generate or --render")


if __name__ == "__main__":
    main()
