"""Baseline 2: grounded VLMs asked to point at waste, with no waste-specific training.

Each model is asked to LOCALISE, not just describe, and the boxes it returns are
scored against the annotations with the same IoU machinery as the Grounding DINO
baseline. That is the whole point of the exercise: a textual answer is cheap and
already known to be unreliable here, whereas a box either lands on the waste or
does not.

Supported backends and why each is here:

  kosmos2   -- generic (non-remote-sensing) grounding, 224px. The control for
               "is remote-sensing pretraining what matters, or just grounding?"
  geochat   -- remote-sensing VLM with referring-expression grounding. Note it
               interpolates CLIP position embeddings to 504px (patch 14, so a
               36x36 grid), which is FINER than this project's own 24x24 LLM
               grid -- the one baseline here with more spatial resolution than
               the model it is being compared against.

Prompt registers matter and differ per model, so each backend owns its own
phrasing rather than being forced into a shared template that suits neither. The
questions follow the plan's ladder: where is the waste, then which regions, then
what material.

Boxes come back in each model's own convention -- Kosmos-2 in normalised 0-1,
GeoChat as {<x1><y1><x2><y2>|<angle>} in 0-100 -- and are converted to absolute
xyxy here, once, so the scorer never sees a model-specific coordinate system.

    python scripts/grounded_vlm_baseline.py --model geochat --dataset aw_m2 \
        --generate --out geochat_aw.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.gdino_baseline import iou, load_gt  # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get(
    "WASTE_VLM_WEIGHTS",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights"))

# The ladder from the plan, phrased once per register.
QUERIES = {
    "waste_generic": "garbage or dumped waste",
    "pile": "a pile of debris or rubble",
    "material": None,      # filled per-dataset from the category names
}


# ---------------------------------------------------------------- box parsing
GEOCHAT_BOX = re.compile(r"\{?<(\d+)><(\d+)><(\d+)><(\d+)>(?:\|<?-?\d+>?)?\}?")


def parse_geochat_boxes(text: str, W: int, H: int) -> list[list[float]]:
    """GeoChat emits {<x1><y1><x2><y2>|<angle>} with coords on a 0-100 grid.

    The grid spans the square the image was padded into, not the image, so the
    padding has to come back off or every box on a non-square image is offset.
    """
    S = max(W, H)
    px, py = (S - W) / 2, (S - H) / 2
    out = []
    for m in GEOCHAT_BOX.finditer(text):
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        out.append([x1 / 100 * S - px, y1 / 100 * S - py,
                    x2 / 100 * S - px, y2 / 100 * S - py])
    return out


# ------------------------------------------------------------------ backends
def run_kosmos2(items, queries, args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

    d = WEIGHTS / "grounding" / "kosmos2"
    proc = AutoProcessor.from_pretrained(str(d))
    model = Kosmos2ForConditionalGeneration.from_pretrained(
        str(d), torch_dtype=torch.float32).to("cuda").eval()

    out = []
    for n, (path, gt, has) in enumerate(items):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        rec = {"image": str(path), "gt": gt, "has_waste": has, "preds": {}, "text": {}}
        for qname, q in queries.items():
            # "<grounding>" switches Kosmos-2 into its grounded-caption mode.
            prompt = f"<grounding><phrase> {q}</phrase>"
            inputs = proc(text=prompt, images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                ids = model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True, max_new_tokens=96)
            txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
            _clean, entities = proc.post_process_generation(txt)
            boxes = []
            for _ent, _span, bbs in entities:
                for (x1, y1, x2, y2) in bbs:       # normalised 0-1
                    boxes.append([x1 * W, y1 * H, x2 * W, y2 * H])
            rec["preds"][qname] = boxes
            rec["text"][qname] = _clean
        out.append(rec)
        if n % 50 == 0:
            print(f"[kosmos2] {n}/{len(items)}", flush=True)
    return out


def run_geochat(items, queries, args):
    import torch
    from PIL import Image

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "vendored" / "GeoChat"))
    from geochat.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from geochat.conversation import conv_templates
    from geochat.mm_utils import process_images, tokenizer_image_token
    from geochat.model.builder import load_pretrained_model

    d = str(WEIGHTS / "grounding" / "geochat-7B")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        d, model_base=None, model_name="geochat-7B", device="cuda")
    model.eval()

    out = []
    for n, (path, gt, has) in enumerate(items):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        # GeoChat runs at 504px on a pad-to-square image; the processor's own
        # defaults are stock CLIP's 336, which is the wrong number of patches.
        px = process_images([img], image_processor, model.config).half().cuda()
        rec = {"image": str(path), "gt": gt, "has_waste": has, "preds": {}, "text": {}}
        for qname, q in queries.items():
            qs = f"{DEFAULT_IMAGE_TOKEN}\n[refer] Give me the location of <p> {q} </p>"
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), tokenizer,
                                        IMAGE_TOKEN_INDEX, return_tensors="pt")
            ids = ids.unsqueeze(0).cuda()
            with torch.no_grad():
                o = model.generate(ids, images=px, do_sample=False,
                                   max_new_tokens=128, use_cache=True)
            txt = tokenizer.batch_decode(o[:, ids.shape[1]:],
                                         skip_special_tokens=True)[0].strip()
            rec["preds"][qname] = parse_geochat_boxes(txt, W, H)
            rec["text"][qname] = txt
        out.append(rec)
        if n % 25 == 0:
            print(f"[geochat] {n}/{len(items)}  last={rec['text'][list(queries)[0]][:70]!r}",
                  flush=True)
    return out


BACKENDS = {"kosmos2": run_kosmos2, "geochat": run_geochat}


# -------------------------------------------------------------------- scoring
def report(args) -> None:
    recs = json.loads(pathlib.Path(args.report).read_text())
    pos = [r for r in recs if r["has_waste"]]
    neg = [r for r in recs if not r["has_waste"]]
    print(f"\n=== {pathlib.Path(args.report).name}: {len(recs)} images "
          f"({len(pos)} positive, {len(neg)} negative)")
    for q in recs[0]["preds"]:
        print(f"\n  --- query: {q}")
        fires_p = sum(1 for r in pos if r["preds"][q]) / max(1, len(pos))
        fires_n = sum(1 for r in neg if r["preds"][q]) / max(1, len(neg))
        nbox = sum(len(r["preds"][q]) for r in pos)
        print(f"    returns a box on {fires_p:.0%} of positives, "
              f"{fires_n:.0%} of negatives  ({nbox} boxes on positives)")
        for t in (0.25, 0.50):
            matched = total = hit = npred = 0
            for r in pos:
                P = r["preds"][q]
                npred += len(P)
                total += len(r["gt"])
                used = set()
                for g in r["gt"]:
                    for i, b in enumerate(P):
                        if i not in used and iou(g, b) >= t:
                            used.add(i); matched += 1; break
                hit += len(used)
            print(f"    IoU>={t:.2f}  box recall {matched/total if total else 0:.3f}  "
                  f"precision(lower bound) {hit/npred if npred else 0:.3f}")
        # A box that merely lands somewhere in the tile is not grounding; compare
        # against a box of the same size placed at the image centre.
        import random
        rng = random.Random(0)
        rnd = 0
        for r in pos:
            for b in r["preds"][q]:
                w, h = b[2] - b[0], b[3] - b[1]
                x, y = rng.uniform(0, max(1, 1000 - w)), rng.uniform(0, max(1, 1000 - h))
                if any(iou(g, [x, y, x + w, y + h]) >= 0.25 for g in r["gt"]):
                    rnd += 1
        print(f"    same boxes placed at random hit {rnd}/{max(1,npred)} "
              f"({rnd/max(1,npred):.3f}) -- the floor for the precision above")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--model", choices=sorted(BACKENDS))
    ap.add_argument("--dataset", default="aw_m2",
                    choices=["aw_m2", "aw_m4", "dronewaste"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="grounded.json")
    args = ap.parse_args()

    if args.generate:
        cats, items = load_gt(args.dataset)
        if args.limit:
            items = items[: args.limit]
        queries = dict(QUERIES)
        queries["material"] = ", ".join(c.lower() for c in cats)
        print(f"[{args.model}] {len(items)} images, "
              f"{sum(1 for _p,_b,h in items if h)} positive")
        for k, v in queries.items():
            print(f"   query[{k}] = {v}")
        out = BACKENDS[args.model](items, queries, args)
        pathlib.Path(args.out).write_text(json.dumps(out))
        print(f"[write] {args.out}")
    if args.report:
        report(args)


if __name__ == "__main__":
    main()
