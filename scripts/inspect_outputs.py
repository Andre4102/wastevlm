"""Render a self-contained HTML gallery of VLM eval outputs for eyeballing.

Reads an eval results dir (test_eval.json + raw_responses.jsonl written by
src/vlm_eval.py), pairs each record with its thumbnail, and writes one HTML
file with the image, ground truth, parsed labels, and the raw generation.

The point is to see *why* a number is what it is: an F1 of 0.31 built from
confident-but-rare predictions looks nothing like one built from spraying every
label, and only the generations show which you have.

  # one run, balanced mix of hit / miss / false-alarm / correct-reject
  python scripts/inspect_outputs.py --run <eval_dir> --out /path/page.html

  # only the cases it got wrong by staying silent
  python scripts/inspect_outputs.py --run <eval_dir> --select fn --n 60 ...

  # same images, two checkpoints side by side (the 150K vs 819K ablation)
  python scripts/inspect_outputs.py --run <next_dir> --compare <150k_dir> ...
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WASTE_DATA_ROOT = Path(
    os.environ.get(
        "WASTE_DATA_ROOT",
        "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data",
    )
)
DATASET_IMAGE_DIR = {
    "dw_paper10": WASTE_DATA_ROOT / "dronewaste" / "images",
    "aw_m2": WASTE_DATA_ROOT / "aerialwaste" / "images",
    "aw_m4": WASTE_DATA_ROOT / "aerialwaste" / "images",
}


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    report = json.loads((run_dir / "test_eval.json").read_text())
    # newline-only split: str.splitlines() also breaks on U+2028, which VLM
    # outputs contain and json.dumps does not escape.
    records = []
    with (run_dir / "raw_responses.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return report, records


def outcome(rec: dict) -> str:
    """Per-image outcome, used both for colouring and for --select."""
    gt, pred = set(rec.get("gt", [])), set(rec.get("parsed", []))
    if not gt and not pred:
        return "tn"          # correctly said nothing
    if gt and pred == gt:
        return "tp"          # exact hit
    if gt and not pred:
        return "fn"          # stayed silent on real waste
    if pred and not gt:
        return "fp"          # invented waste
    return "partial"         # overlapping-but-not-equal label sets


def pick_records(records: list[dict], select: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if select != "mixed":
        pool = [r for r in records if outcome(r) == select]
        rng.shuffle(pool)
        return pool[:n]
    # balanced across outcomes so one failure mode can't fill the page
    buckets: dict[str, list[dict]] = {}
    for r in records:
        buckets.setdefault(outcome(r), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    picked, i = [], 0
    while len(picked) < n and any(len(b) > i for b in buckets.values()):
        for key in ("tp", "partial", "fn", "fp", "tn"):
            b = buckets.get(key, [])
            if len(b) > i and len(picked) < n:
                picked.append(b[i])
        i += 1
    return picked


def thumb_data_uri(path: Path, size: int) -> str | None:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def chips(labels, cls: str) -> str:
    if not labels:
        return '<span class="chip none">—</span>'
    return "".join(f'<span class="chip {cls}">{html.escape(str(x))}</span>'
                   for x in sorted(labels))


def render_answer(rec: dict, label: str) -> str:
    gt, pred = set(rec.get("gt", [])), set(rec.get("parsed", []))
    parts = [f'<div class="who">{html.escape(label)}</div>',
             f'<div class="row"><b>pred</b>{chips(pred, "pred")}</div>']
    if "raw_turn1" in rec:
        parts.append('<div class="raw t1"><b>describe</b> '
                     + html.escape(rec["raw_turn1"]) + "</div>")
    parts.append('<div class="raw"><b>answer</b> '
                 + html.escape(rec.get("raw", "")) + "</div>")
    miss, extra = sorted(gt - pred), sorted(pred - gt)
    if miss or extra:
        d = []
        if miss:  d.append("missed " + ", ".join(miss))
        if extra: d.append("extra " + ", ".join(extra))
        parts.append(f'<div class="delta">{html.escape("; ".join(d))}</div>')
    return f'<div class="answer o-{outcome(rec)}">' + "".join(parts) + "</div>"


def render_html(report: dict, picked: list[dict], img_dir: Path, thumb: int,
                title: str, compare: dict[str, dict] | None,
                run_dir: Path, compare_dir: Path | None) -> str:
    lpi = report.get("labels_per_image", {})
    bp = report.get("binary_presence", {})
    head = [
        f"micro-F1 {report['micro']['f1']:.3f}",
        f"macro-F1 {report['macro']['f1']:.3f}",
        f"P {report['micro']['precision']:.3f}",
        f"R {report['micro']['recall']:.3f}",
        f"n={report['n_test']}",
        f"empty parses {report.get('n_empty_parse', '?')}",
    ]
    if lpi:
        head.append(f"labels/img {lpi['pred_mean']:.2f} (gt {lpi['gt_mean']:.2f})")
    if bp:
        head.append(f"binary F1 {bp['f1']:.3f}")

    cards = []
    for rec in picked:
        uri = thumb_data_uri(img_dir / rec["file"], thumb)
        img = (f'<img src="{uri}" loading="lazy">' if uri
               else '<div class="missing">image not on disk</div>')
        answers = render_answer(rec, "this run")
        if compare is not None:
            other = compare.get(rec["file"])
            answers += (render_answer(other, "compare") if other
                        else '<div class="answer"><div class="who">compare</div>'
                             '<div class="raw">not in the other run</div></div>')
        cards.append(
            '<div class="card">'
            f'<div class="thumb">{img}</div>'
            '<div class="meta">'
            f'<div class="fname">{html.escape(rec["file"])}</div>'
            f'<div class="row"><b>gt</b>{chips(rec.get("gt", []), "gt")}</div>'
            f'{answers}</div></div>'
        )

    prov = f"run: {run_dir}"
    if compare_dir is not None:
        prov += f"<br>compare: {compare_dir}"

    return f"""<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1c1b18; --mut:#6b6862; --line:#e2ded6; --card:#fff; }}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#16150f; --fg:#eceae4; --mut:#9b978d; --line:#33302a; --card:#1e1c16; }} }}
:root[data-theme="dark"] {{ --bg:#16150f; --fg:#eceae4; --mut:#9b978d; --line:#33302a; --card:#1e1c16; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
h1 {{ font-size:19px; margin:0 0 6px; }}
.head {{ color:var(--mut); font-size:13px; margin-bottom:4px; }}
.head span {{ margin-right:14px; white-space:nowrap; }}
.prov {{ color:var(--mut); font-size:12px; font-family:ui-monospace,monospace;
  margin-bottom:20px; word-break:break-all; }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); }}
.card {{ display:flex; gap:14px; background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:12px; }}
.thumb img {{ width:200px; max-width:100%; border-radius:6px; display:block; }}
.missing {{ width:200px; height:120px; display:grid; place-items:center;
  color:var(--mut); font-size:12px; border:1px dashed var(--line); border-radius:6px; }}
.meta {{ flex:1; min-width:0; }}
.fname {{ font-family:ui-monospace,monospace; font-size:12px; color:var(--mut); margin-bottom:6px; }}
.row {{ margin:3px 0; }}
.row b, .raw b, .who {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--mut); margin-right:6px; }}
.who {{ display:block; margin:8px 0 3px; }}
.chip {{ display:inline-block; padding:1px 8px; margin:2px 4px 2px 0; border-radius:999px;
  font-size:12px; border:1px solid var(--line); }}
.chip.gt {{ background:#dbeafe; color:#1e3a8a; border-color:#bfdbfe; }}
.chip.pred {{ background:#dcfce7; color:#14532d; border-color:#bbf7d0; }}
.chip.none {{ color:var(--mut); }}
.raw {{ font-size:13px; margin:3px 0; word-break:break-word; }}
.raw.t1 {{ color:var(--mut); }}
.delta {{ font-size:12px; color:#b45309; margin-top:3px; }}
.answer {{ border-left:3px solid var(--line); padding-left:10px; margin-top:6px; }}
.answer.o-tp {{ border-left-color:#22c55e; }}
.answer.o-partial {{ border-left-color:#eab308; }}
.answer.o-fn {{ border-left-color:#f97316; }}
.answer.o-fp {{ border-left-color:#ef4444; }}
.answer.o-tn {{ border-left-color:#94a3b8; }}
</style>
<h1>{html.escape(title)}</h1>
<div class="head">{"".join(f"<span>{html.escape(h)}</span>" for h in head)}</div>
<div class="prov">{prov}</div>
<div class="grid">{"".join(cards)}</div>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True,
                   help="eval results dir (test_eval.json + raw_responses.jsonl)")
    p.add_argument("--out", type=Path, required=True, help="output .html")
    p.add_argument("--compare", type=Path, default=None,
                   help="second eval dir; its answer on the same image is shown alongside")
    p.add_argument("--n", type=int, default=48, help="cards to render (default 48)")
    p.add_argument("--select", default="mixed",
                   choices=["mixed", "tp", "fn", "fp", "tn", "partial"],
                   help="tp=exact hit, fn=silent on real waste, fp=invented waste, "
                        "tn=correctly silent, partial=overlapping labels (default: mixed)")
    p.add_argument("--thumb", type=int, default=320, help="thumbnail long edge px")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    report, records = load_run(args.run)
    dataset = report["dataset"]
    img_dir = DATASET_IMAGE_DIR[dataset]
    picked = pick_records(records, args.select, args.n, args.seed)
    print(f"[inspect] {args.run.name}: {len(records)} records, "
          f"rendering {len(picked)} ({args.select})")

    compare = None
    if args.compare is not None:
        _, other = load_run(args.compare)
        compare = {r["file"]: r for r in other}
        print(f"[inspect] compare: {args.compare.name} ({len(compare)} records)")

    title = f"{args.run.name} — {dataset} / {report['prompt_style']}"
    page = render_html(report, picked, img_dir, args.thumb, title, compare,
                       args.run, args.compare)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"[inspect] wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
