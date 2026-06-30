"""Side-by-side comparison of baseline models from their JSON reports.

Usage:
    python -m src.compare RESULT_DIR

Reads `*_report.json` files written by report.py and produces overall and
per-source markdown tables comparing each model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ["llava-next", "geochat", "geollava8k"]

# (dataset, model) -> file stem
REPORT_FILES: dict[tuple[str, str], str] = {
    ("aerialwaste", "llava-next"): "aerialwaste_test_report.json",
    ("aerialwaste", "geochat"): "aerialwaste_test_geochat_report.json",
    ("aerialwaste", "geollava8k"): "aerialwaste_test_geollava8k_report.json",
    ("dronewaste", "llava-next"): "dronewaste_report.json",
    ("dronewaste", "geochat"): "dronewaste_geochat_report.json",
    ("dronewaste", "geollava8k"): "dronewaste_geollava8k_report.json",
}


def load_reports(root: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for key, stem in REPORT_FILES.items():
        path = root / stem
        if path.exists():
            with path.open() as f:
                out[key] = json.load(f)
    return out


def _f(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def overall_table(reports: dict[tuple[str, str], dict], question: str) -> str:
    """Markdown overall table for a given question (Q1 or Q3)."""
    lines = [
        f"## {question} overall",
        "",
        "| Dataset | Model | n | acc | prec | rec | F1 | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("aerialwaste", "dronewaste"):
        for model in MODELS:
            r = reports.get((dataset, model))
            if not r:
                continue
            q = r["overall"].get(question, {})
            lines.append(
                f"| {dataset} | {model} | {q.get('n','?')} | "
                f"{_f(q.get('accuracy'))} | {_f(q.get('precision'))} | "
                f"{_f(q.get('recall'))} | {_f(q.get('f1'))} | "
                f"{_f(q.get('brier'))} | {_f(q.get('ece'))} |"
            )
    return "\n".join(lines)


def by_split_table(
    reports: dict[tuple[str, str], dict], dataset: str, question: str
) -> str:
    """Per-source table for one dataset+question."""
    splits: list[str] = []
    for model in MODELS:
        r = reports.get((dataset, model))
        if r:
            splits = list(r.get("by_split", {}).keys())
            break

    lines = [
        f"## {dataset} {question} — by source",
        "",
        "| Source | n | "
        + " | ".join(f"{m} acc / F1 / ECE" for m in MODELS)
        + " |",
        "|" + "---|" * (1 + 1 + len(MODELS)),
    ]
    for src in sorted(splits):
        cells = [f"{src}"]
        n_seen = None
        per_model = []
        for model in MODELS:
            r = reports.get((dataset, model))
            if not r:
                per_model.append("—")
                continue
            row = r.get("by_split", {}).get(src, {}).get(question, {})
            if not row:
                per_model.append("—")
                continue
            if n_seen is None and "n" in r["by_split"][src]:
                n_seen = r["by_split"][src]["n"]
            per_model.append(
                f"{_f(row.get('accuracy'))} / {_f(row.get('f1'))} / {_f(row.get('ece'))}"
            )
        cells.append(str(n_seen) if n_seen is not None else "?")
        cells.extend(per_model)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("/home/ids/diecidue/results/waste_vlm"),
    )
    p.add_argument("--out", type=Path, default=None, help="optional markdown path")
    args = p.parse_args()

    reports = load_reports(args.root)
    sections = [
        overall_table(reports, "Q1"),
        "",
        overall_table(reports, "Q3"),
        "",
        by_split_table(reports, "aerialwaste", "Q1"),
        "",
        by_split_table(reports, "dronewaste", "Q1"),
    ]
    text = "\n\n".join(sections)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
