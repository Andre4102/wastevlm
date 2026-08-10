"""Resumable download of lmms-lab/LLaVA-NeXT-Data (~779k, images embedded in the
parquet -> no multi-source image resolution needed). ~146 GB over 250 shards.

Login-node processes here get reaped, so this retries in a loop; snapshot_download
skips already-complete files, so every restart resumes where it left off. Safe to
launch again from any session if it dies.

    setsid python scripts/download_llava_next.py < /dev/null \
        > logs/download_llava_next.log 2>&1 & disown
"""
from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "lmms-lab/LLaVA-NeXT-Data"
DEST = Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/"
            "data/llava_next")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        print(f"[llava_next] attempt {attempt} -> {DEST}", flush=True)
        try:
            snapshot_download(
                repo_id=REPO,
                repo_type="dataset",
                local_dir=str(DEST),
                allow_patterns=["data/*.parquet", "*.json", "*.md"],
                max_workers=8,
                resume_download=True,
            )
            print("[llava_next] DONE", flush=True)
            return 0
        except Exception as e:  # network blips, reaped children, etc.
            print(f"[llava_next] retry after error: {type(e).__name__}: {e}",
                  flush=True)
            time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
