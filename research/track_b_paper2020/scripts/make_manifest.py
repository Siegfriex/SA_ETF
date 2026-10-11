"""Write MANIFEST.json: sha256 + size of every package file (excluding per-chunk MC jsonl, counted instead)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files, chunks = {}, {}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name == "MANIFEST.json" or ".pytest_cache" in p.parts:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("results/ablation/dgp/"):
            chunks["ablation_dgp_npz"] = chunks.get("ablation_dgp_npz", 0) + 1
            continue
        if rel.startswith("results/mc/"):
            lane = p.parent.name
            chunks[lane] = chunks.get(lane, 0) + 1
            continue
        files[rel] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (ROOT / "MANIFEST.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(), "mc_chunk_files": chunks, "files": files},
        indent=2, ensure_ascii=False))
    print(len(files), "files;", chunks)


if __name__ == "__main__":
    main()
