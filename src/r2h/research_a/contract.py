"""Track A 데이터 계약 로더 · run_id · 입력 데이터 버전.

계약 파일(config/research_a_contract.yaml)이 경로·정의의 단일 출처다. 이 모듈은 그것을 읽고,
산출물마다 붙일 run_id 와 input_data_version 을 만든다. 기존 src/eda_utils 는 수정하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

KST = timezone(timedelta(hours=9))
CONTRACT_REL = Path("config") / "research_a_contract.yaml"


def find_root(start: str | Path | None = None) -> Path:
    """작업 트리 루트 = research_a 계약 파일이 있는 첫 상위 디렉터리."""
    p = Path(start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / CONTRACT_REL).exists():
            return q
    raise FileNotFoundError(f"{CONTRACT_REL} not found walking up from {p}")


@dataclass(frozen=True)
class Contract:
    root: Path
    spec: dict[str, Any]

    # ---- 경로: raw 는 공유 절대경로(READ-ONLY), 나머지는 작업 트리 상대
    @property
    def raw_dir(self) -> Path:
        i = self.spec["inputs"]
        return Path(os.environ.get(i["raw_dir_env"], i["raw_dir"]))

    @property
    def interim_dir(self) -> Path:
        return Path(self.spec["inputs"]["interim_dir"])

    def path(self, key: str) -> Path:
        return self.root / self.spec["inputs"][key]

    @property
    def expected(self) -> dict[str, Any]:
        return self.spec["inputs"]["expected"]

    @property
    def reference(self) -> str:
        return self.spec["analysis_roles"]["reference"]

    @property
    def base_panel(self) -> list[str]:
        return list(self.spec["analysis_roles"]["base_panel"])

    @property
    def roles(self) -> dict[str, str]:
        return dict(self.spec["analysis_roles"]["roles"])

    @property
    def dq_trim_cells(self) -> dict[str, list[str]]:
        return dict(self.spec["returns"]["variants"]["dq_trim_canonical"]["nan_cells"])

    def session_sync_of(self, ticker: str) -> str:
        for k, v in self.spec["session_sync"].items():
            if ticker in v:
                return k
        raise KeyError(ticker)

    def exposure_group_of(self, ticker: str) -> str:
        for g, members in self.spec["analysis_roles"]["exposure_groups"].items():
            if ticker in members:
                return g
        return ticker  # 자기 자신이 독립 exposure

    @property
    def report_dir(self) -> Path:
        return self.root / self.spec["reproducibility"]["report_dir"]

    def derived_dir(self, run_id: str) -> Path:
        return self.root / "data" / "processed" / "research_a" / run_id


def load_contract(root: str | Path | None = None) -> Contract:
    r = find_root(root)
    with open(r / CONTRACT_REL, encoding="utf-8") as f:
        return Contract(root=r, spec=yaml.safe_load(f))


def input_data_version(manifest: pd.DataFrame) -> str:
    """raw 20 파일 내용 지문. ticker 정렬 후 'ticker:sha256' 줄을 해시 → 순서·추가·변경에 민감."""
    lines = sorted(f"{t}:{s}" for t, s in zip(manifest["ticker"], manifest["sha256"]))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def git_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_branch(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "branch", "--show-current"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_run_id(version: str, phase: str = "A1", now: datetime | None = None) -> str:
    now = now or datetime.now(KST)
    return f"{phase}-{now.strftime('%Y%m%dT%H%M%S')}-{version[:8]}"


def runtime_info() -> dict[str, str]:
    import numpy, scipy  # noqa: E401 — 버전 기록용
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "pyyaml": yaml.__version__}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path
