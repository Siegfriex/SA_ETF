from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import nbformat
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "paper2020_contract.yaml"
TRACE_PATH = ROOT / "reports" / "r2h" / "paper2020" / "paper_traceability.csv"
NOTEBOOK_PATH = ROOT / "notebooks" / "paper2020" / "00_SVR_GARCH_CUSUM기본설정.ipynb"


def load_contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_required_b1_artifacts_exist() -> None:
    paths = [
        NOTEBOOK_PATH,
        CONTRACT_PATH,
        TRACE_PATH,
        ROOT / "reports" / "r2h" / "paper2020" / "reproducibility_gap_register.md",
        ROOT / "reports" / "r2h" / "paper2020" / "environment_report.md",
    ]
    assert all(path.is_file() for path in paths)


def test_source_identity_and_equation_inventory() -> None:
    contract = load_contract()
    source = contract["source"]
    assert source["english_original_physical_pages"] == [30, 45]
    assert source["sha256"] == "97c95e308af8dd95e8d283f01674932f52349ab8b4d0ea2a93d4dbe45fcf3cd6"
    assert list(contract["equations"]) == [f"eq_{index}" for index in range(1, 10)]
    pdf = Path(source["local_pdf"])
    if pdf.exists():
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == source["sha256"]


def test_paper_rbf_mapping_fixture() -> None:
    x = np.array([1.0, -2.0])
    y = np.array([0.5, 1.0])
    squared_distance = np.sum((x - y) ** 2)
    for paper_gamma_squared in (0.1, 0.5, 1.0):
        sklearn_gamma = 1.0 / (2.0 * paper_gamma_squared)
        paper_value = np.exp(-squared_distance / (2.0 * paper_gamma_squared))
        sklearn_value = np.exp(-sklearn_gamma * squared_distance)
        assert np.isclose(paper_value, sklearn_value, rtol=0, atol=1e-15)


def test_cusum_golden_contract_arithmetic() -> None:
    golden = load_contract()["cusum"]["golden_test"]
    residuals = np.asarray(golden["residuals"], dtype=float)
    squared = residuals**2
    n = residuals.size
    centered = np.cumsum(squared) - (np.arange(1, n + 1) / n) * squared.sum()
    tau_squared = np.mean(residuals**4) - np.mean(squared) ** 2
    statistic = np.max(np.abs(centered)) / (np.sqrt(n) * np.sqrt(tau_squared))
    assert np.allclose(squared, golden["squared"])
    assert np.allclose(centered, golden["centered_path"])
    assert np.isclose(tau_squared, golden["tau_squared"])
    assert np.isclose(statistic, golden["statistic"])
    assert int(np.argmax(np.abs(centered)) + 1) == golden["argmax_one_based"]
    assert bool(statistic >= load_contract()["cusum"]["critical_value"]) is golden["reject_at_5pct"]


def test_all_table_targets_are_machine_readable() -> None:
    tables = load_contract()["simulation"]["target_tables"]
    assert list(tables) == [
        "table_1_garch",
        "table_2_agarch",
        "table_3_gjr_garch",
        "table_4_tgarch",
        "table_5_log_linear_garch",
        "table_6_model_changes",
    ]
    assert [len(table["rows"]) for table in tables.values()] == [7, 9, 7, 7, 7, 8]
    assert all(len(row["values"]) == 3 for table in tables.values() for row in table["rows"])


def test_empirical_targets_and_data_blocker() -> None:
    empirical = load_contract()["empirical"]
    assert empirical["snapshot_present_locally"] is False
    targets = {
        key: (value["target_change_date"], value["target_statistic"])
        for key, value in empirical["series"].items()
    }
    assert targets == {
        "sp500": ("2015-08-18", 2.4751),
        "kospi": ("2018-01-22", 1.5081),
        "krw_usd": ("2014-09-26", 4.2754),
    }


def test_traceability_covers_equations_tables_figures_and_cusum() -> None:
    with TRACE_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {row["requirement_id"] for row in rows}
    required = {
        *(f"EQ-{index:02d}" for index in range(1, 10)),
        *(f"TABLE-{index:02d}" for index in range(1, 7)),
        *(f"FIGURE-{index:02d}" for index in range(1, 5)),
        "CUSUM-STAT",
        "CUSUM-LOC",
    }
    assert required <= ids
    assert all(row["B1_status"] == "SPECIFIED" for row in rows)


def test_b00_is_valid_unexecuted_notebook_with_phase_gate() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    assert notebook.metadata.kernelspec.name == "hongik_26_2"
    assert notebook.metadata.paper2020.phase == "B1"
    assert notebook.metadata.paper2020.external_io is False
    assert notebook.metadata.paper2020.full_simulation is False
    assert all(cell.execution_count is None and cell.outputs == [] for cell in notebook.cells if cell.cell_type == "code")
    text = "\n".join("".join(cell.source) for cell in notebook.cells)
    assert "B2 NOT STARTED" in text
    code_text = "\n".join("".join(cell.source) for cell in notebook.cells if cell.cell_type == "code")
    assert "SVR(" not in code_text
    assert ".fit(" not in code_text
    assert "requests." not in code_text
