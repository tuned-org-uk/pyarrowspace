import json
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "trec_covid_colab.ipynb"
RUNNER = ROOT / "neurips" / "Trec-covid" / "scripts" / "trec_covid_colab.py"
SPEC = importlib.util.spec_from_file_location("trec_covid_colab", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_trec_covid_colab_notebook_pins_source_and_inputs():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] in {"code", "markdown"}
    )

    assert "875073ccfe15e0f53214bf830a647f72a98caf26" in source
    assert "ce62140cb23feb9becf6270d0d1fe6d1" in source
    assert "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" in source
    assert '"fetch"' in source
    assert '"rev-parse"' in source
    assert "manifest.json" in source
    assert "real_trec_covid_results" in source
    assert "Test-clean/trec-covid" not in source


def test_notebook_cells_are_fresh_and_parseable():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell:{cell['id']}", "exec")


def test_task_validation_rejects_unknown_ids_and_duplicate_ids():
    qrels = {"q1": {"d1": 2, "d2": 0}}
    assert runner.validate_task_inputs(["d1", "d2"], ["q1"], qrels) == 2

    for corpus_ids, query_ids, invalid_qrels in (
        (["d1", "d1"], ["q1"], qrels),
        (["d1", "d2"], ["q1"], {"q1": {"missing": 1}}),
        (["d1"], ["q1"], {"q1": {"d1": -1}}),
        (["d1"], ["q1"], {"q1": {"d1": 0}}),
    ):
        try:
            runner.validate_task_inputs(corpus_ids, query_ids, invalid_qrels)
        except ValueError:
            continue
        raise AssertionError("Invalid task bundle inputs must be rejected")


def test_metrics_use_graded_ndcg_and_full_relevant_set_for_recall():
    result = runner.metrics_at_k(
        ["d2", "d1", "d3"],
        {"d1": 2, "d2": 0, "d3": 1, "d4": 3},
        k=2,
    )
    expected_dcg = 3.0 / math.log2(3)
    ideal_dcg = 7.0 + 3.0 / math.log2(3)
    assert result["ndcg@2"] == pytest.approx(expected_dcg / ideal_dcg)
    assert result["recall@2"] == pytest.approx(1 / 3)
    assert result["precision@2"] == pytest.approx(1 / 2)


def test_corpus_jsonl_is_read_incrementally(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"_id":"d1","text":"first"}\n{"_id":"d2","text":"second"}\n')

    records = runner._iter_jsonl(corpus)

    assert next(records) == {"_id": "d1", "text": "first"}
    assert next(records) == {"_id": "d2", "text": "second"}
