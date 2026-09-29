import copy
import csv
import importlib
from pathlib import Path

import pytest

from ml.betting_network import BETTING_ACTION_NAMES


def test_build_generation_record_flattens_training_and_diversity_metrics(
    capsys,
):
    try:
        training_metrics = importlib.import_module(
            "ml.training_metrics"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create ml/training_metrics.py for graph-ready "
            "generation records."
        )

    build_generation_record = getattr(
        training_metrics,
        "build_generation_record",
        None,
    )
    assert callable(build_generation_record), (
        "ml.training_metrics must expose a callable "
        "build_generation_record operation"
    )

    action_percentages = {
        action_name: (action_index + 1) / 55
        for action_index, action_name in enumerate(
            BETTING_ACTION_NAMES
        )
    }
    generation_result = {
        "generation": 12,
        "best_fitness": 31.5,
        "population_best_fitness": 36.0,
        "average_fitness": 24.0,
        "population_worst_fitness": 13.0,
        "population_fitness_std": 4.25,
        "benchmark_fitness": 0.39,
        "benchmark_normal_fitness": 0.49,
        "benchmark_mid_fitness": 0.39,
        "benchmark_late_fitness": 0.23,
        "benchmark_holdout_fitness": 0.62,
        "benchmark_minimum_holdout_fitness": 0.54,
        "benchmark_selection_fitness": 0.50,
        "league_promoted": True,
        "league_size": 4,
        "strategy_diversity": {
            "total_decisions": 1_680,
            "action_counts": {
                action_name: 0
                for action_name in BETTING_ACTION_NAMES
            },
            "action_percentages": action_percentages,
            "action_entropy": 0.78,
            "unique_policy_count": 41,
            "unique_policy_rate": 41 / 70,
        },
        "contextual_probe_results": {
            "early_tied": {
                "action_index": 1,
                "action_name": "minimum",
                "legal_bet": 100,
            },
        },
    }
    original_result = copy.deepcopy(generation_result)

    record = build_generation_record(
        generation_result,
        verified_score=0.433,
        checkpoint_saved=True,
    )
    default_record = build_generation_record(
        generation_result
    )

    expected_values = {
        "generation": 12,
        "training_best": 36.0,
        "training_mean": 24.0,
        "training_worst": 13.0,
        "training_std": 4.25,
        "selected_candidate_fitness": 31.5,
        "benchmark_score": 0.39,
        "normal_score": 0.49,
        "mid_score": 0.39,
        "late_score": 0.23,
        "full_tournament_score": 0.62,
        "minimum_control_score": 0.54,
        "robust_score": 0.50,
        "verified_score": 0.433,
        "checkpoint_saved": True,
        "league_promoted": True,
        "league_size": 4,
        "total_probe_decisions": 1_680,
        "action_entropy": 0.78,
        "unique_policy_count": 41,
        "unique_policy_rate": 41 / 70,
    }
    for name, expected_value in expected_values.items():
        assert record[name] == pytest.approx(expected_value)

    for action_name, percentage in action_percentages.items():
        assert record[
            f"action_{action_name}"
        ] == pytest.approx(percentage)

    assert record["probe_early_tied_action_index"] == 1
    assert record["probe_early_tied_legal_bet"] == 100

    assert generation_result == original_result
    assert default_record["verified_score"] is None
    assert default_record["checkpoint_saved"] is False
    assert capsys.readouterr().out == ""
    assert not any(
        isinstance(value, (dict, list, tuple, set))
        for value in record.values()
    ), "A graph-ready generation record must be flat"


def test_save_training_history_atomically_replaces_complete_csv(
    tmp_path,
    monkeypatch,
):
    training_metrics = importlib.import_module(
        "ml.training_metrics"
    )
    save_training_history = getattr(
        training_metrics,
        "save_training_history",
        None,
    )
    assert callable(save_training_history), (
        "ml.training_metrics must expose a callable "
        "save_training_history operation"
    )

    output_path = tmp_path / "nested" / "metrics.csv"
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp"
    )
    replace_calls = []
    original_replace = Path.replace

    def tracked_replace(source, destination):
        replace_calls.append(
            (source, Path(destination))
        )
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", tracked_replace)
    first = {
        "generation": 0,
        "training_best": 10.5,
        "verified_score": None,
        "checkpoint_saved": False,
    }
    second = {
        "generation": 1,
        "training_best": 12.0,
        "verified_score": 0.41,
        "checkpoint_saved": True,
    }
    third = {
        "generation": 2,
        "training_best": 13.25,
        "verified_score": 0.43,
        "checkpoint_saved": False,
    }

    save_training_history(output_path, [first, second])

    assert output_path.is_file()
    assert not temporary_path.exists()
    assert replace_calls[0] == (
        temporary_path,
        output_path,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    assert list(rows[0]) == list(first)
    assert [row["generation"] for row in rows] == ["0", "1"]
    assert rows[0]["verified_score"] == ""
    assert rows[1]["checkpoint_saved"] == "True"

    save_training_history(
        output_path,
        [first, second, third],
    )

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as metrics_file:
        replaced_rows = list(csv.DictReader(metrics_file))

    assert len(replaced_rows) == 3
    assert replaced_rows[-1]["generation"] == "2"
    assert not temporary_path.exists()
    assert replace_calls[1] == (
        temporary_path,
        output_path,
    )

    inconsistent_record = {
        "generation": 3,
        "training_best": 14.0,
    }
    with pytest.raises(ValueError):
        save_training_history(
            output_path,
            [first, inconsistent_record],
        )

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as metrics_file:
        preserved_rows = list(csv.DictReader(metrics_file))

    assert preserved_rows == replaced_rows


def test_load_training_history_restores_record_value_types(
    tmp_path,
):
    training_metrics = importlib.import_module(
        "ml.training_metrics"
    )
    load_training_history = getattr(
        training_metrics,
        "load_training_history",
        None,
    )
    assert callable(load_training_history), (
        "ml.training_metrics must expose a callable "
        "load_training_history operation"
    )

    records = [
        {
            "generation": 0,
            "training_best": 10.5,
            "verified_score": None,
            "checkpoint_saved": False,
            "league_promoted": True,
            "league_size": 3,
            "total_probe_decisions": 1680,
            "action_entropy": 0.75,
            "unique_policy_count": 41,
            "unique_policy_rate": 41 / 70,
            "action_minimum": 0.25,
        },
        {
            "generation": 1,
            "training_best": 12.25,
            "verified_score": 0.433,
            "checkpoint_saved": True,
            "league_promoted": False,
            "league_size": 4,
            "total_probe_decisions": 1680,
            "action_entropy": 0.68,
            "unique_policy_count": 35,
            "unique_policy_rate": 0.5,
            "action_minimum": 0.20,
        },
    ]
    metrics_path = tmp_path / "run" / "metrics.csv"

    training_metrics.save_training_history(
        metrics_path,
        records,
    )
    loaded_records = load_training_history(metrics_path)

    assert loaded_records == records
    assert type(loaded_records[0]["generation"]) is int
    assert type(loaded_records[0]["training_best"]) is float
    assert loaded_records[0]["verified_score"] is None
    assert type(loaded_records[0]["checkpoint_saved"]) is bool
    assert type(loaded_records[0]["league_size"]) is int
