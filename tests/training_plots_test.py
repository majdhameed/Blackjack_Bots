import copy
import importlib
from pathlib import Path

import pytest

from ml.betting_network import (
    BETTING_ACTION_NAMES,
    LEGACY_BETTING_ACTION_NAMES,
)
from ml.training_metrics import save_training_history


def test_create_fitness_plot_writes_population_fitness_png(
    tmp_path,
):
    try:
        training_plots = importlib.import_module(
            "ml.training_plots"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create ml/training_plots.py for training graphs."
        )

    create_fitness_plot = getattr(
        training_plots,
        "create_fitness_plot",
        None,
    )
    assert callable(create_fitness_plot), (
        "ml.training_plots must expose a callable "
        "create_fitness_plot operation"
    )

    records = [
        {
            "generation": 0,
            "training_best": 20.0,
            "training_mean": 12.0,
            "training_worst": 5.0,
            "training_std": 3.0,
        },
        {
            "generation": 1,
            "training_best": 24.0,
            "training_mean": 15.0,
            "training_worst": 7.0,
            "training_std": 2.5,
        },
        {
            "generation": 2,
            "training_best": 27.0,
            "training_mean": 18.0,
            "training_worst": 9.0,
            "training_std": 2.0,
        },
    ]
    original_records = copy.deepcopy(records)
    output_path = tmp_path / "nested" / "fitness.png"

    returned_path = create_fitness_plot(
        records,
        output_path,
    )

    assert returned_path == Path(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )
    assert records == original_records


def test_create_evaluation_plot_writes_holdout_scores_png(
    tmp_path,
):
    training_plots = importlib.import_module(
        "ml.training_plots"
    )
    create_evaluation_plot = getattr(
        training_plots,
        "create_evaluation_plot",
        None,
    )
    assert callable(create_evaluation_plot), (
        "ml.training_plots must expose a callable "
        "create_evaluation_plot operation"
    )

    records = [
        {
            "generation": 0,
            "benchmark_score": 0.31,
            "normal_score": 0.42,
            "mid_score": 0.28,
            "late_score": 0.12,
            "full_tournament_score": 0.38,
            "minimum_control_score": 0.45,
            "robust_score": 0.29,
            "verified_score": None,
            "checkpoint_saved": False,
        },
        {
            "generation": 1,
            "benchmark_score": 0.39,
            "normal_score": 0.49,
            "mid_score": 0.39,
            "late_score": 0.23,
            "full_tournament_score": 0.62,
            "minimum_control_score": 0.54,
            "robust_score": 0.50,
            "verified_score": 0.433,
            "checkpoint_saved": True,
        },
        {
            "generation": 2,
            "benchmark_score": 0.37,
            "normal_score": 0.47,
            "mid_score": 0.32,
            "late_score": 0.26,
            "full_tournament_score": 0.62,
            "minimum_control_score": 0.54,
            "robust_score": 0.50,
            "verified_score": None,
            "checkpoint_saved": False,
        },
    ]
    original_records = copy.deepcopy(records)
    output_path = tmp_path / "nested" / "evaluation.png"

    returned_path = create_evaluation_plot(
        records,
        output_path,
    )

    assert returned_path == Path(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )
    assert records == original_records


def test_create_diversity_plot_writes_policy_diversity_png(
    tmp_path,
):
    training_plots = importlib.import_module(
        "ml.training_plots"
    )
    create_diversity_plot = getattr(
        training_plots,
        "create_diversity_plot",
        None,
    )
    assert callable(create_diversity_plot), (
        "ml.training_plots must expose a callable "
        "create_diversity_plot operation"
    )

    records = [
        {
            "generation": 0,
            "action_entropy": 0.91,
            "unique_policy_rate": 0.84,
        },
        {
            "generation": 1,
            "action_entropy": 0.78,
            "unique_policy_rate": 0.63,
        },
        {
            "generation": 2,
            "action_entropy": 0.69,
            "unique_policy_rate": 0.51,
        },
    ]
    original_records = copy.deepcopy(records)
    output_path = tmp_path / "nested" / "diversity.png"

    returned_path = create_diversity_plot(
        records,
        output_path,
    )

    assert returned_path == Path(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )
    assert records == original_records


def test_create_action_plot_writes_stacked_action_percentages_png(
    tmp_path,
):
    training_plots = importlib.import_module(
        "ml.training_plots"
    )
    create_action_plot = getattr(
        training_plots,
        "create_action_plot",
        None,
    )
    assert callable(create_action_plot), (
        "ml.training_plots must expose a callable "
        "create_action_plot operation"
    )

    action_mixes = [
        (0.70, 0.20, 0.10),
        (0.45, 0.35, 0.20),
        (0.25, 0.40, 0.35),
    ]
    records = []
    for generation, action_mix in enumerate(action_mixes):
        record = {
            "generation": generation,
            **{
                f"action_{action_name}": 0.0
                for action_name in BETTING_ACTION_NAMES
            },
        }
        record["action_minimum"] = action_mix[0]
        record["action_bet_05_percent"] = action_mix[1]
        record["action_bet_10_percent"] = action_mix[2]
        records.append(record)

    original_records = copy.deepcopy(records)
    output_path = tmp_path / "nested" / "actions.png"

    returned_path = create_action_plot(
        records,
        output_path,
    )

    assert returned_path == Path(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )
    assert records == original_records


def test_create_action_plot_accepts_legacy_action_columns(
    tmp_path,
):
    training_plots = importlib.import_module(
        "ml.training_plots"
    )
    records = []
    for generation in range(2):
        records.append(
            {
                "generation": generation,
                **{
                    f"action_{action_name}": (
                        1.0 / len(LEGACY_BETTING_ACTION_NAMES)
                    )
                    for action_name in LEGACY_BETTING_ACTION_NAMES
                },
            }
        )

    output_path = tmp_path / "legacy" / "actions.png"
    returned_path = training_plots.create_action_plot(
        records,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )


def test_create_contextual_probe_plot_writes_actions_and_bets_png(
    tmp_path,
):
    training_plots = importlib.import_module("ml.training_plots")
    records = [
        {
            "generation": generation,
            "probe_early_tied_action_index": generation + 1,
            "probe_early_tied_legal_bet": 100 * (generation + 1),
            "probe_late_safe_lead_action_index": 13 - generation,
            "probe_late_safe_lead_legal_bet": 300 - generation * 100,
        }
        for generation in range(3)
    ]
    output_path = tmp_path / "plots" / "contextual_probes.png"

    returned_path = training_plots.create_contextual_probe_plot(
        records,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_create_advantage_plot_writes_minimum_relative_png(tmp_path):
    training_plots = importlib.import_module("ml.training_plots")
    records = []
    for generation in range(3):
        records.append(
            {
                "generation": generation,
                "normal_advantage": -0.02 + generation * 0.02,
                "mid_advantage": -0.01 + generation * 0.02,
                "late_advantage": generation * 0.03,
                "full_tournament_advantage": generation * 0.01,
                "minimum_control_advantage": 0.0,
                "randomized_human_advantage": generation * 0.02,
                "robust_score": -0.01 + generation * 0.02,
            }
        )

    output_path = tmp_path / "advantage.png"
    returned_path = training_plots.create_advantage_plot(
        records,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_training_plots_loads_csv_and_writes_all_graphs(
    tmp_path,
):
    training_plots = importlib.import_module(
        "ml.training_plots"
    )
    generate_training_plots = getattr(
        training_plots,
        "generate_training_plots",
        None,
    )
    assert callable(generate_training_plots), (
        "ml.training_plots must expose a callable "
        "generate_training_plots operation"
    )

    records = []
    for generation in range(3):
        record = {
            "generation": generation,
            "training_best": 20.0 + generation,
            "training_mean": 12.0 + generation,
            "training_worst": 5.0 + generation,
            "training_std": 3.0,
            "benchmark_score": 0.30 + generation * 0.02,
            "normal_score": 0.40 + generation * 0.02,
            "mid_score": 0.30 + generation * 0.02,
            "late_score": 0.20 + generation * 0.02,
            "full_tournament_score": 0.40 + generation * 0.02,
            "minimum_control_score": 0.45,
            "robust_score": 0.35 + generation * 0.02,
            "verified_score": (
                0.37 if generation == 1 else None
            ),
            "checkpoint_saved": generation == 1,
            "league_promoted": False,
            "league_size": 3,
            "total_probe_decisions": 1680,
            "action_entropy": 0.80 - generation * 0.05,
            "unique_policy_count": 40 - generation,
            "unique_policy_rate": 0.60 - generation * 0.05,
            **{
                f"action_{action_name}": 0.0
                for action_name in BETTING_ACTION_NAMES
            },
        }
        record["action_minimum"] = 0.70 - generation * 0.10
        record["action_bet_05_percent"] = 0.20 + generation * 0.05
        record["action_bet_10_percent"] = (
            1.0
            - record["action_minimum"]
            - record["action_bet_05_percent"]
        )
        records.append(record)

    run_directory = tmp_path / "runs" / "test-run"
    metrics_path = run_directory / "metrics.csv"
    save_training_history(metrics_path, records)

    output_paths = generate_training_plots(metrics_path)

    expected_paths = {
        "fitness": run_directory / "fitness.png",
        "evaluation": run_directory / "evaluation.png",
        "diversity": run_directory / "diversity.png",
        "actions": run_directory / "actions.png",
    }
    assert output_paths == expected_paths
    for output_path in output_paths.values():
        assert output_path.is_file()
        assert output_path.stat().st_size > 1_000
        assert output_path.read_bytes()[:8] == (
            b"\x89PNG\r\n\x1a\n"
        )
