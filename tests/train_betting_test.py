from datetime import datetime
from pathlib import Path

import pytest
import numpy as np

import train_betting
from ml.betting_network import BettingNetwork
from ml.population import Population
from ml.trainer import Trainer


# --------------------------------------------------
# Stage 1: required functions exist
# --------------------------------------------------


def test_required_training_functions_exist():
    assert callable(
        getattr(
            train_betting,
            "create_population",
            None,
        )
    )

    assert callable(
        getattr(
            train_betting,
            "create_trainer",
            None,
        )
    )

    assert callable(
        getattr(
            train_betting,
            "run_training",
            None,
        )
    )


def test_create_run_artifact_paths_groups_matching_files(
    tmp_path,
):
    timestamp = datetime(2026, 9, 23, 21, 7, 5)

    paths = train_betting.create_run_artifact_paths(
        base_directory=tmp_path / "runs",
        timestamp=timestamp,
    )

    expected_directory = (
        tmp_path / "runs" / "20260923_210705"
    )
    assert paths["run_directory"] == expected_directory
    assert paths["model_path"] == (
        expected_directory / "best_betting_network.npz"
    )
    assert paths["metrics_path"] == (
        expected_directory / "metrics.csv"
    )
    assert expected_directory.is_dir()


# --------------------------------------------------
# Stage 2: create Population
# --------------------------------------------------


def test_create_population():
    population = train_betting.create_population(
        population_size=14,
        elite_count=2,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=123,
    )

    assert isinstance(population, Population)
    assert population.population_size == 14
    assert population.elite_count == 2
    assert population.mutation_rate == 0.05
    assert population.mutation_strength == 0.1
    assert len(population.networks) == 14


def test_warm_start_population_continues_generation_and_keeps_fresh_half(
    tmp_path,
):
    run_directory = tmp_path / "previous-run"
    incumbent = BettingNetwork(seed=456)
    incumbent.save(run_directory / "best_betting_network.npz")
    train_betting.save_training_history(
        run_directory / "metrics.csv",
        [{"generation": 249}],
    )
    population = Population(14, 2, 0.05, 0.1, seed=123)
    untouched_fresh_network = population.networks[13].clone()

    state = train_betting.warm_start_population(
        population,
        run_directory,
    )

    assert state["next_generation"] == 250
    assert state["warm_network_count"] == 7
    assert state["fresh_network_count"] == 7
    assert population.generation_number == 250
    np.testing.assert_array_equal(
        population.networks[0].weights1,
        incumbent.weights1,
    )
    np.testing.assert_array_equal(
        population.networks[13].weights1,
        untouched_fresh_network.weights1,
    )


def test_carry_forward_league_copies_loadable_champions(tmp_path):
    previous_run = tmp_path / "previous-run"
    previous_league = previous_run / "league"
    network = BettingNetwork(seed=789)
    network.save(previous_league / "champion_0249_0.1234.npz")
    new_league = tmp_path / "new-run" / "league"

    copied_count = train_betting.carry_forward_league(
        previous_run,
        new_league,
    )

    assert copied_count == 1
    copied = BettingNetwork.load(
        new_league / "champion_0249_0.1234.npz"
    )
    np.testing.assert_array_equal(copied.weights1, network.weights1)


def test_parse_arguments_accepts_resume_directory_and_generation_count():
    arguments = train_betting.parse_arguments(
        [
            "--resume-from",
            "runs/old",
            "--generations",
            "100",
            "--power-chip-count",
            "2",
        ]
    )

    assert arguments.resume_from == Path("runs/old")
    assert arguments.generations == 100
    assert arguments.power_chip_count == 2


def test_main_passes_power_chip_count_to_trainer(
    monkeypatch,
):
    received_arguments = {}
    run_directory = Path("unused-test-run")

    class StopAfterTrainerCreation(Exception):
        pass

    monkeypatch.setattr(
        train_betting,
        "create_run_artifact_paths",
        lambda: {
            "run_directory": run_directory,
            "model_path": run_directory / "model.npz",
            "metrics_path": run_directory / "metrics.csv",
        },
    )
    monkeypatch.setattr(
        train_betting,
        "create_population",
        lambda **kwargs: object(),
    )

    def fake_create_trainer(**kwargs):
        received_arguments.update(kwargs)
        raise StopAfterTrainerCreation

    monkeypatch.setattr(
        train_betting,
        "create_trainer",
        fake_create_trainer,
    )

    with pytest.raises(StopAfterTrainerCreation):
        train_betting.main(
            [
                "--generations",
                "1",
                "--power-chip-count",
                "2",
            ]
        )

    assert received_arguments["power_chip_count"] == 2


# --------------------------------------------------
# Stage 3: create Trainer
# --------------------------------------------------


def test_create_trainer():
    population = Population(
        population_size=14,
        elite_count=2,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=123,
    )

    trainer = train_betting.create_trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        power_chip_count=2,
    )

    assert isinstance(trainer, Trainer)
    assert trainer.population is population
    assert trainer.starting_bankroll == 10_000
    assert trainer.rounds_per_tournament == 12
    assert trainer.decks == 6
    assert trainer.minimum_bet == 100
    assert trainer.hit_soft_17 is True
    assert trainer.max_hands == 4
    assert trainer.power_chip_count == 2


# --------------------------------------------------
# Fake objects for fast orchestration tests
# --------------------------------------------------


class FakeNetwork:
    def __init__(self, identifier):
        self.identifier = identifier
        self.saved_paths = []

    def save(self, file_path):
        self.saved_paths.append(Path(file_path))


class FakeTrainer:
    def __init__(self, generation_results):
        self.generation_results = (
            generation_results
        )

        self.call_count = 0
        self.received_tournament_counts = []

    def train_generation(
        self,
        tournaments_per_network,
        **kwargs,
    ):
        self.received_tournament_counts.append(
            tournaments_per_network
        )

        result = self.generation_results[
            self.call_count
        ]

        self.call_count += 1

        return result


def make_generation_results():
    return [
        {
            "generation": 0,
            "best_network_index": 3,
            "best_fitness": 10.0,
            "average_fitness": 5.0,
            "best_network": FakeNetwork("zero"),
        },
        {
            "generation": 1,
            "best_network_index": 8,
            "best_fitness": 15.0,
            "average_fitness": 7.0,
            "best_network": FakeNetwork("one"),
        },
        {
            "generation": 2,
            "best_network_index": 2,
            "best_fitness": 12.0,
            "average_fitness": 8.0,
            "best_network": FakeNetwork("two"),
        },
    ]


# --------------------------------------------------
# Stage 4: repeat requested generations
# --------------------------------------------------


def test_run_training_trains_requested_generations(
    tmp_path,
):
    trainer = FakeTrainer(
        make_generation_results()
    )

    train_betting.run_training(
        trainer=trainer,
        generations=3,
        tournaments_per_network=5,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
    )

    assert trainer.call_count == 3

    assert trainer.received_tournament_counts == [
        5,
        5,
        5,
    ]


# --------------------------------------------------
# Stage 5: print every generation
# --------------------------------------------------


def test_run_training_prints_generation_results(
    tmp_path,
):
    trainer = FakeTrainer(
        make_generation_results()
    )

    printed_messages = []

    train_betting.run_training(
        trainer=trainer,
        generations=3,
        tournaments_per_network=5,
        output_path=tmp_path / "winner.npz",
        print_fn=printed_messages.append,
    )

    combined_output = "\n".join(
        str(message)
        for message in printed_messages
    )

    assert "0" in combined_output
    assert "10" in combined_output
    assert "5" in combined_output

    assert "1" in combined_output
    assert "15" in combined_output
    assert "7" in combined_output

    assert "2" in combined_output
    assert "12" in combined_output
    assert "8" in combined_output


# --------------------------------------------------
# Stage 6: track the best network across generations
# --------------------------------------------------


def test_run_training_tracks_overall_best_network(
    tmp_path,
):
    results = make_generation_results()
    trainer = FakeTrainer(results)

    summary = train_betting.run_training(
        trainer=trainer,
        generations=3,
        tournaments_per_network=5,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
    )

    assert summary["best_generation"] == 1
    assert summary["best_fitness"] == 15.0

    assert (
        summary["best_network"].identifier
        == "one"
    )


# --------------------------------------------------
# Stage 7: save the overall best network
# --------------------------------------------------


def test_run_training_saves_overall_best_network(
    tmp_path,
):
    results = make_generation_results()
    trainer = FakeTrainer(results)

    output_path = tmp_path / "winner.npz"

    summary = train_betting.run_training(
        trainer=trainer,
        generations=3,
        tournaments_per_network=5,
        output_path=output_path,
        print_fn=lambda message: None,
    )

    winning_network = results[1][
        "best_network"
    ]

    assert winning_network.saved_paths == [
        output_path,
        tmp_path / "checkpoints" / "generation_0001_+15.0000.npz",
    ]

    assert (
        results[0]["best_network"].saved_paths
        == [
            output_path,
            tmp_path / "checkpoints" / "generation_0000_+10.0000.npz",
        ]
    )

    assert (
        results[2]["best_network"].saved_paths
        == []
    )

    assert summary["output_path"] == output_path


# --------------------------------------------------
# Stage 8: return generation history
# --------------------------------------------------


def test_run_training_returns_history(
    tmp_path,
):
    trainer = FakeTrainer(
        make_generation_results()
    )

    summary = train_betting.run_training(
        trainer=trainer,
        generations=3,
        tournaments_per_network=5,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
    )

    assert len(summary["history"]) == 3

    assert summary["history"][0][
        "generation"
    ] == 0

    assert summary["history"][1][
        "generation"
    ] == 1

    assert summary["history"][2][
        "generation"
    ] == 2


def test_saved_generation_prefers_raw_advancement_over_advantage(
    tmp_path,
):
    common = {
        "best_network_index": 0,
        "best_fitness": 10.0,
        "average_fitness": 5.0,
        "benchmark_fitness": 0.5,
        "benchmark_normal_fitness": 0.5,
        "benchmark_mid_fitness": 0.5,
        "benchmark_holdout_fitness": 0.5,
        "benchmark_minimum_holdout_fitness": 0.5,
    }
    results = [
        {
            **common,
            "generation": 0,
            "best_network": FakeNetwork("no-late"),
            "benchmark_late_fitness": 0.0,
            "benchmark_selection_fitness": 0.0,
            "benchmark_raw_selection_fitness": 0.9,
        },
        {
            **common,
            "generation": 1,
            "best_network": FakeNetwork("late-success"),
            "benchmark_late_fitness": 0.1,
            "benchmark_selection_fitness": 0.1,
            "benchmark_raw_selection_fitness": 0.5,
        },
    ]
    trainer = FakeTrainer(results)

    summary = train_betting.run_training(
        trainer=trainer,
        generations=2,
        tournaments_per_network=1,
        benchmark_tournaments_per_generation=10,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
    )

    assert summary["best_generation"] == 0
    assert summary["best_benchmark_late_fitness"] == 0.0


def test_independent_verification_rejects_lucky_checkpoint(
    tmp_path,
):
    def generation_result(generation, score):
        return {
            "generation": generation,
            "best_network_index": generation,
            "best_fitness": 10.0,
            "average_fitness": 5.0,
            "best_network": FakeNetwork(str(generation)),
            "benchmark_fitness": score,
            "benchmark_normal_fitness": score,
            "benchmark_mid_fitness": score,
            "benchmark_late_fitness": score,
            "benchmark_holdout_fitness": score,
            "benchmark_minimum_holdout_fitness": score,
            "benchmark_selection_fitness": score,
        }

    class VerifyingTrainer(FakeTrainer):
        def compare_checkpoint_networks(
            self,
            challenger,
            incumbent,
            tournament_count,
        ):
            challenger_score = (
                0.40 if challenger.identifier == "0" else 0.30
            )

            def metrics(score):
                return {
                    "benchmark_fitness": score,
                    "benchmark_normal_fitness": score,
                    "benchmark_mid_fitness": score,
                    "benchmark_late_fitness": score,
                    "benchmark_holdout_fitness": score,
                    "benchmark_minimum_holdout_fitness": score,
                    "benchmark_selection_fitness": score,
                }

            incumbent_metrics = (
                None if incumbent is None else metrics(0.40)
            )
            return metrics(challenger_score), incumbent_metrics

    trainer = VerifyingTrainer(
        [
            generation_result(0, 0.50),
            generation_result(1, 0.90),
        ]
    )

    summary = train_betting.run_training(
        trainer=trainer,
        generations=2,
        tournaments_per_network=1,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
        benchmark_tournaments_per_generation=10,
        checkpoint_verification_tournaments=100,
    )

    assert summary["best_generation"] == 0
    assert summary["best_benchmark_selection_fitness"] == 0.40


def test_resumed_incumbent_is_verified_before_replacement(tmp_path):
    incumbent = FakeNetwork("incumbent")
    challenger = FakeNetwork("challenger")
    result = {
        "generation": 250,
        "best_network_index": 0,
        "best_fitness": 10.0,
        "average_fitness": 5.0,
        "best_network": challenger,
        "benchmark_fitness": 0.4,
        "benchmark_normal_fitness": 0.4,
        "benchmark_mid_fitness": 0.4,
        "benchmark_late_fitness": 0.4,
        "benchmark_holdout_fitness": 0.4,
        "benchmark_minimum_holdout_fitness": 0.4,
        "benchmark_selection_fitness": 0.1,
        "benchmark_raw_selection_fitness": 0.4,
    }

    class ResumeTrainer(FakeTrainer):
        def compare_checkpoint_networks(
            self, candidate, current_incumbent, tournament_count
        ):
            assert candidate is challenger
            assert current_incumbent is incumbent
            return {
                "benchmark_selection_fitness": 0.30,
                "benchmark_raw_selection_fitness": 0.30,
                "checkpoint_accepted": False,
                "checkpoint_batches_won": 1,
                "checkpoint_batch_count": 3,
                "checkpoint_improvement": -0.05,
            }, {
                "benchmark_selection_fitness": 0.35,
                "benchmark_raw_selection_fitness": 0.35,
            }

    output_path = tmp_path / "new-run" / "winner.npz"
    summary = train_betting.run_training(
        trainer=ResumeTrainer([result]),
        generations=1,
        tournaments_per_network=1,
        output_path=output_path,
        print_fn=lambda message: None,
        benchmark_tournaments_per_generation=10,
        checkpoint_verification_tournaments=30,
        initial_best_network=incumbent,
        initial_best_generation=249,
    )

    assert summary["best_network"] is incumbent
    assert summary["best_generation"] == 249
    assert incumbent.saved_paths == [output_path]
    assert challenger.saved_paths == []


def test_rotating_verification_archives_risk_style_champion(tmp_path):
    result = {
        "generation": 0,
        "best_network_index": 0,
        "best_fitness": 10.0,
        "average_fitness": 5.0,
        "best_network": FakeNetwork("aggressive"),
        "benchmark_fitness": 0.4,
        "benchmark_normal_fitness": 0.4,
        "benchmark_mid_fitness": 0.4,
        "benchmark_late_fitness": 0.4,
        "benchmark_holdout_fitness": 0.4,
        "benchmark_minimum_holdout_fitness": 0.4,
        "benchmark_selection_fitness": 0.1,
        "benchmark_raw_selection_fitness": 0.4,
    }

    class PortfolioTrainer(FakeTrainer):
        def compare_checkpoint_networks(
            self, challenger, incumbent, tournament_count
        ):
            return {
                "benchmark_selection_fitness": 0.1,
                "benchmark_raw_selection_fitness": 0.4,
                "benchmark_top_two_rate": 0.4,
                "benchmark_first_place_rate": 0.2,
                "benchmark_average_position": 3.8,
                "benchmark_bankruptcy_rate": 0.15,
                "checkpoint_accepted": True,
                "checkpoint_batches_won": 3,
                "checkpoint_batch_count": 3,
                "checkpoint_improvement": None,
            }, None

    trainer = PortfolioTrainer([result])
    summary = train_betting.run_training(
        trainer=trainer,
        generations=1,
        tournaments_per_network=1,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
        benchmark_tournaments_per_generation=10,
        checkpoint_verification_tournaments=30,
    )

    expected_portfolio_path = (
        tmp_path
        / "champions"
        / "aggressive"
        / "generation_0000_0.4000.npz"
    )
    assert summary["portfolio_best_scores"] == {"aggressive": 0.4}
    assert summary["portfolio_paths"] == [expected_portfolio_path]
    assert expected_portfolio_path in result["best_network"].saved_paths


def test_final_audit_reports_without_changing_winner(tmp_path):
    results = make_generation_results()

    class AuditTrainer(FakeTrainer):
        def evaluate_robust_checkpoint(self, network, tournament_count):
            assert network.identifier == "one"
            assert tournament_count == 1_000
            return {
                "benchmark_raw_selection_fitness": 0.37,
                "benchmark_first_place_rate": 0.18,
                "benchmark_bankruptcy_rate": 0.06,
            }

    summary = train_betting.run_training(
        trainer=AuditTrainer(results),
        generations=3,
        tournaments_per_network=1,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
        final_audit_tournaments=1_000,
    )

    assert summary["best_generation"] == 1
    assert summary["final_audit_metrics"][
        "benchmark_raw_selection_fitness"
    ] == pytest.approx(0.37)


def test_run_training_stops_after_verified_patience(tmp_path):
    def generation_result(generation):
        return {
            "generation": generation,
            "best_network_index": generation,
            "best_fitness": 10.0,
            "average_fitness": 5.0,
            "best_network": FakeNetwork(str(generation)),
            "benchmark_fitness": 0.5,
            "benchmark_normal_fitness": 0.5,
            "benchmark_mid_fitness": 0.5,
            "benchmark_late_fitness": 0.5,
            "benchmark_holdout_fitness": 0.5,
            "benchmark_minimum_holdout_fitness": 0.5,
            "benchmark_selection_fitness": 0.5,
        }

    class StableTrainer(FakeTrainer):
        def compare_checkpoint_networks(
            self,
            challenger,
            incumbent,
            tournament_count,
        ):
            score = 0.5 if incumbent is None else 0.49
            challenger_metrics = {
                "benchmark_selection_fitness": score,
                "benchmark_raw_selection_fitness": score,
                "checkpoint_accepted": incumbent is None,
                "checkpoint_batches_won": 0 if incumbent else 3,
                "checkpoint_batch_count": 3,
                "checkpoint_improvement": None,
            }
            incumbent_metrics = (
                None
                if incumbent is None
                else {
                    "benchmark_selection_fitness": 0.5,
                    "benchmark_raw_selection_fitness": 0.5,
                }
            )
            return challenger_metrics, incumbent_metrics

    trainer = StableTrainer(
        [generation_result(generation) for generation in range(10)]
    )
    summary = train_betting.run_training(
        trainer=trainer,
        generations=10,
        tournaments_per_network=1,
        output_path=tmp_path / "winner.npz",
        print_fn=lambda message: None,
        benchmark_tournaments_per_generation=10,
        checkpoint_verification_tournaments=30,
        early_stopping_patience=2,
    )

    assert summary["early_stopped"] is True
    assert summary["completed_generations"] == 3
    assert summary["best_generation"] == 0


def test_run_training_persists_metrics_after_every_generation(
    tmp_path,
    monkeypatch,
):
    build_calls = []
    save_calls = []

    def fake_build_generation_record(
        generation_result,
        verified_score=None,
        checkpoint_saved=False,
    ):
        record = {
            "generation": generation_result["generation"],
            "verified_score": verified_score,
            "checkpoint_saved": checkpoint_saved,
        }
        build_calls.append(record.copy())
        return record

    def fake_save_training_history(output_path, records):
        save_calls.append(
            (
                Path(output_path),
                [record.copy() for record in records],
            )
        )

    monkeypatch.setattr(
        train_betting,
        "build_generation_record",
        fake_build_generation_record,
        raising=False,
    )
    monkeypatch.setattr(
        train_betting,
        "save_training_history",
        fake_save_training_history,
        raising=False,
    )

    metrics_output_path = tmp_path / "run" / "metrics.csv"
    trainer = FakeTrainer(make_generation_results())

    summary = train_betting.run_training(
        trainer=trainer,
        generations=3,
        tournaments_per_network=5,
        output_path=tmp_path / "winner.npz",
        metrics_output_path=metrics_output_path,
        print_fn=lambda message: None,
    )

    assert build_calls == [
        {
            "generation": 0,
            "verified_score": None,
            "checkpoint_saved": True,
        },
        {
            "generation": 1,
            "verified_score": None,
            "checkpoint_saved": True,
        },
        {
            "generation": 2,
            "verified_score": None,
            "checkpoint_saved": False,
        },
    ]
    assert [
        len(records)
        for _, records in save_calls
    ] == [1, 2, 3]
    assert all(
        output_path == metrics_output_path
        for output_path, _ in save_calls
    )
    assert summary["metrics_history"] == save_calls[-1][1]


# --------------------------------------------------
# Stage 9: validate training arguments
# --------------------------------------------------


@pytest.mark.parametrize(
    "generations",
    [0, -1],
)
def test_run_training_rejects_invalid_generations(
    tmp_path,
    generations,
):
    trainer = FakeTrainer(
        make_generation_results()
    )

    with pytest.raises(ValueError):
        train_betting.run_training(
            trainer=trainer,
            generations=generations,
            tournaments_per_network=5,
            output_path=tmp_path / "winner.npz",
            print_fn=lambda message: None,
        )


@pytest.mark.parametrize(
    "generations",
    [True, 1.5, "3"],
)
def test_run_training_rejects_invalid_generation_type(
    tmp_path,
    generations,
):
    trainer = FakeTrainer(
        make_generation_results()
    )

    with pytest.raises(TypeError):
        train_betting.run_training(
            trainer=trainer,
            generations=generations,
            tournaments_per_network=5,
            output_path=tmp_path / "winner.npz",
            print_fn=lambda message: None,
        )


@pytest.mark.parametrize(
    "tournaments_per_network",
    [0, -1],
)
def test_run_training_rejects_invalid_tournament_count(
    tmp_path,
    tournaments_per_network,
):
    trainer = FakeTrainer(
        make_generation_results()
    )

    with pytest.raises(ValueError):
        train_betting.run_training(
            trainer=trainer,
            generations=3,
            tournaments_per_network=(
                tournaments_per_network
            ),
            output_path=tmp_path / "winner.npz",
            print_fn=lambda message: None,
        )
