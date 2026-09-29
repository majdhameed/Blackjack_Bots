import copy
import random

import numpy as np
import pytest

from ml.population import Population
from ml.trainer import Trainer


class MinimumActionNetwork:
    def forward(self, features):
        return 0.01

    def preferred_action(self, features):
        return 1


def make_trainer(
    population_size=14,
    seed=123,
):
    population = Population(
        population_size=population_size,
        elite_count=2,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=seed,
    )

    return Trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
    )


def test_evaluate_generation_rejects_invalid_count():
    trainer = make_trainer()

    for invalid_count in (
        0,
        -1,
        1.5,
        True,
    ):
        expected_error = (
            TypeError
            if isinstance(invalid_count, bool)
            or not isinstance(invalid_count, int)
            else ValueError
        )

        with pytest.raises(expected_error):
            trainer.evaluate_generation(
                invalid_count
            )


def test_matched_minimum_control_has_zero_benchmark_advantage():
    trainer = make_trainer()

    result = trainer.evaluate_advantage_over_minimum(
        MinimumActionNetwork(),
        number_of_tournaments=4,
        stage="normal",
        lineup_kind="disciplined",
        full_tournament=True,
    )

    assert result["candidate_score"] == result["minimum_score"]
    assert result["advantage"] == 0.0
    assert result["candidate_top_two_rate"] == pytest.approx(
        result["minimum_top_two_rate"]
    )
    assert result["top_two_advantage"] == pytest.approx(0.0)
    assert 0.0 <= result["candidate_bankruptcy_rate"] <= 1.0
    assert 1.0 <= result["candidate_average_position"] <= 7.0


def test_matched_minimum_control_adds_zero_baseline_fitness():
    trainer = make_trainer()
    trainer.population.networks[0] = MinimumActionNetwork()

    score = trainer.evaluate_network_against_baselines(
        network_index=0,
        number_of_tournaments=4,
    )

    assert score == 0.0
    assert trainer.population.fitness_scores[0] == 0.0


def test_evaluate_generation_uses_every_network():
    trainer = make_trainer()
    evaluated_groups = []

    def fake_evaluate_group(network_indices):
        evaluated_groups.append(
            list(network_indices)
        )

        for network_index in network_indices:
            trainer.population.add_fitness(
                network_index,
                1,
            )

    trainer.evaluate_group = fake_evaluate_group

    scores = trainer.evaluate_generation(
        tournaments_per_network=3
    )

    assert len(evaluated_groups) == 6

    for network_index in range(14):
        assert scores[network_index] == 3


def test_evaluate_generation_makes_groups_of_seven():
    trainer = make_trainer()
    evaluated_groups = []

    def fake_evaluate_group(network_indices):
        evaluated_groups.append(
            list(network_indices)
        )

    trainer.evaluate_group = fake_evaluate_group

    trainer.evaluate_generation(
        tournaments_per_network=2
    )

    assert len(evaluated_groups) == 4

    for group in evaluated_groups:
        assert len(group) == 7
        assert len(set(group)) == 7


def test_evaluate_generation_resets_old_fitness():
    trainer = make_trainer()

    trainer.population.fitness_scores[:] = 100

    def fake_evaluate_group(network_indices):
        for network_index in network_indices:
            trainer.population.add_fitness(
                network_index,
                2,
            )

    trainer.evaluate_group = fake_evaluate_group

    scores = trainer.evaluate_generation(
        tournaments_per_network=1
    )

    np.testing.assert_array_equal(
        scores,
        np.full(14, 2.0),
    )


def test_returned_fitness_is_a_copy():
    trainer = make_trainer()

    def fake_evaluate_group(network_indices):
        for network_index in network_indices:
            trainer.population.add_fitness(
                network_index,
                1,
            )

    trainer.evaluate_group = fake_evaluate_group

    scores = trainer.evaluate_generation(1)

    scores[0] = 999

    assert (
        trainer.population.fitness_scores[0]
        == 1
    )


def test_train_generation_returns_summary():
    trainer = make_trainer(
        population_size=14
    )

    expected_scores = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ],
        dtype=float,
    )

    def fake_evaluate_generation(
        tournaments_per_network,
    ):
        trainer.population.fitness_scores = (
            expected_scores.copy()
        )

        return expected_scores.copy()

    trainer.evaluate_generation = (
        fake_evaluate_generation
    )

    result = trainer.train_generation(
        tournaments_per_network=1
    )

    assert result["generation"] == 0
    assert result["best_network_index"] == 13
    assert result["best_fitness"] == 14
    assert result["average_fitness"] == pytest.approx(
        7.5
    )

    assert result["best_network"] is not None

    assert (
        trainer.population.generation_number
        == 1
    )

    assert np.all(
        trainer.population.fitness_scores == 0
    )


def test_train_generation_best_network_is_snapshot():
    trainer = make_trainer()

    def fake_evaluate_generation(
        tournaments_per_network,
    ):
        scores = np.arange(
            14,
            dtype=float,
        )

        trainer.population.fitness_scores = (
            scores.copy()
        )

        return scores.copy()

    trainer.evaluate_generation = (
        fake_evaluate_generation
    )

    winning_network = (
        trainer.population.networks[13]
    )

    original_weights = (
        winning_network.weights1.copy()
    )

    result = trainer.train_generation(1)

    np.testing.assert_array_equal(
        result["best_network"].weights1,
        original_weights,
    )

    assert (
        result["best_network"]
        is not winning_network
    )


def test_benchmark_selects_best_of_top_training_candidates():
    trainer = make_trainer()
    scores = np.arange(14, dtype=float)

    trainer.population.networks[13].biases3[0] = 0.1
    trainer.population.networks[12].biases3[0] = 0.9
    trainer.population.networks[11].biases3[0] = 0.2

    def fake_evaluate_generation(
        tournaments_per_network,
    ):
        trainer.population.fitness_scores = scores.copy()
        return scores.copy()

    trainer.evaluate_generation = (
        fake_evaluate_generation
    )
    trainer.evaluate_network_against_fixed_benchmark = (
        lambda network, tournament_count, stage=None, **kwargs: float(
            network.biases3[0]
        )
    )

    result = trainer.train_generation(
        tournaments_per_network=1,
        benchmark_tournaments=10,
        benchmark_candidate_count=3,
    )

    assert result["best_network_index"] == 12
    assert result["best_fitness"] == 12
    assert result["benchmark_fitness"] == pytest.approx(
        0.9
    )
    assert result["benchmark_parent_indices"] == [12, 11]


def test_benchmark_reproduction_preserves_risk_families():
    population = Population(
        population_size=14,
        elite_count=3,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=123,
    )
    trainer = Trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
    )
    scores = np.arange(14, dtype=float)
    markers = {13: (0.9, 0.15), 12: (0.8, 0.05), 11: (0.7, 0.0)}
    for index, (score, _) in markers.items():
        trainer.population.networks[index].biases3[0] = score

    def fake_evaluate_generation(tournaments_per_network):
        trainer.population.fitness_scores = scores.copy()
        return scores.copy()

    def fake_benchmark(network, tournament_count, **kwargs):
        score = float(network.biases3[0])
        bankruptcy = next(
            risk for marker, risk in markers.values() if marker == score
        )
        return {
            "candidate_score": score,
            "minimum_score": 0.0,
            "advantage": score,
            "candidate_bankruptcy_rate": bankruptcy,
        }

    trainer.evaluate_generation = fake_evaluate_generation
    trainer.evaluate_advantage_over_minimum = fake_benchmark

    result = trainer.train_generation(
        tournaments_per_network=1,
        benchmark_tournaments=10,
        benchmark_candidate_count=3,
    )

    assert result["benchmark_parent_indices"] == [13, 11, 12]
    assert result["benchmark_parent_styles"] == [
        "aggressive",
        "safe",
        "balanced",
    ]


def test_benchmark_prefers_higher_advancement_over_relative_advantage():
    trainer = make_trainer()
    scores = np.arange(14, dtype=float)
    trainer.population.networks[13].biases3[0] = 0.9
    trainer.population.networks[12].biases3[0] = 0.5

    def fake_evaluate_generation(tournaments_per_network):
        trainer.population.fitness_scores = scores.copy()
        return scores.copy()

    def fake_paired_benchmark(
        network,
        tournament_count,
        stage=None,
        **kwargs,
    ):
        marker = float(network.biases3[0])
        advantage = 0.0
        if stage == "late" and marker < 0.7:
            advantage = 0.1
        return {
            "candidate_score": marker,
            "minimum_score": marker - advantage,
            "advantage": advantage,
        }

    trainer.evaluate_generation = fake_evaluate_generation
    trainer.evaluate_advantage_over_minimum = fake_paired_benchmark

    result = trainer.train_generation(
        tournaments_per_network=1,
        benchmark_tournaments=10,
        benchmark_candidate_count=2,
    )

    assert result["best_network_index"] == 13
    assert result["benchmark_late_fitness"] == pytest.approx(
        0.9
    )
    assert result["benchmark_late_advantage"] == pytest.approx(0.0)


def test_benchmark_selection_includes_full_tournament_holdout():
    trainer = make_trainer()
    scores = np.arange(14, dtype=float)
    trainer.population.networks[13].biases3[0] = 0.6
    trainer.population.networks[12].biases3[0] = 0.5

    def fake_evaluate_generation(tournaments_per_network):
        trainer.population.fitness_scores = scores.copy()
        return scores.copy()

    def fake_benchmark(
        network,
        tournament_count,
        stage=None,
        full_tournament=False,
        **kwargs,
    ):
        marker = float(network.biases3[0])
        if full_tournament and marker > 0.55:
            return 0.0
        return marker

    trainer.evaluate_generation = fake_evaluate_generation
    trainer.evaluate_network_against_fixed_benchmark = (
        fake_benchmark
    )

    result = trainer.train_generation(
        tournaments_per_network=1,
        benchmark_tournaments=10,
        benchmark_candidate_count=2,
    )

    assert result["best_network_index"] == 12
    assert result["benchmark_holdout_fitness"] == pytest.approx(
        0.5
    )


def test_benchmark_candidates_receive_common_random_numbers():
    trainer = make_trainer()
    scores = np.arange(14, dtype=float)
    calls_by_marker = {}

    trainer.population.networks[13].biases3[0] = 0.6
    trainer.population.networks[12].biases3[0] = 0.5

    def fake_evaluate_generation(tournaments_per_network):
        trainer.population.fitness_scores = scores.copy()
        return scores.copy()

    def random_benchmark(network, tournament_count, **kwargs):
        marker = float(network.biases3[0])
        calls_by_marker.setdefault(marker, []).append(
            float(trainer.population.random_generator.random())
        )
        return 0.5

    trainer.evaluate_generation = fake_evaluate_generation
    trainer.evaluate_network_against_fixed_benchmark = random_benchmark

    trainer.train_generation(
        tournaments_per_network=1,
        benchmark_tournaments=10,
        benchmark_candidate_count=2,
    )

    assert calls_by_marker[0.6] == calls_by_marker[0.5]


def test_checkpoint_verification_uses_rotating_matched_batches():
    trainer = make_trainer()

    class MarkedNetwork:
        def __init__(self, marker):
            self.marker = marker

    def fake_robust_checkpoint(network, tournament_count):
        noise = (
            float(trainer.population.random_generator.random())
            + random.random()
        )
        score = network.marker + noise
        return {
            "benchmark_selection_fitness": score,
            "benchmark_raw_selection_fitness": score,
        }

    trainer.evaluate_robust_checkpoint = fake_robust_checkpoint
    numpy_state = copy.deepcopy(
        trainer.population.random_generator.bit_generator.state
    )
    python_state = random.getstate()

    first, incumbent = trainer.compare_checkpoint_networks(
        MarkedNetwork(0.01),
        MarkedNetwork(0.0),
        benchmark_tournaments=30,
    )
    second, _ = trainer.compare_checkpoint_networks(
        MarkedNetwork(0.01),
        MarkedNetwork(0.0),
        benchmark_tournaments=30,
    )

    assert first["benchmark_selection_fitness"] != pytest.approx(
        second["benchmark_selection_fitness"]
    )
    assert first["checkpoint_batches_won"] == 3
    assert first["checkpoint_accepted"] is True
    assert first["checkpoint_improvement"] == pytest.approx(0.01)
    assert incumbent is not None
    expected_generator = np.random.default_rng()
    expected_generator.bit_generator.state = copy.deepcopy(numpy_state)
    expected_generator.integers(0, 2**32, size=6)
    assert (
        trainer.population.random_generator.bit_generator.state
        == expected_generator.bit_generator.state
    )
    assert random.getstate() == python_state


def test_checkpoint_verification_requires_minimum_improvement():
    trainer = make_trainer()

    class MarkedNetwork:
        def __init__(self, marker):
            self.marker = marker

    trainer.evaluate_robust_checkpoint = (
        lambda network, tournament_count: {
            "benchmark_selection_fitness": network.marker,
            "benchmark_raw_selection_fitness": network.marker,
        }
    )

    challenger, _ = trainer.compare_checkpoint_networks(
        MarkedNetwork(0.004),
        MarkedNetwork(0.0),
        benchmark_tournaments=30,
    )

    assert challenger["checkpoint_batches_won"] == 3
    assert challenger["checkpoint_accepted"] is False


def test_checkpoint_verification_requires_majority_of_batches():
    trainer = make_trainer()

    class MarkedNetwork:
        def __init__(self, challenger):
            self.challenger = challenger

    challenger_differences = (0.05, -0.01, -0.01)
    call_count = 0

    def fake_robust_checkpoint(network, tournament_count):
        nonlocal call_count
        batch_index = call_count // 2
        call_count += 1
        score = (
            challenger_differences[batch_index]
            if network.challenger
            else 0.0
        )
        return {
            "benchmark_selection_fitness": score,
            "benchmark_raw_selection_fitness": score,
        }

    trainer.evaluate_robust_checkpoint = fake_robust_checkpoint
    challenger, _ = trainer.compare_checkpoint_networks(
        MarkedNetwork(True),
        MarkedNetwork(False),
        benchmark_tournaments=30,
    )

    assert challenger["checkpoint_improvement"] == pytest.approx(0.01)
    assert challenger["checkpoint_batches_won"] == 1
    assert challenger["checkpoint_accepted"] is False


def test_checkpoint_verification_requires_unanimous_batches():
    trainer = make_trainer()

    class MarkedNetwork:
        def __init__(self, challenger):
            self.challenger = challenger

    challenger_differences = (0.03, 0.03, -0.01)
    call_count = 0

    def fake_robust_checkpoint(network, tournament_count):
        nonlocal call_count
        batch_index = call_count // 2
        call_count += 1
        score = (
            challenger_differences[batch_index]
            if network.challenger
            else 0.0
        )
        return {
            "benchmark_selection_fitness": score,
            "benchmark_raw_selection_fitness": score,
        }

    trainer.evaluate_robust_checkpoint = fake_robust_checkpoint
    challenger, _ = trainer.compare_checkpoint_networks(
        MarkedNetwork(True),
        MarkedNetwork(False),
        benchmark_tournaments=30,
    )

    assert challenger["checkpoint_improvement"] > 0.01
    assert challenger["checkpoint_batches_won"] == 2
    assert challenger["checkpoint_unanimous_batches"] is False
    assert challenger["checkpoint_accepted"] is False


def test_checkpoint_verification_rejects_category_regression():
    trainer = make_trainer()

    class MarkedNetwork:
        def __init__(self, challenger):
            self.challenger = challenger

    def fake_robust_checkpoint(network, tournament_count):
        score = 0.52 if network.challenger else 0.50
        human_score = 0.44 if network.challenger else 0.50
        return {
            "benchmark_selection_fitness": score,
            "benchmark_raw_selection_fitness": score,
            "benchmark_normal_fitness": score,
            "benchmark_mid_fitness": score,
            "benchmark_late_fitness": score,
            "benchmark_holdout_fitness": score,
            "benchmark_minimum_holdout_fitness": score,
            "benchmark_human_holdout_fitness": human_score,
            "benchmark_bankruptcy_rate": 0.0,
        }

    trainer.evaluate_robust_checkpoint = fake_robust_checkpoint
    challenger, _ = trainer.compare_checkpoint_networks(
        MarkedNetwork(True),
        MarkedNetwork(False),
        benchmark_tournaments=30,
    )

    assert challenger["checkpoint_batches_won"] == 3
    assert challenger["checkpoint_category_gate_passed"] is False
    assert challenger["checkpoint_worst_category_delta"] == pytest.approx(
        -0.06
    )
    assert "benchmark_human_holdout_fitness" in challenger[
        "checkpoint_regressed_categories"
    ]
    assert challenger["checkpoint_accepted"] is False


def test_checkpoint_verification_rejects_bankruptcy_regression():
    trainer = make_trainer()

    class MarkedNetwork:
        def __init__(self, challenger):
            self.challenger = challenger

    def fake_robust_checkpoint(network, tournament_count):
        score = 0.52 if network.challenger else 0.50
        bankruptcy = 0.03 if network.challenger else 0.0
        return {
            "benchmark_selection_fitness": score,
            "benchmark_raw_selection_fitness": score,
            "benchmark_bankruptcy_rate": bankruptcy,
        }

    trainer.evaluate_robust_checkpoint = fake_robust_checkpoint
    challenger, _ = trainer.compare_checkpoint_networks(
        MarkedNetwork(True),
        MarkedNetwork(False),
        benchmark_tournaments=30,
    )

    assert challenger["checkpoint_bankruptcy_gate_passed"] is False
    assert challenger["checkpoint_bankruptcy_delta"] == pytest.approx(0.03)
    assert challenger["checkpoint_accepted"] is False


def test_broad_benchmark_winner_is_added_to_champion_league():
    trainer = make_trainer()
    scores = np.arange(14, dtype=float)

    def fake_evaluate_generation(tournaments_per_network):
        trainer.population.fitness_scores = scores.copy()
        return scores.copy()

    trainer.evaluate_generation = fake_evaluate_generation
    trainer.evaluate_network_against_fixed_benchmark = (
        lambda network, tournament_count, **kwargs: 0.5
    )

    result = trainer.train_generation(
        tournaments_per_network=1,
        benchmark_tournaments=10,
        benchmark_candidate_count=1,
    )

    assert result["league_promoted"] is True
    assert result["league_size"] == 1


def test_league_training_table_contains_archived_champion():
    trainer = make_trainer()
    trainer.champion_league.append(
        trainer.population.networks[0].clone()
    )

    competitors = trainer.create_training_baseline_competitors(
        trainer.population.networks[1],
        table_kind="league",
    )

    names = [name for name, _ in competitors]
    assert len(competitors) == 7
    assert "neural" in names
    assert "league_champion_1" in names
