import numpy as np
import pytest

from ml.population import Population
from ml.trainer import (
    Trainer,
    advancement_fitness_for_tie,
    training_progress_fitness,
)


def make_trainer():
    population = Population(
        population_size=14,
        elite_count=2,
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

    return trainer


def test_holdout_advancement_scale_remains_comparable():
    assert advancement_fitness_for_tie(0, 1) == pytest.approx(
        1.05
    )
    assert advancement_fitness_for_tie(1, 2) == pytest.approx(
        1.0
    )


def test_awards_only_first_and_second_place():
    trainer = make_trainer()

    network_indices = [
        7,
        2,
        10,
        4,
        13,
        0,
        8,
    ]

    # Each tuple is:
    # (local tournament seat, ending bankroll)
    rankings = [
        (3, 16_000),
        (0, 15_000),
        (5, 14_000),
        (1, 13_000),
        (6, 12_000),
        (2, 11_000),
        (4, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    expected_scores = np.zeros(14)

    expected_scores[4] = 1.5
    expected_scores[7] = 1.0

    np.testing.assert_array_equal(
        trainer.population.fitness_scores,
        expected_scores,
    )


def test_two_way_first_place_tie_shares_points():
    trainer = make_trainer()

    network_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    rankings = [
        (2, 15_000),
        (5, 15_000),
        (0, 14_000),
        (1, 13_000),
        (3, 12_000),
        (4, 11_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    assert (
        trainer.population.fitness_scores[2]
        == pytest.approx(1.25)
    )

    assert (
        trainer.population.fitness_scores[5]
        == pytest.approx(1.25)
    )

    assert trainer.population.fitness_scores[0] == 0
    assert trainer.population.fitness_scores[1] == 0
    assert trainer.population.fitness_scores[3] == 0
    assert trainer.population.fitness_scores[4] == 0
    assert trainer.population.fitness_scores[6] == 0


def test_three_way_middle_tie_shares_points():
    trainer = make_trainer()

    network_indices = list(range(7))

    rankings = [
        (0, 17_000),
        (1, 16_000),
        (2, 14_000),
        (3, 14_000),
        (4, 14_000),
        (5, 12_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    assert trainer.population.fitness_scores[2] == 0
    assert trainer.population.fitness_scores[3] == 0
    assert trainer.population.fitness_scores[4] == 0

    assert trainer.population.fitness_scores[0] == 1.5
    assert trainer.population.fitness_scores[1] == 1.0
    assert trainer.population.fitness_scores[5] == 0
    assert trainer.population.fitness_scores[6] == 0


def test_all_players_tied_receive_average_points():
    trainer = make_trainer()

    network_indices = list(range(7))

    rankings = [
        (0, 10_000),
        (1, 10_000),
        (2, 10_000),
        (3, 10_000),
        (4, 10_000),
        (5, 10_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    # First- and second-place credit is shared by all seven players.
    expected_fitness = 2.5 / 7

    for network_index in network_indices:
        assert (
            trainer.population.fitness_scores[
                network_index
            ]
            == expected_fitness
        )


def test_fitness_accumulates_across_tournaments():
    trainer = make_trainer()

    network_indices = list(range(7))

    rankings = [
        (0, 16_000),
        (1, 15_000),
        (2, 14_000),
        (3, 13_000),
        (4, 12_000),
        (5, 11_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    assert trainer.population.fitness_scores[0] == 3.0
    assert trainer.population.fitness_scores[1] == 2.0
    assert trainer.population.fitness_scores[2] == 0
    assert trainer.population.fitness_scores[3] == 0
    assert trainer.population.fitness_scores[4] == 0
    assert trainer.population.fitness_scores[5] == 0
    assert trainer.population.fitness_scores[6] == 0


def test_training_reward_credits_closing_gap_to_second():
    trainer = make_trainer()
    network_indices = list(range(7))
    starting_bankrolls = [
        12_000,
        11_000,
        10_000,
        9_000,
        8_000,
        7_000,
        6_000,
    ]
    rankings = [
        (0, 12_000),
        (1, 11_000),
        (2, 10_900),
        (3, 9_000),
        (4, 8_000),
        (5, 7_000),
        (6, 6_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
        starting_bankrolls=starting_bankrolls,
    )

    assert trainer.population.fitness_scores[2] == pytest.approx(
        0.135
    )
    assert trainer.population.fitness_scores[3] == 0


def test_training_reward_credits_rank_improvement():
    trainer = make_trainer()
    network_indices = list(range(7))
    starting_bankrolls = [
        12_000,
        11_000,
        10_000,
        9_000,
        8_000,
        7_000,
        6_000,
    ]
    rankings = [
        (0, 12_000),
        (1, 11_000),
        (3, 10_500),
        (2, 10_000),
        (4, 8_000),
        (5, 7_000),
        (6, 6_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
        starting_bankrolls=starting_bankrolls,
    )

    # 0.05 for moving fourth to third, plus 0.1125 for
    # closing 75% of the original gap to second.
    assert trainer.population.fitness_scores[3] == pytest.approx(
        0.1625
    )


def test_training_reward_targets_first_when_already_second():
    trainer = make_trainer()
    network_indices = list(range(7))
    starting_bankrolls = [
        12_000,
        11_000,
        10_000,
        9_000,
        8_000,
        7_000,
        6_000,
    ]
    rankings = [
        (0, 12_000),
        (1, 11_900),
        (2, 10_000),
        (3, 9_000),
        (4, 8_000),
        (5, 7_000),
        (6, 6_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
        starting_bankrolls=starting_bankrolls,
    )

    # 0.18 for closing 90% of the gap to first, plus
    # 0.15 for improving the safety margin over third.
    assert trainer.population.fitness_scores[1] == pytest.approx(
        1.33
    )


def test_training_reward_credits_expanding_first_place_lead():
    trainer = make_trainer()
    network_indices = list(range(7))
    starting_bankrolls = [
        12_000,
        11_000,
        10_000,
        9_000,
        8_000,
        7_000,
        6_000,
    ]
    rankings = [
        (0, 13_000),
        (1, 11_000),
        (2, 10_000),
        (3, 9_000),
        (4, 8_000),
        (5, 7_000),
        (6, 6_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
        starting_bankrolls=starting_bankrolls,
    )

    assert trainer.population.fitness_scores[0] == pytest.approx(
        1.7
    )


def test_training_shaping_fades_out_by_generation_40():
    trainer = make_trainer()

    trainer.population.generation_number = 20
    assert trainer.training_shaping_multiplier() == pytest.approx(
        0.5
    )

    trainer.population.generation_number = 40
    assert trainer.training_shaping_multiplier() == 0


def test_gap_closure_reward_is_disabled_for_early_full_tournaments():
    starting = [12_000, 11_000, 10_000, 9_000, 8_000, 7_000, 6_000]
    final = [12_000, 11_000, 10_900, 9_000, 8_000, 7_000, 6_000]

    assert training_progress_fitness(
        starting,
        final,
        seat_index=2,
        gap_closure_multiplier=0.0,
    ) == 0.0
    assert training_progress_fitness(
        starting,
        final,
        seat_index=2,
        gap_closure_multiplier=1.0,
    ) == pytest.approx(0.135)


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("normal", 0.0),
        ("mid", 0.25),
        ("late", 1.0),
        ("extreme", 0.5),
    ],
)
def test_gap_closure_reward_is_stage_gated(stage, expected):
    assert Trainer.gap_closure_multiplier_for_stage(stage) == expected


@pytest.mark.parametrize(
    "roll,expected_stage",
    [
        (0.0, "normal"),
        (0.6499, "normal"),
        (0.65, "mid"),
        (0.8499, "mid"),
        (0.85, "late"),
        (0.9999, "late"),
    ],
)
def test_training_stage_distribution_boundaries(roll, expected_stage):
    trainer = make_trainer()

    class FixedRandom:
        def random(self):
            return roll

    trainer.population.random_generator = FixedRandom()

    assert trainer.select_training_stage() == expected_stage
