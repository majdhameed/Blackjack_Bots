import random

import pytest

from evaluate_betting_network import (
    RISK_TAKER_STRATEGY_NAMES,
    STRATEGY_NAMES,
    create_statistics,
    create_league_competitors,
    evaluate_network,
    record_top_two_credits,
)
from ml.betting_network import BettingNetwork


def test_risk_taker_statistics_include_aggressive_leaders():
    statistics = create_statistics(
        RISK_TAKER_STRATEGY_NAMES
    )

    assert "half_bankroll_leader_1" in statistics
    assert "half_bankroll_leader_2" in statistics
    assert "opening_all_in_leader" in statistics


def test_top_two_credit_goes_to_first_and_second():
    statistics = create_statistics()
    rankings = [
        (seat_index, 10_000 - seat_index * 100)
        for seat_index in range(7)
    ]

    record_top_two_credits(
        rankings,
        list(STRATEGY_NAMES),
        statistics,
    )

    assert statistics[STRATEGY_NAMES[0]][
        "top_two_credit"
    ] == 1.0
    assert statistics[STRATEGY_NAMES[1]][
        "top_two_credit"
    ] == 1.0

    for strategy_name in STRATEGY_NAMES[2:]:
        assert statistics[strategy_name][
            "top_two_credit"
        ] == 0.0


def test_three_way_first_tie_shares_two_slots():
    statistics = create_statistics()
    rankings = [
        (0, 10_000),
        (1, 10_000),
        (2, 10_000),
        (3, 9_000),
        (4, 8_000),
        (5, 7_000),
        (6, 6_000),
    ]

    record_top_two_credits(
        rankings,
        list(STRATEGY_NAMES),
        statistics,
    )

    for strategy_name in STRATEGY_NAMES[:3]:
        assert statistics[strategy_name][
            "top_two_credit"
        ] == pytest.approx(2 / 3)


def test_league_evaluation_loads_archived_champions(tmp_path):
    saved_network_path = tmp_path / "best.npz"
    league_directory = tmp_path / "league"
    BettingNetwork(seed=1).save(saved_network_path)
    BettingNetwork(seed=2).save(
        league_directory / "champion_0001_0.3000.npz"
    )
    BettingNetwork(seed=3).save(
        league_directory / "champion_0002_0.3200.npz"
    )

    competitors = create_league_competitors(
        saved_network_path,
        league_directory,
    )
    names = [name for name, _ in competitors]

    assert len(competitors) == 7
    assert "league_champion_1" in names
    assert "league_champion_2" in names


def test_evaluation_seed_covers_cards_and_restores_global_randomness(
    tmp_path,
):
    model_path = tmp_path / "network.npz"
    BettingNetwork(seed=7).save(model_path)

    random.seed(91)
    original_state = random.getstate()
    first = evaluate_network(
        model_path,
        number_of_tournaments=3,
        seed=2468,
    )
    assert random.getstate() == original_state

    random.seed(1357)
    second = evaluate_network(
        model_path,
        number_of_tournaments=3,
        seed=2468,
    )

    assert first == second
