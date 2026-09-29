from dataclasses import replace

import pytest

from ml.betting_encoder import (
    BETTING_FEATURE_COUNT,
    encode_betting_observation,
)
from tournament.observation import (
    BettingObservation,
)


def make_observation():
    return BettingObservation(
        round_number=4,
        total_rounds=12,
        rounds_remaining=8,

        player_index=2,
        round_player_index=2,
        betting_position=3,

        minimum_bet=100,
        bankroll=8_000,

        bankrolls=(
            10_000,
            9_000,
            8_000,
            12_000,
            11_000,
            7_000,
            14_000,
        ),

        active_players=(
            True,
            True,
            True,
            True,
            True,
            False,
            True,
        ),

        current_bets=(
            100,
            500,
            0,
            0,
            0,
            0,
            0,
        ),

        bets_placed=(
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ),

        betting_order=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ),

        card_value_counts=(
            1,  # Ace
            1,  # 2
            1,  # 3
            1,  # 4
            1,  # 5
            1,  # 6
            1,  # 7
            1,  # 8
            1,  # 9
            1,  # ten-valued
        ),

        cards_seen=10,
        running_count=2,
        true_count=1.5,

        cards_remaining=250,
        decks_remaining=250 / 52,
        shoe_penetration=0.20,
    )


def test_encoder_returns_expected_float_features():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    assert isinstance(features, tuple)
    assert len(features) == BETTING_FEATURE_COUNT


def test_previous_bet_is_kept_but_own_outcome_history_is_neutralized():
    observation = replace(
        make_observation(),
        previous_bet=500,
        previous_bankroll_change=-500,
        previous_result=-1.0,
        consecutive_losses=2,
        has_previous_round=True,
    )

    features = encode_betting_observation(observation)

    assert features[52] == pytest.approx(500 / 14_000)
    assert features[53:56] == (0.0, 0.0, 0.0)
    assert features[56] == 1.0

    assert all(
        isinstance(value, float)
        for value in features
    )


def test_own_win_and_loss_histories_encode_identically():
    base = replace(
        make_observation(),
        previous_bet=500,
        has_previous_round=True,
    )
    after_win = replace(
        base,
        previous_bankroll_change=500,
        previous_result=1.0,
        consecutive_losses=0,
    )
    after_losses = replace(
        base,
        previous_bankroll_change=-2_000,
        previous_result=-1.0,
        consecutive_losses=4,
    )

    assert encode_betting_observation(after_win) == (
        encode_betting_observation(after_losses)
    )


def test_opponent_risk_features_are_explicit():
    observation = replace(
        make_observation(),
        largest_opponent_previous_bet_fraction=0.50,
        average_opponent_previous_bet_fraction=0.18,
        opponents_over_ten_percent=4,
        opponents_over_twenty_five_percent=2,
        opponent_bet_volatility=0.16,
        opponents_with_loss_streak=3,
    )

    features = encode_betting_observation(observation)

    assert features[57:63] == pytest.approx(
        (0.50, 0.18, 4 / 6, 2 / 6, 0.16, 3 / 6)
    )


def test_general_features():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    money_scale = 14_000

    assert features[0] == pytest.approx(
        4 / 12
    )

    assert features[1] == pytest.approx(
        8 / 12
    )

    assert features[2] == pytest.approx(
        8_000 / money_scale
    )

    assert features[3] == pytest.approx(
        100 / money_scale
    )

    assert features[4] == pytest.approx(
        3 / 6
    )


def test_each_bankroll_is_encoded_separately():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    money_scale = 14_000

    expected_bankroll_features = tuple(
        bankroll / money_scale
        for bankroll in observation.bankrolls
    )

    # Features 5 through 11 are bankrolls.
    assert features[5:12] == pytest.approx(
        expected_bankroll_features
    )

    # Ensures the acting bankroll was not
    # accidentally repeated seven times.
    assert len(set(features[5:12])) > 1


def test_active_players_become_binary_features():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    # Features 12 through 18 are active flags.
    assert features[12:19] == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
    )


def test_current_bets_are_normalized():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    money_scale = 14_000

    expected_bets = tuple(
        bet / money_scale
        for bet in observation.current_bets
    )

    # Features 19 through 25 are current bets.
    assert features[19:26] == pytest.approx(
        expected_bets
    )


def test_bets_placed_become_binary_features():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    # Features 26 through 32 are placed flags.
    assert features[26:33] == (
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_card_counts_are_divided_by_cards_seen():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    # Features 33 through 42 are card buckets.
    assert features[33:43] == pytest.approx(
        (0.1,) * 10
    )


def test_running_count_is_normalized():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    # Feature 43 is running_count / cards_seen.
    assert features[43] == pytest.approx(
        2 / 10
    )


def test_true_count_is_normalized():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    # Feature 44 is true count divided by 10.
    assert features[44] == pytest.approx(
        1.5 / 10
    )


def test_shoe_penetration_is_included():
    observation = make_observation()

    features = encode_betting_observation(
        observation
    )

    assert features[45] == pytest.approx(
        0.20
    )


def test_tournament_situation_features_are_explicit():
    features = encode_betting_observation(
        make_observation()
    )

    money_scale = 14_000

    assert features[46] == pytest.approx(5 / 5)
    assert features[47] == pytest.approx(
        6_000 / money_scale
    )
    assert features[48] == pytest.approx(
        4_000 / money_scale
    )
    assert features[49] == 0.0
    assert features[50] == pytest.approx(
        500 / money_scale
    )
    assert features[51] == 0.0


def test_final_round_and_top_two_are_explicit():
    observation = replace(
        make_observation(),
        round_number=12,
        rounds_remaining=0,
        bankroll=13_000,
        bankrolls=(
            13_000,
            9_000,
            8_000,
            12_000,
            11_000,
            7_000,
            14_000,
        ),
    )

    features = encode_betting_observation(
        observation
    )

    assert features[46] == pytest.approx(1 / 5)
    assert features[49] == 1.0
    assert features[51] == 1.0


def test_zero_cards_seen_does_not_divide_by_zero():
    observation = replace(
        make_observation(),
        card_value_counts=(0,) * 10,
        cards_seen=0,
        running_count=0,
        true_count=0.0,
    )

    features = encode_betting_observation(
        observation
    )

    assert features[33:43] == (0.0,) * 10
    assert features[43] == 0.0
    assert features[44] == 0.0


def test_positive_true_count_is_clamped():
    observation = replace(
        make_observation(),
        true_count=20.0,
    )

    features = encode_betting_observation(
        observation
    )

    assert features[44] == 1.0


def test_negative_true_count_is_clamped():
    observation = replace(
        make_observation(),
        true_count=-20.0,
    )

    features = encode_betting_observation(
        observation
    )

    assert features[44] == -1.0


def test_wrong_player_count_raises_error():
    observation = replace(
        make_observation(),
        bankrolls=(10_000,) * 6,
    )

    with pytest.raises(ValueError):
        encode_betting_observation(
            observation
        )


def test_wrong_active_player_length_raises_error():
    observation = replace(
        make_observation(),
        active_players=(True,) * 6,
    )

    with pytest.raises(ValueError):
        encode_betting_observation(
            observation
        )


def test_wrong_current_bet_length_raises_error():
    observation = replace(
        make_observation(),
        current_bets=(0,) * 6,
    )

    with pytest.raises(ValueError):
        encode_betting_observation(
            observation
        )


def test_wrong_bets_placed_length_raises_error():
    observation = replace(
        make_observation(),
        bets_placed=(False,) * 6,
    )

    with pytest.raises(ValueError):
        encode_betting_observation(
            observation
        )


def test_wrong_card_count_length_raises_error():
    observation = replace(
        make_observation(),
        card_value_counts=(0,) * 9,
    )

    with pytest.raises(ValueError):
        encode_betting_observation(
            observation
        )


def test_zero_total_rounds_raises_error():
    observation = replace(
        make_observation(),
        total_rounds=0,
    )

    with pytest.raises(ValueError):
        encode_betting_observation(
            observation
        )


def test_invalid_observation_type_raises_error():
    with pytest.raises(TypeError):
        encode_betting_observation(
            "not an observation"
        )
