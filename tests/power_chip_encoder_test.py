from dataclasses import replace

import numpy as np

from ml.power_chip_encoder import (
    POWER_CHIP_FEATURE_COUNT,
    encode_power_chip_observation,
)
from tournament.observation import PowerChipObservation
from tournament.power_chips import PowerChipAction


def make_observation():
    return PowerChipObservation(
        round_number=4,
        total_rounds=12,
        rounds_remaining=8,
        player_index=0,
        round_player_index=0,
        hand_index=0,
        bankroll=9_900,
        bankrolls=(9_900, 10_100, 8_500),
        current_bet=100,
        hand_total=16,
        hand_is_soft=False,
        hand_card_values=(10, 6),
        dealer_upcard_value=10,
        power_chip_counts=(2, 1, 0),
        action=PowerChipAction.REPLACE,
        legal_targets=(0, 1),
    )


def test_power_chip_encoder_returns_fixed_finite_vector():
    features = encode_power_chip_observation(
        make_observation()
    )

    assert isinstance(features, np.ndarray)
    assert features.shape == (POWER_CHIP_FEATURE_COUNT,)
    assert np.issubdtype(features.dtype, np.floating)
    assert np.isfinite(features).all()


def test_power_chip_encoder_changes_for_action_type():
    observation = make_observation()
    replace_features = encode_power_chip_observation(observation)
    rehit_features = encode_power_chip_observation(
        replace(
            observation,
            action=PowerChipAction.REHIT,
        )
    )

    assert not np.array_equal(replace_features, rehit_features)


def test_power_chip_encoder_changes_for_wager_size():
    observation = make_observation()
    small_wager = encode_power_chip_observation(observation)
    large_wager = encode_power_chip_observation(
        replace(
            observation,
            bankroll=8_000,
            current_bet=2_000,
        )
    )

    assert not np.array_equal(small_wager, large_wager)


def test_power_chip_encoder_changes_for_visible_chip_counts():
    observation = make_observation()
    original_counts = encode_power_chip_observation(observation)
    changed_counts = encode_power_chip_observation(
        replace(
            observation,
            power_chip_counts=(1, 3, 2),
        )
    )

    assert not np.array_equal(original_counts, changed_counts)
