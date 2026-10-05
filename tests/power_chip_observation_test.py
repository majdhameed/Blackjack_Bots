from dataclasses import FrozenInstanceError

import pytest

from tournament.observation import PowerChipObservation
from tournament.power_chips import PowerChipAction


def test_power_chip_observation_is_an_immutable_bot_snapshot():
    observation = PowerChipObservation(
        round_number=4,
        total_rounds=12,
        rounds_remaining=8,
        player_index=2,
        round_player_index=1,
        hand_index=0,
        bankroll=9_900,
        bankrolls=(10_200, 9_500, 9_900),
        current_bet=100,
        hand_total=17,
        hand_is_soft=False,
        hand_card_values=(8, 7, 2),
        dealer_upcard_value=10,
        power_chip_counts=(2, 0, 1),
        action=PowerChipAction.REHIT,
        legal_targets=(2,),
    )

    assert observation.action is PowerChipAction.REHIT
    assert observation.legal_targets == (2,)
    assert observation.power_chip_counts == (2, 0, 1)
    assert observation.power_chip_counts[observation.player_index] == 1

    with pytest.raises(FrozenInstanceError):
        observation.hand_total = 18
