import numpy as np

from tournament.observation import PowerChipObservation
from tournament.power_chips import PowerChipAction


POWER_CHIP_FEATURE_COUNT = 15


def encode_power_chip_observation(observation):
    if not isinstance(observation, PowerChipObservation):
        raise TypeError("Expected a PowerChipObservation instance")

    total_rounds = max(1, observation.total_rounds)
    largest_bankroll = max(
        1.0,
        *(float(bankroll) for bankroll in observation.bankrolls),
    )
    bankroll_before_wager = max(
        1.0,
        float(observation.bankroll + observation.current_bet),
    )

    chip_counts = tuple(
        float(count)
        for count in observation.power_chip_counts
    )
    chip_scale = max(1.0, *chip_counts)
    own_chip_count = (
        chip_counts[observation.player_index]
        if 0 <= observation.player_index < len(chip_counts)
        else 0.0
    )
    opponent_chip_counts = tuple(
        count
        for player_index, count in enumerate(chip_counts)
        if player_index != observation.player_index
    )
    maximum_opponent_chips = max(
        opponent_chip_counts,
        default=0.0,
    )
    average_opponent_chips = (
        sum(opponent_chip_counts) / len(opponent_chip_counts)
        if opponent_chip_counts
        else 0.0
    )

    def legal_target_card_value(target_position):
        if target_position >= len(observation.legal_targets):
            return 0.0

        card_index = observation.legal_targets[target_position]
        if not 0 <= card_index < len(observation.hand_card_values):
            return 0.0

        return float(observation.hand_card_values[card_index]) / 11.0

    return np.asarray(
        [
            observation.round_number / total_rounds,
            observation.rounds_remaining / total_rounds,
            observation.bankroll / largest_bankroll,
            observation.current_bet / bankroll_before_wager,
            min(max(observation.hand_total, 0), 31) / 31.0,
            float(observation.hand_is_soft),
            observation.dealer_upcard_value / 11.0,
            own_chip_count / chip_scale,
            maximum_opponent_chips / chip_scale,
            average_opponent_chips / chip_scale,
            float(observation.action is PowerChipAction.REPLACE),
            float(observation.action is PowerChipAction.REHIT),
            min(len(observation.legal_targets), 2) / 2.0,
            legal_target_card_value(0),
            legal_target_card_value(1),
        ],
        dtype=float,
    )
    
