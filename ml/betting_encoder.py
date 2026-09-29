# Reserved for converting BettingObservation values into model-ready numeric features. 
from tournament.observation import BettingObservation


BETTING_FEATURE_COUNT = 63

def encode_betting_observation(observation: BettingObservation) -> tuple[float, ...]:
    if not isinstance(observation, BettingObservation):
        raise TypeError("Observation must be a betting observation")

    player_count = len(observation.bankrolls)

    if player_count != 7:
        raise ValueError("player count incorrect")

    if len(observation.active_players) != 7:
        raise ValueError("active_players must contain 7 values")

    if len(observation.current_bets) != 7:
        raise ValueError("current_bets must contain 7 values")

    if len(observation.bets_placed) != 7:
        raise ValueError("bets_placed must contain 7 values")

    if len(observation.card_value_counts) != 10:
        raise ValueError(
            "card_value_counts must contain 10 values"
        )

    if observation.total_rounds <= 0:
        raise ValueError(
            "total_rounds must be positive"
        )

    money_scale = max(max(observation.bankrolls), observation.minimum_bet, 1)

    features = [
        observation.round_number / observation.total_rounds,
        observation.rounds_remaining / observation.total_rounds,
        observation.bankroll / money_scale,
        observation.minimum_bet / money_scale,
        observation.betting_position / 6
    ]

    features.extend(
        bankroll / money_scale
        for bankroll in observation.bankrolls
    )

    features.extend(
        1.0 if active else 0.0
        for active in observation.active_players
    )

    features.extend(
        bet / money_scale
        for bet in observation.current_bets
    )

    features.extend(
        1.0 if bet_placed else 0.0
        for bet_placed in observation.bets_placed
    )

    visible_card_scaled = max(observation.cards_seen, 1)

    features.extend(
        card_count / visible_card_scaled
        for card_count in observation.card_value_counts
    )

    features.append(observation.running_count / visible_card_scaled)

    features.append(max(-10, min(observation.true_count, 10))/10)

    features.append(observation.shoe_penetration)

    active_bankrolls = [
        bankroll
        for player_index, bankroll in enumerate(
            observation.bankrolls
        )
        if observation.active_players[player_index]
    ]

    players_ahead = sum(
        bankroll > observation.bankroll
        for bankroll in active_bankrolls
    )

    rank_scale = max(len(active_bankrolls) - 1, 1)
    leader_bankroll = max(active_bankrolls)

    sorted_bankrolls = sorted(
        active_bankrolls,
        reverse=True,
    )

    advancement_cutoff = (
        sorted_bankrolls[1]
        if len(sorted_bankrolls) > 1
        else sorted_bankrolls[0]
    )

    features.extend(
        (
            players_ahead / rank_scale,
            (
                leader_bankroll
                - observation.bankroll
            ) / money_scale,
            (
                advancement_cutoff
                - observation.bankroll
            ) / money_scale,
            1.0 if players_ahead <= 1 else 0.0,
            max(observation.current_bets) / money_scale,
            1.0 if observation.rounds_remaining == 0 else 0.0,
        )
    )

    features.extend(
        (
            observation.previous_bet / money_scale,
            # A player's prior outcome does not alter the odds of the next
            # blackjack hand.  Keep these legacy positions neutral so saved
            # network shapes remain compatible without rewarding loss chasing.
            0.0,
            0.0,
            0.0,
            1.0 if observation.has_previous_round else 0.0,
        )
    )

    opponent_scale = max(player_count - 1, 1)
    features.extend(
        (
            min(1.0, observation.largest_opponent_previous_bet_fraction),
            min(1.0, observation.average_opponent_previous_bet_fraction),
            observation.opponents_over_ten_percent / opponent_scale,
            observation.opponents_over_twenty_five_percent / opponent_scale,
            min(1.0, observation.opponent_bet_volatility),
            observation.opponents_with_loss_streak / opponent_scale,
        )
    )

    if len(features) != BETTING_FEATURE_COUNT:
        raise ValueError(
            f"Expected {BETTING_FEATURE_COUNT} features, got {len(features)}"
        )
    return tuple(float(value) for value in features)
