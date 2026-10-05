# Immutable snapshots passed to bots. They expose the information a bot may
# use without giving it mutable access to the tournament engine.
from dataclasses import dataclass

from tournament.power_chips import PowerChipAction


# State available while a player is selecting a wager.
@dataclass(frozen=True)
class BettingObservation:
    round_number: int
    total_rounds: int
    rounds_remaining: int
    player_index: int
    round_player_index: int
    betting_position: int
    minimum_bet: int
    bankroll: float
    bankrolls: tuple[float]
    active_players: tuple[bool]
    current_bets: tuple[float]
    bets_placed: tuple[bool]
    betting_order : tuple[int]
    card_value_counts: tuple
    cards_seen: int
    running_count: int
    true_count: float
    cards_remaining: int
    decks_remaining: float
    shoe_penetration: float
    previous_bet: float = 0.0
    previous_bankroll_change: float = 0.0
    previous_result: float = 0.0
    consecutive_losses: int = 0
    has_previous_round: bool = False
    largest_opponent_previous_bet_fraction: float = 0.0
    average_opponent_previous_bet_fraction: float = 0.0
    opponents_over_ten_percent: int = 0
    opponents_over_twenty_five_percent: int = 0
    opponent_bet_volatility: float = 0.0
    opponents_with_loss_streak: int = 0

# State available while a player is selecting a blackjack action.
@dataclass(frozen=True)
class ActionObservation:
    round_number: int
    total_rounds: int
    rounds_remaining: int
    player_index: int
    round_player_index: int
    hand_index: int
    bankroll: float
    bankrolls: tuple
    current_bet: float
    player_bets: tuple
    hand_total: int
    hand_is_soft: bool
    hand_card_values: tuple
    hand_came_from_split: bool
    dealer_upcard_value: int
    legal_actions: tuple
    betting_order: tuple
    active_players: tuple
    hit_soft_17: bool
    card_value_counts: tuple
    cards_seen: int
    running_count: int
    true_count: float
    cards_remaining: int
    decks_remaining: float
    shoe_penetration: float


@dataclass(frozen=True)
class PowerChipObservation:
    round_number: int
    total_rounds: int
    rounds_remaining: int
    player_index: int
    round_player_index: int
    hand_index: int
    bankroll: float
    bankrolls: tuple
    current_bet: float
    hand_total: int
    hand_is_soft: bool
    hand_card_values: tuple
    dealer_upcard_value: int
    power_chip_counts: tuple
    action: PowerChipAction
    legal_targets: tuple

