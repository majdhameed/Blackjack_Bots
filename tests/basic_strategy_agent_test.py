from agents.basic_strategy_agent import BasicStrategyAgent
from blackjack.actions import Action
from tournament.observation import ActionObservation


def test_basic_strategy_agent_uses_every_card_in_hand():
    observation = ActionObservation(
        round_number=1,
        total_rounds=12,
        rounds_remaining=11,
        player_index=0,
        round_player_index=0,
        hand_index=0,
        bankroll=9_900,
        bankrolls=(9_900, 10_000),
        current_bet=100,
        player_bets=(100, 100),
        hand_total=18,
        hand_is_soft=False,
        hand_card_values=(10, 8),
        hand_came_from_split=False,
        dealer_upcard_value=10,
        legal_actions=(Action.HIT, Action.STAND, Action.DOUBLE),
        betting_order=(0, 1),
        active_players=(True, True),
        hit_soft_17=False,
        card_value_counts=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        cards_seen=0,
        running_count=0,
        true_count=0.0,
        cards_remaining=312,
        decks_remaining=6.0,
        shoe_penetration=0.0,
    )

    assert BasicStrategyAgent().choose_action(observation) is Action.STAND
