from blackjack.actions import Action
import pytest
from agents.all_in_agent import AllInAgent
from blackjack.player import Player
from tournament.tournament import Tournament


from blackjack.actions import Action


class StandBot:
    def choose_bet(self, betting_observation):
        return min(
            betting_observation.bankroll,
            betting_observation.minimum_bet,
        )

    def choose_action(self, action_observation):
        legal_actions = (
            action_observation.legal_actions
        )

        if Action.STAND in legal_actions:
            return Action.STAND

        raise ValueError(
            "StandBot has no legal action"
        )


def make_tournament(
    number_of_rounds=1,
    player_count=3,
):
    players = [
        Player(10_000)
        for _ in range(player_count)
    ]

    bots = [
        StandBot()
        for _ in range(player_count)
    ]

    tournament = Tournament(
        players=players,
        bots=bots,
        number_of_rounds=number_of_rounds,
        decks=6,
        minimum_bet=100,
        hit_soft_17=False,
        max_hands=4,
    )

    return tournament


def test_active_player_indices():
    tournament = make_tournament()

    tournament.players[1].bankroll = 0

    assert (
        tournament.get_active_player_indices()
        == [0, 2]
    )


def test_bot_matches_round_player_after_elimination():
    tournament = make_tournament()

    tournament.players[1].bankroll = 0

    tournament.start_next_round()

    # Local round index 0 maps to tournament player 0.
    assert (
        tournament.get_bot_for_round_player(0)
        is tournament.bots[0]
    )

    # Local round index 1 maps to tournament player 2.
    assert (
        tournament.get_bot_for_round_player(1)
        is tournament.bots[2]
    )


def test_start_next_round_creates_table_round():
    tournament = make_tournament()

    table_round = tournament.start_next_round()

    assert table_round is not None
    assert tournament.current_table_round is table_round
    assert tournament.current_round_number == 1
    assert tournament.active_player_indices == [
        0,
        1,
        2,
    ]


def test_place_round_bets():
    tournament = make_tournament()

    tournament.start_next_round()
    tournament.place_round_bets()

    table_round = tournament.current_table_round

    assert table_round.betting_complete() is True

    for player in table_round.players:
        assert len(player.hands) == 1
        assert player.get_hand(0).bet == 100
        assert player.bankroll == 9_900


def test_all_in_bot_can_bet_fractional_bankroll():
    tournament = make_tournament(player_count=2)
    tournament.players[0].bankroll = 150.5
    tournament.bots[0] = AllInAgent()

    tournament.start_next_round()
    tournament.place_round_bets()

    assert tournament.players[0].get_hand(0).bet == 150.5
    assert tournament.players[0].bankroll == 0


def test_betting_observation_includes_previous_round_state():
    tournament = make_tournament(number_of_rounds=2)

    tournament.play_one_round()
    tournament.start_next_round()
    round_player_index = (
        tournament.current_table_round.betting_order[0]
    )
    observation = tournament.build_betting_observation(
        round_player_index
    )

    assert observation.has_previous_round is True
    assert observation.previous_bet == 100
    assert observation.previous_bankroll_change == (
        observation.bankroll - 10_000
    )
    assert observation.previous_result in (-1.0, 0.0, 1.0)


def test_betting_observation_summarizes_opponent_risk():
    tournament = make_tournament(number_of_rounds=2)
    tournament.start_next_round()
    tournament.previous_bets = [100, 200, 3_000]
    tournament.consecutive_losses = [0, 2, 0]

    observation = tournament.build_betting_observation(
        tournament.active_player_indices.index(0)
    )

    assert observation.largest_opponent_previous_bet_fraction == 0.30
    assert observation.average_opponent_previous_bet_fraction == pytest.approx(
        0.16
    )
    assert observation.opponents_over_ten_percent == 1
    assert observation.opponents_over_twenty_five_percent == 1
    assert observation.opponent_bet_volatility == pytest.approx(0.14)
    assert observation.opponents_with_loss_streak == 1


def test_begin_player_actions_deals_cards():
    tournament = make_tournament()

    tournament.start_next_round()
    tournament.place_round_bets()

    tournament.begin_player_actions()

    table_round = tournament.current_table_round

    for player in table_round.players:
        assert len(player.get_hand(0).hand.cards) == 2

    assert len(table_round.dealer.hand.cards) == 2
    assert table_round.naturals_checked is True


def test_play_one_round():
    tournament = make_tournament(
        number_of_rounds=1
    )

    outcomes = tournament.play_one_round()

    assert outcomes is not None
    assert tournament.current_round_number == 1
    assert len(tournament.round_history) == 1
    assert tournament.current_table_round is None
    assert tournament.is_over is True


def test_complete_twelve_round_tournament():
    tournament = make_tournament(
        number_of_rounds=12
    )

    rankings = tournament.play_tournament()

    assert tournament.is_over is True
    assert tournament.current_round_number == 12
    assert len(tournament.round_history) == 12
    assert len(rankings) == 3


def test_rankings_are_sorted_by_bankroll():
    tournament = make_tournament()

    tournament.players[0].bankroll = 8_000
    tournament.players[1].bankroll = 12_000
    tournament.players[2].bankroll = 10_000

    rankings = tournament.get_rankings()

    assert rankings == [
        (1, 12_000),
        (2, 10_000),
        (0, 8_000),
    ]


def test_starting_bettor_rotates():
    tournament = make_tournament(
        number_of_rounds=2
    )

    tournament.play_one_round()

    first_round = tournament.round_history[0]

    tournament.play_one_round()

    second_round = tournament.round_history[1]

    assert first_round.betting_order == [
        0,
        1,
        2,
    ]

    assert second_round.betting_order == [
        1,
        2,
        0,
    ]


def test_tournament_ends_with_one_active_player():
    tournament = make_tournament(
        number_of_rounds=12
    )

    tournament.players[1].bankroll = 0
    tournament.players[2].bankroll = 0

    rankings = tournament.play_tournament()

    assert tournament.is_over is True
    assert tournament.current_round_number == 0
    assert len(tournament.round_history) == 0

    assert rankings[0] == (0, 10_000)
