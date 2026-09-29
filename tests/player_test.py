import pytest

from blackjack.cards import Card
from blackjack.player import Player
from blackjack.player_hand import PlayerHand
from tournament.power_chips import PowerChipAction


def give_two_cards(player, hand_index=0):
    player.add_card(hand_index, Card(5, "Hearts"))
    player.add_card(hand_index, Card(6, "Spades"))


def test_invalid_starting_bankroll():
    with pytest.raises(ValueError):
        Player(0)


def test_place_bet_creates_player_hand():
    player = Player(10_000)

    player.place_bet(1_000, minimum_bet=100)

    assert player.bankroll == 9_000
    assert len(player.hands) == 1
    assert isinstance(player.hands[0], PlayerHand)
    assert player.hands[0].bet == 1_000
    assert player.active_hand_index == 0


def test_bet_below_minimum():
    player = Player(10_000)

    with pytest.raises(ValueError):
        player.place_bet(50, minimum_bet=100)


def test_short_all_in_below_minimum():
    player = Player(50)

    player.place_bet(50, minimum_bet=100)

    assert player.bankroll == 0
    assert player.hands[0].bet == 50


def test_cannot_bet_more_than_bankroll():
    player = Player(1_000)

    with pytest.raises(ValueError):
        player.place_bet(1_001, minimum_bet=100)


def test_bet_must_use_table_minimum_chip_increment():
    player = Player(10_000)

    with pytest.raises(ValueError):
        player.place_bet(155, minimum_bet=100)


def test_cannot_place_second_bet():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    with pytest.raises(ValueError):
        player.place_bet(500, minimum_bet=100)


def test_add_card_to_selected_hand():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    card = Card("A", "Spades")

    player.add_card(0, card)

    assert player.hands[0].hand.cards[0] is card


def test_reject_invalid_card():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    with pytest.raises(TypeError):
        player.add_card(0, "Ace of Spades")


def test_invalid_hand_index():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    with pytest.raises(IndexError):
        player.stand(1)


def test_stand_selected_hand():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    give_two_cards(player)

    player.stand(0)

    assert player.hands[0].has_stood is True


def test_double_down():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    give_two_cards(player)

    player.double_down(0, 1_000)

    assert player.bankroll == 8_000
    assert player.hands[0].bet == 2_000
    assert player.hands[0].has_doubled is True
    assert player.hands[0].has_stood is False


def test_finish_double():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    give_two_cards(player)
    player.double_down(0, 1_000)

    player.add_card(0, Card(10, "Clubs"))
    player.finish_double(0)

    assert player.hands[0].has_stood is True


def test_cannot_double_more_than_original_bet():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    give_two_cards(player)

    with pytest.raises(ValueError):
        player.double_down(0, 1_001)


def test_cannot_double_more_than_bankroll():
    player = Player(1_500)
    player.place_bet(1_000, minimum_bet=100)
    give_two_cards(player)

    with pytest.raises(ValueError):
        player.double_down(0, 1_000)


def test_surrender():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    give_two_cards(player)

    player.surrender(0)

    assert player.bankroll == 9_500
    assert player.hands[0].has_surrendered is True
    assert player.hands[0].has_stood is True


def test_split_creates_two_hands():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    player.add_card(0, Card(8, "Hearts"))
    player.add_card(0, Card(8, "Spades"))

    player.split_hand(0)

    assert len(player.hands) == 2
    assert player.bankroll == 8_000

    assert player.hands[0].bet == 1_000
    assert player.hands[1].bet == 1_000

    assert player.hands[0].hand.cards[0].rank == 8
    assert player.hands[1].hand.cards[0].rank == 8

    assert player.hands[0].came_from_split is True
    assert player.hands[1].came_from_split is True


def test_split_aces_are_marked():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    player.add_card(0, Card("A", "Hearts"))
    player.add_card(0, Card("A", "Spades"))

    player.split_hand(0)

    assert player.hands[0].split_aces is True
    assert player.hands[1].split_aces is True


def test_cannot_split_different_cards():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    player.add_card(0, Card(8, "Hearts"))
    player.add_card(0, Card(9, "Spades"))

    with pytest.raises(ValueError):
        player.split_hand(0)


def test_cannot_split_without_enough_bankroll():
    player = Player(1_000)
    player.place_bet(1_000, minimum_bet=100)
    player.add_card(0, Card(8, "Hearts"))
    player.add_card(0, Card(8, "Spades"))

    with pytest.raises(ValueError):
        player.split_hand(0)


def test_cannot_exceed_maximum_hands():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    player.add_card(0, Card(8, "Hearts"))
    player.add_card(0, Card(8, "Spades"))

    with pytest.raises(ValueError):
        player.split_hand(0, max_hands=1)


def test_standing_one_split_hand_does_not_stop_other_hand():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    player.add_card(0, Card(8, "Hearts"))
    player.add_card(0, Card(8, "Spades"))
    player.split_hand(0)

    player.stand(0)

    assert player.hands[0].has_stood is True
    assert player.hands[1].has_stood is False


def test_win_payout():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    player.win(0)

    assert player.bankroll == 11_000


def test_blackjack_payout():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    player.win_blackjack(0)

    assert player.bankroll == 11_500


def test_split_blackjack_pays_even_money():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)
    player.add_card(0, Card("A", "Hearts"))
    player.add_card(0, Card("A", "Spades"))
    player.split_hand(0)

    player.add_card(0, Card("K", "Clubs"))
    player.win_blackjack(0)

    # 10,000 - 1,000 original bet - 1,000 split bet
    # + 2,000 even-money return for the winning split hand.
    assert player.bankroll == 10_000


def test_push_returns_selected_hands_bet():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    player.push(0)

    assert player.bankroll == 10_000


def test_reset_round():
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    player.reset_round()

    assert player.hands == []
    assert player.active_hand_index == 0


def test_player_defaults_to_no_power_chips():
    player = Player(10_000)

    assert player.power_chips.remaining() == 0


def test_power_chips_survive_round_reset():
    player = Player(
        bankroll=10_000,
        power_chip_count=2,
    )

    player.power_chips.consume(PowerChipAction.REPLACE)
    player.place_bet(1_000, minimum_bet=100)
    player.reset_round()

    assert player.power_chips.remaining() == 1
    assert player.hands == []
