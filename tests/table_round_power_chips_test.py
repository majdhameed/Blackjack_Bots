import pytest

from blackjack.actions import Action
from blackjack.card_counter import CardCounter
from blackjack.cards import Card, Shoe
from blackjack.player import Player
from tournament.power_chips import PowerChipAction
from tournament.table_round import TableRound


def make_power_chip_table():
    players = [
        Player(10_000, power_chip_count=1),
        Player(10_000, power_chip_count=0),
    ]
    shoe = Shoe(1)
    replacement_card = Card("A", "Spades")
    deal_order = [
        Card(10, "Hearts"),
        Card(8, "Hearts"),
        Card(9, "Clubs"),
        Card(6, "Diamonds"),
        Card(8, "Diamonds"),
        Card(7, "Clubs"),
        replacement_card,
    ]
    shoe.cards = list(reversed(deal_order))
    table = TableRound(
        players=players,
        shoe=shoe,
        minimum_bet=100,
        starting_player=1,
        hit_soft_17=False,
        max_hands=4,
        card_counter=CardCounter(),
    )
    for player_index in table.betting_order:
        table.place_bet(player_index, 100)
    table.deal_initial_cards()
    return table, replacement_card


def test_table_rejects_power_chip_before_dealer_blackjack_check():
    table, replacement_card = make_power_chip_table()
    player = table.players[0]
    original_cards = list(player.get_hand(0).hand.cards)

    with pytest.raises(ValueError):
        table.player_use_power_chip(
            player_index=0,
            hand_index=0,
            action=PowerChipAction.REPLACE,
            target_index=0,
        )

    assert player.get_hand(0).hand.cards == original_cards
    assert player.power_chips.remaining() == 1
    assert table.shoe.cards == [replacement_card]


def test_table_applies_power_chip_after_dealer_blackjack_check():
    table, replacement_card = make_power_chip_table()
    player = table.players[0]
    removed_card = player.get_hand(0).hand.cards[0]
    cards_seen_before = table.card_counter.cards_seen
    table.resolve_naturals()

    returned_card = table.player_use_power_chip(
        player_index=0,
        hand_index=0,
        action=PowerChipAction.REPLACE,
        target_index=0,
    )

    assert returned_card is removed_card
    assert player.get_hand(0).hand.cards[0] is replacement_card
    assert player.power_chips.remaining() == 0
    assert table.shoe.cards_remaining() == 0
    assert table.card_counter.cards_seen == cards_seen_before + 1


def test_busting_hit_waits_for_rehit_and_replacement_can_save_hand():
    table, _ = make_power_chip_table()
    table.resolve_naturals()
    bust_card = Card(10, "Spades")
    saving_card = Card(5, "Spades")
    table.shoe.cards = [saving_card, bust_card]

    table.player_hit(player_index=0, hand_index=0)

    assert table.players[0].get_hand(0).hand.is_bust() is True
    assert table.get_current_player_index() == 0

    table.player_use_power_chip(
        player_index=0,
        hand_index=0,
        action=PowerChipAction.REHIT,
        target_index=2,
    )

    assert table.players[0].get_hand(0).hand.get_total() == 21
    assert table.players[0].power_chips.remaining() == 0
    assert table.get_current_player_index() == 1


def test_declining_rehit_restores_normal_actions_for_playable_hand():
    table, _ = make_power_chip_table()
    table.resolve_naturals()
    table.shoe.cards = [Card(2, "Spades")]

    table.player_hit(player_index=0, hand_index=0)

    assert table.get_current_player_index() == 0
    assert table.get_legal_actions(0, 0) == set()

    table.player_decline_rehit(
        player_index=0,
        hand_index=0,
    )

    assert table.get_current_player_index() == 0
    assert Action.HIT in table.get_legal_actions(0, 0)
    assert Action.STAND in table.get_legal_actions(0, 0)


def test_rehit_is_rejected_after_pending_rehit_is_declined():
    table, _ = make_power_chip_table()
    table.resolve_naturals()
    hit_card = Card(2, "Spades")
    replacement_card = Card(3, "Clubs")
    table.shoe.cards = [replacement_card, hit_card]

    table.player_hit(player_index=0, hand_index=0)
    table.player_decline_rehit(player_index=0, hand_index=0)

    player = table.players[0]
    cards_before = list(player.get_hand(0).hand.cards)

    with pytest.raises(ValueError, match="pending rehit"):
        table.player_use_power_chip(
            player_index=0,
            hand_index=0,
            action=PowerChipAction.REHIT,
            target_index=2,
        )

    assert player.get_hand(0).hand.cards == cards_before
    assert player.power_chips.remaining() == 1


def test_table_exposes_rehit_target_only_while_decision_is_pending():
    table, _ = make_power_chip_table()
    table.resolve_naturals()
    table.shoe.cards = [Card(2, "Spades")]

    table.player_hit(player_index=0, hand_index=0)

    assert table.get_legal_power_chip_targets(
        player_index=0,
        hand_index=0,
        action=PowerChipAction.REHIT,
    ) == (2,)

    table.player_decline_rehit(player_index=0, hand_index=0)

    assert table.get_legal_power_chip_targets(
        player_index=0,
        hand_index=0,
        action=PowerChipAction.REHIT,
    ) == ()


def test_split_hand_can_replace_only_new_card_and_then_resplit_pair():
    players = [
        Player(10_000, power_chip_count=1),
        Player(10_000, power_chip_count=0),
    ]
    shoe = Shoe(1)
    replacement_card = Card(3, "Spades")
    deal_order = [
        Card(3, "Hearts"),
        Card(9, "Hearts"),
        Card(10, "Clubs"),
        Card(3, "Diamonds"),
        Card(8, "Diamonds"),
        Card(6, "Spades"),
        Card(5, "Clubs"),
        Card(7, "Clubs"),
        replacement_card,
    ]
    shoe.cards = list(reversed(deal_order))
    table = TableRound(
        players=players,
        shoe=shoe,
        minimum_bet=100,
        starting_player=1,
        hit_soft_17=False,
        max_hands=4,
        card_counter=CardCounter(),
    )
    for player_index in table.betting_order:
        table.place_bet(player_index, 100)
    table.deal_initial_cards()
    table.resolve_naturals()
    table.player_split(player_index=0, hand_index=0)

    assert table.get_legal_power_chip_targets(
        player_index=0,
        hand_index=0,
        action=PowerChipAction.REPLACE,
    ) == (1,)
    assert table.get_legal_power_chip_targets(
        player_index=0,
        hand_index=1,
        action=PowerChipAction.REPLACE,
    ) == (1,)

    shoe_size_before = table.shoe.cards_remaining()
    with pytest.raises(ValueError):
        table.player_use_power_chip(
            player_index=0,
            hand_index=0,
            action=PowerChipAction.REPLACE,
            target_index=0,
        )

    assert table.shoe.cards_remaining() == shoe_size_before
    assert players[0].power_chips.remaining() == 1

    table.player_use_power_chip(
        player_index=0,
        hand_index=0,
        action=PowerChipAction.REPLACE,
        target_index=1,
    )

    assert players[0].get_hand(0).hand.cards[1] is replacement_card
    assert players[0].get_hand(0).hand.can_split() is True
    assert Action.SPLIT in table.get_legal_actions(0, 0)
    assert table.get_legal_power_chip_targets(
        player_index=0,
        hand_index=1,
        action=PowerChipAction.REPLACE,
    ) == ()
