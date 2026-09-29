import pytest

from blackjack.cards import Card, Shoe
from blackjack.hand import Hand
from tournament.power_chips import (
    PowerChipAction,
    PowerChipInventory,
    apply_power_chip_to_hand,
    apply_power_chip_from_shoe,
    legal_power_chip_targets,
)


def test_power_chip_inventory_tracks_usage():
    inventory = PowerChipInventory(chip_count=2)

    assert inventory.remaining() == 2
    assert inventory.can_use(PowerChipAction.REHIT) is True
    assert inventory.can_use(PowerChipAction.REPLACE) is True

    inventory.consume(PowerChipAction.REHIT)

    assert inventory.remaining() == 1
    assert inventory.can_use(PowerChipAction.REHIT) is True
    assert inventory.can_use(PowerChipAction.REPLACE) is True

    inventory.consume(PowerChipAction.REPLACE)

    assert inventory.remaining() == 0
    assert inventory.can_use(PowerChipAction.REHIT) is False
    assert inventory.can_use(PowerChipAction.REPLACE) is False

    with pytest.raises(ValueError):
        inventory.consume(PowerChipAction.REHIT)


def test_legal_targets_change_after_dealer_check_and_hit():
    inventory = PowerChipInventory(chip_count=1)
    common_state = {
        "dealt_card_count": 2,
        "inventory": inventory,
    }

    assert legal_power_chip_targets(
        action=PowerChipAction.REPLACE,
        card_count=2,
        dealer_blackjack_checked=False,
        **common_state,
    ) == ()
    assert legal_power_chip_targets(
        action=PowerChipAction.REHIT,
        card_count=2,
        dealer_blackjack_checked=False,
        **common_state,
    ) == ()

    assert legal_power_chip_targets(
        action=PowerChipAction.REPLACE,
        card_count=2,
        dealer_blackjack_checked=True,
        **common_state,
    ) == (0, 1)
    assert legal_power_chip_targets(
        action=PowerChipAction.REHIT,
        card_count=2,
        dealer_blackjack_checked=True,
        **common_state,
    ) == ()

    assert legal_power_chip_targets(
        action=PowerChipAction.REPLACE,
        card_count=3,
        dealer_blackjack_checked=True,
        **common_state,
    ) == ()
    assert legal_power_chip_targets(
        action=PowerChipAction.REHIT,
        card_count=3,
        dealer_blackjack_checked=True,
        **common_state,
    ) == (2,)

    assert legal_power_chip_targets(
        action=PowerChipAction.REHIT,
        card_count=4,
        dealer_blackjack_checked=True,
        **common_state,
    ) == (3,)

    assert legal_power_chip_targets(
        action=PowerChipAction.REPLACE,
        card_count=2,
        dealt_card_count=2,
        dealer_blackjack_checked=True,
        inventory=PowerChipInventory(chip_count=0),
    ) == ()


@pytest.mark.parametrize(
    (
        "action",
        "starting_cards",
        "target_index",
        "expected_ranks",
    ),
    [
        (
            PowerChipAction.REPLACE,
            [Card(10, "Hearts"), Card(6, "Clubs")],
            0,
            ["A", 6],
        ),
        (
            PowerChipAction.REHIT,
            [
                Card(10, "Hearts"),
                Card(6, "Clubs"),
                Card(9, "Diamonds"),
            ],
            2,
            [10, 6, "A"],
        ),
    ],
)
def test_apply_power_chip_replaces_target_and_consumes_one_chip(
    action,
    starting_cards,
    target_index,
    expected_ranks,
):
    hand = Hand()
    for card in starting_cards:
        hand.add_card(card)
    removed_card = hand.cards[target_index]
    replacement_card = Card("A", "Spades")
    inventory = PowerChipInventory(chip_count=2)

    returned_card = apply_power_chip_to_hand(
        hand=hand,
        action=action,
        target_index=target_index,
        replacement_card=replacement_card,
        dealt_card_count=2,
        dealer_blackjack_checked=True,
        inventory=inventory,
    )

    assert returned_card is removed_card
    assert [card.rank for card in hand.cards] == expected_ranks
    assert hand.cards[target_index] is replacement_card
    assert inventory.remaining() == 1


def test_illegal_power_chip_target_changes_nothing():
    hand = Hand()
    first_card = Card(10, "Hearts")
    second_card = Card(6, "Clubs")
    hand.add_card(first_card)
    hand.add_card(second_card)
    inventory = PowerChipInventory(chip_count=1)

    with pytest.raises(ValueError):
        apply_power_chip_to_hand(
            hand=hand,
            action=PowerChipAction.REHIT,
            target_index=1,
            replacement_card=Card(5, "Spades"),
            dealt_card_count=2,
            dealer_blackjack_checked=True,
            inventory=inventory,
        )

    assert hand.cards == [first_card, second_card]
    assert inventory.remaining() == 1


def test_apply_power_chip_from_shoe_deals_one_replacement_card():
    hand = Hand()
    removed_card = Card(10, "Hearts")
    other_card = Card(6, "Clubs")
    replacement_card = Card("A", "Spades")
    hand.add_card(removed_card)
    hand.add_card(other_card)
    shoe = Shoe(1)
    shoe.cards = [replacement_card]
    inventory = PowerChipInventory(chip_count=1)

    returned_card = apply_power_chip_from_shoe(
        hand=hand,
        action=PowerChipAction.REPLACE,
        target_index=0,
        shoe=shoe,
        dealt_card_count=2,
        dealer_blackjack_checked=True,
        inventory=inventory,
    )

    assert returned_card is removed_card
    assert hand.cards == [replacement_card, other_card]
    assert shoe.cards_remaining() == 0
    assert inventory.remaining() == 0


def test_illegal_power_chip_request_does_not_draw_from_shoe():
    hand = Hand()
    first_card = Card(10, "Hearts")
    second_card = Card(6, "Clubs")
    replacement_card = Card(5, "Spades")
    hand.add_card(first_card)
    hand.add_card(second_card)
    shoe = Shoe(1)
    shoe.cards = [replacement_card]
    inventory = PowerChipInventory(chip_count=1)

    with pytest.raises(ValueError):
        apply_power_chip_from_shoe(
            hand=hand,
            action=PowerChipAction.REHIT,
            target_index=1,
            shoe=shoe,
            dealt_card_count=2,
            dealer_blackjack_checked=True,
            inventory=inventory,
        )

    assert hand.cards == [first_card, second_card]
    assert shoe.cards == [replacement_card]
    assert inventory.remaining() == 1
