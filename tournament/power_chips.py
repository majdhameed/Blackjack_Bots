from enum import Enum

class PowerChipAction(Enum):
    REHIT = "REHIT"
    REPLACE = "REPLACE"

class PowerChipInventory:
    def __init__(self, chip_count):
        self.chip_count = chip_count

    def remaining(self):
        return self.chip_count

    def can_use(self, action):
        if self.chip_count > 0:
            return True
        return False
   
    def consume(self, action):
        if self.chip_count <= 0:
            raise ValueError("No power chips available")
        self.chip_count -=1

def legal_power_chip_targets(action, card_count, dealt_card_count, dealer_blackjack_checked, inventory):

    if not inventory.can_use(action):
        return ()

    if not dealer_blackjack_checked:
        return ()

    hit_occurred = card_count > dealt_card_count

    if action == PowerChipAction.REPLACE and not hit_occurred:
        return tuple(range(dealt_card_count))

    if action == PowerChipAction.REHIT and hit_occurred:
        return (card_count - 1, )

    return tuple()

def apply_power_chip_to_hand(hand, action, target_index, replacement_card, dealt_card_count, dealer_blackjack_checked, inventory):
    targets = legal_power_chip_targets(action, len(hand.cards), dealt_card_count, dealer_blackjack_checked, inventory)

    if target_index not in targets:
        raise ValueError("Action can not be performed on that card")

    removed_card = hand.cards[target_index]

    hand.cards[target_index] = replacement_card

    inventory.consume(action)

    return removed_card
    
def apply_power_chip_from_shoe(hand, action, target_index, shoe, dealt_card_count, dealer_blackjack_checked, inventory):
    targets = legal_power_chip_targets(action, len(hand.cards), dealt_card_count, dealer_blackjack_checked, inventory)

    if target_index not in targets:
        raise ValueError("This action can not be performed on that card")

    replacement_card = shoe.deal_card()

    removed_card = apply_power_chip_to_hand(hand, action, target_index, replacement_card, dealt_card_count, dealer_blackjack_checked, inventory)

    return removed_card

