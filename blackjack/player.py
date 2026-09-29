# Player bankroll and hand management. This layer applies wagers and payouts;
# Round and TableRound decide when each operation is legal in the game flow.
from blackjack.cards import Card
from blackjack.player_hand import PlayerHand

from tournament.power_chips import PowerChipInventory

class Player:
    def __init__(self, bankroll, power_chip_count=0):
        if bankroll <= 0:
            raise ValueError("Bankroll must be greater than 0")

        self.bankroll = bankroll
        self.hands = []
        self.active_hand_index = 0
        self.power_chips = PowerChipInventory(power_chip_count)


    def get_hand(self, hand_index):
        if not isinstance(hand_index, int):
            raise TypeError("Hand index must be an integer")

        if hand_index < 0 or hand_index >= len(self.hands):
            raise IndexError("Invalid hand index")

        return self.hands[hand_index]

    def place_bet(self, bet, minimum_bet):
        if self.hands:
            raise ValueError("Player has already placed a bet")

        if bet <= 0:
            raise ValueError("Bet must be greater than 0")

        if bet > self.bankroll:
            raise ValueError(
                "Bankroll is not large enough to place this bet"
            )

        # A short bankroll may be wagered in full even below the table minimum.
        if bet < minimum_bet and bet != self.bankroll:
            raise ValueError(
                "Bet must meet the minimum or equal the remaining bankroll"
            )

        if (
            bet != self.bankroll
            and bet % minimum_bet != 0
        ):
            raise ValueError(
                "Bet must use table-minimum chip increments"
            )

        self.bankroll -= bet
        self.hands.append(PlayerHand(bet))
        self.active_hand_index = 0

    def add_card(self, hand_index, card):
        if not isinstance(card, Card):
            raise TypeError("Not a card")

        player_hand = self.get_hand(hand_index)
        player_hand.add_card(card)

    def stand(self, hand_index):
        player_hand = self.get_hand(hand_index)
        player_hand.stand()

    def double_down(self, hand_index, additional_bet):
        # A player may double for less when the remaining bankroll cannot match
        # the original wager.
        player_hand = self.get_hand(hand_index)

        if player_hand.is_finished():
            raise ValueError("This hand is already finished")

        if len(player_hand.hand.cards) != 2:
            raise ValueError(
                "Player can only double on the first two cards"
            )

        if additional_bet <= 0:
            raise ValueError(
                "Double-down bet must be greater than 0"
            )

        if additional_bet > player_hand.bet:
            raise ValueError(
                "Double-down bet cannot exceed the original bet"
            )

        if additional_bet > self.bankroll:
            raise ValueError(
                "Bankroll is not large enough to double down"
            )

        player_hand.mark_doubled()

        self.bankroll -= additional_bet
        player_hand.bet += additional_bet

    def finish_double(self, hand_index):
        player_hand = self.get_hand(hand_index)
        player_hand.finish_double()

    def surrender(self, hand_index):
        player_hand = self.get_hand(hand_index)

        player_hand.mark_surrendered()
        # Surrender returns half of the original wager.
        self.bankroll += player_hand.bet * 0.5

    def split_hand(self, hand_index, max_hands=4):
        player_hand = self.get_hand(hand_index)

        if player_hand.is_finished():
            raise ValueError("This hand is already finished")

        if not player_hand.hand.can_split():
            raise ValueError("This hand cannot be split")

        if len(self.hands) >= max_hands:
            raise ValueError("Maximum number of hands reached")

        if self.bankroll < player_hand.bet:
            raise ValueError(
                "Bankroll is not large enough to split"
            )

        first_card = player_hand.hand.cards[0]
        second_card = player_hand.hand.cards[1]
        original_bet = player_hand.bet

        splitting_aces = (
            first_card.rank == "A"
            and second_card.rank == "A"
        )

        first_hand = PlayerHand(
            bet=original_bet,
            came_from_split=True,
            split_aces=splitting_aces,
        )

        second_hand = PlayerHand(
            bet=original_bet,
            came_from_split=True,
            split_aces=splitting_aces,
        )

        first_hand.add_card(first_card)
        second_hand.add_card(second_card)

        self.bankroll -= original_bet

        # Replace the original hand in place so play order remains stable.
        self.hands[hand_index:hand_index + 1] = [
            first_hand,
            second_hand,
        ]

        self.active_hand_index = hand_index

    def win(self, hand_index):
        player_hand = self.get_hand(hand_index)
        self.bankroll += player_hand.bet * 2

    def win_blackjack(self, hand_index):
        player_hand = self.get_hand(hand_index)

        # Natural blackjack pays 3:2; a split 21 is only a normal 1:1 win.
        if player_hand.came_from_split:
            self.bankroll += player_hand.bet * 2
        else:
            self.bankroll += player_hand.bet * 2.5

    def push(self, hand_index):
        player_hand = self.get_hand(hand_index)
        self.bankroll += player_hand.bet

    def reset_round(self):
        self.hands = []
        self.active_hand_index = 0
