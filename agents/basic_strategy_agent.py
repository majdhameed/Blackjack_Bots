# Bot that wagers the table minimum and chooses card-play actions from the
# basic-strategy chart using immutable tournament observations.
from blackjack.player import Player
from blackjack.round import Round

from blackjack.actions import Action
from blackjack.basic_strategy import (
    choose_action as choose_basic_strategy_action,
)

from blackjack.cards import Card
from blackjack.hand import Hand
from tournament.observation import BettingObservation, ActionObservation, PowerChipObservation


class BasicStrategyAgent:
    def choose_bet(self, betting_observation: BettingObservation):
        if betting_observation.bankroll <= 0:
            raise ValueError("No money")

        # A player below the minimum is allowed to wager the remaining balance.
        return min(betting_observation.minimum_bet,
                   betting_observation.bankroll)
    
    def choose_action(self, observation: ActionObservation):
        if not isinstance(
            observation,
            ActionObservation,
        ):
            raise TypeError(
                "observation must be an ActionObservation"
            )

        legal_actions = set(
            observation.legal_actions
        )

        if not legal_actions:
            raise ValueError(
                "This hand has no legal actions"
            )

        # Observations contain values rather than engine objects, so rebuild a
        # temporary hand for the shared basic-strategy function.
        reconstructed_hand = Hand()

        for card_value in observation.hand_card_values:
            if card_value == 11:
                rank = "A"
            else:
                rank = card_value

            reconstructed_hand.add_card(
                Card(rank, "Hearts")
            )

        dealer_value = (
            observation.dealer_upcard_value
        )

        if dealer_value == 11:
            dealer_rank = "A"
        else:
            dealer_rank = dealer_value

        # Suit has no strategic effect; Hearts is a harmless placeholder.
        dealer_upcard = Card(
            dealer_rank,
            "Hearts",
        )

        action = choose_basic_strategy_action(
            hand=reconstructed_hand,
            dealer_upcard=dealer_upcard,
            can_double=(
                Action.DOUBLE in legal_actions
            ),
            can_split=(
                Action.SPLIT in legal_actions
            ),
            can_surrender=(
                Action.SURRENDER in legal_actions
            ),
            hit_soft_17=observation.hit_soft_17,
        )

        if action not in legal_actions:
            raise ValueError(
                "Basic strategy returned an "
                f"illegal action: {action}"
            )

        return action

    def choose_power_chip(self, observation: PowerChipObservation):
        if not isinstance(observation, PowerChipObservation):
            raise TypeError("wrong observation type")

        policy = getattr(self, "power_chip_policy", None)

        if policy is None:
            return None

        chooser = getattr(policy, "choose_power_chip", None)
        if not callable(chooser):
            raise TypeError("Invalid policy type")

        return chooser(observation)

    def set_power_chip_policy(self, policy):
        chooser = getattr(policy, "choose_power_chip", None)

        if not callable(chooser):
            raise TypeError ("Invalid chooser")

        self.power_chip_policy = policy

        return self
