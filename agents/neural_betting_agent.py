from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import legal_bet
from ml.betting_encoder import encode_betting_observation
from ml.betting_network import BETTING_ACTION_NAMES
from tournament.observation import BettingObservation


class NeuralBettingAgent(BasicStrategyAgent):
    def __init__(self, network):
        if not callable(getattr(network, "forward", None)):
            raise TypeError("network must have a forward method")

        self.network = network
        self.last_action_name = "legacy_fraction"

    def _select_action(self, features):
        preferred_action = getattr(
            self.network,
            "preferred_action",
            None,
        )
        if not callable(preferred_action):
            return 0

        action_index = preferred_action(features)
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 0 <= action_index < len(BETTING_ACTION_NAMES)
        ):
            raise ValueError(
                "network returned an invalid betting action"
            )
        return action_index

    def _requested_bet_for_action(
        self,
        action_name,
        observation,
        features,
    ):
        bankroll = observation.bankroll

        if action_name == "legacy_fraction":
            return bankroll * self.network.forward(features)
        if action_name == "minimum":
            return observation.minimum_bet
        if action_name.startswith("bet_") and action_name.endswith("_percent"):
            percentage = int(action_name.split("_")[1])
            return bankroll * percentage / 100

        raise ValueError(f"Unknown betting action: {action_name}")

    def choose_bet(self, betting_observation):
        if not isinstance(
            betting_observation,
            BettingObservation,
        ):
            raise TypeError(
                "observation must be a BettingObservation"
            )

        # Convert the observation into model-ready numeric features.
        features = encode_betting_observation(
            betting_observation
        )

        action_index = self._select_action(features)
        action_name = BETTING_ACTION_NAMES[action_index]
        self.last_action_name = action_name

        requested_bet = self._requested_bet_for_action(
            action_name,
            betting_observation,
            features,
        )
        return legal_bet(
            betting_observation,
            requested_bet,
        )
