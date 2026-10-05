import pytest

from agents.neural_betting_agent import NeuralBettingAgent
from ml.betting_encoder import BETTING_FEATURE_COUNT
from ml.betting_network import BETTING_ACTION_NAMES, BETTING_PERCENTAGES
from tournament.observation import BettingObservation, PowerChipObservation
from tournament.power_chips import PowerChipAction


class FakeNetwork:
    def __init__(self, output):
        self.output = output
        self.received_features = None

    def forward(self, features):
        self.received_features = features
        return self.output


class FakeActionNetwork(FakeNetwork):
    def __init__(self, action_index, output=0.01):
        super().__init__(output)
        self.action_index = action_index

    def preferred_action(self, features):
        self.received_features = features
        return self.action_index


class FakePowerChipNetwork(FakeNetwork):
    def __init__(self, scores):
        super().__init__(0.01)
        self.scores = scores

    def power_chip_scores(self, features):
        self.received_features = features
        return self.scores


def make_observation(
    bankroll=1_000,
    minimum_bet=10,
):
    return BettingObservation(
        round_number=1,
        total_rounds=12,
        rounds_remaining=11,
        player_index=0,
        round_player_index=0,
        betting_position=0,
        minimum_bet=minimum_bet,
        bankroll=bankroll,
        bankrolls=(
            bankroll,
            1_000,
            1_000,
            1_000,
            1_000,
            1_000,
            1_000,
        ),
        active_players=(
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ),
        current_bets=(0, 0, 0, 0, 0, 0, 0),
        bets_placed=(
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        betting_order=(0, 1, 2, 3, 4, 5, 6),
        card_value_counts=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        cards_seen=0,
        running_count=0,
        true_count=0.0,
        cards_remaining=312,
        decks_remaining=6.0,
        shoe_penetration=0.0,
    )


def test_network_output_controls_bet_percentage():
    network = FakeNetwork(0.25)
    agent = NeuralBettingAgent(network)

    observation = make_observation(
        bankroll=1_000,
        minimum_bet=10,
    )

    bet = agent.choose_bet(observation)

    assert bet == 250
    assert network.received_features is not None
    assert len(network.received_features) == BETTING_FEATURE_COUNT


def test_bet_is_floored_to_table_chip_increment():
    network = FakeNetwork(0.0155)
    agent = NeuralBettingAgent(network)

    observation = make_observation(
        bankroll=10_000,
        minimum_bet=100,
    )


def make_power_chip_observation(legal_targets=(0, 1)):
    return PowerChipObservation(
        round_number=1,
        total_rounds=12,
        rounds_remaining=11,
        player_index=0,
        round_player_index=0,
        hand_index=0,
        bankroll=900,
        bankrolls=(900, 1_000),
        current_bet=100,
        hand_total=16,
        hand_is_soft=False,
        hand_card_values=(10, 6, 5),
        dealer_upcard_value=10,
        power_chip_counts=(1, 1),
        action=PowerChipAction.REPLACE,
        legal_targets=legal_targets,
    )

    assert agent.choose_bet(observation) == 100


def test_bet_cannot_be_below_minimum():
    network = FakeNetwork(0.001)
    agent = NeuralBettingAgent(network)

    observation = make_observation(
        bankroll=1_000,
        minimum_bet=10,
    )

    bet = agent.choose_bet(observation)

    assert bet == 10


def test_output_one_bets_entire_bankroll():
    network = FakeNetwork(1.0)
    agent = NeuralBettingAgent(network)

    observation = make_observation(
        bankroll=1_000,
        minimum_bet=10,
    )

    bet = agent.choose_bet(observation)

    assert bet == 1_000


def test_player_below_minimum_bets_entire_bankroll():
    network = FakeNetwork(0.25)
    agent = NeuralBettingAgent(network)

    observation = make_observation(
        bankroll=7,
        minimum_bet=10,
    )

    bet = agent.choose_bet(observation)

    assert bet == 7


def test_choose_bet_rejects_invalid_observation():
    network = FakeNetwork(0.25)
    agent = NeuralBettingAgent(network)

    with pytest.raises(TypeError):
        agent.choose_bet("not an observation")


def test_constructor_rejects_object_without_forward():
    with pytest.raises(TypeError):
        NeuralBettingAgent(object())


@pytest.mark.parametrize(
    "percentage",
    BETTING_PERCENTAGES,
)
def test_network_can_select_percentage_bet_buckets(percentage):
    action_name = f"bet_{percentage:02d}_percent"
    agent = NeuralBettingAgent(
        FakeActionNetwork(BETTING_ACTION_NAMES.index(action_name))
    )

    assert agent.choose_bet(make_observation()) == percentage * 10


def test_network_can_select_true_table_minimum():
    agent = NeuralBettingAgent(
        FakeActionNetwork(
            BETTING_ACTION_NAMES.index("minimum")
        )
    )

    assert agent.choose_bet(make_observation()) == 10


def test_neural_agent_can_save_power_chip():
    agent = NeuralBettingAgent(
        FakePowerChipNetwork([0.9, 0.2, 0.1])
    )

    assert agent.choose_power_chip(
        make_power_chip_observation()
    ) is None


def test_neural_agent_can_choose_card_at_index_zero():
    agent = NeuralBettingAgent(
        FakePowerChipNetwork([0.1, 0.9, 0.2])
    )

    assert agent.choose_power_chip(
        make_power_chip_observation((0, 1))
    ) == 0


def test_neural_agent_can_choose_second_legal_target():
    agent = NeuralBettingAgent(
        FakePowerChipNetwork([0.1, 0.2, 0.9])
    )

    assert agent.choose_power_chip(
        make_power_chip_observation((0, 2))
    ) == 2


def test_neural_agent_ignores_unavailable_target_choice():
    agent = NeuralBettingAgent(
        FakePowerChipNetwork([0.1, 0.2, 0.9])
    )

    assert agent.choose_power_chip(
        make_power_chip_observation((2,))
    ) == 2


def test_neural_agent_saves_chip_when_no_targets_are_legal():
    agent = NeuralBettingAgent(
        FakePowerChipNetwork([0.1, 0.8, 0.9])
    )

    assert agent.choose_power_chip(
        make_power_chip_observation(())
    ) is None
