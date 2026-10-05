import pytest

from agents.all_in_agent import AllInAgent
from agents.basic_strategy_agent import BasicStrategyAgent
from agents.power_chip_policies import BustSavingPowerChipPolicy
from tournament.observation import PowerChipObservation
from tournament.power_chips import PowerChipAction


def make_power_chip_observation(action, legal_targets, hand_total=18):
    return PowerChipObservation(
        round_number=4,
        total_rounds=12,
        rounds_remaining=8,
        player_index=0,
        round_player_index=0,
        hand_index=0,
        bankroll=9_900,
        bankrolls=(9_900, 10_100),
        current_bet=100,
        hand_total=hand_total,
        hand_is_soft=False,
        hand_card_values=(10, 8),
        dealer_upcard_value=10,
        power_chip_counts=(1, 1),
        action=action,
        legal_targets=legal_targets,
    )


def test_basic_strategy_agent_declines_power_chip_by_default():
    observation = make_power_chip_observation(
        action=PowerChipAction.REPLACE,
        legal_targets=(0, 1),
    )

    assert BasicStrategyAgent().choose_power_chip(observation) is None


def test_betting_agent_inherits_safe_power_chip_default():
    observation = make_power_chip_observation(
        action=PowerChipAction.REHIT,
        legal_targets=(2,),
    )

    assert AllInAgent().choose_power_chip(observation) is None


def test_basic_strategy_agent_rejects_wrong_power_chip_observation_type():
    with pytest.raises(TypeError):
        BasicStrategyAgent().choose_power_chip(object())


def test_betting_agent_delegates_to_attached_power_chip_policy():
    observation = make_power_chip_observation(
        action=PowerChipAction.REHIT,
        legal_targets=(2,),
        hand_total=26,
    )
    agent = AllInAgent()
    agent.power_chip_policy = BustSavingPowerChipPolicy()

    assert agent.choose_power_chip(observation) == 2


def test_agent_can_set_power_chip_policy_and_chain_configuration():
    observation = make_power_chip_observation(
        action=PowerChipAction.REHIT,
        legal_targets=(2,),
        hand_total=26,
    )
    agent = AllInAgent()

    returned_agent = agent.set_power_chip_policy(
        BustSavingPowerChipPolicy()
    )

    assert returned_agent is agent
    assert agent.choose_power_chip(observation) == 2


def test_agent_rejects_invalid_power_chip_policy():
    with pytest.raises(TypeError):
        BasicStrategyAgent().set_power_chip_policy(object())
