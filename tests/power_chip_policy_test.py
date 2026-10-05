from agents.power_chip_policies import (
    BustSavingPowerChipPolicy,
    CompositePowerChipPolicy,
    HighStakesPowerChipPolicy,
    LateRoundPowerChipPolicy,
    StiffHandReplacementPowerChipPolicy,
)
import pytest
from tournament.observation import PowerChipObservation
from tournament.power_chips import PowerChipAction


def make_observation(
    action,
    hand_total,
    legal_targets,
    hand_card_values=(10, 6, 10),
    dealer_upcard_value=10,
    hand_is_soft=False,
    bankroll=9_900,
    current_bet=100,
    rounds_remaining=8,
):
    return PowerChipObservation(
        round_number=4,
        total_rounds=12,
        rounds_remaining=rounds_remaining,
        player_index=0,
        round_player_index=0,
        hand_index=0,
        bankroll=bankroll,
        bankrolls=(9_900, 10_100),
        current_bet=current_bet,
        hand_total=hand_total,
        hand_is_soft=hand_is_soft,
        hand_card_values=hand_card_values,
        dealer_upcard_value=dealer_upcard_value,
        power_chip_counts=(1, 1),
        action=action,
        legal_targets=legal_targets,
    )


def test_bust_saving_policy_rehits_the_only_legal_target():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=26,
        legal_targets=(2,),
    )

    assert BustSavingPowerChipPolicy().choose_power_chip(observation) == 2


def test_bust_saving_policy_preserves_chip_when_rehit_did_not_bust():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=18,
        legal_targets=(2,),
    )

    assert BustSavingPowerChipPolicy().choose_power_chip(observation) is None


def test_bust_saving_policy_does_not_use_replace():
    observation = make_observation(
        action=PowerChipAction.REPLACE,
        hand_total=16,
        legal_targets=(0, 1),
    )

    assert BustSavingPowerChipPolicy().choose_power_chip(observation) is None


def test_stiff_replacement_policy_replaces_lowest_legal_card():
    observation = make_observation(
        action=PowerChipAction.REPLACE,
        hand_total=16,
        legal_targets=(0, 1),
        hand_card_values=(10, 6),
        dealer_upcard_value=10,
    )

    assert StiffHandReplacementPowerChipPolicy().choose_power_chip(
        observation
    ) == 1


def test_stiff_replacement_policy_respects_split_hand_targets():
    observation = make_observation(
        action=PowerChipAction.REPLACE,
        hand_total=13,
        legal_targets=(1,),
        hand_card_values=(3, 10),
        dealer_upcard_value=10,
    )

    assert StiffHandReplacementPowerChipPolicy().choose_power_chip(
        observation
    ) == 1


def test_stiff_replacement_policy_preserves_chip_against_weak_dealer():
    observation = make_observation(
        action=PowerChipAction.REPLACE,
        hand_total=16,
        legal_targets=(0, 1),
        hand_card_values=(10, 6),
        dealer_upcard_value=6,
    )

    assert StiffHandReplacementPowerChipPolicy().choose_power_chip(
        observation
    ) is None


def test_stiff_replacement_policy_declines_soft_hand():
    observation = make_observation(
        action=PowerChipAction.REPLACE,
        hand_total=16,
        legal_targets=(0, 1),
        hand_card_values=(11, 5),
        dealer_upcard_value=10,
        hand_is_soft=True,
    )

    assert StiffHandReplacementPowerChipPolicy().choose_power_chip(
        observation
    ) is None


def make_composite_policy():
    return CompositePowerChipPolicy(
        policies=(
            BustSavingPowerChipPolicy(),
            StiffHandReplacementPowerChipPolicy(),
        )
    )


def test_composite_policy_uses_bust_saving_rule():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=26,
        legal_targets=(2,),
    )

    assert make_composite_policy().choose_power_chip(observation) == 2


def test_composite_policy_uses_stiff_replacement_rule():
    observation = make_observation(
        action=PowerChipAction.REPLACE,
        hand_total=16,
        legal_targets=(0, 1),
        hand_card_values=(10, 6),
        dealer_upcard_value=10,
    )

    assert make_composite_policy().choose_power_chip(observation) == 1


def test_composite_policy_declines_when_every_rule_declines():
    observation = make_observation(
        action=PowerChipAction.REPLACE,
        hand_total=18,
        legal_targets=(0, 1),
        hand_card_values=(10, 8),
        dealer_upcard_value=6,
    )

    assert make_composite_policy().choose_power_chip(observation) is None


def test_high_stakes_policy_preserves_chip_on_small_wager():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=26,
        legal_targets=(2,),
        bankroll=9_900,
        current_bet=100,
    )
    policy = HighStakesPowerChipPolicy(
        policy=BustSavingPowerChipPolicy(),
        minimum_bet_fraction=0.10,
    )

    assert policy.choose_power_chip(observation) is None


def test_high_stakes_policy_uses_chip_on_large_wager():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=26,
        legal_targets=(2,),
        bankroll=8_000,
        current_bet=2_000,
    )
    policy = HighStakesPowerChipPolicy(
        policy=BustSavingPowerChipPolicy(),
        minimum_bet_fraction=0.10,
    )

    assert policy.choose_power_chip(observation) == 2


def test_high_stakes_policy_cannot_override_tactical_decline():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=18,
        legal_targets=(2,),
        bankroll=8_000,
        current_bet=2_000,
    )
    policy = HighStakesPowerChipPolicy(
        policy=BustSavingPowerChipPolicy(),
        minimum_bet_fraction=0.10,
    )

    assert policy.choose_power_chip(observation) is None


def test_late_round_policy_preserves_chip_early():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=26,
        legal_targets=(2,),
        rounds_remaining=8,
    )
    policy = LateRoundPowerChipPolicy(
        policy=BustSavingPowerChipPolicy(),
        maximum_rounds_remaining=2,
    )

    assert policy.choose_power_chip(observation) is None


def test_late_round_policy_uses_chip_near_tournament_end():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=26,
        legal_targets=(2,),
        rounds_remaining=1,
    )
    policy = LateRoundPowerChipPolicy(
        policy=BustSavingPowerChipPolicy(),
        maximum_rounds_remaining=2,
    )

    assert policy.choose_power_chip(observation) == 2


def test_late_round_policy_cannot_override_tactical_decline():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=18,
        legal_targets=(2,),
        rounds_remaining=1,
    )
    policy = LateRoundPowerChipPolicy(
        policy=BustSavingPowerChipPolicy(),
        maximum_rounds_remaining=2,
    )

    assert policy.choose_power_chip(observation) is None


def test_late_round_policy_allows_final_round_only_configuration():
    observation = make_observation(
        action=PowerChipAction.REHIT,
        hand_total=26,
        legal_targets=(2,),
        rounds_remaining=0,
    )
    policy = LateRoundPowerChipPolicy(
        policy=BustSavingPowerChipPolicy(),
        maximum_rounds_remaining=0,
    )

    assert policy.choose_power_chip(observation) == 2


@pytest.mark.parametrize("invalid_policy", [object(), None])
def test_policy_wrappers_reject_invalid_nested_policy(invalid_policy):
    with pytest.raises(TypeError, match="choose_power_chip"):
        HighStakesPowerChipPolicy(invalid_policy, 0.1)

    with pytest.raises(TypeError, match="choose_power_chip"):
        LateRoundPowerChipPolicy(invalid_policy, 1)
