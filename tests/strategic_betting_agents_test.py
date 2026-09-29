from dataclasses import replace
import math

import pytest

from agents.chasing_agent import ChasingAgent
from agents.count_aware_agent import CountAwareAgent
from agents.controlled_lead_martingale_agent import (
    ControlledLeadMartingaleAgent,
)
from agents.early_lead_agent import EarlyLeadAgent
from agents.lead_protection_agent import (
    LeadProtectionAgent,
)
from agents.human_behavior_agent import HumanBehaviorAgent
from agents.unpredictable_betting_agent import (
    UnpredictableBettingAgent,
)
from ml.population import Population
from ml.trainer import Trainer
from tournament.observation import BettingObservation


def make_observation(
    round_number=1,
    total_rounds=12,
    bankroll=10_000,
    bankrolls=None,
    player_index=0,
    current_bets=None,
    bets_placed=None,
    true_count=0.0,
):
    if bankrolls is None:
        bankrolls = [10_000] * 7

    if current_bets is None:
        current_bets = [0] * 7

    if bets_placed is None:
        bets_placed = [False] * 7

    return BettingObservation(
        round_number=round_number,
        total_rounds=total_rounds,
        rounds_remaining=(
            total_rounds - round_number
        ),
        player_index=player_index,
        round_player_index=player_index,
        betting_position=0,
        minimum_bet=100,
        bankroll=bankroll,
        bankrolls=tuple(bankrolls),
        active_players=(True,) * 7,
        current_bets=tuple(current_bets),
        bets_placed=tuple(bets_placed),
        betting_order=tuple(range(7)),
        card_value_counts=(0,) * 10,
        cards_seen=0,
        running_count=0,
        true_count=true_count,
        cards_remaining=312,
        decks_remaining=6.0,
        shoe_penetration=0.0,
    )


def test_chaser_waits_early_and_increases_late_when_behind():
    agent = ChasingAgent()
    bankrolls = [7_000, 12_000, 11_000, 10_000, 9_000, 8_000, 7_500]

    early_bet = agent.choose_bet(
        make_observation(
            round_number=3,
            bankroll=7_000,
            bankrolls=bankrolls,
        )
    )

    late_bet = agent.choose_bet(
        make_observation(
            round_number=10,
            bankroll=7_000,
            bankrolls=bankrolls,
        )
    )

    assert early_bet == 100
    assert late_bet > early_bet


def test_early_lead_agent_attacks_then_defends_a_lead():
    agent = EarlyLeadAgent()

    opening_bet = agent.choose_bet(
        make_observation()
    )

    leading_bet = agent.choose_bet(
        make_observation(
            round_number=4,
            bankroll=12_000,
            bankrolls=[
                12_000,
                10_000,
                10_000,
                10_000,
                10_000,
                10_000,
                10_000,
            ],
        )
    )

    assert opening_bet == 2_500
    assert leading_bet == 100


def test_early_lead_agent_keeps_attacking_a_small_lead():
    agent = EarlyLeadAgent()

    bet = agent.choose_bet(
        make_observation(
            round_number=4,
            bankroll=10_500,
            bankrolls=[
                10_500,
                10_000,
                9_900,
                9_800,
                9_700,
                9_600,
                9_500,
            ],
        )
    )

    assert bet == 2_600


def test_aggressive_early_leaders_bet_half_or_all_in():
    observation = make_observation()

    half_bet = EarlyLeadAgent(
        0.25,
        0.50,
    ).choose_bet(observation)
    all_in_bet = EarlyLeadAgent(
        0.10,
        1.00,
    ).choose_bet(observation)

    assert half_bet == 5_000
    assert all_in_bet == 10_000


def test_controlled_martingale_builds_then_protects_lead():
    agent = ControlledLeadMartingaleAgent()
    tied = make_observation()
    after_loss = replace(
        tied,
        bankroll=9_500,
        bankrolls=(9_500,) + tied.bankrolls[1:],
        previous_bet=500,
        previous_bankroll_change=-500,
        previous_result=-1.0,
        consecutive_losses=1,
        has_previous_round=True,
    )
    leading = replace(
        tied,
        bankroll=10_500,
        bankrolls=(10_500,) + tied.bankrolls[1:],
    )

    assert agent.choose_bet(tied) == 500
    assert agent.choose_bet(after_loss) == 1_000
    assert agent.choose_bet(leading) == 100


def test_human_behavior_agent_reacts_to_a_loss_legally():
    agent = HumanBehaviorAgent(
        seed=123,
        impulse_probability=0,
        copy_probability=0,
        protect_probability=0,
    )
    observation = replace(
        make_observation(),
        previous_bet=500,
        previous_bankroll_change=-500,
        previous_result=-1.0,
        consecutive_losses=1,
        has_previous_round=True,
    )

    bet = agent.choose_bet(observation)

    assert bet >= 600
    assert bet % 100 == 0


def test_human_behavior_agent_can_persist_in_rare_all_in_mode():
    agent = HumanBehaviorAgent(
        seed=123,
        impulse_probability=0,
        copy_probability=0,
        protect_probability=0,
        all_in_probability=1,
        mode_persistence=3,
    )
    observation = make_observation()

    first_bet = agent.choose_bet(observation)
    second_bet = agent.choose_bet(observation)

    assert first_bet == observation.bankroll
    assert second_bet == observation.bankroll
    assert agent.current_mode == "all_in"


def test_human_behavior_agent_can_become_desperate_on_final_round():
    agent = HumanBehaviorAgent(
        seed=123,
        protect_probability=0,
        desperation_probability=1,
    )
    observation = replace(
        make_observation(),
        round_number=12,
        rounds_remaining=0,
        bankroll=8_000,
        bankrolls=(12_000, 10_000, 9_500, 9_000, 8_500, 8_200, 8_000),
        player_index=6,
        round_player_index=6,
    )

    bet = agent.choose_bet(observation)

    assert bet >= observation.bankroll * 0.50
    assert agent.current_mode == "desperate"


def test_lead_protector_covers_a_visible_winning_bet():
    agent = LeadProtectionAgent()

    bet = agent.choose_bet(
        make_observation(
            round_number=12,
            bankroll=12_000,
            bankrolls=[
                12_000,
                11_000,
                9_000,
                9_000,
                9_000,
                9_000,
                9_000,
            ],
            current_bets=[
                0,
                1_000,
                0,
                0,
                0,
                0,
                0,
            ],
            bets_placed=[
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )

    assert bet == 1_000


def test_lead_protector_builds_an_unsafe_early_lead():
    agent = LeadProtectionAgent()

    bet = agent.choose_bet(
        make_observation(
            round_number=4,
            bankroll=10_500,
            bankrolls=[
                10_500,
                10_000,
                9_900,
                9_800,
                9_700,
                9_600,
                9_500,
            ],
        )
    )

    assert bet == 2_100


def test_count_aware_agent_spreads_only_on_positive_counts():
    agent = CountAwareAgent()

    assert agent.choose_bet(
        make_observation(true_count=1.9)
    ) == 100

    assert agent.choose_bet(
        make_observation(true_count=4.0)
    ) == 1_000


@pytest.mark.parametrize(
    "agent",
    [
        ChasingAgent(),
        CountAwareAgent(),
        EarlyLeadAgent(),
        LeadProtectionAgent(),
    ],
)
def test_strategic_agents_never_bet_more_than_bankroll(agent):
    observation = make_observation(
        round_number=12,
        bankroll=75,
        bankrolls=[75, 20_000, 19_000, 18_000, 17_000, 16_000, 15_000],
        true_count=10,
    )

    assert agent.choose_bet(observation) == 75


def test_unpredictable_agent_changes_bet_sizes():
    agent = UnpredictableBettingAgent(
        seed=123,
        minimum_bet_probability=0.25,
        maximum_bankroll_fraction=0.60,
    )
    observation = make_observation()

    bets = {
        agent.choose_bet(observation)
        for _ in range(30)
    }

    assert len(bets) > 3
    assert min(bets) >= observation.minimum_bet
    assert max(bets) <= observation.bankroll


def make_trainer(randomize_training_rounds=True):
    population = Population(
        population_size=14,
        elite_count=2,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=123,
    )

    return Trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        randomize_training_rounds=(
            randomize_training_rounds
        ),
    )


def test_population_starts_with_varied_betting_biases():
    trainer = make_trainer()
    biases = [
        network.biases3[0]
        for network in trainer.population.networks
    ]

    fractions = [
        1 / (1 + math.exp(-bias))
        for bias in biases
    ]
    assert min(fractions) < 0.03
    assert max(fractions) > 0.80
    assert len(set(biases)) > 1


def test_training_round_lengths_vary_within_configured_range():
    trainer = make_trainer()
    round_counts = {
        trainer.training_round_count()
        for _ in range(50)
    }

    assert min(round_counts) >= 8
    assert max(round_counts) <= 16
    assert len(round_counts) > 1


def test_adaptive_table_contains_only_strong_adaptive_opponents():
    trainer = make_trainer()
    competitors = (
        trainer.create_training_baseline_competitors(
            trainer.population.networks[0],
            table_kind="adaptive",
        )
    )

    strategy_names = [
        strategy_name
        for strategy_name, _ in competitors
    ]

    assert len(competitors) == 7
    assert strategy_names.count("neural") == 1
    assert len(set(strategy_names[1:])) == 6
    assert set(strategy_names[1:]) == {
        "conservative_chaser",
        "aggressive_chaser",
        "lead_protector",
        "controlled_lead_martingale",
        "half_bankroll_leader",
        "adaptive_human",
    }


def test_disciplined_benchmark_has_minimum_control():
    trainer = make_trainer()
    competitors = trainer.create_fixed_benchmark_competitors(
        trainer.population.networks[0],
        "disciplined",
    )

    strategy_names = [
        strategy_name
        for strategy_name, _ in competitors
    ]

    assert len(competitors) == 7
    assert "neural" in strategy_names
    assert "minimum_control" in strategy_names


def test_risk_taker_benchmark_has_half_and_all_in_leaders():
    trainer = make_trainer()
    competitors = trainer.create_fixed_benchmark_competitors(
        trainer.population.networks[0],
        "risk_taker",
    )
    strategy_names = {
        strategy_name
        for strategy_name, _ in competitors
    }

    assert "half_bankroll_leader_1" in strategy_names
    assert "half_bankroll_leader_2" in strategy_names
    assert "opening_all_in_leader" in strategy_names


def test_randomized_human_holdout_samples_six_personalities():
    trainer = make_trainer()
    competitors = trainer.create_fixed_benchmark_competitors(
        trainer.population.networks[0],
        "randomized_human",
    )
    opponents = [agent for name, agent in competitors if name != "neural"]

    assert len(opponents) == 6
    human_opponents = [
        agent
        for agent in opponents
        if isinstance(agent, HumanBehaviorAgent)
    ]
    assert len(human_opponents) == 3
    assert len({agent.maximum_fraction for agent in human_opponents}) > 1
    assert len({agent.all_in_probability for agent in human_opponents}) > 1
    assert {name for name, _ in competitors} >= {
        "minimum_control",
        "balanced_chaser",
        "lead_protector",
    }


def test_extreme_strategies_are_isolated_to_extreme_tables():
    trainer = make_trainer()
    competitors = (
        trainer.create_training_baseline_competitors(
            trainer.population.networks[0],
            table_kind="extreme",
        )
    )

    strategy_names = {
        strategy_name
        for strategy_name, _ in competitors
    }

    assert "all_in" in strategy_names
    assert "five_percent" in strategy_names
    assert "fifteen_percent" in strategy_names


def test_late_stage_scenario_puts_focal_player_behind():
    trainer = make_trainer()
    players, total_rounds, completed_rounds = (
        trainer.create_tournament_scenario(
            player_count=7,
            focal_seat=3,
            stage="late",
        )
    )

    focal_bankroll = players[3].bankroll
    players_ahead = sum(
        player.bankroll > focal_bankroll
        for seat, player in enumerate(players)
        if seat != 3
    )

    assert 2 <= total_rounds - completed_rounds <= 3
    assert players_ahead == 2
    second_bankroll = sorted(
        (player.bankroll for player in players),
        reverse=True,
    )[1]
    relative_gap = (
        second_bankroll - focal_bankroll
    ) / focal_bankroll
    assert 0.029 <= relative_gap <= 0.081


def test_normal_scenario_starts_everyone_tied():
    trainer = make_trainer()
    players, total_rounds, completed_rounds = (
        trainer.create_tournament_scenario(
            player_count=7,
            focal_seat=3,
            stage="normal",
        )
    )

    assert {
        player.bankroll
        for player in players
    } == {10_000}
    assert 8 <= total_rounds <= 16
    assert completed_rounds == 0


def test_mid_stage_has_two_or_three_established_leaders():
    trainer = make_trainer()
    players, total_rounds, completed_rounds = (
        trainer.create_tournament_scenario(
            player_count=7,
            focal_seat=3,
            stage="mid",
        )
    )

    focal_bankroll = players[3].bankroll
    players_ahead = sum(
        player.bankroll > focal_bankroll
        for seat, player in enumerate(players)
        if seat != 3
    )

    assert 5 <= total_rounds - completed_rounds <= 8
    assert players_ahead in (2, 3)
    second_bankroll = sorted(
        (player.bankroll for player in players),
        reverse=True,
    )[1]
    relative_gap = (
        second_bankroll - focal_bankroll
    ) / focal_bankroll
    assert 0.049 <= relative_gap <= 0.201
