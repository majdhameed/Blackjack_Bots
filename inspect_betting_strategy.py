import argparse
from collections import Counter
import random

from agents.basic_strategy_agent import BasicStrategyAgent
from agents.chasing_agent import ChasingAgent
from agents.controlled_lead_martingale_agent import (
    ControlledLeadMartingaleAgent,
)
from agents.early_lead_agent import EarlyLeadAgent
from agents.lead_protection_agent import (
    LeadProtectionAgent,
)
from agents.neural_betting_agent import (
    NeuralBettingAgent,
)
from ml.betting_encoder import (
    encode_betting_observation,
)
from ml.betting_network import (
    BETTING_ACTION_NAMES,
    BettingNetwork,
)
from tournament.observation import (
    BettingObservation,
)

from ml.betting_probes import (
    create_betting_probes,
)
from evaluate_betting_network import (
    STRATEGY_NAMES,
    create_competitors,
    create_statistics,
    run_one_tournament,
)




def rank_and_gap_to_second(observation):
    active_bankrolls = [
        bankroll
        for bankroll, active in zip(
            observation.bankrolls,
            observation.active_players,
        )
        if active
    ]
    rank = 1 + sum(
        bankroll > observation.bankroll
        for bankroll in active_bankrolls
    )
    second_bankroll = sorted(
        active_bankrolls,
        reverse=True,
    )[1]
    gap_fraction = max(
        0,
        second_bankroll - observation.bankroll,
    ) / observation.bankroll
    return rank, gap_fraction


class RecordingNeuralBettingAgent(NeuralBettingAgent):
    """Neural agent that retains immutable observations for inspection."""

    def __init__(self, network, decisions):
        super().__init__(network)
        self.decisions = decisions

    def choose_bet(self, betting_observation):
        final_bet = super().choose_bet(betting_observation)
        features = encode_betting_observation(betting_observation)
        self.decisions.append(
            {
                "observation": betting_observation,
                "action_name": self.last_action_name,
                "legal_bet": final_bet,
                "raw_output": self.network.forward(features),
            }
        )
        return final_bet


def sample_tournament_decisions(
    network_path,
    number_of_tournaments=100,
    seed=123,
):
    """Collect neural decisions from reproducible, realistic mixed tables."""
    if (
        isinstance(number_of_tournaments, bool)
        or not isinstance(number_of_tournaments, int)
        or number_of_tournaments <= 0
    ):
        raise ValueError("number_of_tournaments must be positive")

    network = BettingNetwork.load(network_path)
    decisions = []
    competitors = []
    for name, factory in create_competitors(network_path):
        if name == "neural":
            competitors.append(
                (
                    name,
                    lambda network=network, decisions=decisions: (
                        RecordingNeuralBettingAgent(network, decisions)
                    ),
                )
            )
        else:
            competitors.append((name, factory))
    statistics = create_statistics(STRATEGY_NAMES)
    head_to_head = {
        name: 0.0
        for name in STRATEGY_NAMES
        if name != "neural"
    }
    seating_random = random.Random(seed)
    previous_random_state = random.getstate()
    random.seed(seed)
    try:
        for _ in range(number_of_tournaments):
            run_one_tournament(
                competitors=competitors,
                statistics=statistics,
                head_to_head=head_to_head,
                random_generator=seating_random,
                starting_bankroll=10_000,
                rounds_per_tournament=12,
                decks=6,
                minimum_bet=100,
                hit_soft_17=True,
                max_hands=4,
            )
    finally:
        random.setstate(previous_random_state)

    return decisions


def print_sampled_tournament_decisions(decisions, example_limit=15):
    """Summarize real states and show diverse non-minimum examples."""
    action_counts = Counter(
        decision["action_name"]
        for decision in decisions
    )
    non_minimum = [
        decision
        for decision in decisions
        if decision["legal_bet"]
        > decision["observation"].minimum_bet
    ]

    print()
    print("Sampled real-tournament decisions")
    print("---------------------------------")
    print(f"Decisions observed: {len(decisions):,}")
    print(
        "Above-minimum decisions: "
        f"{len(non_minimum):,} "
        f"({len(non_minimum) / len(decisions):.1%})"
    )
    print("Action distribution:")
    for action_name in BETTING_ACTION_NAMES:
        count = action_counts.get(action_name, 0)
        if count:
            print(
                f"  {action_name:<18} "
                f"{count:>6,} ({count / len(decisions):>6.1%})"
            )

    examples_by_context = {}
    for decision in non_minimum:
        observation = decision["observation"]
        rank, gap_fraction = rank_and_gap_to_second(observation)
        stage = min(
            2,
            (observation.round_number - 1) * 3
            // observation.total_rounds,
        )
        gap_bucket = int(gap_fraction * 20) * 5
        signature = (
            stage,
            rank,
            gap_bucket,
            decision["action_name"],
        )
        current_fraction = (
            decision["legal_bet"] / observation.bankroll
        )
        previous = examples_by_context.get(signature)
        if (
            previous is None
            or current_fraction
            > previous[0]
        ):
            examples_by_context[signature] = (
                current_fraction,
                decision,
                rank,
                gap_fraction,
            )

    ranked_examples = sorted(
        examples_by_context.items(),
        key=lambda item: item[1][0],
        reverse=True,
    )
    selected_signatures = set()
    examples = []
    for action_name in BETTING_ACTION_NAMES:
        if action_name in {"legacy_fraction", "minimum"}:
            continue
        matching_example = next(
            (
                (signature, details)
                for signature, details in ranked_examples
                if signature[3] == action_name
            ),
            None,
        )
        if matching_example is not None:
            signature, details = matching_example
            selected_signatures.add(signature)
            examples.append(details)

    for signature, details in ranked_examples:
        if len(examples) >= example_limit:
            break
        if signature not in selected_signatures:
            selected_signatures.add(signature)
            examples.append(details)

    examples = examples[:example_limit]

    if not examples:
        print("No above-minimum decisions were observed.")
        return

    print()
    print("Representative above-minimum states")
    header = (
        f"{'Rnd':>4}{'Left':>6}{'Rank':>6}{'Gap 2nd':>10}"
        f"{'Bankroll':>12}{'Action':>18}{'Bet':>10}"
        f"{'Bet %':>8}{'Opp avg':>10}{'Prev':>7}{'Losses':>8}"
    )
    print(header)
    print("-" * len(header))
    for fraction, decision, rank, gap_fraction in examples:
        observation = decision["observation"]
        print(
            f"{observation.round_number:>4}"
            f"{observation.rounds_remaining:>6}"
            f"{rank:>6}"
            f"{gap_fraction:>9.1%}"
            f"{observation.bankroll:>12,.0f}"
            f"{decision['action_name']:>18}"
            f"{decision['legal_bet']:>10,.0f}"
            f"{fraction:>7.1%}"
            f"{observation.average_opponent_previous_bet_fraction:>9.1%}"
            f"{observation.previous_result:>7.0f}"
            f"{observation.consecutive_losses:>8}"
        )


def inspect_strategy(
    network_path,
    sample_tournaments=100,
    sample_seed=123,
    sample_limit=15,
):
    network = BettingNetwork.load(network_path)
    agent = NeuralBettingAgent(network)
    minimum_agent = BasicStrategyAgent()
    chaser = ChasingAgent()
    controlled_lead = ControlledLeadMartingaleAgent()
    half_lead = EarlyLeadAgent(0.25, 0.50)
    all_in_lead = EarlyLeadAgent(0.10, 1.00)
    protector = LeadProtectionAgent()

    scenarios = create_betting_probes()

    print()
    print("Learned neural betting strategy")
    print("-------------------------------")

    header = (
        f"{'Scenario':<48}"
        f"{'Rank':>6}"
        f"{'Gap 2nd':>10}"
        f"{'Output':>10}"
        f"{'Action':>22}"
        f"{'Neural':>12}"
        f"{'Minimum':>12}"
        f"{'Chaser':>12}"
        f"{'Controlled':>12}"
        f"{'Half lead':>12}"
        f"{'All-in lead':>12}"
        f"{'Protector':>12}"
    )

    print(header)
    print("-" * len(header))

    minimum_matches = 0

    for scenario in scenarios:
        observation = scenario["observation"]
        rank, gap_fraction = rank_and_gap_to_second(
            observation
        )

        features = encode_betting_observation(
            observation
        )

        raw_fraction = network.forward(features)
        action_name = BETTING_ACTION_NAMES[
            network.preferred_action(features)
        ]

        final_bet = agent.choose_bet(
            observation
        )

        minimum_bet = minimum_agent.choose_bet(
            observation
        )
        chaser_bet = chaser.choose_bet(observation)
        controlled_bet = controlled_lead.choose_bet(
            observation
        )
        half_lead_bet = half_lead.choose_bet(
            observation
        )
        all_in_lead_bet = all_in_lead.choose_bet(
            observation
        )
        protector_bet = protector.choose_bet(
            observation
        )
        if final_bet == minimum_bet:
            minimum_matches += 1

        print(
            f"{scenario['name']:<48}"
            f"{rank:>6}"
            f"{gap_fraction:>9.1%}"
            f"{raw_fraction:>10.4f}"
            f"{action_name:>22}"
            f"{final_bet:>12,}"
            f"{minimum_bet:>12,}"
            f"{chaser_bet:>12,}"
            f"{controlled_bet:>12,}"
            f"{half_lead_bet:>12,}"
            f"{all_in_lead_bet:>12,}"
            f"{protector_bet:>12,}"
        )

    print()
    print(
        "Neural matches the minimum bettor in "
        f"{minimum_matches} of {len(scenarios)} scenarios."
    )

    if sample_tournaments > 0:
        decisions = sample_tournament_decisions(
            network_path,
            number_of_tournaments=sample_tournaments,
            seed=sample_seed,
        )
        print_sampled_tournament_decisions(
            decisions,
            example_limit=sample_limit,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect a saved betting-network checkpoint."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/best_betting_network.npz",
    )
    parser.add_argument(
        "--sample-tournaments",
        type=int,
        default=100,
        help="Number of realistic mixed tournaments to inspect; zero disables.",
    )
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--sample-limit", type=int, default=15)
    args = parser.parse_args(argv)
    if args.sample_tournaments < 0:
        parser.error("--sample-tournaments cannot be negative")
    if args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    inspect_strategy(
        args.model_path,
        sample_tournaments=args.sample_tournaments,
        sample_seed=args.sample_seed,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
