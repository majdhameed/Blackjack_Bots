import argparse
from functools import wraps
import inspect
import random
from pathlib import Path

from agents.basic_strategy_agent import (
    BasicStrategyAgent,
)
from agents.chasing_agent import ChasingAgent
from agents.controlled_lead_martingale_agent import (
    ControlledLeadMartingaleAgent,
)
from agents.early_lead_agent import EarlyLeadAgent
from agents.lead_protection_agent import (
    LeadProtectionAgent,
)
from agents.human_behavior_agent import HumanBehaviorAgent
from agents.neural_betting_agent import (
    NeuralBettingAgent,
)
from agents.unpredictable_betting_agent import (
    UnpredictableBettingAgent,
)
from blackjack.player import Player
from ml.betting_network import BettingNetwork
from tournament.tournament import Tournament


STRATEGY_NAMES = (
    "neural",
    "minimum_control",
    "controlled_lead_martingale",
    "aggressive_chaser",
    "lead_protector",
    "early_lead",
    "adaptive_human",
)

RISK_TAKER_STRATEGY_NAMES = (
    "neural",
    "minimum_control",
    "half_bankroll_leader_1",
    "half_bankroll_leader_2",
    "opening_all_in_leader",
    "lead_protector",
    "aggressive_chaser",
)


def isolated_evaluation_randomness(function):
    """Make every evaluation seed cover shoes and table randomness too."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        bound_arguments = signature.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()
        seed = bound_arguments.arguments["seed"]
        previous_random_state = random.getstate()
        random.seed(seed)
        try:
            return function(*args, **kwargs)
        finally:
            random.setstate(previous_random_state)

    return wrapper


def create_competitors(saved_network_path):
    network = BettingNetwork.load(
        saved_network_path
    )

    # Factories create a fresh agent for each tournament.
    return [
        (
            "neural",
            lambda: NeuralBettingAgent(network),
        ),
        (
            "minimum_control",
            lambda: BasicStrategyAgent(),
        ),
        (
            "controlled_lead_martingale",
            lambda: ControlledLeadMartingaleAgent(),
        ),
        (
            "aggressive_chaser",
            lambda: ChasingAgent(0.35, 0.65),
        ),
        ("lead_protector", LeadProtectionAgent),
        (
            "early_lead",
            lambda: EarlyLeadAgent(0.25, 0.50),
        ),
        (
            "adaptive_human",
            lambda: HumanBehaviorAgent(
                seed=12345,
                base_fraction=0.05,
                maximum_fraction=0.25,
            ),
        ),
    ]


def create_risk_taker_competitors(saved_network_path):
    network = BettingNetwork.load(saved_network_path)

    return [
        (
            "neural",
            lambda: NeuralBettingAgent(network),
        ),
        (
            "minimum_control",
            lambda: BasicStrategyAgent(),
        ),
        (
            "half_bankroll_leader_1",
            lambda: EarlyLeadAgent(0.25, 0.50),
        ),
        (
            "half_bankroll_leader_2",
            lambda: EarlyLeadAgent(0.25, 0.50),
        ),
        (
            "opening_all_in_leader",
            lambda: EarlyLeadAgent(0.10, 1.00),
        ),
        ("lead_protector", LeadProtectionAgent),
        (
            "aggressive_chaser",
            lambda: ChasingAgent(0.35, 0.65),
        ),
    ]


def create_league_competitors(
    saved_network_path,
    league_directory="models/league",
):
    network = BettingNetwork.load(saved_network_path)
    league_paths = sorted(
        Path(league_directory).glob("champion_*.npz")
    )[-6:]

    competitors = [
        (
            "neural",
            lambda: NeuralBettingAgent(network),
        )
    ]
    for champion_number, league_path in enumerate(
        league_paths,
        start=1,
    ):
        champion = BettingNetwork.load(league_path)
        competitors.append(
            (
                f"league_champion_{champion_number}",
                lambda champion=champion: NeuralBettingAgent(
                    champion
                ),
            )
        )

    fillers = (
        (
            "league_controlled",
            ControlledLeadMartingaleAgent,
        ),
        ("league_protector", LeadProtectionAgent),
        (
            "league_chaser",
            lambda: ChasingAgent(0.35, 0.65),
        ),
        (
            "league_human",
            lambda: HumanBehaviorAgent(
                seed=54321,
                base_fraction=0.05,
                maximum_fraction=0.30,
            ),
        ),
        (
            "league_early_lead",
            lambda: EarlyLeadAgent(0.25, 0.50),
        ),
        ("league_minimum", BasicStrategyAgent),
    )
    for name, factory in fillers:
        if len(competitors) == 7:
            break
        competitors.append((name, factory))

    return competitors


def create_statistics(strategy_names=STRATEGY_NAMES):
    return {
        strategy_name: {
            "win_credit": 0.0,
            "top_two_credit": 0.0,
            "position_total": 0.0,
            "bankroll_total": 0.0,
            "bankruptcies": 0,
        }
        for strategy_name in strategy_names
    }


def record_win_credits(
    rankings,
    seat_names,
    statistics,
):
    highest_bankroll = rankings[0][1]

    winning_seats = [
        seat_index
        for seat_index, bankroll in rankings
        if bankroll == highest_bankroll
    ]

    shared_credit = 1.0 / len(winning_seats)

    for seat_index in winning_seats:
        strategy_name = seat_names[seat_index]

        statistics[strategy_name][
            "win_credit"
        ] += shared_credit


def record_bankrolls(
    rankings,
    seat_names,
    statistics,
):
    for seat_index, bankroll in rankings:
        strategy_name = seat_names[seat_index]

        statistics[strategy_name][
            "bankroll_total"
        ] += bankroll

        if bankroll <= 0:
            statistics[strategy_name][
                "bankruptcies"
            ] += 1


def record_top_two_credits(
    rankings,
    seat_names,
    statistics,
):
    ranking_index = 0

    while ranking_index < len(rankings):
        bankroll = rankings[ranking_index][1]
        tie_end = ranking_index

        while (
            tie_end < len(rankings)
            and rankings[tie_end][1] == bankroll
        ):
            tie_end += 1

        tie_size = tie_end - ranking_index
        top_two_slots = max(
            0,
            min(tie_end, 2) - ranking_index,
        )
        shared_credit = top_two_slots / tie_size

        for tied_index in range(
            ranking_index,
            tie_end,
        ):
            seat_index = rankings[tied_index][0]
            strategy_name = seat_names[seat_index]
            statistics[strategy_name][
                "top_two_credit"
            ] += shared_credit

        ranking_index = tie_end


def record_finishing_positions(
    rankings,
    seat_names,
    statistics,
):
    ranking_index = 0

    while ranking_index < len(rankings):
        bankroll = rankings[ranking_index][1]

        tie_end = ranking_index

        while (
            tie_end < len(rankings)
            and rankings[tie_end][1] == bankroll
        ):
            tie_end += 1

        # Positions are one-indexed.
        occupied_positions = range(
            ranking_index + 1,
            tie_end + 1,
        )

        average_position = (
            sum(occupied_positions)
            / (tie_end - ranking_index)
        )

        for tied_index in range(
            ranking_index,
            tie_end,
        ):
            seat_index = rankings[tied_index][0]
            strategy_name = seat_names[
                seat_index
            ]

            statistics[strategy_name][
                "position_total"
            ] += average_position

        ranking_index = tie_end


def record_head_to_head(
    rankings,
    seat_names,
    head_to_head,
):
    bankroll_by_strategy = {}

    for seat_index, bankroll in rankings:
        strategy_name = seat_names[seat_index]
        bankroll_by_strategy[strategy_name] = (
            bankroll
        )

    neural_bankroll = bankroll_by_strategy[
        "neural"
    ]

    for strategy_name in head_to_head:
        opponent_bankroll = bankroll_by_strategy[
            strategy_name
        ]

        if neural_bankroll > opponent_bankroll:
            head_to_head[strategy_name] += 1.0
        elif neural_bankroll == opponent_bankroll:
            head_to_head[strategy_name] += 0.5


def run_one_tournament(
    competitors,
    statistics,
    head_to_head,
    random_generator,
    starting_bankroll,
    rounds_per_tournament,
    decks,
    minimum_bet,
    hit_soft_17,
    max_hands,
):
    shuffled_competitors = competitors.copy()

    random_generator.shuffle(
        shuffled_competitors
    )

    seat_names = [
        strategy_name
        for strategy_name, _ in (
            shuffled_competitors
        )
    ]

    bots = [
        agent_factory()
        for _, agent_factory in (
            shuffled_competitors
        )
    ]

    players = [
        Player(starting_bankroll)
        for _ in shuffled_competitors
    ]

    tournament = Tournament(
        players=players,
        bots=bots,
        number_of_rounds=rounds_per_tournament,
        decks=decks,
        minimum_bet=minimum_bet,
        hit_soft_17=hit_soft_17,
        max_hands=max_hands,
    )

    rankings = tournament.play_tournament()

    record_win_credits(
        rankings,
        seat_names,
        statistics,
    )

    record_bankrolls(
        rankings,
        seat_names,
        statistics,
    )

    record_top_two_credits(
        rankings,
        seat_names,
        statistics,
    )

    record_finishing_positions(
        rankings,
        seat_names,
        statistics,
    )

    record_head_to_head(
        rankings,
        seat_names,
        head_to_head,
    )

    return rankings


@isolated_evaluation_randomness
def evaluate_network(
    saved_network_path,
    number_of_tournaments,
    starting_bankroll=10_000,
    rounds_per_tournament=12,
    decks=6,
    minimum_bet=100,
    hit_soft_17=True,
    max_hands=4,
    seed=123,
    table_kind="standard",
    league_directory="models/league",
):
    if (
        isinstance(number_of_tournaments, bool)
        or not isinstance(
            number_of_tournaments,
            int,
        )
    ):
        raise TypeError(
            "number_of_tournaments must be an integer"
        )

    if number_of_tournaments <= 0:
        raise ValueError(
            "number_of_tournaments must be positive"
        )

    if table_kind == "standard":
        competitors = create_competitors(
            saved_network_path
        )
        strategy_names = STRATEGY_NAMES
        title = "Betting strategy evaluation"
    elif table_kind == "risk_taker":
        competitors = create_risk_taker_competitors(
            saved_network_path
        )
        strategy_names = RISK_TAKER_STRATEGY_NAMES
        title = "Early risk-taker table evaluation"
    elif table_kind == "league":
        competitors = create_league_competitors(
            saved_network_path,
            league_directory,
        )
        strategy_names = tuple(
            name for name, _ in competitors
        )
        title = "Archived champion league evaluation"
    else:
        raise ValueError(
            "table_kind must be standard, risk_taker, or league"
        )

    statistics = create_statistics(strategy_names)

    head_to_head = {
        strategy_name: 0.0
        for strategy_name in strategy_names
        if strategy_name != "neural"
    }

    random_generator = random.Random(seed)

    progress_interval = max(
        1,
        number_of_tournaments // 10,
    )

    for tournament_number in range(
        1,
        number_of_tournaments + 1,
    ):
        run_one_tournament(
            competitors=competitors,
            statistics=statistics,
            head_to_head=head_to_head,
            random_generator=random_generator,
            starting_bankroll=starting_bankroll,
            rounds_per_tournament=(
                rounds_per_tournament
            ),
            decks=decks,
            minimum_bet=minimum_bet,
            hit_soft_17=hit_soft_17,
            max_hands=max_hands,
        )

        if (
            tournament_number % progress_interval
            == 0
            or tournament_number
            == number_of_tournaments
        ):
            print(
                f"Completed {tournament_number:,} "
                f"of {number_of_tournaments:,} "
                "tournaments"
            )

    return {
        "number_of_tournaments": (
            number_of_tournaments
        ),
        "statistics": statistics,
        "head_to_head": head_to_head,
        "strategy_names": strategy_names,
        "title": title,
    }


def print_results(results):
    number_of_tournaments = results[
        "number_of_tournaments"
    ]

    statistics = results["statistics"]

    print()
    title = results.get(
        "title",
        "Betting strategy evaluation",
    )
    print(title)
    print("-" * len(title))
    print(
        f"Tournaments: "
        f"{number_of_tournaments:,}"
    )
    print()

    header = (
        f"{'Strategy':<28}"
        f"{'Win rate':>12}"
        f"{'Top two':>12}"
        f"{'Avg position':>16}"
        f"{'Avg bankroll':>18}"
        f"{'Bankruptcy':>14}"
    )

    print(header)
    print("-" * len(header))

    for strategy_name in results.get(
        "strategy_names",
        STRATEGY_NAMES,
    ):
        strategy_statistics = statistics[
            strategy_name
        ]

        win_rate = (
            strategy_statistics["win_credit"]
            / number_of_tournaments
            * 100
        )

        top_two_rate = (
            strategy_statistics["top_two_credit"]
            / number_of_tournaments
            * 100
        )

        average_position = (
            strategy_statistics["position_total"]
            / number_of_tournaments
        )

        average_bankroll = (
            strategy_statistics["bankroll_total"]
            / number_of_tournaments
        )

        bankruptcy_rate = (
            strategy_statistics["bankruptcies"]
            / number_of_tournaments
            * 100
        )

        print(
            f"{strategy_name:<28}"
            f"{win_rate:>11.2f}%"
            f"{top_two_rate:>11.2f}%"
            f"{average_position:>16.2f}"
            f"{average_bankroll:>18,.2f}"
            f"{bankruptcy_rate:>13.2f}%"
        )

    print()
    print("Neural head-to-head results")
    print("---------------------------")

    for strategy_name, credit in results[
        "head_to_head"
    ].items():
        rate = (
            credit
            / number_of_tournaments
            * 100
        )

        print(
            f"Neural vs {strategy_name:<10}: "
            f"{rate:6.2f}%"
        )


def ask_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))

            if value <= 0:
                raise ValueError

            return value

        except ValueError:
            print(
                "Please enter a positive integer."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a saved betting-network checkpoint."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/best_betting_network.npz",
    )
    parser.add_argument(
        "--tournaments",
        type=int,
        help="Skip the interactive tournament-count prompt.",
    )
    parser.add_argument(
        "--league-directory",
        help=(
            "League checkpoint directory; defaults to the league folder "
            "beside the selected model."
        ),
    )
    args = parser.parse_args(argv)
    saved_network_path = Path(args.model_path)

    if not saved_network_path.exists():
        raise FileNotFoundError(
            "Saved network was not found at "
            f"{saved_network_path}"
        )

    number_of_tournaments = args.tournaments
    if number_of_tournaments is None:
        number_of_tournaments = ask_positive_integer(
            "Number of evaluation tournaments: "
        )
    elif number_of_tournaments <= 0:
        parser.error("--tournaments must be positive")

    results = evaluate_network(
        saved_network_path=saved_network_path,
        number_of_tournaments=(
            number_of_tournaments
        ),
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        seed=123,
    )

    print_results(results)

    risk_results = evaluate_network(
        saved_network_path=saved_network_path,
        number_of_tournaments=(
            number_of_tournaments
        ),
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        seed=321,
        table_kind="risk_taker",
    )

    print_results(risk_results)

    league_directory = (
        Path(args.league_directory)
        if args.league_directory is not None
        else saved_network_path.parent / "league"
    )
    if any(league_directory.glob("champion_*.npz")):
        league_results = evaluate_network(
            saved_network_path=saved_network_path,
            number_of_tournaments=number_of_tournaments,
            starting_bankroll=10_000,
            rounds_per_tournament=12,
            decks=6,
            minimum_bet=100,
            hit_soft_17=True,
            max_hands=4,
            seed=777,
            table_kind="league",
            league_directory=league_directory,
        )
        print_results(league_results)

    results = evaluate_against_six_minimum(
        saved_network_path=saved_network_path,
        number_of_tournaments=(
            number_of_tournaments
        ),
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        seed=123,
    )

    print_six_minimum_results(results)

@isolated_evaluation_randomness
def evaluate_against_six_minimum(
    saved_network_path,
    number_of_tournaments,
    starting_bankroll=10_000,
    rounds_per_tournament=12,
    decks=6,
    minimum_bet=100,
    hit_soft_17=True,
    max_hands=4,
    seed=123,
):
    if (
        isinstance(number_of_tournaments, bool)
        or not isinstance(
            number_of_tournaments,
            int,
        )
    ):
        raise TypeError(
            "number_of_tournaments must be an integer"
        )

    if number_of_tournaments <= 0:
        raise ValueError(
            "number_of_tournaments must be positive"
        )

    network = BettingNetwork.load(
        saved_network_path
    )

    random_generator = random.Random(seed)

    first_place_credit = 0.0
    top_two_credit = 0.0
    position_total = 0.0
    bankroll_total = 0.0
    bankruptcies = 0

    progress_interval = max(
        1,
        number_of_tournaments // 10,
    )

    for tournament_number in range(
        1,
        number_of_tournaments + 1,
    ):
        competitors = [
            (
                "neural",
                NeuralBettingAgent(network),
            ),
            (
                "minimum_1",
                BasicStrategyAgent(),
            ),
            (
                "minimum_2",
                BasicStrategyAgent(),
            ),
            (
                "minimum_3",
                BasicStrategyAgent(),
            ),
            (
                "minimum_4",
                BasicStrategyAgent(),
            ),
            (
                "minimum_5",
                BasicStrategyAgent(),
            ),
            (
                "minimum_6",
                BasicStrategyAgent(),
            ),
        ]

        random_generator.shuffle(competitors)

        seat_names = [
            name
            for name, _ in competitors
        ]

        bots = [
            bot
            for _, bot in competitors
        ]

        players = [
            Player(starting_bankroll)
            for _ in competitors
        ]

        tournament = Tournament(
            players=players,
            bots=bots,
            number_of_rounds=(
                rounds_per_tournament
            ),
            decks=decks,
            minimum_bet=minimum_bet,
            hit_soft_17=hit_soft_17,
            max_hands=max_hands,
        )

        rankings = tournament.play_tournament()

        neural_seat_index = seat_names.index(
            "neural"
        )

        neural_ranking_index = None
        neural_bankroll = None

        for ranking_index, (
            seat_index,
            bankroll,
        ) in enumerate(rankings):
            if seat_index == neural_seat_index:
                neural_ranking_index = (
                    ranking_index
                )

                neural_bankroll = bankroll
                break

        if neural_ranking_index is None:
            raise RuntimeError(
                "Neural player was missing "
                "from tournament rankings"
            )

        # Find every player tied with the neural bot.
        tie_start = neural_ranking_index

        while (
            tie_start > 0
            and rankings[tie_start - 1][1]
            == neural_bankroll
        ):
            tie_start -= 1

        tie_end = neural_ranking_index + 1

        while (
            tie_end < len(rankings)
            and rankings[tie_end][1]
            == neural_bankroll
        ):
            tie_end += 1

        tie_size = tie_end - tie_start

        # Rankings use zero-based indexes here.
        # A tie occupying indexes 0 and 1 shares
        # first-place credit equally.
        if tie_start == 0:
            first_place_credit += (
                1.0 / tie_size
            )

        # Determine how many of the tied positions
        # fall within the top two.
        top_two_slots = max(
            0,
            min(tie_end, 2) - tie_start,
        )

        top_two_credit += (
            top_two_slots / tie_size
        )

        # Convert occupied indexes to one-based
        # finishing positions.
        occupied_positions = range(
            tie_start + 1,
            tie_end + 1,
        )

        average_position = (
            sum(occupied_positions)
            / tie_size
        )

        position_total += average_position
        bankroll_total += neural_bankroll

        if neural_bankroll <= 0:
            bankruptcies += 1

        if (
            tournament_number % progress_interval
            == 0
            or tournament_number
            == number_of_tournaments
        ):
            print(
                f"Completed {tournament_number:,} "
                f"of {number_of_tournaments:,} "
                "six-minimum tournaments"
            )

    return {
        "number_of_tournaments": (
            number_of_tournaments
        ),
        "first_place_rate": (
            first_place_credit
            / number_of_tournaments
            * 100
        ),
        "top_two_rate": (
            top_two_credit
            / number_of_tournaments
            * 100
        ),
        "average_position": (
            position_total
            / number_of_tournaments
        ),
        "average_bankroll": (
            bankroll_total
            / number_of_tournaments
        ),
        "bankruptcy_rate": (
            bankruptcies
            / number_of_tournaments
            * 100
        ),
    }

def print_six_minimum_results(results):
    print()
    print("Neural versus six minimum bettors")
    print("----------------------------------")

    print(
        f"Tournaments: "
        f"{results['number_of_tournaments']:,}"
    )

    print(
        f"First-place rate: "
        f"{results['first_place_rate']:.2f}%"
    )

    print(
        f"Top-two advancement rate: "
        f"{results['top_two_rate']:.2f}%"
    )

    print(
        f"Average position: "
        f"{results['average_position']:.2f}"
    )

    print(
        f"Average bankroll: "
        f"${results['average_bankroll']:,.2f}"
    )

    print(
        f"Bankruptcy rate: "
        f"{results['bankruptcy_rate']:.2f}%"
    )

    print()
    print("Equal-strategy reference values")
    print("-------------------------------")
    print("First-place rate: 14.29%")
    print("Top-two rate: 28.57%")
    print("Average position: 4.00")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation stopped.")

