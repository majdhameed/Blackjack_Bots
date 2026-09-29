import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from ml.betting_network import BettingNetwork
from ml.population import Population
from ml.trainer import Trainer
from ml.training_metrics import (
    save_training_history,
    build_generation_record,
    load_training_history,
)


def create_run_artifact_paths(base_directory="runs", timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()

    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    run_directory = Path(base_directory) / run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    return {
        "run_directory": run_directory,
        "model_path": run_directory / "best_betting_network.npz",
        "metrics_path": run_directory / "metrics.csv",
    }

def create_population(population_size, elite_count, mutation_rate, mutation_strength, seed=None):

    population = Population(population_size, elite_count, mutation_rate, mutation_strength, seed)
    return population


def warm_start_population(population, resume_directory):
    """Seed half a fresh population from a completed run's champions."""
    resume_directory = Path(resume_directory)
    model_path = resume_directory / "best_betting_network.npz"
    metrics_path = resume_directory / "metrics.csv"

    if not resume_directory.is_dir():
        raise FileNotFoundError(
            f"Resume directory does not exist: {resume_directory}"
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Resume model does not exist: {model_path}"
        )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Resume metrics do not exist: {metrics_path}"
        )

    metrics_history = load_training_history(metrics_path)
    if not metrics_history:
        raise ValueError("Resume metrics contain no generations")
    next_generation = max(
        record["generation"] for record in metrics_history
    ) + 1

    source_networks = [BettingNetwork.load(model_path)]
    champions_directory = resume_directory / "champions"
    if champions_directory.is_dir():
        for style in ("safe", "balanced", "aggressive"):
            style_paths = sorted(
                (champions_directory / style).glob("*.npz")
            )
            if style_paths:
                source_networks.append(
                    BettingNetwork.load(style_paths[-1])
                )

    warm_network_count = max(
        population.elite_count,
        population.population_size // 2,
    )
    warm_network_count = min(
        warm_network_count,
        population.population_size,
    )

    for network_index in range(warm_network_count):
        network = source_networks[
            network_index % len(source_networks)
        ].clone()
        # Keep one exact copy of every available source. The remaining warm
        # networks are independently mutated descendants.
        if network_index >= len(source_networks):
            mutation_seed = int(
                population.random_generator.integers(0, 2**32)
            )
            network.random_generator = np.random.default_rng(
                mutation_seed
            )
            network.mutate(
                population.mutation_rate,
                population.mutation_strength,
            )
        population.networks[network_index] = network

    population.generation_number = next_generation
    population.reset_fitness()
    return {
        "incumbent_network": source_networks[0].clone(),
        "next_generation": next_generation,
        "warm_network_count": warm_network_count,
        "fresh_network_count": (
            population.population_size - warm_network_count
        ),
        "source_network_count": len(source_networks),
    }


def carry_forward_league(resume_directory, league_directory):
    """Copy validated league checkpoints into the new run directory."""
    source_directory = Path(resume_directory) / "league"
    destination_directory = Path(league_directory)
    if not source_directory.is_dir():
        return 0

    league_paths = sorted(
        source_directory.glob("champion_*.npz")
    )[-8:]
    for league_path in league_paths:
        network = BettingNetwork.load(league_path)
        network.save(destination_directory / league_path.name)
    return len(league_paths)

def create_trainer(
    population,
    starting_bankroll,
    rounds_per_tournament,
    decks,
    minimum_bet,
    hit_soft_17,
    max_hands,
    randomize_training_rounds=False,
    league_directory=None,
):
    trainer = Trainer(
        population,
        starting_bankroll,
        rounds_per_tournament,
        decks,
        minimum_bet,
        hit_soft_17,
        max_hands,
        randomize_training_rounds,
        league_directory=league_directory,
    )
    return trainer

def run_training(
    trainer,
    generations,
    tournaments_per_network,
    output_path,
    baseline_tournaments_per_network=0,
    print_fn=print,
    benchmark_tournaments_per_generation=0,
    benchmark_candidate_count=1,
    checkpoint_verification_tournaments=0,
    metrics_output_path=None,
    early_stopping_patience=0,
    final_audit_tournaments=0,
    initial_best_network=None,
    initial_best_generation=None,
):

    if type(generations) is not int:
        raise TypeError("Generations must be an integer")
    if generations <= 0:
        raise ValueError("There must be atleast 1 generation")

    if type(tournaments_per_network) is not int:
        raise TypeError("Generations must be an integer")
    if tournaments_per_network <= 0:
        raise ValueError("There must be atleast 1 generation")

    if (
        isinstance(
            benchmark_tournaments_per_generation,
            bool,
        )
        or not isinstance(
            benchmark_tournaments_per_generation,
            int,
        )
    ):
        raise TypeError(
            "benchmark_tournaments_per_generation must be an integer"
        )

    if benchmark_tournaments_per_generation < 0:
        raise ValueError(
            "benchmark_tournaments_per_generation cannot be negative"
        )

    if (
        isinstance(benchmark_candidate_count, bool)
        or not isinstance(benchmark_candidate_count, int)
    ):
        raise TypeError(
            "benchmark_candidate_count must be an integer"
        )

    if benchmark_candidate_count <= 0:
        raise ValueError(
            "benchmark_candidate_count must be positive"
        )

    if (
        isinstance(checkpoint_verification_tournaments, bool)
        or not isinstance(
            checkpoint_verification_tournaments,
            int,
        )
    ):
        raise TypeError(
            "checkpoint_verification_tournaments must be an integer"
        )
    if checkpoint_verification_tournaments < 0:
        raise ValueError(
            "checkpoint_verification_tournaments cannot be negative"
        )
    if 0 < checkpoint_verification_tournaments < 4:
        raise ValueError(
            "checkpoint_verification_tournaments must be zero or at least four"
        )
    if (
        isinstance(early_stopping_patience, bool)
        or not isinstance(early_stopping_patience, int)
    ):
        raise TypeError("early_stopping_patience must be an integer")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    if (
        isinstance(final_audit_tournaments, bool)
        or not isinstance(final_audit_tournaments, int)
    ):
        raise TypeError("final_audit_tournaments must be an integer")
    if final_audit_tournaments < 0:
        raise ValueError("final_audit_tournaments cannot be negative")
    
    history = []
    best_network = initial_best_network
    best_fitness = float('-inf')
    best_comparison_key = None
    best_benchmark_fitness = None
    best_benchmark_normal_fitness = None
    best_benchmark_mid_fitness = None
    best_benchmark_late_fitness = None
    best_benchmark_holdout_fitness = None
    best_benchmark_minimum_holdout_fitness = None
    best_benchmark_human_holdout_fitness = None
    best_benchmark_selection_fitness = None
    best_benchmark_raw_selection_fitness = None
    best_benchmark_advantage_fitness = None
    best_generation = initial_best_generation
    metrics_history = []
    checkpoint_paths = []
    portfolio_best_scores = {}
    portfolio_paths = []
    early_stopped = False

    if best_network is not None:
        best_network.save(output_path)


    for _ in range(generations):
        verified_score = None
        if (
            baseline_tournaments_per_network > 0
            or benchmark_tournaments_per_generation > 0
        ):
            result = trainer.train_generation(
                tournaments_per_network=(
                    tournaments_per_network
                ),
                baseline_tournaments_per_network=(
                    baseline_tournaments_per_network
                ),
                benchmark_tournaments=(
                    benchmark_tournaments_per_generation
                ),
                benchmark_candidate_count=(
                    benchmark_candidate_count
                ),
            )
        else:
            result = trainer.train_generation(
                tournaments_per_network
            )

        history_entry = {
            "generation": result["generation"],
            "best_network_index": result[
                "best_network_index"
            ],
            "best_fitness": result["best_fitness"],
            "average_fitness": result[
                "average_fitness"
            ],
            "benchmark_fitness": result.get(
                "benchmark_fitness"
            ),
            "benchmark_normal_fitness": result.get(
                "benchmark_normal_fitness"
            ),
            "benchmark_mid_fitness": result.get(
                "benchmark_mid_fitness"
            ),
            "benchmark_late_fitness": result.get(
                "benchmark_late_fitness"
            ),
            "benchmark_holdout_fitness": result.get(
                "benchmark_holdout_fitness"
            ),
            "benchmark_minimum_holdout_fitness": result.get(
                "benchmark_minimum_holdout_fitness"
            ),
            "benchmark_human_holdout_fitness": result.get(
                "benchmark_human_holdout_fitness"
            ),
            "benchmark_selection_fitness": result.get(
                "benchmark_selection_fitness"
            ),
            "benchmark_raw_selection_fitness": result.get(
                "benchmark_raw_selection_fitness"
            ),
            "benchmark_advantage_fitness": result.get(
                "benchmark_advantage_fitness"
            ),
        }

        history.append(history_entry)

        
        print_fn(
            f"Generation {result['generation']}: "
            f"best fitness = "
            f"{result['best_fitness']:.2f}, "
            f"average fitness = "
            f"{result['average_fitness']:.2f}"
        )

        if result.get("benchmark_fitness") is not None:
            print_fn(
                "Rotating benchmark average top-two "
                f"rate = {result['benchmark_fitness']:.2f}"
            )

            print_fn(
                "  normal tables = "
                f"{result['benchmark_normal_fitness']:.2f}, "
                "mid-stage tables = "
                f"{result['benchmark_mid_fitness']:.2f}, "
                "late-stage tables = "
                f"{result['benchmark_late_fitness']:.2f}"
            )

            print_fn(
                "  full-tournament holdout = "
                f"{result['benchmark_holdout_fitness']:.2f}, "
                "six-minimum holdout = "
                f"{result['benchmark_minimum_holdout_fitness']:.2f}, "
                "minimum-relative selection score = "
                f"{result['benchmark_selection_fitness']:+.3f}"
            )
            if result.get("benchmark_raw_selection_fitness") is not None:
                print_fn(
                    "  raw weighted score = "
                    f"{result['benchmark_raw_selection_fitness']:.2f}, "
                    "advantage over matched minimum = "
                    f"{result['benchmark_selection_fitness']:+.3f}"
                )
            if result.get("benchmark_normal_advantage") is not None:
                print_fn(
                    "  matched-minimum advantage: normal = "
                    f"{result['benchmark_normal_advantage']:+.3f}, "
                    "mid = "
                    f"{result['benchmark_mid_advantage']:+.3f}, "
                    "late = "
                    f"{result['benchmark_late_advantage']:+.3f}"
                )
            if result.get("benchmark_human_holdout_fitness") is not None:
                print_fn(
                    "  randomized-human holdout = "
                    f"{result['benchmark_human_holdout_fitness']:.2f}"
                )

            parent_styles = result.get("benchmark_parent_styles", [])
            if parent_styles:
                style_counts = {
                    style: parent_styles.count(style)
                    for style in (
                        "safe",
                        "balanced",
                        "aggressive",
                        "self_play_fallback",
                    )
                }
                print_fn(
                    "  benchmark-selected reproduction parents: "
                    f"safe = {style_counts['safe']}, "
                    f"balanced = {style_counts['balanced']}, "
                    f"aggressive = {style_counts['aggressive']}, "
                    "self-play fallback = "
                    f"{style_counts['self_play_fallback']}"
                )

            if result.get("league_promoted"):
                print_fn(
                    "  promoted to opponent league; "
                    f"league size = {result['league_size']}"
                )

        selection_fitness = result.get(
            "benchmark_selection_fitness"
        )
        if selection_fitness is None:
            selection_fitness = result["best_fitness"]
            comparison_key = (selection_fitness, selection_fitness)
        else:
            comparison_key = (
                result.get("benchmark_raw_selection_fitness", 0.0),
                selection_fitness,
            )

        should_save = (
            selection_fitness is not None
            if checkpoint_verification_tournaments > 0
            else (
                best_comparison_key is None
                or comparison_key > best_comparison_key
            )
        )

    
        metrics_for_checkpoint = result


        if (
            selection_fitness is not None
            and checkpoint_verification_tournaments > 0
        ):
            compare_networks = getattr(
                trainer,
                "compare_checkpoint_networks",
                None,
            )
            if callable(compare_networks):
                challenger_metrics, incumbent_metrics = (
                    compare_networks(
                        result["best_network"],
                        best_network,
                        checkpoint_verification_tournaments,
                    )
                )

                challenger_key = (
                    challenger_metrics.get(
                        "benchmark_raw_selection_fitness",
                        0.0,
                    ),
                    challenger_metrics[
                        "benchmark_selection_fitness"
                    ],
                )
                incumbent_key = None
                if incumbent_metrics is not None:
                    incumbent_key = (
                        incumbent_metrics.get(
                            "benchmark_raw_selection_fitness",
                            0.0,
                        ),
                        incumbent_metrics[
                            "benchmark_selection_fitness"
                        ],
                    )
                should_save = challenger_metrics.get(
                    "checkpoint_accepted",
                    (
                        incumbent_key is None
                        or challenger_key > incumbent_key
                    ),
                )
                comparison_key = challenger_key
                metrics_for_checkpoint = challenger_metrics
                incumbent_text = (
                    "none"
                    if incumbent_metrics is None
                    else f"{incumbent_key[0]:+.3f}"
                )
                batches_won = challenger_metrics.get(
                    "checkpoint_batches_won"
                )
                batch_count = challenger_metrics.get(
                    "checkpoint_batch_count"
                )
                print_fn(
                    "  rotating checkpoint verification: "
                    "challenger = "
                    f"{challenger_key[0]:+.3f}, incumbent = "
                    f"{incumbent_text}"
                )
                if batches_won is not None and batch_count is not None:
                    print_fn(
                        "  validation batches won = "
                        f"{batches_won}/{batch_count}"
                    )
                if not should_save:
                    print_fn(
                        "  checkpoint rejected by independent verification"
                    )
                verified_score = challenger_metrics.get(
                    "benchmark_raw_selection_fitness",
                    challenger_metrics["benchmark_selection_fitness"],
                )
                result["checkpoint_batches_won"] = batches_won
                result["checkpoint_batch_count"] = batch_count
                result["checkpoint_improvement"] = challenger_metrics.get(
                    "checkpoint_improvement"
                )
                for gate_name in (
                    "checkpoint_worst_category_delta",
                    "checkpoint_bankruptcy_delta",
                    "checkpoint_category_gate_passed",
                    "checkpoint_bankruptcy_gate_passed",
                    "checkpoint_unanimous_batches",
                ):
                    result[gate_name] = challenger_metrics.get(gate_name)
                if incumbent_metrics is not None:
                    worst_delta = challenger_metrics.get(
                        "checkpoint_worst_category_delta"
                    )
                    bankruptcy_delta = challenger_metrics.get(
                        "checkpoint_bankruptcy_delta"
                    )
                    print_fn(
                        "  conservative checkpoint gates: "
                        "unanimous batches = "
                        f"{challenger_metrics.get('checkpoint_unanimous_batches')}, "
                        "category floor = "
                        + (
                            "n/a"
                            if worst_delta is None
                            else f"{worst_delta:+.3f}"
                        )
                        + ", bankruptcy change = "
                        + (
                            "n/a"
                            if bankruptcy_delta is None
                            else f"{bankruptcy_delta:+.3f}"
                        )
                    )
                    regressed_categories = challenger_metrics.get(
                        "checkpoint_regressed_categories",
                        (),
                    )
                    if regressed_categories:
                        print_fn(
                            "  rejected category regressions: "
                            + ", ".join(regressed_categories)
                        )
                for diagnostic_name in (
                    "benchmark_top_two_rate",
                    "benchmark_first_place_rate",
                    "benchmark_average_position",
                    "benchmark_bankruptcy_rate",
                ):
                    result[diagnostic_name] = challenger_metrics.get(
                        diagnostic_name
                    )

                bankruptcy_rate = challenger_metrics.get(
                    "benchmark_bankruptcy_rate"
                )
                if bankruptcy_rate is not None:
                    if bankruptcy_rate <= 0.02:
                        risk_style = "safe"
                    elif bankruptcy_rate <= 0.10:
                        risk_style = "balanced"
                    else:
                        risk_style = "aggressive"
                    portfolio_score = challenger_key[0]
                    if portfolio_score > portfolio_best_scores.get(
                        risk_style,
                        float("-inf"),
                    ):
                        portfolio_best_scores[risk_style] = portfolio_score
                        portfolio_path = (
                            Path(output_path).parent
                            / "champions"
                            / risk_style
                            / (
                                f"generation_{result['generation']:04d}_"
                                f"{portfolio_score:.4f}.npz"
                            )
                        )
                        result["best_network"].save(portfolio_path)
                        portfolio_paths.append(portfolio_path)
                        print_fn(
                            f"  new {risk_style} champion archived: "
                            f"{portfolio_path}"
                        )
            
                

        if should_save:
            best_comparison_key = comparison_key
            best_fitness = result["best_fitness"]
            best_benchmark_fitness = metrics_for_checkpoint.get(
                "benchmark_fitness"
            )
            best_benchmark_normal_fitness = metrics_for_checkpoint.get(
                "benchmark_normal_fitness"
            )
            best_benchmark_mid_fitness = metrics_for_checkpoint.get(
                "benchmark_mid_fitness"
            )
            best_benchmark_late_fitness = metrics_for_checkpoint.get(
                "benchmark_late_fitness"
            )
            best_benchmark_holdout_fitness = metrics_for_checkpoint.get(
                "benchmark_holdout_fitness"
            )
            best_benchmark_minimum_holdout_fitness = metrics_for_checkpoint.get(
                "benchmark_minimum_holdout_fitness"
            )
            best_benchmark_human_holdout_fitness = metrics_for_checkpoint.get(
                "benchmark_human_holdout_fitness"
            )
            best_benchmark_selection_fitness = metrics_for_checkpoint.get(
                "benchmark_selection_fitness"
            )
            best_benchmark_raw_selection_fitness = metrics_for_checkpoint.get(
                "benchmark_raw_selection_fitness"
            )
            best_benchmark_advantage_fitness = metrics_for_checkpoint.get(
                "benchmark_advantage_fitness"
            )
            best_generation = result["generation"]
            best_network = result["best_network"]
            best_network.save(output_path)
            archive_path = (
                Path(output_path).parent
                / "checkpoints"
                / (
                    f"generation_{best_generation:04d}_"
                    f"{comparison_key[0]:+.4f}.npz"
                )
            )
            best_network.save(archive_path)
            checkpoint_paths.append(archive_path)
            print_fn(
                "Checkpoint saved: "
                f"{output_path} "
                f"(generation {best_generation})"
            )
            print_fn(f"Checkpoint archived: {archive_path}")
       

        if metrics_output_path:
            record = build_generation_record(result, verified_score, should_save, )
            metrics_history.append(record)
            save_training_history(metrics_output_path, metrics_history)

        if (
            early_stopping_patience > 0
            and best_generation is not None
            and result["generation"] - best_generation
            >= early_stopping_patience
        ):
            early_stopped = True
            print_fn(
                "Early stopping: no verified checkpoint improvement for "
                f"{early_stopping_patience} generations."
            )
            break


    final_audit_metrics = None
    if final_audit_tournaments > 0 and best_network is not None:
        final_audit_metrics = trainer.evaluate_robust_checkpoint(
            best_network,
            final_audit_tournaments,
        )
        print_fn(
            "Final untouched audit: top-two rate = "
            f"{final_audit_metrics['benchmark_raw_selection_fitness']:.3f}, "
            "first-place rate = "
            f"{final_audit_metrics.get('benchmark_first_place_rate', 0.0):.3f}, "
            "bankruptcy rate = "
            f"{final_audit_metrics.get('benchmark_bankruptcy_rate', 0.0):.3f}"
        )

    return{
        "best_generation": best_generation,
        "best_fitness": best_fitness,
        "best_benchmark_fitness": (
            best_benchmark_fitness
        ),
        "best_benchmark_normal_fitness": (
            best_benchmark_normal_fitness
        ),
        "best_benchmark_mid_fitness": (
            best_benchmark_mid_fitness
        ),
        "best_benchmark_late_fitness": (
            best_benchmark_late_fitness
        ),
        "best_benchmark_holdout_fitness": (
            best_benchmark_holdout_fitness
        ),
        "best_benchmark_minimum_holdout_fitness": (
            best_benchmark_minimum_holdout_fitness
        ),
        "best_benchmark_human_holdout_fitness": (
            best_benchmark_human_holdout_fitness
        ),
        "best_benchmark_selection_fitness": (
            best_benchmark_selection_fitness
        ),
        "best_benchmark_raw_selection_fitness": (
            best_benchmark_raw_selection_fitness
        ),
        "best_benchmark_advantage_fitness": (
            best_benchmark_advantage_fitness
        ),
        "best_network": best_network,
        "output_path": output_path,
        "history": history,
        "metrics_history": metrics_history,
        "checkpoint_paths": checkpoint_paths,
        "portfolio_paths": portfolio_paths,
        "portfolio_best_scores": portfolio_best_scores,
        "early_stopped": early_stopped,
        "completed_generations": len(history),
        "final_audit_metrics": final_audit_metrics,
    }


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description="Train the tournament betting network."
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Warm-start from a previous run directory containing "
            "best_betting_network.npz and metrics.csv."
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=250,
        help="Number of additional generations to train (default: 250).",
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_arguments(arguments)
    artifact_paths = create_run_artifact_paths()

    population = create_population(
        population_size=70,
        elite_count=7,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=123,
    )

    resume_state = None
    league_directory = artifact_paths["run_directory"] / "league"
    if args.resume_from is not None:
        resume_state = warm_start_population(
            population,
            args.resume_from,
        )
        league_count = carry_forward_league(
            args.resume_from,
            league_directory,
        )
        print(
            "Warm-started generation "
            f"{resume_state['next_generation']} from {args.resume_from}"
        )
        print(
            "Population seeds: "
            f"{resume_state['warm_network_count']} resumed, "
            f"{resume_state['fresh_network_count']} fresh; "
            f"{league_count} league champions carried forward"
        )

    trainer = create_trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        randomize_training_rounds=True,
        league_directory=league_directory,
    )
    summary = run_training(
        trainer=trainer,
        generations=args.generations,
        tournaments_per_network=30,
        baseline_tournaments_per_network=20,
        benchmark_tournaments_per_generation=150,
        benchmark_candidate_count=14,
        checkpoint_verification_tournaments=500,
        early_stopping_patience=0,
        final_audit_tournaments=1_000,
        output_path=artifact_paths["model_path"],
        metrics_output_path=artifact_paths["metrics_path"],
        initial_best_network=(
            None
            if resume_state is None
            else resume_state["incumbent_network"]
        ),
        initial_best_generation=(
            None
            if resume_state is None
            else resume_state["next_generation"] - 1
        ),
    )

    print()
    print(
        "Best generation:",
        summary["best_generation"],
    )
    print(
        "Best fitness:",
        summary["best_fitness"],
    )
    print(
        "Best strict benchmark:",
        summary["best_benchmark_fitness"],
    )
    print(
        "Best benchmark stages:",
        "normal =",
        summary["best_benchmark_normal_fitness"],
        "mid =",
        summary["best_benchmark_mid_fitness"],
        "late =",
        summary["best_benchmark_late_fitness"],
    )
    print(
        "Best full-tournament holdout:",
        summary["best_benchmark_holdout_fitness"],
    )
    print(
        "Best six-minimum holdout:",
        summary["best_benchmark_minimum_holdout_fitness"],
    )
    print(
        "Best randomized-human holdout:",
        summary["best_benchmark_human_holdout_fitness"],
    )
    print(
        "Best minimum-relative selection score:",
        summary["best_benchmark_selection_fitness"],
    )
    print(
        "Saved network:",
        summary["output_path"],
    )


if __name__ == "__main__":
    main()
