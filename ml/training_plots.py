import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from pathlib import Path
from collections import defaultdict


from ml.betting_network import (
    BETTING_ACTION_NAMES,
    FIVE_PERCENT_BETTING_ACTION_NAMES,
    LEGACY_BETTING_ACTION_NAMES,
    MACRO_BETTING_ACTION_NAMES,
)
from ml.betting_probes import CONTEXTUAL_PROBE_LABELS

from ml.training_metrics import load_training_history

FITNESS_VALUES = [
    "training_best",
    "training_mean",
    "training_worst",
]

EVALUATION_VALUES = [
    "benchmark_score",
    "normal_score",
    "mid_score",
    "late_score",
    "full_tournament_score",
    "minimum_control_score",
    "randomized_human_score",
    "robust_score",
    "verified_score",
]

DIVERSITY_VALUES = [
    "action_entropy",
    "unique_policy_rate",
    "contextual_policy_rate",
]

ADVANTAGE_VALUES = [
    "normal_advantage",
    "mid_advantage",
    "late_advantage",
    "full_tournament_advantage",
    "minimum_control_advantage",
    "randomized_human_advantage",
    "robust_score",
]

EVALUATION_TREND_VALUES = [
    value
    for value in EVALUATION_VALUES
    if value != "verified_score"
]


def _rolling_average(values, window_size=10):
    averages = []

    for index in range(len(values)):
        window = values[
            max(0, index - window_size + 1):index + 1
        ]
        available_values = [
            value
            for value in window
            if value is not None
        ]
        averages.append(
            sum(available_values) / len(available_values)
            if available_values
            else None
        )

    return averages


def create_fitness_plot(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lists = defaultdict(list)

    for record in records:
        lists["generation"].append(record["generation"])
        for value in FITNESS_VALUES:
            
            lists[value].append(record[value])
        
        lists["lower"].append(
            record["training_mean"] - record["training_std"]
        )
        lists["upper"].append(
            record["training_mean"] + record["training_std"]
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    for value in FITNESS_VALUES:
        ax.plot(lists["generation"], lists[value], label=value, linewidth=2)

    ax.fill_between(
        lists["generation"],
        lists["lower"],
        lists["upper"],
        alpha=0.2,
        label="mean +/- 1 std",
    )

    ax.set_title("Population Fitness")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def create_evaluation_plot(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lists = defaultdict(list)

    available_evaluation_values = [
        value
        for value in EVALUATION_VALUES
        if value in records[0]
    ]

    for record in records:
        lists["generation"].append(record["generation"])
        for value in available_evaluation_values:
            lists[value].append(record[value])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    for value in EVALUATION_TREND_VALUES:
        if value not in available_evaluation_values:
            continue
        rolling_line, = ax.plot(
            lists["generation"],
            _rolling_average(lists[value]),
            label=f"{value} (10-gen avg)",
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            lists["generation"],
            lists[value],
            color=rolling_line.get_color(),
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
        )

    ax.set_title("Evaluation and Holdout Scores")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Score")
    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.5,
    )

    checkpoint_generations = []
    checkpoint_scores = []

    for record in records:
        if record["checkpoint_saved"]:
            checkpoint_generations.append(record["generation"])
            checkpoint_scores.append(
                record["verified_score"]
                if record["verified_score"] is not None
                else record["robust_score"]
            )

    ax.scatter(
        checkpoint_generations,
        checkpoint_scores,
        color="black",
        marker="o",
        label="saved checkpoint",
        zorder=5,
    )

    verified_generations = []
    verified_scores = []
    for record in records:
        if record["verified_score"] is not None:
            verified_generations.append(record["generation"])
            verified_scores.append(record["verified_score"])

    ax.scatter(
        verified_generations,
        verified_scores,
        color="gray",
        marker="x",
        label="independent verification",
        zorder=5,
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def create_diversity_plot(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lists = defaultdict(list)

    available_diversity_values = [
        value
        for value in DIVERSITY_VALUES
        if value in records[0]
    ]

    for record in records:
        lists["generation"].append(record["generation"])
        for value in available_diversity_values:
            lists[value].append(record[value])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    for value in available_diversity_values:
        ax.plot(lists["generation"], lists[value], label=value, linewidth=2)

    ax.set_ybound(lower=0, upper=1)

    ax.set_title("Population Strategy Diversity")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized Diversity")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def create_action_plot(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )    

    genlist = []

    lists = defaultdict(list)

    available_action_names = []
    for action_name in (
        *BETTING_ACTION_NAMES,
        *FIVE_PERCENT_BETTING_ACTION_NAMES,
        *MACRO_BETTING_ACTION_NAMES,
        *LEGACY_BETTING_ACTION_NAMES,
    ):
        if (
            action_name not in available_action_names
            and f"action_{action_name}" in records[0]
        ):
            available_action_names.append(action_name)

    action_lists = [
        lists[action]
        for action in available_action_names
    ]

    for record in records:
        genlist.append(record["generation"])
        for action in available_action_names:
            lists[action].append(record["action_"+action])


    fig, ax = plt.subplots(figsize=(12, 7))

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )


    ax.stackplot(
        genlist,
        *action_lists,
        labels=available_action_names,
    )

    ax.legend(loc="upper left")


    ax.set_ybound(lower=0, upper=1)

    ax.set_title("Population Betting Actions")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Action Percentage")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def create_advantage_plot(records, output_path):
    """Plot improvement relative to a matched minimum-betting control."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    available_values = [
        value
        for value in ADVANTAGE_VALUES
        if value in records[0]
    ]
    if not available_values:
        raise ValueError("records do not contain minimum-relative metrics")

    generations = [record["generation"] for record in records]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for value in available_values:
        values = [record[value] for record in records]
        rolling_line, = ax.plot(
            generations,
            _rolling_average(values),
            label=f"{value} (10-gen avg)",
            linewidth=2,
        )
        ax.plot(
            generations,
            values,
            color=rolling_line.get_color(),
            alpha=0.18,
            linewidth=0.8,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_title("Advantage Over Matched Minimum Control")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Advancement fitness difference")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_contextual_probe_plot(records, output_path):
    """Plot the selected candidate's response to fixed situations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available_probes = [
        slug
        for slug in CONTEXTUAL_PROBE_LABELS
        if (
            f"probe_{slug}_action_index" in records[0]
            and f"probe_{slug}_legal_bet" in records[0]
        )
    ]
    if not available_probes:
        raise ValueError("records do not contain contextual probe metrics")

    generations = [record["generation"] for record in records]
    fig, (action_ax, bet_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        sharex=True,
    )

    for slug in available_probes:
        label = CONTEXTUAL_PROBE_LABELS[slug]
        action_ax.step(
            generations,
            [
                record[f"probe_{slug}_action_index"]
                for record in records
            ],
            where="mid",
            label=label,
            linewidth=1.6,
        )
        bet_ax.plot(
            generations,
            [record[f"probe_{slug}_legal_bet"] for record in records],
            label=label,
            linewidth=1.6,
        )

    action_ax.set_title("Selected Candidate: Fixed-Situation Actions")
    action_ax.set_ylabel("Chosen action")
    action_ax.set_yticks(range(len(BETTING_ACTION_NAMES)))
    action_ax.set_yticklabels(BETTING_ACTION_NAMES)
    action_ax.set_ylim(-0.5, len(BETTING_ACTION_NAMES) - 0.5)
    action_ax.grid(True, alpha=0.3)
    action_ax.legend(loc="upper left", ncols=2)

    bet_ax.set_title("Selected Candidate: Legal Bets")
    bet_ax.set_xlabel("Generation")
    bet_ax.set_ylabel("Bet (chips)")
    bet_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    bet_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    bet_ax.grid(True, alpha=0.3)
    bet_ax.legend(loc="upper left", ncols=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def generate_training_plots(metrics_path):
    metrics_path = Path(metrics_path)

    records = load_training_history(metrics_path)

    directory = metrics_path.parent

    output_path_dict = {
        "fitness": directory / "fitness.png",
        "evaluation": directory / "evaluation.png",
        "diversity": directory / "diversity.png",
        "actions": directory / "actions.png",
    }

    if any(
        key.startswith("probe_") and key.endswith("_action_index")
        for key in records[0]
    ):
        output_path_dict["contextual_probes"] = (
            directory / "contextual_probes.png"
        )
    if any(
        key.endswith("_advantage")
        for key in records[0]
    ):
        output_path_dict["advantage"] = directory / "advantage.png"

    create_fitness_plot(records, output_path_dict["fitness"])
    create_evaluation_plot(records, output_path_dict["evaluation"])
    create_diversity_plot(records, output_path_dict["diversity"])
    create_action_plot(records, output_path_dict["actions"])
    if "contextual_probes" in output_path_dict:
        create_contextual_probe_plot(
            records,
            output_path_dict["contextual_probes"],
        )
    if "advantage" in output_path_dict:
        create_advantage_plot(records, output_path_dict["advantage"])

    return output_path_dict

