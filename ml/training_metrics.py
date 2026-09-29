import csv
from pathlib import Path

from ml.betting_network import BETTING_ACTION_NAMES

INTEGER_FIELDS = {
    "generation",
    "league_size",
    "total_probe_decisions",
    "unique_policy_count",
    "checkpoint_batches_won",
    "checkpoint_batch_count",
    "benchmark_safe_parent_count",
    "benchmark_balanced_parent_count",
    "benchmark_aggressive_parent_count",
    "self_play_fallback_parent_count",
}

BOOLEAN_FIELDS = {
    "checkpoint_saved",
    "league_promoted",
    "checkpoint_category_gate_passed",
    "checkpoint_bankruptcy_gate_passed",
    "checkpoint_unanimous_batches",
}

def build_generation_record(generation_result, verified_score=None, checkpoint_saved=False):
    strategy_diversity = generation_result["strategy_diversity"]
    parent_styles = generation_result.get("benchmark_parent_styles", [])

    
    record = {
        "generation": generation_result["generation"],
        "training_best": generation_result["population_best_fitness"],
        "training_mean": generation_result["average_fitness"],
        "training_worst": generation_result["population_worst_fitness"],
        "training_std": generation_result["population_fitness_std"],
        "selected_candidate_fitness": generation_result["best_fitness"],
        "benchmark_score": generation_result["benchmark_fitness"],
        "normal_score": generation_result["benchmark_normal_fitness"],
        "mid_score": generation_result["benchmark_mid_fitness"],
        "late_score": generation_result["benchmark_late_fitness"],
        "full_tournament_score": generation_result["benchmark_holdout_fitness"],
        "minimum_control_score": generation_result["benchmark_minimum_holdout_fitness"],
        "randomized_human_score": generation_result.get(
            "benchmark_human_holdout_fitness"
        ),
        "raw_selection_score": generation_result.get(
            "benchmark_raw_selection_fitness"
        ),
        "advantage_score": generation_result.get(
            "benchmark_advantage_fitness"
        ),
        "normal_advantage": generation_result.get(
            "benchmark_normal_advantage"
        ),
        "mid_advantage": generation_result.get(
            "benchmark_mid_advantage"
        ),
        "late_advantage": generation_result.get(
            "benchmark_late_advantage"
        ),
        "full_tournament_advantage": generation_result.get(
            "benchmark_holdout_advantage"
        ),
        "minimum_control_advantage": generation_result.get(
            "benchmark_minimum_holdout_advantage"
        ),
        "randomized_human_advantage": generation_result.get(
            "benchmark_human_holdout_advantage"
        ),
        "robust_score": generation_result["benchmark_selection_fitness"],
        "verified_score": verified_score,
        "verified_top_two_rate": generation_result.get(
            "benchmark_top_two_rate"
        ),
        "verified_first_place_rate": generation_result.get(
            "benchmark_first_place_rate"
        ),
        "verified_average_position": generation_result.get(
            "benchmark_average_position"
        ),
        "verified_bankruptcy_rate": generation_result.get(
            "benchmark_bankruptcy_rate"
        ),
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_batches_won": generation_result.get(
            "checkpoint_batches_won"
        ),
        "checkpoint_batch_count": generation_result.get(
            "checkpoint_batch_count"
        ),
        "checkpoint_improvement": generation_result.get(
            "checkpoint_improvement"
        ),
        "checkpoint_worst_category_delta": generation_result.get(
            "checkpoint_worst_category_delta"
        ),
        "checkpoint_bankruptcy_delta": generation_result.get(
            "checkpoint_bankruptcy_delta"
        ),
        "checkpoint_category_gate_passed": generation_result.get(
            "checkpoint_category_gate_passed"
        ),
        "checkpoint_bankruptcy_gate_passed": generation_result.get(
            "checkpoint_bankruptcy_gate_passed"
        ),
        "checkpoint_unanimous_batches": generation_result.get(
            "checkpoint_unanimous_batches"
        ),
        "league_promoted": generation_result["league_promoted"],
        "league_size": generation_result["league_size"],
        "benchmark_safe_parent_count": parent_styles.count("safe"),
        "benchmark_balanced_parent_count": parent_styles.count("balanced"),
        "benchmark_aggressive_parent_count": parent_styles.count(
            "aggressive"
        ),
        "self_play_fallback_parent_count": parent_styles.count(
            "self_play_fallback"
        ),
        "total_probe_decisions": strategy_diversity["total_decisions"],
        "action_entropy": strategy_diversity["action_entropy"],
        "unique_policy_count": strategy_diversity["unique_policy_count"],
        "unique_policy_rate": strategy_diversity["unique_policy_rate"],
        "mean_distinct_actions": strategy_diversity.get(
            "mean_distinct_actions",
            1.0,
        ),
        "contextual_policy_rate": strategy_diversity.get(
            "contextual_policy_rate",
            0.0,
        ),
    }

    for action in BETTING_ACTION_NAMES:
        name = f"action_{action}"
        value = strategy_diversity["action_percentages"][action]
        record[name] = value

    for probe_slug, probe_result in generation_result.get(
        "contextual_probe_results",
        {},
    ).items():
        prefix = f"probe_{probe_slug}"
        record[f"{prefix}_action_index"] = probe_result[
            "action_index"
        ]
        record[f"{prefix}_legal_bet"] = probe_result["legal_bet"]

    return record

def save_training_history(output_path, records):

    output_path = Path(output_path)
    if len(records) == 0:
        raise ValueError("no records")
    keys = records[0].keys()

    for record in records:
        if record.keys() != keys:

            raise ValueError("Record keys don't match")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = output_path.with_name(f".{output_path.name}.tmp")

    with temp.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)

        writer.writeheader()
        writer.writerows(records)
    
    temp.replace(output_path)



def load_training_history(input_path):
    input_path = Path(input_path)
    records = []
    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            record = {}
            for header, raw_value in row.items():
                raw_value = raw_value.strip()

                if raw_value == "":
                    value = None
                elif (
                    header in INTEGER_FIELDS
                    or header.endswith("_action_index")
                    or header.endswith("_legal_bet")
                ):
                    value = int(raw_value)
                elif header in BOOLEAN_FIELDS:
                    if raw_value not in {"True", "False"}:
                        raise ValueError(
                            f"Invalid boolean value for {header}: "
                            f"{raw_value!r}"
                        )
                    value = raw_value == "True"
                else:
                    value = float(raw_value)

                record[header] = value

            records.append(record)

    return records
