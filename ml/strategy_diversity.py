import math

from ml.betting_probes import create_betting_probes
from ml.betting_encoder import encode_betting_observation
from ml.betting_network import BETTING_ACTION_NAMES

def _calculate_action_entropy(action_percentages):
    entropy = 0

    for action_name, action_percentage in action_percentages.items():
        if action_percentage > 0:
            entropy -= (action_percentage * math.log(action_percentage))

    entropy /= math.log(len(BETTING_ACTION_NAMES))

    return entropy

def measure_strategy_diversity(networks):
    probes = create_betting_probes()

    action_counts = {action:0 for action in BETTING_ACTION_NAMES}
    policy_signatures = []

    for network in networks:
        policy_signature = []

        for probe in probes:
            observation = probe["observation"]
            features = encode_betting_observation(observation)

            action_name = BETTING_ACTION_NAMES[
                network.preferred_action(features)
            ]
            action_counts[action_name] += 1
            policy_signature.append(action_name)

        policy_signatures.append(tuple(policy_signature))

    total_decisions = sum(action_counts.values())
    action_percentages = {action_name:0 for action_name in BETTING_ACTION_NAMES}

    for action_name in BETTING_ACTION_NAMES:
        percentage = action_counts[action_name] / total_decisions
        action_percentages[action_name] = percentage

    action_entropy = _calculate_action_entropy(action_percentages)

    unique_policy_count = len(set(policy_signatures))

    unique_policy_rate = unique_policy_count / len(networks)

    distinct_action_counts = [
        len(set(signature))
        for signature in policy_signatures
    ]
    mean_distinct_actions = (
        sum(distinct_action_counts) / len(distinct_action_counts)
    )
    contextual_policy_rate = sum(
        count > 1
        for count in distinct_action_counts
    ) / len(distinct_action_counts)

    return {
        "total_decisions": total_decisions,
        "action_counts": action_counts,
        "action_percentages": action_percentages,
        "action_entropy": action_entropy,
        "unique_policy_count": unique_policy_count,
        "unique_policy_rate": unique_policy_rate,
        "mean_distinct_actions": mean_distinct_actions,
        "contextual_policy_rate": contextual_policy_rate,
    }
        

        
