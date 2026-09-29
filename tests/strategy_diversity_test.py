import importlib

import pytest

from ml.betting_network import BETTING_ACTION_NAMES
from ml.betting_encoder import BETTING_FEATURE_COUNT
from ml.betting_probes import create_betting_probes


class FixedActionNetwork:
    def __init__(self, action_index):
        self.action_index = action_index
        self.probe_count = 0

    def preferred_action(self, features):
        assert len(features) == BETTING_FEATURE_COUNT
        self.probe_count += 1
        return self.action_index


class RoundSensitiveNetwork:
    def preferred_action(self, features):
        return 1 if features[0] < 0.5 else 6


def test_strategy_diversity_distinguishes_collapse_from_variety():
    probe_count = len(create_betting_probes())
    try:
        strategy_diversity = importlib.import_module(
            "ml.strategy_diversity"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create ml/strategy_diversity.py for population "
            "behavior analysis."
        )

    measure_strategy_diversity = getattr(
        strategy_diversity,
        "measure_strategy_diversity",
        None,
    )
    assert callable(measure_strategy_diversity), (
        "ml.strategy_diversity must expose a callable "
        "measure_strategy_diversity operation"
    )

    collapsed_networks = [
        FixedActionNetwork(3)
        for _ in range(4)
    ]
    collapsed = measure_strategy_diversity(
        collapsed_networks
    )

    assert collapsed["total_decisions"] == 4 * probe_count
    collapsed_action = BETTING_ACTION_NAMES[3]
    assert (
        collapsed["action_counts"][collapsed_action]
        == 4 * probe_count
    )
    assert collapsed["action_percentages"][collapsed_action] == pytest.approx(1.0)
    assert collapsed["action_entropy"] == pytest.approx(0.0)
    assert collapsed["unique_policy_count"] == 1
    assert collapsed["unique_policy_rate"] == pytest.approx(0.25)
    assert collapsed["mean_distinct_actions"] == pytest.approx(1.0)
    assert collapsed["contextual_policy_rate"] == pytest.approx(0.0)
    assert all(
        network.probe_count == probe_count
        for network in collapsed_networks
    )

    diverse_networks = [
        FixedActionNetwork(action_index)
        for action_index in range(len(BETTING_ACTION_NAMES))
    ]
    diverse = measure_strategy_diversity(diverse_networks)

    assert diverse["total_decisions"] == (
        len(BETTING_ACTION_NAMES) * probe_count
    )
    assert set(diverse["action_counts"]) == set(
        BETTING_ACTION_NAMES
    )
    assert all(
        count == probe_count
        for count in diverse["action_counts"].values()
    )
    assert sum(
        diverse["action_percentages"].values()
    ) == pytest.approx(1.0)
    assert diverse["action_entropy"] == pytest.approx(1.0)
    assert diverse["unique_policy_count"] == len(
        BETTING_ACTION_NAMES
    )
    assert diverse["unique_policy_rate"] == pytest.approx(1.0)
    assert diverse["mean_distinct_actions"] == pytest.approx(1.0)
    assert diverse["contextual_policy_rate"] == pytest.approx(0.0)

    contextual = measure_strategy_diversity(
        [RoundSensitiveNetwork(), FixedActionNetwork(1)]
    )
    assert contextual["mean_distinct_actions"] > 1.0
    assert contextual["contextual_policy_rate"] == pytest.approx(0.5)
