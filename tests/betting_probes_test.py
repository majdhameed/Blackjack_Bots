import importlib
import random

import pytest

from ml.betting_encoder import encode_betting_observation
from tournament.observation import BettingObservation
from ml.betting_network import BettingNetwork


def test_shared_betting_probes_cover_inspection_states():
    try:
        betting_probes = importlib.import_module(
            "ml.betting_probes"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create ml/betting_probes.py for the shared "
            "inspection scenarios."
        )

    create_betting_probes = getattr(
        betting_probes,
        "create_betting_probes",
        None,
    )
    assert callable(create_betting_probes), (
        "ml.betting_probes must expose a callable "
        "create_betting_probes operation"
    )

    probes = create_betting_probes()

    assert len(probes) == 32

    names = [probe["name"] for probe in probes]
    observations = [
        probe["observation"] for probe in probes
    ]

    assert len(set(names)) == len(names), (
        "Every betting probe must have a unique name"
    )
    assert all(
        isinstance(observation, BettingObservation)
        for observation in observations
    )

    round_numbers = {
        observation.round_number
        for observation in observations
    }
    assert 1 in round_numbers
    assert 6 in round_numbers
    assert 11 in round_numbers
    assert 12 in round_numbers

    assert any(
        observation.has_previous_round
        and observation.previous_result < 0
        for observation in observations
    ), "Include at least one previous-loss probe"

    assert any(
        "gap to second" in name.lower()
        for name in names
    ), "Include probes that vary the gap to second place"

    assert sum(
        name.startswith("Loss-history counterfactual")
        for name in names
    ) == 3
    assert sum(
        name.startswith("Gap response")
        for name in names
    ) == 5

    inspector = importlib.import_module(
        "inspect_betting_strategy"
    )
    assert getattr(
        inspector,
        "create_betting_probes",
        None,
    ) is create_betting_probes, (
        "inspect_betting_strategy must import and use the shared "
        "create_betting_probes operation"
    )
    assert not callable(
        getattr(inspector, "create_scenarios", None)
    ), (
        "Remove the old create_scenarios operation from the inspector "
        "instead of duplicating the probes"
    )


def test_contextual_training_probes_have_stable_unique_slugs():
    betting_probes = importlib.import_module("ml.betting_probes")

    probes = betting_probes.create_contextual_training_probes()

    assert [probe["slug"] for probe in probes] == list(
        betting_probes.CONTEXTUAL_PROBE_LABELS
    )
    assert len(probes) == 10
    assert len({probe["slug"] for probe in probes}) == 10
    assert all(
        isinstance(probe["observation"], BettingObservation)
        for probe in probes
    )


def test_contextual_probe_evaluation_records_action_and_legal_bet():
    betting_probes = importlib.import_module("ml.betting_probes")

    class MinimumNetwork:
        def forward(self, features):
            return 0.5

        def preferred_action(self, features):
            return 1

    results = betting_probes.evaluate_contextual_training_probes(
        MinimumNetwork()
    )

    assert list(results) == list(betting_probes.CONTEXTUAL_PROBE_LABELS)
    assert all(
        result == {
            "action_index": 1,
            "action_name": "minimum",
            "legal_bet": 100,
        }
        for result in results.values()
    )


def test_paired_risk_probes_change_only_opponent_risk_features():
    betting_probes = importlib.import_module("ml.betting_probes")
    probes = {
        probe["slug"]: probe["observation"]
        for probe in betting_probes.create_contextual_training_probes()
    }

    for stage in ("early", "late"):
        cautious = encode_betting_observation(
            probes[f"{stage}_gap_cautious"]
        )
        volatile = encode_betting_observation(
            probes[f"{stage}_gap_volatile"]
        )

        assert cautious[:57] == volatile[:57]
        assert cautious[57:] != volatile[57:]


def test_loss_history_counterfactuals_encode_identically():
    betting_probes = importlib.import_module("ml.betting_probes")
    loss_probes = [
        probe["observation"]
        for probe in betting_probes.create_betting_probes()
        if probe["name"].startswith("Loss-history counterfactual")
    ]

    encoded = [encode_betting_observation(probe) for probe in loss_probes]
    assert len(encoded) == 3
    assert encoded[0] == encoded[1] == encoded[2]


def test_gap_response_probes_increase_advancement_deficit():
    betting_probes = importlib.import_module("ml.betting_probes")
    gap_probes = [
        probe["observation"]
        for probe in betting_probes.create_betting_probes()
        if probe["name"].startswith("Gap response")
    ]

    deficits = [
        encode_betting_observation(probe)[48]
        for probe in gap_probes
    ]
    assert deficits == sorted(deficits)
    assert len(set(deficits)) == 5


def test_inspector_samples_reproducible_real_tournament_states(tmp_path):
    inspector = importlib.import_module("inspect_betting_strategy")
    model_path = tmp_path / "network.npz"
    BettingNetwork(seed=11).save(model_path)

    random.seed(1234)
    original_state = random.getstate()
    first = inspector.sample_tournament_decisions(
        model_path,
        number_of_tournaments=3,
        seed=99,
    )
    assert random.getstate() == original_state

    random.seed(5678)
    second = inspector.sample_tournament_decisions(
        model_path,
        number_of_tournaments=3,
        seed=99,
    )

    assert first == second
    assert 3 <= len(first) <= 36
    assert all(
        isinstance(decision["observation"], BettingObservation)
        for decision in first
    )
    assert all(
        decision["legal_bet"] >= decision["observation"].minimum_bet
        for decision in first
    )
