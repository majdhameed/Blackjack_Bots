import numpy as np

from ml.betting_network import (
    POWER_CHIP_CHOICE_COUNT,
    BettingNetwork,
)
from ml.power_chip_encoder import POWER_CHIP_FEATURE_COUNT


def test_power_chip_network_has_correct_parameter_shapes():
    network = BettingNetwork(seed=123)

    assert network.power_chip_hidden_weights.shape == (
        16,
        POWER_CHIP_FEATURE_COUNT,
    )
    assert network.power_chip_hidden_biases.shape == (16,)
    assert network.power_chip_output_weights.shape == (
        POWER_CHIP_CHOICE_COUNT,
        16,
    )
    assert network.power_chip_output_biases.shape == (
        POWER_CHIP_CHOICE_COUNT,
    )


def test_power_chip_network_returns_one_finite_score_per_choice():
    network = BettingNetwork(seed=123)
    features = np.zeros(POWER_CHIP_FEATURE_COUNT)

    scores = network.power_chip_scores(features)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (POWER_CHIP_CHOICE_COUNT,)
    assert np.all(np.isfinite(scores))


def test_power_chip_network_selects_highest_scoring_choice():
    network = BettingNetwork(seed=123)
    network.power_chip_hidden_weights.fill(0.0)
    network.power_chip_hidden_biases.fill(0.0)
    network.power_chip_output_weights.fill(0.0)
    network.power_chip_output_biases[:] = [0.1, 0.9, 0.2]

    choice = network.preferred_power_chip_action(
        np.zeros(POWER_CHIP_FEATURE_COUNT)
    )

    assert choice == 1


def test_clone_copies_power_chip_parameters_without_sharing_memory():
    original = BettingNetwork(seed=123)
    cloned = original.clone()

    parameter_names = (
        "power_chip_hidden_weights",
        "power_chip_hidden_biases",
        "power_chip_output_weights",
        "power_chip_output_biases",
    )

    for parameter_name in parameter_names:
        original_parameter = getattr(original, parameter_name)
        cloned_parameter = getattr(cloned, parameter_name)

        assert np.array_equal(cloned_parameter, original_parameter)
        assert not np.shares_memory(cloned_parameter, original_parameter)


def test_mutation_changes_power_chip_parameters():
    network = BettingNetwork(seed=123)
    parameter_names = (
        "power_chip_hidden_weights",
        "power_chip_hidden_biases",
        "power_chip_output_weights",
        "power_chip_output_biases",
    )
    original_parameters = {
        parameter_name: getattr(network, parameter_name).copy()
        for parameter_name in parameter_names
    }

    network.mutate(
        mutation_rate=1.0,
        mutation_strength=0.5,
    )

    for parameter_name in parameter_names:
        assert not np.array_equal(
            getattr(network, parameter_name),
            original_parameters[parameter_name],
        )


def test_get_parameters_includes_power_chip_parameters():
    network = BettingNetwork(seed=123)

    parameters = network.get_parameters()

    for parameter_name in (
        "power_chip_hidden_weights",
        "power_chip_hidden_biases",
        "power_chip_output_weights",
        "power_chip_output_biases",
    ):
        assert parameter_name in parameters
        assert np.array_equal(
            parameters[parameter_name],
            getattr(network, parameter_name),
        )
        assert not np.shares_memory(
            parameters[parameter_name],
            getattr(network, parameter_name),
        )


def test_save_and_load_preserve_power_chip_parameters(tmp_path):
    original = BettingNetwork(seed=123)
    checkpoint_path = tmp_path / "network.npz"

    original.save(checkpoint_path)
    loaded = BettingNetwork.load(checkpoint_path)

    for parameter_name in (
        "power_chip_hidden_weights",
        "power_chip_hidden_biases",
        "power_chip_output_weights",
        "power_chip_output_biases",
    ):
        assert np.array_equal(
            getattr(loaded, parameter_name),
            getattr(original, parameter_name),
        )


def test_old_checkpoint_defaults_to_saving_power_chip(tmp_path):
    original = BettingNetwork(seed=123)
    old_parameters = {
        name: value
        for name, value in original.get_parameters().items()
        if not name.startswith("power_chip_")
    }
    checkpoint_path = tmp_path / "old-network.npz"
    np.savez(checkpoint_path, **old_parameters)

    loaded = BettingNetwork.load(checkpoint_path)

    assert np.array_equal(
        loaded.power_chip_output_weights,
        np.zeros_like(loaded.power_chip_output_weights),
    )
    assert loaded.power_chip_output_biases[0] > np.max(
        loaded.power_chip_output_biases[1:]
    )
