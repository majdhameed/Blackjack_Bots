import numpy as np
import pytest

from ml.betting_network import BettingNetwork


PARAMETER_NAMES = (
    "weights1",
    "biases1",
    "weights2",
    "biases2",
    "weights3",
    "biases3",
    "action_weights",
    "action_biases",
    "power_chip_hidden_weights",
    "power_chip_hidden_biases",
    "power_chip_output_weights",
    "power_chip_output_biases",
)


def assert_networks_equal(first, second):
    for name in PARAMETER_NAMES:
        first_parameter = getattr(first, name)
        second_parameter = getattr(second, name)

        np.testing.assert_array_equal(
            first_parameter,
            second_parameter,
        )


def test_clone_has_identical_parameters():
    original = BettingNetwork(seed=123)
    cloned = original.clone()

    assert_networks_equal(original, cloned)


def test_clone_parameters_are_independent():
    original = BettingNetwork(seed=123)
    cloned = original.clone()

    cloned.weights1[0, 0] += 100
    cloned.biases2[0] += 100

    assert (
        original.weights1[0, 0]
        != cloned.weights1[0, 0]
    )

    assert (
        original.biases2[0]
        != cloned.biases2[0]
    )


def test_clone_is_a_different_object():
    original = BettingNetwork(seed=123)
    cloned = original.clone()

    assert cloned is not original


def test_zero_mutation_rate_changes_nothing():
    network = BettingNetwork(seed=123)
    original = network.clone()

    network.mutate(
        mutation_rate=0.0,
        mutation_strength=1.0,
    )

    assert_networks_equal(network, original)


def test_zero_mutation_strength_changes_nothing():
    network = BettingNetwork(seed=123)
    original = network.clone()

    network.mutate(
        mutation_rate=1.0,
        mutation_strength=0.0,
    )

    assert_networks_equal(network, original)


def test_full_mutation_changes_parameters():
    network = BettingNetwork(seed=123)
    original = network.clone()

    network.mutate(
        mutation_rate=1.0,
        mutation_strength=0.1,
    )

    changed_parameters = []

    for name in PARAMETER_NAMES:
        original_parameter = getattr(
            original,
            name,
        )

        mutated_parameter = getattr(
            network,
            name,
        )

        changed_parameters.append(
            not np.array_equal(
                original_parameter,
                mutated_parameter,
            )
        )

    assert all(changed_parameters)


def test_action_head_uses_higher_mutation_rate_than_shared_layers():
    network = BettingNetwork(seed=123)
    original = network.clone()

    network.mutate(mutation_rate=0.34, mutation_strength=0.1)

    assert np.all(network.action_biases != original.action_biases)
    assert np.any(network.weights1 == original.weights1)


@pytest.mark.parametrize(
    "mutation_rate",
    [-0.01, 1.01, -1, 2],
)
def test_invalid_mutation_rate_raises(
    mutation_rate,
):
    network = BettingNetwork(seed=123)

    with pytest.raises(ValueError):
        network.mutate(
            mutation_rate=mutation_rate,
            mutation_strength=0.1,
        )


def test_negative_mutation_strength_raises():
    network = BettingNetwork(seed=123)

    with pytest.raises(ValueError):
        network.mutate(
            mutation_rate=0.1,
            mutation_strength=-0.1,
        )


def test_get_parameters_returns_all_parameters():
    network = BettingNetwork(seed=123)

    parameters = network.get_parameters()

    assert set(parameters) == set(PARAMETER_NAMES)


def test_get_parameters_returns_copies():
    network = BettingNetwork(seed=123)

    parameters = network.get_parameters()

    original_value = network.weights1[0, 0]

    parameters["weights1"][0, 0] += 100

    assert network.weights1[0, 0] == original_value


def test_clones_begin_with_same_random_state():
    original = BettingNetwork(seed=123)
    cloned = original.clone()

    original.mutate(
        mutation_rate=0.5,
        mutation_strength=0.1,
    )

    cloned.mutate(
        mutation_rate=0.5,
        mutation_strength=0.1,
    )

    assert_networks_equal(original, cloned)
