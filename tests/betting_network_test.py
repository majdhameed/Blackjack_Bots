import numpy as np
import pytest

from ml.betting_network import (
    BETTING_ACTION_COUNT,
    BettingNetwork,
)
from ml.betting_encoder import BETTING_FEATURE_COUNT


def make_features():
    return tuple(
        index / BETTING_FEATURE_COUNT
        for index in range(BETTING_FEATURE_COUNT)
    )


def test_network_has_correct_parameter_shapes():
    network = BettingNetwork(seed=1)

    assert network.weights1.shape == (32, BETTING_FEATURE_COUNT)
    assert network.biases1.shape == (32,)

    assert network.weights2.shape == (16, 32)
    assert network.biases2.shape == (16,)

    assert network.weights3.shape == (1, 16)
    assert network.biases3.shape == (1,)
    assert network.action_weights.shape == (
        BETTING_ACTION_COUNT,
        16,
    )
    assert network.action_biases.shape == (
        BETTING_ACTION_COUNT,
    )


def test_strategy_seed_selects_requested_action():
    network = BettingNetwork(seed=1)
    network.seed_preferred_action(3)

    assert network.preferred_action((0.0,) * BETTING_FEATURE_COUNT) == 3


def test_forward_returns_float():
    network = BettingNetwork(seed=1)

    result = network.forward(
        make_features()
    )

    assert isinstance(result, float)


def test_output_is_between_zero_and_one():
    network = BettingNetwork(seed=1)

    result = network.forward(
        make_features()
    )

    assert 0.0 <= result <= 1.0


def test_zero_features_produce_half_with_zero_biases():
    network = BettingNetwork(seed=1)

    features = (0.0,) * BETTING_FEATURE_COUNT

    result = network.forward(features)

    # Every hidden value is zero, so the final
    # pre-sigmoid value is also zero.
    assert result == pytest.approx(0.5)


def test_same_seed_produces_same_output():
    first_network = BettingNetwork(seed=123)
    second_network = BettingNetwork(seed=123)

    features = make_features()

    first_result = first_network.forward(features)
    second_result = second_network.forward(features)

    assert first_result == pytest.approx(
        second_result
    )


def test_same_seed_produces_same_weights():
    first_network = BettingNetwork(seed=123)
    second_network = BettingNetwork(seed=123)

    assert np.array_equal(
        first_network.weights1,
        second_network.weights1,
    )

    assert np.array_equal(
        first_network.weights2,
        second_network.weights2,
    )

    assert np.array_equal(
        first_network.weights3,
        second_network.weights3,
    )


def test_different_seeds_produce_different_outputs():
    first_network = BettingNetwork(seed=1)
    second_network = BettingNetwork(seed=2)

    features = make_features()

    first_result = first_network.forward(features)
    second_result = second_network.forward(features)

    assert first_result != pytest.approx(
        second_result
    )


def test_positive_final_bias_produces_output_above_half():
    network = BettingNetwork(seed=1)

    network.biases3[0] = 2.0

    result = network.forward(
        (0.0,) * BETTING_FEATURE_COUNT
    )

    assert result > 0.5


def test_negative_final_bias_produces_output_below_half():
    network = BettingNetwork(seed=1)

    network.biases3[0] = -2.0

    result = network.forward(
        (0.0,) * BETTING_FEATURE_COUNT
    )

    assert result < 0.5


def test_large_positive_value_does_not_overflow():
    network = BettingNetwork(seed=1)

    network.biases3[0] = 1_000_000

    result = network.forward(
        (0.0,) * BETTING_FEATURE_COUNT
    )

    assert 0.0 <= result <= 1.0
    assert result > 0.99


def test_large_negative_value_does_not_overflow():
    network = BettingNetwork(seed=1)

    network.biases3[0] = -1_000_000

    result = network.forward(
        (0.0,) * BETTING_FEATURE_COUNT
    )

    assert 0.0 <= result <= 1.0
    assert result < 0.01


def test_too_few_features_raises_error():
    network = BettingNetwork(seed=1)

    with pytest.raises(ValueError):
        network.forward(
            (0.0,) * 51
        )


def test_too_many_features_raises_error():
    network = BettingNetwork(seed=1)

    with pytest.raises(ValueError):
        network.forward(
            (0.0,) * 53
        )


def test_nan_feature_raises_error():
    network = BettingNetwork(seed=1)

    features = list(
        (0.0,) * BETTING_FEATURE_COUNT
    )
    features[10] = float("nan")

    with pytest.raises(ValueError):
        network.forward(features)


def test_positive_infinity_raises_error():
    network = BettingNetwork(seed=1)

    features = list(
        (0.0,) * BETTING_FEATURE_COUNT
    )
    features[10] = float("inf")

    with pytest.raises(ValueError):
        network.forward(features)


def test_negative_infinity_raises_error():
    network = BettingNetwork(seed=1)

    features = list(
        (0.0,) * BETTING_FEATURE_COUNT
    )
    features[10] = float("-inf")

    with pytest.raises(ValueError):
        network.forward(features)
