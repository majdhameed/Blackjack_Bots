import numpy as np
import pytest

from ml.betting_encoder import BETTING_FEATURE_COUNT
from ml.betting_network import (
    BETTING_ACTION_NAMES,
    FIVE_PERCENT_BETTING_ACTION_NAMES,
    LEGACY_ACTION_RENAMES,
    LEGACY_BETTING_ACTION_NAMES,
    MACRO_BETTING_ACTION_NAMES,
    BettingNetwork,
)


PARAMETER_NAMES = (
    "weights1",
    "biases1",
    "weights2",
    "biases2",
    "weights3",
    "biases3",
    "action_weights",
    "action_biases",
)


def assert_networks_equal(first, second):
    for parameter_name in PARAMETER_NAMES:
        np.testing.assert_array_equal(
            getattr(first, parameter_name),
            getattr(second, parameter_name),
        )


def test_save_creates_npz_file(tmp_path):
    network = BettingNetwork(seed=123)

    file_path = tmp_path / "network.npz"

    network.save(file_path)

    assert file_path.exists()
    assert file_path.is_file()


def test_save_and_load_preserves_parameters(
    tmp_path,
):
    original = BettingNetwork(seed=123)

    original.mutate(
        mutation_rate=0.5,
        mutation_strength=0.2,
    )

    file_path = tmp_path / "network.npz"

    original.save(file_path)

    loaded = BettingNetwork.load(file_path)

    assert isinstance(loaded, BettingNetwork)
    assert_networks_equal(original, loaded)


def test_loaded_network_produces_same_output(
    tmp_path,
):
    original = BettingNetwork(seed=123)

    features = np.linspace(
        0.0,
        1.0,
        BETTING_FEATURE_COUNT,
    )

    expected_output = original.forward(features)

    file_path = tmp_path / "network.npz"
    original.save(file_path)

    loaded = BettingNetwork.load(file_path)

    actual_output = loaded.forward(features)

    assert actual_output == pytest.approx(
        expected_output
    )


def test_loaded_network_is_independent(
    tmp_path,
):
    original = BettingNetwork(seed=123)

    file_path = tmp_path / "network.npz"
    original.save(file_path)

    loaded = BettingNetwork.load(file_path)

    loaded.weights1[0, 0] += 100

    assert (
        loaded.weights1[0, 0]
        != original.weights1[0, 0]
    )


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        BettingNetwork.load(
            "file_that_does_not_exist.npz"
        )


def test_load_rejects_missing_parameters(
    tmp_path,
):
    file_path = tmp_path / "incomplete.npz"

    np.savez(
        file_path,
        weights1=np.zeros((32, BETTING_FEATURE_COUNT)),
    )

    with pytest.raises(ValueError):
        BettingNetwork.load(file_path)


@pytest.mark.parametrize(
    "parameter_name,wrong_shape",
    [
        ("weights1", (31, BETTING_FEATURE_COUNT)),
        ("biases1", (31,)),
        ("weights2", (15, 32)),
        ("biases2", (15,)),
        ("weights3", (2, 16)),
        ("biases3", (2,)),
        ("action_weights", (9, 16)),
        ("action_biases", (9,)),
    ],
)
def test_load_rejects_wrong_parameter_shapes(
    tmp_path,
    parameter_name,
    wrong_shape,
):
    network = BettingNetwork(seed=123)

    parameters = network.get_parameters()

    parameters[parameter_name] = np.zeros(
        wrong_shape
    )

    file_path = tmp_path / "wrong_shape.npz"

    np.savez(
        file_path,
        **parameters,
    )

    with pytest.raises(ValueError):
        BettingNetwork.load(file_path)


def test_load_rejects_nonfinite_parameters(
    tmp_path,
):
    network = BettingNetwork(seed=123)

    parameters = network.get_parameters()
    parameters["weights1"][0, 0] = np.nan

    file_path = tmp_path / "nonfinite.npz"

    np.savez(
        file_path,
        **parameters,
    )

    with pytest.raises(ValueError):
        BettingNetwork.load(file_path)


def test_load_upgrades_legacy_46_feature_network(
    tmp_path,
):
    network = BettingNetwork(seed=123)
    parameters = network.get_parameters()
    legacy_weights = parameters["weights1"][:, :46]
    parameters["weights1"] = legacy_weights
    parameters.pop("action_weights")
    parameters.pop("action_biases")

    file_path = tmp_path / "legacy_network.npz"
    np.savez(file_path, **parameters)

    loaded = BettingNetwork.load(file_path)

    assert loaded.weights1.shape == (32, BETTING_FEATURE_COUNT)

    np.testing.assert_array_equal(
        loaded.weights1[:, :46],
        legacy_weights,
    )

    np.testing.assert_array_equal(
        loaded.weights1[:, 46:],
        np.zeros((32, BETTING_FEATURE_COUNT - 46)),
    )

    assert loaded.preferred_action((0.0,) * BETTING_FEATURE_COUNT) == 0


def test_load_maps_legacy_action_head_to_percentage_actions(
    tmp_path,
):
    network = BettingNetwork(seed=123)
    parameters = network.get_parameters()
    legacy_weights = np.arange(
        len(LEGACY_BETTING_ACTION_NAMES) * 16,
        dtype=float,
    ).reshape(len(LEGACY_BETTING_ACTION_NAMES), 16)
    legacy_biases = np.arange(
        len(LEGACY_BETTING_ACTION_NAMES),
        dtype=float,
    )
    parameters["action_weights"] = legacy_weights
    parameters["action_biases"] = legacy_biases

    file_path = tmp_path / "legacy_actions.npz"
    np.savez(file_path, **parameters)

    loaded = BettingNetwork.load(file_path)

    expected_source_by_action = {}
    for legacy_index, legacy_name in enumerate(LEGACY_BETTING_ACTION_NAMES):
        current_name = LEGACY_ACTION_RENAMES[legacy_name]
        expected_source_by_action[current_name] = legacy_index

    for current_name, legacy_index in expected_source_by_action.items():
        current_index = BETTING_ACTION_NAMES.index(current_name)
        np.testing.assert_array_equal(
            loaded.action_weights[current_index],
            legacy_weights[legacy_index],
        )
        assert loaded.action_biases[current_index] == pytest.approx(
            legacy_biases[legacy_index]
        )

    for new_action_name in (
        set(BETTING_ACTION_NAMES) - set(expected_source_by_action)
    ):
        new_action_index = BETTING_ACTION_NAMES.index(
            new_action_name
        )
        assert loaded.action_biases[new_action_index] < -100_000


def test_load_maps_macro_action_checkpoint_to_percentage_actions(
    tmp_path,
):
    network = BettingNetwork(seed=123)
    parameters = network.get_parameters()
    old_weights = np.arange(
        len(MACRO_BETTING_ACTION_NAMES) * 16,
        dtype=float,
    ).reshape(len(MACRO_BETTING_ACTION_NAMES), 16)
    old_biases = np.arange(
        len(MACRO_BETTING_ACTION_NAMES),
        dtype=float,
    )
    parameters["action_weights"] = old_weights
    parameters["action_biases"] = old_biases
    file_path = tmp_path / "macro_actions.npz"
    np.savez(file_path, **parameters)

    loaded = BettingNetwork.load(file_path)

    expected_source_by_action = {}
    for old_index, old_name in enumerate(MACRO_BETTING_ACTION_NAMES):
        expected_source_by_action[
            LEGACY_ACTION_RENAMES[old_name]
        ] = old_index

    for action_name, old_index in expected_source_by_action.items():
        action_index = BETTING_ACTION_NAMES.index(action_name)
        np.testing.assert_array_equal(
            loaded.action_weights[action_index],
            old_weights[old_index],
        )
        assert loaded.action_biases[action_index] == pytest.approx(
            old_biases[old_index]
        )


def test_load_maps_five_percent_checkpoint_to_smaller_action_set(
    tmp_path,
):
    network = BettingNetwork(seed=123)
    parameters = network.get_parameters()
    old_weights = np.arange(
        len(FIVE_PERCENT_BETTING_ACTION_NAMES) * 16,
        dtype=float,
    ).reshape(len(FIVE_PERCENT_BETTING_ACTION_NAMES), 16)
    old_biases = np.arange(
        len(FIVE_PERCENT_BETTING_ACTION_NAMES),
        dtype=float,
    )
    parameters["weights1"] = parameters["weights1"][:, :57]
    parameters["action_weights"] = old_weights
    parameters["action_biases"] = old_biases
    file_path = tmp_path / "five_percent_actions.npz"
    np.savez(file_path, **parameters)

    loaded = BettingNetwork.load(file_path)

    assert loaded.weights1.shape == (32, BETTING_FEATURE_COUNT)
    expected_source_by_action = {}
    for old_index, old_name in enumerate(
        FIVE_PERCENT_BETTING_ACTION_NAMES
    ):
        expected_source_by_action[
            LEGACY_ACTION_RENAMES[old_name]
        ] = old_index

    for action_name, old_index in expected_source_by_action.items():
        action_index = BETTING_ACTION_NAMES.index(action_name)
        np.testing.assert_array_equal(
            loaded.action_weights[action_index],
            old_weights[old_index],
        )
        assert loaded.action_biases[action_index] == pytest.approx(
            old_biases[old_index]
        )
