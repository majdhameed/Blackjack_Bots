import copy
from pathlib import Path

import numpy as np

from ml.betting_encoder import BETTING_FEATURE_COUNT


LEGACY_BETTING_ACTION_NAMES = (
    "legacy_fraction",
    "minimum",
    "five_percent",
    "controlled_recovery",
    "take_second",
    "take_first",
    "cover_visible_bets",
    "half_bankroll",
    "all_in",
    "protect_lead",
)

MACRO_BETTING_ACTION_NAMES = (
    "legacy_fraction",
    "minimum",
    "five_percent",
    "controlled_recovery",
    "quarter_gap_to_second",
    "half_gap_to_second",
    "full_gap_to_second",
    "quarter_gap_to_first",
    "half_gap_to_first",
    "full_gap_to_first",
    "cover_visible_bets",
    "half_bankroll",
    "all_in",
    "protect_lead",
)

FIVE_PERCENT_BETTING_ACTION_NAMES = (
    "legacy_fraction",
    "minimum",
    *tuple(
        f"bet_{percentage:02d}_percent"
        for percentage in range(5, 101, 5)
    ),
)

BETTING_PERCENTAGES = (
    2,
    3,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    75,
    100,
)

PERCENTAGE_BET_ACTION_NAMES = tuple(
    f"bet_{percentage:02d}_percent"
    for percentage in BETTING_PERCENTAGES
)

BETTING_ACTION_NAMES = (
    "legacy_fraction",
    "minimum",
    *PERCENTAGE_BET_ACTION_NAMES,
)
BETTING_ACTION_COUNT = len(BETTING_ACTION_NAMES)
TRAINABLE_BETTING_ACTION_INDICES = tuple(
    range(1, BETTING_ACTION_COUNT)
)

LEGACY_ACTION_RENAMES = {
    "legacy_fraction": "legacy_fraction",
    "minimum": "minimum",
    "five_percent": "bet_05_percent",
    "controlled_recovery": "bet_10_percent",
    "take_second": "bet_100_percent",
    "take_first": "bet_100_percent",
    "quarter_gap_to_second": "bet_20_percent",
    "half_gap_to_second": "bet_50_percent",
    "full_gap_to_second": "bet_100_percent",
    "quarter_gap_to_first": "bet_20_percent",
    "half_gap_to_first": "bet_50_percent",
    "full_gap_to_first": "bet_100_percent",
    "cover_visible_bets": "bet_30_percent",
    "half_bankroll": "bet_50_percent",
    "all_in": "bet_100_percent",
    "protect_lead": "minimum",
}

for old_percentage in range(5, 101, 5):
    old_name = f"bet_{old_percentage:02d}_percent"
    closest_percentage = min(
        BETTING_PERCENTAGES,
        key=lambda percentage: (
            abs(percentage - old_percentage),
            percentage,
        ),
    )
    LEGACY_ACTION_RENAMES[old_name] = (
        f"bet_{closest_percentage:02d}_percent"
    )


class BettingNetwork:
    def __init__(self, seed=None):
        if (
            seed is not None
            and not isinstance(seed, int)
        ):
            raise TypeError(
                "seed must be an integer or None"
            )

        self.random_generator = (
            np.random.default_rng(seed)
        )

        self.weights1 = (
            self.random_generator.normal(
                loc=0.0,
                scale=0.1,
                size=(32, BETTING_FEATURE_COUNT),
            )
        )

        self.biases1 = np.zeros(32)

        self.weights2 = (
            self.random_generator.normal(
                loc=0.0,
                scale=0.1,
                size=(16, 32),
            )
        )

        self.biases2 = np.zeros(16)

        self.weights3 = (
            self.random_generator.normal(
                loc=0.0,
                scale=0.1,
                size=(1,16)
            )
        )

        self.biases3 = np.zeros(1)

        # A second head chooses among strategically meaningful betting
        # actions.  The original fractional output is retained both as one
        # available action and for loading networks created by older runs.
        self.action_weights = (
            self.random_generator.normal(
                loc=0.0,
                scale=0.1,
                size=(BETTING_ACTION_COUNT, 16),
            )
        )
        self.action_biases = np.zeros(
            BETTING_ACTION_COUNT
        )
        self.action_biases[0] = -1_000_000.0

    def _hidden_values(self, features):
        if len(features) != BETTING_FEATURE_COUNT:
            raise ValueError(
                "features must have a length of "
                f"{BETTING_FEATURE_COUNT}"
            )

        np_features = np.asarray(features, dtype=float)

        if not np.all(np.isfinite(np_features)):
            raise ValueError(
                "features must contain finite numbers"
            )

        hidden1 = self.weights1 @ np_features + self.biases1
        hidden1 = np.maximum(0, hidden1)

        hidden2 = self.weights2 @ hidden1 + self.biases2
        return np.maximum(0, hidden2)

    def forward(self, features):
        hidden2 = self._hidden_values(features)

        output = self.weights3 @ hidden2 + self.biases3

        output = np.clip(output, -60, 60)

        output = 1 / (1 + np.exp(-output))

        return float(output.item())

    def action_scores(self, features):
        """Return one score for each high-level betting action."""
        hidden2 = self._hidden_values(features)
        scores = (
            self.action_weights @ hidden2
            + self.action_biases
        )
        return np.asarray(scores, dtype=float)

    def preferred_action(self, features):
        scores = self.action_scores(features)
        return int(np.argmax(scores))

    def seed_preferred_action(self, action_index):
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
        ):
            raise TypeError("action_index must be an integer")
        if not 0 <= action_index < BETTING_ACTION_COUNT:
            raise ValueError("action_index is outside the action set")

        # Cover every action without locking a policy into one global choice.
        self.action_biases.fill(-0.05)
        if action_index != 0:
            self.action_biases[0] = -1_000_000.0
        self.action_biases[action_index] = 0.05

    def clone(self):

        cloned_net = BettingNetwork()

        cloned_net.random_generator.bit_generator.state = (
            copy.deepcopy(
                self.random_generator.bit_generator.state
            )
        )

        cloned_net.weights1 = np.copy(self.weights1)
        cloned_net.biases1 = np.copy(self.biases1)
        
        cloned_net.weights2 = np.copy(self.weights2)
        cloned_net.biases2 = np.copy(self.biases2)
        
        cloned_net.weights3 = np.copy(self.weights3)
        cloned_net.biases3 = np.copy(self.biases3)
        cloned_net.action_weights = np.copy(
            self.action_weights
        )
        cloned_net.action_biases = np.copy(
            self.action_biases
        )
        
        cloned_net.random_generator = np.random.default_rng()
        cloned_net.random_generator.bit_generator.state = self.random_generator.bit_generator.state
        
        return cloned_net

    def mutate(self, mutation_rate, mutation_strength):

        if not (0 <= mutation_rate <= 1):
            raise ValueError("mutation rate must be between 0 and 1")

        if mutation_strength < 0:
            raise ValueError("mutation strength must be a postive number")

        parameter_names = (
            "weights1",
            "biases1",
            "weights2",
            "biases2",
            "weights3",
            "biases3",
            "action_weights",
            "action_biases",
        )

        for parameter_name in parameter_names:
            parameter = getattr(self, parameter_name)

            parameter_mutation_rate = mutation_rate
            parameter_mutation_strength = mutation_strength
            if parameter_name in {"action_weights", "action_biases"}:
                parameter_mutation_rate = min(1.0, mutation_rate * 3.0)
                parameter_mutation_strength = mutation_strength * 1.5

            mutation_mask = (
                self.random_generator.random(
                     size=parameter.shape
                )
                < parameter_mutation_rate
            )

            mutation_noise = (
                self.random_generator.normal(
                    loc=0.0,
                    scale=parameter_mutation_strength,
                    size=parameter.shape,
                )
            )

            parameter += mutation_mask * mutation_noise
    def get_parameters(self):
        return {
            "weights1": self.weights1.copy(),
            "biases1": self.biases1.copy(),
            "weights2": self.weights2.copy(),
            "biases2": self.biases2.copy(),
            "weights3": self.weights3.copy(),
            "biases3": self.biases3.copy(),
            "action_weights": self.action_weights.copy(),
            "action_biases": self.action_biases.copy(),
        }
    
    def save(self, file_path):
        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = file_path.with_name(
            f".{file_path.name}.tmp.npz"
        )

        np.savez(
            temporary_path,
            **self.get_parameters(),
        )
        temporary_path.replace(file_path)


    @classmethod
    def load(cls, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Network file does not exist: {file_path}"
            )

        expected_shapes = {
            "weights1": (
                32,
                BETTING_FEATURE_COUNT,
            ),
            "biases1": (32,),
            "weights2": (16, 32),
            "biases2": (16,),
            "weights3": (1, 16),
            "biases3": (1,),
            "action_weights": (
                BETTING_ACTION_COUNT,
                16,
            ),
            "action_biases": (BETTING_ACTION_COUNT,),
        }

        legacy_optional_parameters = {
            "action_weights",
            "action_biases",
        }

        with np.load(
            file_path,
            allow_pickle=False,
        ) as data:
            has_action_weights = "action_weights" in data.files
            has_action_biases = "action_biases" in data.files
            if has_action_weights != has_action_biases:
                raise ValueError(
                    "Network file must contain both action_weights "
                    "and action_biases"
                )
            missing_parameters = (
                set(expected_shapes)
                - legacy_optional_parameters
                - set(data.files)
            )

            if missing_parameters:
                raise ValueError(
                    "Network file is missing parameters: "
                    f"{sorted(missing_parameters)}"
                )

            loaded_parameters = {}

            for parameter_name, expected_shape in (
                expected_shapes.items()
            ):
                if parameter_name not in data.files:
                    continue
                parameter = np.asarray(
                    data[parameter_name],
                    dtype=float,
                )

                historical_action_names = None
                for action_names in (
                    LEGACY_BETTING_ACTION_NAMES,
                    MACRO_BETTING_ACTION_NAMES,
                    FIVE_PERCENT_BETTING_ACTION_NAMES,
                ):
                    historical_shape = (
                        (len(action_names), 16)
                        if parameter_name == "action_weights"
                        else (len(action_names),)
                    )
                    if (
                        parameter_name in {"action_weights", "action_biases"}
                        and parameter.shape == historical_shape
                    ):
                        historical_action_names = action_names
                        break

                if historical_action_names is not None:
                    fill_value = (
                        0.0
                        if parameter_name == "action_weights"
                        else -1_000_000.0
                    )
                    expanded_parameter = np.full(
                        expected_shape,
                        fill_value,
                        dtype=float,
                    )
                    for old_index, old_name in enumerate(
                        historical_action_names
                    ):
                        current_name = LEGACY_ACTION_RENAMES[old_name]
                        current_index = BETTING_ACTION_NAMES.index(current_name)
                        expanded_parameter[current_index] = parameter[old_index]
                    parameter = expanded_parameter

                if (
                    parameter_name == "weights1"
                    and parameter.ndim == 2
                    and parameter.shape[0] == 32
                    and parameter.shape[1] in {46, 52, 57}
                ):
                    parameter = np.pad(
                        parameter,
                        (
                            (0, 0),
                            (
                                0,
                                BETTING_FEATURE_COUNT
                                - parameter.shape[1],
                            ),
                        ),
                    )

                if parameter.shape != expected_shape:
                    raise ValueError(
                        f"{parameter_name} must have shape "
                        f"{expected_shape}, but has shape "
                        f"{parameter.shape}"
                    )

                if not np.all(np.isfinite(parameter)):
                    raise ValueError(
                        f"{parameter_name} contains "
                        "non-finite values"
                    )

                loaded_parameters[
                    parameter_name
                ] = parameter.copy()

        network = cls()

        # Old one-output checkpoints keep their exact original behavior by
        # selecting the legacy fraction action until evolution changes it.
        if "action_weights" not in loaded_parameters:
            network.action_weights = np.zeros(
                (BETTING_ACTION_COUNT, 16),
            )
            network.action_biases = np.full(
                BETTING_ACTION_COUNT,
                -10.0,
            )
            network.action_biases[0] = 10.0

        network.weights1 = loaded_parameters[
            "weights1"
        ]

        network.biases1 = loaded_parameters[
            "biases1"
        ]

        network.weights2 = loaded_parameters[
            "weights2"
        ]

        network.biases2 = loaded_parameters[
            "biases2"
        ]

        network.weights3 = loaded_parameters[
            "weights3"
        ]

        network.biases3 = loaded_parameters[
            "biases3"
        ]

        if "action_weights" in loaded_parameters:
            network.action_weights = loaded_parameters[
                "action_weights"
            ]
            network.action_biases = loaded_parameters[
                "action_biases"
            ]

        return network
