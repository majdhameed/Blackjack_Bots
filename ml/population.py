import math
import random

import numpy as np

from ml.betting_network import (
    TRAINABLE_BETTING_ACTION_INDICES,
    BettingNetwork,
)

class Population:
    def __init__(self, population_size, elite_count, mutation_rate, mutation_strength, seed=None):
        if isinstance(population_size, bool) or not isinstance(
            population_size,
            int,
        ):
            raise TypeError(
                "population_size must be an integer"
            )

        if population_size <= 0:
            raise ValueError(
                "population_size must be greater than zero"
            )

        if population_size % 7 != 0:
            raise ValueError(
                "population_size must be a multiple of seven"
            )

        if isinstance(elite_count, bool) or not isinstance(
            elite_count,
            int,
        ):
            raise TypeError(
                "elite_count must be an integer"
            )

        if elite_count <= 0:
            raise ValueError(
                "elite_count must be greater than zero"
            )

        if elite_count > population_size:
            raise ValueError(
                "elite_count cannot exceed population_size"
            )

        if isinstance(mutation_rate, bool) or not isinstance(
            mutation_rate,
            (int, float),
        ):
            raise TypeError(
                "mutation_rate must be a number"
            )

        if not math.isfinite(mutation_rate):
            raise ValueError(
                "mutation_rate must be finite"
            )

        if not 0 <= mutation_rate <= 1:
            raise ValueError(
                "mutation_rate must be between zero and one"
            )

        if isinstance(
            mutation_strength,
            bool,
        ) or not isinstance(
            mutation_strength,
            (int, float),
        ):
            raise TypeError(
                "mutation_strength must be a number"
            )

        if not math.isfinite(mutation_strength):
            raise ValueError(
                "mutation_strength must be finite"
            )

        if mutation_strength < 0:
            raise ValueError(
                "mutation_strength cannot be negative"
            )

        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise TypeError(
                "seed must be an integer or None"
            )
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        self.random_generator = (
            np.random.default_rng(seed)
        )

        self.networks = []

        self.seeded_bet_fractions = (
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.35,
            0.50,
            0.90,
        )

        for network_index in range(population_size):
            self.networks.append(
                self._create_strategy_seeded_network(
                    network_index
                )
            )

        self.fitness_scores = np.zeros(
            population_size,
            dtype=float
        )

        self.generation_number = 0

    def _create_strategy_seeded_network(self, seed_index):
        network_seed = int(
            self.random_generator.integers(
                0,
                2**32,
            )
        )
        network = BettingNetwork(seed=network_seed)

        target_fraction = self.seeded_bet_fractions[
            seed_index % len(self.seeded_bet_fractions)
        ]
        network.biases3[0] = (
            math.log(
                target_fraction
                / (1.0 - target_fraction)
            )
            + self.random_generator.uniform(-0.10, 0.10)
        )
        network.seed_preferred_action(
            TRAINABLE_BETTING_ACTION_INDICES[
                seed_index % len(TRAINABLE_BETTING_ACTION_INDICES)
            ]
        )
        return network

    def reset_fitness(self):
        self.fitness_scores = np.zeros(self.population_size,dtype=float)

    def add_fitness(self, network_index, amount):
        if not isinstance(network_index, int):
            raise TypeError("network index must be integer")

        if network_index >= self.population_size or network_index < 0:
            raise IndexError("network index outside of bounds")

        if not isinstance(amount, (float, int)):
            raise TypeError("amount must be a number")

        if not math.isfinite(amount):
            raise ValueError("amount must not be infinite or NaN")

        self.fitness_scores[network_index] += amount

    def get_ranked_indices(self):

        ranked_indices = np.argsort(self.fitness_scores)[::-1]

        return [int(index) for index in ranked_indices]

    def create_next_generation(self, elite_indices=None):
        if elite_indices is None:
            sorted_indices = self.get_ranked_indices()
            elite_indices = sorted_indices[:self.elite_count]
        else:
            elite_indices = list(elite_indices)
            if len(elite_indices) != self.elite_count:
                raise ValueError(
                    "elite_indices must contain exactly elite_count indices"
                )
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in elite_indices
            ):
                raise TypeError("elite indices must be integers")
            if len(set(elite_indices)) != len(elite_indices):
                raise ValueError("elite indices must be unique")
            if any(
                index < 0 or index >= self.population_size
                for index in elite_indices
            ):
                raise IndexError("elite index outside of bounds")

        next_networks = []

        for elite_index in elite_indices:
            cloned_network = self.networks[elite_index].clone()
            next_networks.append(cloned_network)

        # Keep a small stream of strategy-seeded immigrants so the population
        # cannot permanently collapse into a single passive betting family.
        immigrant_count = min(
            7,
            self.population_size - len(next_networks),
        )
        offspring_target = (
            self.population_size - immigrant_count
        )

        while len(next_networks) < offspring_target:
            parent_index = int(self.random_generator.choice(elite_indices))

            child = self.networks[parent_index].clone()

            child_seed = int(self.random_generator.integers(0, 2**32))
            child.random_generator = (
                np.random.default_rng(child_seed)
            )

            child.mutate(
                self.mutation_rate,
                self.mutation_strength,
            )

            next_networks.append(child)

        for immigrant_index in range(immigrant_count):
            next_networks.append(
                self._create_strategy_seeded_network(
                    self.generation_number
                    * immigrant_count
                    + immigrant_index
                )
            )


        self.networks = next_networks

        self.generation_number += 1

        self.reset_fitness()

