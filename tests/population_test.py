import numpy as np
import pytest

from ml.population import Population
from ml.betting_encoder import BETTING_FEATURE_COUNT


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


def networks_are_equal(first, second):
    return all(
        np.array_equal(
            getattr(first, name),
            getattr(second, name),
        )
        for name in PARAMETER_NAMES
    )


def test_population_creates_requested_networks():
    population = Population(
        population_size=14,
        elite_count=2,
        mutation_rate=0.1,
        mutation_strength=0.1,
        seed=123,
    )

    assert len(population.networks) == 14
    assert len(population.fitness_scores) == 14
    assert population.generation_number == 0
    assert np.all(
        np.asarray(population.fitness_scores) == 0
    )
    assert all(
        network.preferred_action((0.0,) * BETTING_FEATURE_COUNT) != 0
        for network in population.networks
    )


def test_same_seed_creates_same_population():
    first = Population(14, 2, 0.1, 0.1, seed=123)
    second = Population(14, 2, 0.1, 0.1, seed=123)

    for first_network, second_network in zip(
        first.networks,
        second.networks,
    ):
        assert networks_are_equal(
            first_network,
            second_network,
        )


def test_add_and_reset_fitness():
    population = Population(7, 2, 0.1, 0.1, seed=1)

    population.add_fitness(3, 10)
    population.add_fitness(3, 2.5)
    population.add_fitness(0, -4)

    assert population.fitness_scores[3] == 12.5
    assert population.fitness_scores[0] == -4

    population.reset_fitness()

    assert np.all(
        np.asarray(population.fitness_scores) == 0
    )


def test_get_ranked_indices():
    population = Population(7, 2, 0.1, 0.1, seed=1)

    population.fitness_scores = np.array(
        [4, 12, -2, 8, 1, 0, 6],
        dtype=float,
    )

    assert population.get_ranked_indices() == [
        1,
        3,
        6,
        0,
        4,
        5,
        2,
    ]


def test_next_generation_preserves_elites():
    population = Population(
        population_size=7,
        elite_count=2,
        mutation_rate=1.0,
        mutation_strength=1.0,
        seed=123,
    )

    population.fitness_scores = np.array(
        [0, 10, 0, 20, 0, 0, 0],
        dtype=float,
    )

    best = population.networks[3].clone()
    second_best = population.networks[1].clone()

    population.create_next_generation()

    assert networks_are_equal(
        population.networks[0],
        best,
    )
    assert networks_are_equal(
        population.networks[1],
        second_best,
    )


def test_next_generation_can_use_explicit_elites():
    population = Population(
        population_size=7,
        elite_count=2,
        mutation_rate=1.0,
        mutation_strength=1.0,
        seed=123,
    )
    population.fitness_scores[0] = 100
    selected_first = population.networks[5].clone()
    selected_second = population.networks[2].clone()

    population.create_next_generation(elite_indices=[5, 2])

    assert networks_are_equal(population.networks[0], selected_first)
    assert networks_are_equal(population.networks[1], selected_second)


def test_explicit_elites_must_be_unique_and_complete():
    population = Population(7, 2, 0.1, 0.1, seed=123)

    with pytest.raises(ValueError):
        population.create_next_generation(elite_indices=[1])
    with pytest.raises(ValueError):
        population.create_next_generation(elite_indices=[1, 1])


def test_children_are_mutated():
    population = Population(
        population_size=7,
        elite_count=1,
        mutation_rate=1.0,
        mutation_strength=0.5,
        seed=123,
    )

    population.fitness_scores[4] = 100
    parent = population.networks[4].clone()

    population.create_next_generation()

    assert networks_are_equal(
        population.networks[0],
        parent,
    )

    for child in population.networks[1:]:
        assert not networks_are_equal(
            child,
            parent,
        )


def test_children_are_independent_objects():
    population = Population(
        population_size=7,
        elite_count=1,
        mutation_rate=0.0,
        mutation_strength=0.0,
        seed=123,
    )

    population.fitness_scores[2] = 100
    population.create_next_generation()

    original_value = population.networks[0].weights1[
        0,
        0,
    ]

    population.networks[1].weights1[0, 0] += 100

    assert (
        population.networks[0].weights1[0, 0]
        == original_value
    )


def test_next_generation_updates_state():
    population = Population(7, 2, 0.1, 0.1, seed=1)

    population.fitness_scores[0] = 10
    population.create_next_generation()

    assert population.generation_number == 1
    assert len(population.networks) == 7
    assert np.all(
        np.asarray(population.fitness_scores) == 0
    )


@pytest.mark.parametrize(
    "population_size",
    [0, -7, 1, 8],
)
def test_invalid_population_size(population_size):
    with pytest.raises(ValueError):
        Population(
            population_size,
            1,
            0.1,
            0.1,
            seed=1,
        )


@pytest.mark.parametrize(
    "elite_count",
    [0, -1, 8],
)
def test_invalid_elite_count(elite_count):
    with pytest.raises(ValueError):
        Population(
            7,
            elite_count,
            0.1,
            0.1,
            seed=1,
        )


@pytest.mark.parametrize(
    "mutation_rate",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_mutation_rate(mutation_rate):
    with pytest.raises(ValueError):
        Population(
            7,
            1,
            mutation_rate,
            0.1,
            seed=1,
        )


@pytest.mark.parametrize(
    "mutation_strength",
    [-0.1, float("nan"), float("inf")],
)
def test_invalid_mutation_strength(
    mutation_strength,
):
    with pytest.raises(ValueError):
        Population(
            7,
            1,
            0.1,
            mutation_strength,
            seed=1,
        )


@pytest.mark.parametrize(
    "argument,value",
    [
        ("population_size", 7.0),
        ("population_size", True),
        ("elite_count", 1.0),
        ("elite_count", True),
        ("mutation_rate", True),
        ("mutation_strength", True),
        ("seed", 1.5),
        ("seed", True),
    ],
)
def test_invalid_types(argument, value):
    arguments = {
        "population_size": 7,
        "elite_count": 1,
        "mutation_rate": 0.1,
        "mutation_strength": 0.1,
        "seed": 1,
    }

    arguments[argument] = value

    with pytest.raises(TypeError):
        Population(**arguments)
