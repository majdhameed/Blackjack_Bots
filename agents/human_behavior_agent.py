import random

from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import (
    active_opponent_bankrolls,
    legal_bet,
    top_two_cutoff,
)


class HumanBehaviorAgent(BasicStrategyAgent):
    """A configurable bettor with tilt, win-pressing, copying, and impulses."""

    def __init__(
        self,
        seed=None,
        base_fraction=0.05,
        maximum_fraction=0.25,
        loss_multiplier=1.5,
        win_multiplier=1.25,
        impulse_probability=0.08,
        copy_probability=0.12,
        protect_probability=0.80,
        tilt_probability=0.20,
        all_in_probability=0.02,
        desperation_probability=0.35,
        mode_persistence=3,
    ):
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if not 0 < base_fraction <= maximum_fraction <= 1:
            raise ValueError(
                "fractions must satisfy 0 < base <= maximum <= 1"
            )
        if loss_multiplier < 1 or win_multiplier < 1:
            raise ValueError("multipliers must be at least one")
        for name, probability in (
            ("impulse_probability", impulse_probability),
            ("copy_probability", copy_probability),
            ("protect_probability", protect_probability),
            ("tilt_probability", tilt_probability),
            ("all_in_probability", all_in_probability),
            ("desperation_probability", desperation_probability),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(
                    f"{name} must be between zero and one"
                )
        if (
            isinstance(mode_persistence, bool)
            or not isinstance(mode_persistence, int)
            or mode_persistence < 1
        ):
            raise ValueError("mode_persistence must be a positive integer")

        self.random_generator = random.Random(seed)
        self.base_fraction = base_fraction
        self.maximum_fraction = maximum_fraction
        self.loss_multiplier = loss_multiplier
        self.win_multiplier = win_multiplier
        self.impulse_probability = impulse_probability
        self.copy_probability = copy_probability
        self.protect_probability = protect_probability
        self.tilt_probability = tilt_probability
        self.all_in_probability = all_in_probability
        self.desperation_probability = desperation_probability
        self.mode_persistence = mode_persistence
        self.current_mode = "cautious"
        self.mode_rounds_remaining = 0

    def _enter_mode(self, mode):
        self.current_mode = mode
        self.mode_rounds_remaining = self.random_generator.randint(
            1,
            self.mode_persistence,
        )
        return mode

    def _select_mode(self, observation, is_leading):
        outside_top_two = (
            observation.bankroll < top_two_cutoff(observation)
        )

        if (
            outside_top_two
            and observation.rounds_remaining <= 3
            and self.random_generator.random()
            < self.desperation_probability
        ):
            return self._enter_mode("desperate")

        if is_leading and self.random_generator.random() < self.protect_probability:
            return self._enter_mode("protecting")

        if (
            observation.consecutive_losses >= 2
            and self.random_generator.random() < self.tilt_probability
        ):
            return self._enter_mode("tilted")

        if self.mode_rounds_remaining > 0:
            self.mode_rounds_remaining -= 1
            return self.current_mode

        if self.random_generator.random() < self.all_in_probability:
            return self._enter_mode("all_in")

        if observation.has_previous_round:
            if observation.previous_result < 0:
                return self._enter_mode("chasing")
            if (
                observation.previous_result > 0
                and self.random_generator.random() < 0.45
            ):
                return self._enter_mode("pressing")

        if self.random_generator.random() < self.impulse_probability:
            return self._enter_mode(
                self.random_generator.choice(("pressing", "tilted"))
            )

        return self._enter_mode("cautious")

    def choose_bet(self, observation):
        opponents = active_opponent_bankrolls(
            observation
        )
        is_leading = (
            not opponents
            or observation.bankroll > max(opponents)
        )
        mode = self._select_mode(observation, is_leading)

        if mode == "protecting":
            requested_bet = observation.minimum_bet
        elif mode == "all_in":
            requested_bet = observation.bankroll
        elif mode == "desperate":
            deficit_fraction = max(
                0.0,
                (
                    top_two_cutoff(observation)
                    - observation.bankroll
                ) / observation.bankroll,
            )
            if (
                observation.rounds_remaining == 0
                and deficit_fraction >= 0.25
            ):
                requested_bet = observation.bankroll * (
                    self.random_generator.choice((0.50, 0.75, 1.00))
                )
            else:
                requested_bet = observation.bankroll * (
                    self.random_generator.uniform(0.10, 0.35)
                )
        elif mode == "tilted":
            requested_bet = observation.bankroll * self.random_generator.uniform(
                max(self.base_fraction, 0.10),
                max(self.maximum_fraction, 0.50),
            )
        else:
            requested_bet = (
                observation.bankroll * self.base_fraction
            )
            if mode == "chasing" and observation.has_previous_round:
                requested_bet = max(
                    requested_bet,
                    observation.previous_bet * self.loss_multiplier,
                )
            elif mode == "pressing":
                previous_bet = max(
                    observation.minimum_bet,
                    observation.previous_bet,
                )
                requested_bet = max(
                    requested_bet,
                    previous_bet * self.win_multiplier,
                )

            if (
                max(observation.current_bets, default=0) > 0
                and self.random_generator.random()
                < self.copy_probability
            ):
                requested_bet = max(
                    requested_bet,
                    max(observation.current_bets),
                )

            requested_bet *= self.random_generator.uniform(
                0.85,
                1.15,
            )
            requested_bet = min(
                requested_bet,
                observation.bankroll
                * self.maximum_fraction,
            )

        return legal_bet(observation, requested_bet)
