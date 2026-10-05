from tournament.observation import PowerChipObservation
from tournament.power_chips import PowerChipAction


def _get_chooser(policy):
    chooser = getattr(policy, "choose_power_chip", None)
    if not callable(chooser):
        raise TypeError("Policy must define choose_power_chip")
    return chooser


class BustSavingPowerChipPolicy:
    def choose_power_chip(self, observation):
        if not isinstance(observation, PowerChipObservation):
            raise TypeError("Wrong observation type")

        if observation.action != PowerChipAction.REHIT:
            return None

        if observation.hand_total <= 21:
            return None

        if len(observation.legal_targets) < 1:
            return None

        return observation.legal_targets[0]


class StiffHandReplacementPowerChipPolicy:
    def choose_power_chip(self, observation):
        if not isinstance(observation, PowerChipObservation):
            raise TypeError("Wrong observation type")

        if observation.action != PowerChipAction.REPLACE:
            return None

        if len(observation.legal_targets) < 1:
            return None

        if observation.hand_is_soft:
            return None

        if observation.hand_total > 16 or observation.hand_total < 12:
            return None

        if observation.dealer_upcard_value < 7:
            return None

        best_target = observation.legal_targets[0]

        for target in observation.legal_targets:
            value = observation.hand_card_values[target]

            if value < observation.hand_card_values[best_target]:
                best_target = target

        return best_target


class CompositePowerChipPolicy:
    def __init__(self, policies):
        self.policies = tuple(policies)
        for policy in self.policies:
            _get_chooser(policy)

    def choose_power_chip(self, observation):
        if not isinstance(observation, PowerChipObservation):
            raise TypeError("Wrong observation type")

        for policy in self.policies:
            chooser = _get_chooser(policy)
            target = chooser(observation)

            if target is not None:
                return target

        return None


class HighStakesPowerChipPolicy:
    def __init__(self, policy, minimum_bet_fraction):
        if minimum_bet_fraction < 0 or minimum_bet_fraction > 1:
            raise ValueError("Fraction must be between 0 and 1")

        _get_chooser(policy)
        self.policy = policy
        self.minimum_bet_fraction = minimum_bet_fraction

    def choose_power_chip(self, observation):
        if not isinstance(observation, PowerChipObservation):
            raise TypeError("Wrong observation type")

        chooser = _get_chooser(self.policy)

        stack_before_wager = observation.bankroll + observation.current_bet

        if stack_before_wager < 1:
            return None

        bet_fraction = observation.current_bet / stack_before_wager

        if bet_fraction < self.minimum_bet_fraction:
            return None

        return chooser(observation)


class LateRoundPowerChipPolicy:
    def __init__(self, policy, maximum_rounds_remaining):
        if (
            isinstance(maximum_rounds_remaining, bool)
            or not isinstance(maximum_rounds_remaining, int)
        ):
            raise TypeError("maximum_rounds_remaining must be an integer")
        if maximum_rounds_remaining < 0:
            raise ValueError("maximum_rounds_remaining cannot be negative")

        _get_chooser(policy)
        self.policy = policy
        self.max_rounds_remaining = maximum_rounds_remaining

    def choose_power_chip(self, observation):
        if not isinstance(observation, PowerChipObservation):
            raise TypeError("Wrong observation type")

        chooser = _get_chooser(self.policy)

        if observation.rounds_remaining > self.max_rounds_remaining:
            return None

        return chooser(observation)
