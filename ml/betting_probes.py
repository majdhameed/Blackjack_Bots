from tournament.observation import BettingObservation


TOTAL_ROUNDS = 12
MINIMUM_BET = 100

CONTEXTUAL_PROBE_LABELS = {
    "early_tied": "Early: tied",
    "early_behind_second": "Early: just behind 2nd",
    "mid_one_bet_behind": "Mid: one bet behind 2nd",
    "mid_far_behind": "Mid: far behind 2nd",
    "late_behind_second": "Late: just behind 2nd",
    "late_safe_lead": "Late: safe lead",
    "early_gap_cautious": "Early: 5% behind cautious table",
    "early_gap_volatile": "Early: 5% behind volatile table",
    "late_gap_cautious": "Late: 5% behind cautious table",
    "late_gap_volatile": "Late: 5% behind volatile table",
}


def make_observation(
    round_number,
    bankrolls,
    player_index,
    betting_position,
    current_bets,
    bets_placed,
    true_count=0.0,
    previous_bet=0.0,
    previous_bankroll_change=0.0,
    previous_result=0.0,
    consecutive_losses=0,
    has_previous_round=False,
    largest_opponent_previous_bet_fraction=0.0,
    average_opponent_previous_bet_fraction=0.0,
    opponents_over_ten_percent=0,
    opponents_over_twenty_five_percent=0,
    opponent_bet_volatility=0.0,
    opponents_with_loss_streak=0,
):
    return BettingObservation(
        round_number=round_number,
        total_rounds=TOTAL_ROUNDS,
        rounds_remaining=(
            TOTAL_ROUNDS - round_number
        ),
        player_index=player_index,
        round_player_index=player_index,
        betting_position=betting_position,
        minimum_bet=MINIMUM_BET,
        bankroll=bankrolls[player_index],
        bankrolls=tuple(bankrolls),
        active_players=(
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ),
        current_bets=tuple(current_bets),
        bets_placed=tuple(bets_placed),
        betting_order=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ),
        card_value_counts=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        cards_seen=0,
        running_count=0,
        true_count=true_count,
        cards_remaining=312,
        decks_remaining=6.0,
        shoe_penetration=0.0,
        previous_bet=previous_bet,
        previous_bankroll_change=(
            previous_bankroll_change
        ),
        previous_result=previous_result,
        consecutive_losses=consecutive_losses,
        has_previous_round=has_previous_round,
        largest_opponent_previous_bet_fraction=(
            largest_opponent_previous_bet_fraction
        ),
        average_opponent_previous_bet_fraction=(
            average_opponent_previous_bet_fraction
        ),
        opponents_over_ten_percent=opponents_over_ten_percent,
        opponents_over_twenty_five_percent=(
            opponents_over_twenty_five_percent
        ),
        opponent_bet_volatility=opponent_bet_volatility,
        opponents_with_loss_streak=opponents_with_loss_streak,
    )

def create_betting_probes():
    tied_bankrolls = [
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
    ]

    leading_bankrolls = [
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        12_000,
    ]

    behind_bankrolls = [
        11_000,
        10_500,
        10_000,
        10_000,
        10_000,
        10_000,
        8_000,
    ]

    slightly_behind_bankrolls = [
        10_500,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
    ]

    scenarios = [
        {
            "name": "Round 1, tied, betting first",
            "observation": make_observation(
                round_number=1,
                bankrolls=tied_bankrolls,
                player_index=0,
                betting_position=0,
                current_bets=[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                bets_placed=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ),
        },
        {
            "name": "Round 1, tied, betting last",
            "observation": make_observation(
                round_number=1,
                bankrolls=tied_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 4, three leaders established",
            "observation": make_observation(
                round_number=4,
                bankrolls=[
                    16_000,
                    14_500,
                    13_000,
                    10_000,
                    9_500,
                    9_200,
                    9_000,
                ],
                player_index=6,
                betting_position=6,
                current_bets=[
                    500,
                    300,
                    200,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 6, leading, betting first",
            "observation": make_observation(
                round_number=6,
                bankrolls=[
                    12_000,
                    10_000,
                    10_000,
                    10_000,
                    10_000,
                    10_000,
                    10_000,
                ],
                player_index=0,
                betting_position=0,
                current_bets=[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                bets_placed=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ),
        },
        {
            "name": "Round 6, behind, betting last",
            "observation": make_observation(
                round_number=6,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 11, leading, betting last",
            "observation": make_observation(
                round_number=11,
                bankrolls=leading_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 11, far behind, betting last",
            "observation": make_observation(
                round_number=11,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Final round, tied, betting first",
            "observation": make_observation(
                round_number=12,
                bankrolls=tied_bankrolls,
                player_index=0,
                betting_position=0,
                current_bets=[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                bets_placed=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ),
        },
        {
            "name": "Final round, tied, betting last",
            "observation": make_observation(
                round_number=12,
                bankrolls=tied_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": (
                "Final round, slightly behind, "
                "betting last"
            ),
            "observation": make_observation(
                round_number=12,
                bankrolls=(
                    slightly_behind_bankrolls
                ),
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": (
                "Final round, far behind, "
                "betting last"
            ),
            "observation": make_observation(
                round_number=12,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": (
                "Final round, behind, opponents "
                "bet large"
            ),
            "observation": make_observation(
                round_number=12,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    2_000,
                    1_500,
                    1_000,
                    500,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Final round, leading, betting last",
            "observation": make_observation(
                round_number=12,
                bankrolls=leading_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
    ]

    scenarios.append(
        {
            "name": "Round 4, narrow 5% lead",
            "observation": make_observation(
                round_number=4,
                bankrolls=[
                    10_500,
                    10_000,
                    9_900,
                    9_800,
                    9_700,
                    9_600,
                    9_500,
                ],
                player_index=0,
                betting_position=0,
                current_bets=[0] * 7,
                bets_placed=[False] * 7,
            ),
        }
    )

    gap_checks = (
        (6, 5),
        (6, 10),
        (6, 20),
        (11, 3),
        (11, 5),
        (11, 8),
        (12, 3),
        (12, 8),
    )

    for round_number, gap_percent in gap_checks:
        focal_bankroll = 10_000
        second_bankroll = round(
            focal_bankroll
            * (1 + gap_percent / 100)
        )
        leader_bankroll = max(
            second_bankroll + 500,
            11_500,
        )
        scenarios.append(
            {
                "name": (
                    f"Round {round_number}, 3rd, "
                    f"{gap_percent}% gap to second"
                ),
                "observation": make_observation(
                    round_number=round_number,
                    bankrolls=[
                        leader_bankroll,
                        second_bankroll,
                        9_800,
                        9_600,
                        9_400,
                        9_200,
                        focal_bankroll,
                    ],
                    player_index=6,
                    betting_position=6,
                    current_bets=[
                        100,
                        100,
                        100,
                        100,
                        100,
                        100,
                        0,
                    ],
                    bets_placed=[
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                    ],
                ),
            }
        )

    for loss_count, previous_bet in (
        (1, 500),
        (2, 1_000),
    ):
        scenarios.append(
            {
                "name": (
                    f"Round 6, after {loss_count} "
                    f"loss{'es' if loss_count > 1 else ''}"
                ),
                "observation": make_observation(
                    round_number=6,
                    bankrolls=[
                        10_600,
                        10_100,
                        9_900,
                        9_800,
                        9_700,
                        9_600,
                        9_500,
                    ],
                    player_index=6,
                    betting_position=6,
                    current_bets=[100] * 6 + [0],
                    bets_placed=[True] * 6 + [False],
                    previous_bet=previous_bet,
                    previous_bankroll_change=-previous_bet,
                    previous_result=-1.0,
                    consecutive_losses=loss_count,
                    has_previous_round=True,
                ),
            }
        )

    loss_history_variants = (
        ("no recent loss", 0.0, 0.0, 0),
        ("one recent loss", -500.0, -1.0, 1),
        ("three recent losses", -1_500.0, -1.0, 3),
    )
    for label, bankroll_change, result, losses in loss_history_variants:
        scenarios.append(
            {
                "name": f"Loss-history counterfactual, {label}",
                "observation": make_observation(
                    round_number=6,
                    bankrolls=[
                        11_000,
                        10_500,
                        9_800,
                        9_600,
                        9_400,
                        9_200,
                        10_000,
                    ],
                    player_index=6,
                    betting_position=6,
                    current_bets=[100] * 6 + [0],
                    bets_placed=[True] * 6 + [False],
                    previous_bet=500,
                    previous_bankroll_change=bankroll_change,
                    previous_result=result,
                    consecutive_losses=losses,
                    has_previous_round=True,
                ),
            }
        )

    for gap_percent in (2, 5, 10, 20, 40):
        second_bankroll = 10_000 * (1 + gap_percent / 100)
        scenarios.append(
            {
                "name": (
                    "Gap response, "
                    f"{gap_percent}% behind second"
                ),
                "observation": make_observation(
                    round_number=11,
                    bankrolls=[
                        second_bankroll + 1_000,
                        second_bankroll,
                        9_800,
                        9_600,
                        9_400,
                        9_200,
                        10_000,
                    ],
                    player_index=6,
                    betting_position=6,
                    current_bets=[100] * 6 + [0],
                    bets_placed=[True] * 6 + [False],
                    previous_bet=500,
                    has_previous_round=True,
                ),
            }
        )

    return scenarios


def create_contextual_training_probes():
    """Return stable situations used to track one candidate over time."""
    probe_states = (
        (
            "early_tied",
            1,
            [10_000] * 7,
        ),
        (
            "early_behind_second",
            4,
            [10_500, 10_000, 9_800, 9_700, 9_600, 9_500, 9_900],
        ),
        (
            "mid_one_bet_behind",
            6,
            [10_500, 10_000, 9_800, 9_700, 9_600, 9_500, 9_900],
        ),
        (
            "mid_far_behind",
            6,
            [12_000, 10_500, 9_800, 9_500, 9_200, 9_000, 8_000],
        ),
        (
            "late_behind_second",
            11,
            [10_500, 10_000, 9_800, 9_700, 9_600, 9_500, 9_900],
        ),
        (
            "late_safe_lead",
            11,
            [10_000, 9_900, 9_800, 9_700, 9_600, 9_500, 12_000],
        ),
    )

    probes = []
    for slug, round_number, bankrolls in probe_states:
        probes.append(
            {
                "slug": slug,
                "name": CONTEXTUAL_PROBE_LABELS[slug],
                "observation": make_observation(
                    round_number=round_number,
                    bankrolls=bankrolls,
                    player_index=6,
                    betting_position=6,
                    current_bets=[100] * 6 + [0],
                    bets_placed=[True] * 6 + [False],
                ),
            }
        )

    gap_bankrolls = [
        11_500,
        10_500,
        9_800,
        9_600,
        9_400,
        9_200,
        10_000,
    ]
    risk_profiles = {
        "cautious": {
            "largest_opponent_previous_bet_fraction": 0.02,
            "average_opponent_previous_bet_fraction": 0.01,
            "opponents_over_ten_percent": 0,
            "opponents_over_twenty_five_percent": 0,
            "opponent_bet_volatility": 0.005,
            "opponents_with_loss_streak": 0,
        },
        "volatile": {
            "largest_opponent_previous_bet_fraction": 0.50,
            "average_opponent_previous_bet_fraction": 0.18,
            "opponents_over_ten_percent": 4,
            "opponents_over_twenty_five_percent": 2,
            "opponent_bet_volatility": 0.16,
            "opponents_with_loss_streak": 3,
        },
    }
    for stage_name, round_number in (("early", 3), ("late", 11)):
        for risk_name, risk_values in risk_profiles.items():
            slug = f"{stage_name}_gap_{risk_name}"
            probes.append(
                {
                    "slug": slug,
                    "name": CONTEXTUAL_PROBE_LABELS[slug],
                    "observation": make_observation(
                        round_number=round_number,
                        bankrolls=gap_bankrolls,
                        player_index=6,
                        betting_position=6,
                        current_bets=[100] * 6 + [0],
                        bets_placed=[True] * 6 + [False],
                        **risk_values,
                    ),
                }
            )

    return probes


def evaluate_contextual_training_probes(network):
    """Record the selected action and final legal wager for each probe."""
    from agents.neural_betting_agent import NeuralBettingAgent
    from ml.betting_network import BETTING_ACTION_NAMES

    agent = NeuralBettingAgent(network)
    results = {}

    for probe in create_contextual_training_probes():
        legal_wager = agent.choose_bet(probe["observation"])
        action_name = agent.last_action_name
        results[probe["slug"]] = {
            "action_index": BETTING_ACTION_NAMES.index(action_name),
            "action_name": action_name,
            "legal_bet": legal_wager,
        }

    return results
