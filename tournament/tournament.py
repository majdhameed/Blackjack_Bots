# Tournament coordinator. It creates successive TableRound instances, maps
# temporary table indices back to permanent seats, and presents safe snapshots
# to each bot before collecting its bet or action.
import math

from blackjack.actions import Action
from blackjack.cards import Shoe
from blackjack.player import Player
from tournament.power_chips import PowerChipAction
from tournament.table_round import TableRound
from tournament.observation import BettingObservation, ActionObservation, PowerChipObservation
from blackjack.card_counter import CardCounter


class Tournament:
    def __init__(self, players, bots, number_of_rounds, decks, minimum_bet, hit_soft_17, max_hands):

        if not isinstance(players, list) or not isinstance(bots, list):
            raise TypeError("Players and bots must be provided as a list")
        if len(bots) != len(players):
            raise ValueError("There are unequal numbers of players and bots")
        if len(players) < 2:
            raise ValueError("There must be atleast 2 players")
        if number_of_rounds <= 0:
            raise ValueError("The number of rounds must be a positive integer")
        for player in players:
            if not isinstance(player, Player):
                raise TypeError("player must be of type Player") 
        for bot in bots:
            if not callable(getattr(bot, "choose_bet", None)):
                raise TypeError(
                    "Each bot must have a choose_bet method"
                )

            if not callable(getattr(bot, "choose_action", None)):
                raise TypeError(
                    "Each bot must have a choose_action method"
                )        
        
        self.players = players
        self.bots = bots
        self.number_of_rounds = number_of_rounds
        self.minimum_bet = minimum_bet
        self.hit_soft_17 = hit_soft_17
        self.max_hands = max_hands

        self.shoe = Shoe(decks)
        self.current_round_number = 0
        self.starting_player_index = 0
        self.round_history = []
        self.is_over = False
        self.current_table_round = None
        self.active_player_indices = []
        self.card_counter = CardCounter()
        self.decks = decks
        self.reshuffle_threshold = int(decks * 52 * .2)
        player_count = len(players)
        self.previous_bets = [0.0] * player_count
        self.previous_bankroll_changes = [0.0] * player_count
        self.previous_results = [0.0] * player_count
        self.consecutive_losses = [0] * player_count
        self.has_previous_round = [False] * player_count
        self.current_round_starting_bankrolls = None
        self.current_round_bets = [0.0] * player_count





    def get_active_player_indices(self):
        # Permanent seat indices are retained even after other players bust out.
        indices = []
        for player_index, player in enumerate(self.players):
            if player.bankroll > 0:
                indices.append(player_index)

        return indices
    
    def tournament_should_end(self):
        # A tournament stops at the round limit or when at most one player remains.
        active_players = self.get_active_player_indices()

        return (
            self.current_round_number
            >= self.number_of_rounds
            or len(active_players) <= 1
        )

    def start_next_round(self):
        # Build a compact table containing only funded players while preserving
        # the rotating starting seat from the full tournament.
        if (
            self.current_table_round is not None
            and not self.current_table_round.is_over
        ):
            raise ValueError(
                "The current round is not finished"
            )

        if self.tournament_should_end():
            self.is_over = True
            return None

        if self.shoe.cards_remaining() < self.reshuffle_threshold:
            self.shoe = Shoe(self.decks)
            self.card_counter.reset()

        active_indices = (
            self.get_active_player_indices()
        )

        # Find the next active starting player.
        while (
            self.starting_player_index
            not in active_indices
        ):
            self.starting_player_index = (
                self.starting_player_index + 1
            ) % len(self.players)

        # TableRound uses local player indices.
        starting_local_index = active_indices.index(
            self.starting_player_index
        )

        round_players = [
            self.players[player_index]
            for player_index in active_indices
        ]

        self.current_round_starting_bankrolls = [
            player.bankroll
            for player in self.players
        ]
        self.current_round_bets = [
            0.0
            for _ in self.players
        ]

        for player in round_players:
            player.reset_round()

        self.active_player_indices = active_indices

        self.current_table_round = TableRound(
            players=round_players,
            shoe=self.shoe,
            minimum_bet=self.minimum_bet,
            starting_player=starting_local_index + 1,
            hit_soft_17=self.hit_soft_17,
            max_hands=self.max_hands,
            card_counter = self.card_counter
        )

        self.current_round_number += 1

        return self.current_table_round

    def get_bot_for_round_player(self, round_player_index):
        # Convert a current-round index to its permanent tournament bot.
        if round_player_index < 0 or round_player_index >= len(self.active_player_indices):
            raise ValueError("Player index is outside the range of valid players")

        tournament_index = self.active_player_indices[round_player_index]

        bot = self.bots[tournament_index]

        return bot

    def place_round_bets(self):
        # Bots act in table order and receive only an immutable observation.
        if self.current_table_round is None:
            raise TypeError("Table round must be of type TableRound")
        if self.current_table_round.is_over:
            raise ValueError("This round is already over")

        for round_player_index in self.current_table_round.betting_order:
            bot = self.get_bot_for_round_player(round_player_index)

            betting_observation = self.build_betting_observation(round_player_index)

            bet = bot.choose_bet(betting_observation)

            if isinstance(bet, bool) or not isinstance(bet, (int, float)):
                raise TypeError("Bet must be a number")
            if not math.isfinite(bet):
                raise ValueError("Bet must be finite")

            permanent_player_index = (
                self.active_player_indices[
                    round_player_index
                ]
            )
            self.current_round_bets[
                permanent_player_index
            ] = bet

            self.current_table_round.place_bet(round_player_index, bet)

        if not self.current_table_round.betting_complete():
            raise ValueError("Bets have not all been placed")

    def begin_player_actions(self):
        # Initial dealing and natural resolution form the boundary between the
        # betting phase and ordinary player actions.
        if self.current_table_round is None:
            raise ValueError("Player round does not exist")
        if not self.current_table_round.betting_complete():
            raise ValueError("betting is not complete")

        self.current_table_round.deal_initial_cards()
        naturals_ended =self.current_table_round.resolve_naturals()

        return naturals_ended

    def get_active_hand_index(self, round_player_index):
        active_hand_index = None

        player = self.current_table_round.get_player(round_player_index)

        for i in range(len(player.hands)):
            player_hand = player.get_hand(i)
            if not player_hand.is_settled and not player_hand.is_finished():
                active_hand_index = i
                return active_hand_index

        return active_hand_index

    def play_all_players(self):
        # Continue until every active hand at the table is finished.
        if self.current_table_round is None:
            raise ValueError("Table round does not exist")
        if not self.current_table_round.naturals_checked:
            raise ValueError("Naturals have not been checked")
        if self.current_table_round.is_over:
            raise ValueError("Round is already over")

        while True:
            round_player_index = self.current_table_round.get_current_player_index()

            if round_player_index is None:
                break

            if self.current_table_round.pending_rehit is not None:
                pending_player_index, pending_hand_index = (
                    self.current_table_round.pending_rehit
                )

                pending_bot = self.get_bot_for_round_player(pending_player_index)

                rehit_observation = self.build_power_chip_observation(
                    pending_player_index,
                    pending_hand_index,
                    PowerChipAction.REHIT,
                )

                target = pending_bot.choose_power_chip(rehit_observation)

                if target is None:
                    self.current_table_round.player_decline_rehit(
                        pending_player_index,
                        pending_hand_index,
                    )
                else:
                    self.current_table_round.player_use_power_chip(
                        pending_player_index,
                        pending_hand_index,
                        PowerChipAction.REHIT,
                        target,
                    )

                continue


            hand_index = self.get_active_hand_index(round_player_index)

            if hand_index is None:
                self.current_table_round.advance_action_turn()
                continue

            player = self.current_table_round.get_player(round_player_index)

            player_hand = player.get_hand(hand_index)

            bot = self.get_bot_for_round_player(round_player_index)

            legal_replace_targets = (
                self.current_table_round.get_legal_power_chip_targets(
                    round_player_index,
                    hand_index,
                    PowerChipAction.REPLACE,
                )
            )

            if len(legal_replace_targets) != 0:
                replace_observation = self.build_power_chip_observation(
                    round_player_index,
                    hand_index,
                    PowerChipAction.REPLACE,
                )

                target = bot.choose_power_chip(replace_observation)

                if target is not None:
                    self.current_table_round.player_use_power_chip(
                        round_player_index,
                        hand_index,
                        PowerChipAction.REPLACE,
                        target,
                    )
                    continue



            legal_actions = self.current_table_round.get_legal_actions(round_player_index, hand_index)

            action_observation = self.build_action_observation(round_player_index, hand_index)
            chosen_action = bot.choose_action(action_observation)

            if chosen_action not in legal_actions:
                raise ValueError("Chosen action is not legal")

            if chosen_action == Action.HIT:
                self.current_table_round.player_hit(round_player_index, hand_index)
            elif chosen_action == Action.STAND:
                self.current_table_round.player_stand(round_player_index, hand_index)
            elif chosen_action == Action.DOUBLE:
                additional_bet = min(player_hand.bet, player.bankroll)
                self.current_table_round.player_double(round_player_index, hand_index, additional_bet)
            elif chosen_action == Action.SPLIT:
                self.current_table_round.player_split(round_player_index, hand_index)
            elif chosen_action == Action.SURRENDER:
                self.current_table_round.player_surrender(round_player_index, hand_index)
            else:
                raise ValueError("Unknown action")


    def finish_current_round(self):
        # Run the dealer and settlement phases, then archive the completed round
        if self.current_table_round is None:
            raise ValueError("The current round does not exist")

        table_round = self.current_table_round

        if table_round.is_over:
            outcomes = table_round.outcomes
        else:
            if table_round.get_current_player_index() is not None:
                raise ValueError("The current round is not over yet")

            table_round.play_dealer()

            outcomes = table_round.settle_round()

        if self.current_round_starting_bankrolls is None:
            raise RuntimeError(
                "Round starting bankrolls were not recorded"
            )

        for permanent_player_index in (
            self.active_player_indices
        ):
            bankroll_change = (
                self.players[
                    permanent_player_index
                ].bankroll
                - self.current_round_starting_bankrolls[
                    permanent_player_index
                ]
            )
            if bankroll_change > 0:
                previous_result = 1.0
                self.consecutive_losses[
                    permanent_player_index
                ] = 0
            elif bankroll_change < 0:
                previous_result = -1.0
                self.consecutive_losses[
                    permanent_player_index
                ] += 1
            else:
                previous_result = 0.0
                self.consecutive_losses[
                    permanent_player_index
                ] = 0

            self.previous_bets[
                permanent_player_index
            ] = self.current_round_bets[
                permanent_player_index
            ]
            self.previous_bankroll_changes[
                permanent_player_index
            ] = bankroll_change
            self.previous_results[
                permanent_player_index
            ] = previous_result
            self.has_previous_round[
                permanent_player_index
            ] = True

        self.round_history.append(table_round)

        self.current_table_round = None

        self.starting_player_index = (self.starting_player_index + 1) % len(self.players)

        if self.tournament_should_end():
            self.is_over = True

        return outcomes

    def play_one_round(self):
        # Drive all phases of one table round in their required order.
        table_round = self.start_next_round()

        if table_round is None:
            return None

        self.place_round_bets()

        naturals_ended = self.begin_player_actions()

        if not naturals_ended:
            self.play_all_players()

        outcomes = self.finish_current_round()

        return outcomes

    def get_rankings(self):
        # Rankings retain permanent player indices and sort by bankroll descending.
        rankings = sorted(
            [(player_index, player.bankroll) for player_index, player in enumerate(self.players)],
            key=lambda x: x[1],
            reverse=True,
        )
        return rankings

    def play_tournament(self):
        # Repeatedly play rounds until an end condition is reached.
        while not self.is_over:
            outcomes = self.play_one_round()

            if outcomes is None:
                self.is_over = True
                break


        return self.get_rankings()
        


    def build_betting_observation(self, round_player_index):
        # Convert round-local state into permanent-seat tuples for the acting bot.
        if self.current_table_round is None:
            raise ValueError("The current round does not exist")

        table_round = self.current_table_round

        if table_round.betting_complete():
            raise ValueError("Betting is already complete")

        expected_player_index = table_round.betting_order[table_round.next_bettor_index]

        if round_player_index != expected_player_index:
            raise ValueError("It's not this player's turn to bet")

        permanent_player_index = self.active_player_indices[round_player_index]

        current_bets = tuple(0 for _ in range(len(self.players)))

        bets_placed = tuple(False for _ in range(len(self.players)))

        for local_index, player in enumerate(table_round.players):
            permanent_index = self.active_player_indices[local_index]
            if len(player.hands) > 0:
                first_hand = player.get_hand(0)
                current_bets = current_bets[:permanent_index] + (first_hand.bet,) + current_bets[permanent_index + 1:]
                bets_placed = bets_placed[:permanent_index] + (True,) + bets_placed[permanent_index + 1:]

        bankrolls = tuple(player.bankroll for player in self.players)

        active_players = tuple(
            player_index in self.active_player_indices
            for player_index in range(len(self.players)))


        permanent_betting_order = tuple(
            self.active_player_indices[local_index]
            for local_index in table_round.betting_order
        )
        count_information = (
            self.get_count_information()
        )

        opponent_indices = [
            player_index
            for player_index in self.active_player_indices
            if player_index != permanent_player_index
        ]
        opponent_previous_bet_fractions = [
            self.previous_bets[player_index]
            / max(
                self.players[player_index].bankroll,
                self.minimum_bet,
            )
            for player_index in opponent_indices
        ]
        if opponent_previous_bet_fractions:
            average_opponent_fraction = sum(
                opponent_previous_bet_fractions
            ) / len(opponent_previous_bet_fractions)
            opponent_variance = sum(
                (fraction - average_opponent_fraction) ** 2
                for fraction in opponent_previous_bet_fractions
            ) / len(opponent_previous_bet_fractions)
            opponent_volatility = math.sqrt(opponent_variance)
        else:
            average_opponent_fraction = 0.0
            opponent_volatility = 0.0

        return BettingObservation(
            round_number=self.current_round_number,
            total_rounds=self.number_of_rounds,
            rounds_remaining=self.number_of_rounds - self.current_round_number,
            player_index=permanent_player_index,
            round_player_index=round_player_index,
            betting_position=table_round.next_bettor_index,
            minimum_bet=self.minimum_bet,
            bankroll = self.players[permanent_player_index].bankroll,
            bankrolls=bankrolls,
            active_players=active_players,
            current_bets=current_bets,
            bets_placed=bets_placed,
            betting_order=permanent_betting_order,
            previous_bet=self.previous_bets[
                permanent_player_index
            ],
            previous_bankroll_change=(
                self.previous_bankroll_changes[
                    permanent_player_index
                ]
            ),
            previous_result=self.previous_results[
                permanent_player_index
            ],
            consecutive_losses=self.consecutive_losses[
                permanent_player_index
            ],
            has_previous_round=self.has_previous_round[
                permanent_player_index
            ],
            largest_opponent_previous_bet_fraction=max(
                opponent_previous_bet_fractions,
                default=0.0,
            ),
            average_opponent_previous_bet_fraction=(
                average_opponent_fraction
            ),
            opponents_over_ten_percent=sum(
                fraction >= 0.10
                for fraction in opponent_previous_bet_fractions
            ),
            opponents_over_twenty_five_percent=sum(
                fraction >= 0.25
                for fraction in opponent_previous_bet_fractions
            ),
            opponent_bet_volatility=opponent_volatility,
            opponents_with_loss_streak=sum(
                self.consecutive_losses[player_index] > 0
                for player_index in opponent_indices
            ),
            **count_information
        )

    def build_action_observation(self, round_player_index, hand_index):
        # Expose values and legal choices without exposing mutable engine objects.
        table_round = self.current_table_round

        if table_round is None:
            raise ValueError("no table_round")

        current_player_index = table_round.get_current_player_index()

        if current_player_index is None:
            raise ValueError("All players have finished acting")

        if round_player_index != current_player_index:
            raise ValueError("It is not this player's turn")

        player = table_round.get_player(round_player_index)

        player_hand = player.get_hand(hand_index)

        legal_actions = table_round.get_legal_actions(round_player_index, hand_index)

        if len(legal_actions) == 0:
            raise ValueError("No legal actions available")

        permanent_player_index = self.active_player_indices[round_player_index]

        bankrolls = tuple(player.bankroll for player in self.players)

        active_players = tuple(
            player_index in self.active_player_indices
            for player_index in range(len(self.players))
        )

        player_bets = list(0 for _ in range(len(self.players)))

        for local_index, local_player in enumerate(table_round.players):
            permanent_index = self.active_player_indices[local_index]

            total_bet = 0

            for local_player_hand in local_player.hands:
                total_bet += local_player_hand.bet

            player_bets[permanent_index] = total_bet

        player_bets = tuple(player_bets)

        current_bet = player_hand.bet

        hand_total = player_hand.hand.get_total()

        hand_is_soft = player_hand.hand.is_soft()

        hand_card_values = tuple(
            card.get_value()
            for card in player_hand.hand.cards
        )

        hand_came_from_split = player_hand.came_from_split

        dealer_upcard_value = table_round.dealer.hand.cards[0].get_value()

        permanent_betting_order = tuple(
            self.active_player_indices[local_index]
            for local_index in table_round.betting_order
        )

        ordered_legal_actions = tuple(
            action
            for action in Action
            if action in legal_actions
        )

        count_information = (
            self.get_count_information()
        )

        return ActionObservation(
            round_number=self.current_round_number,
            total_rounds=self.number_of_rounds,
            rounds_remaining=self.number_of_rounds - self.current_round_number,
            player_index=permanent_player_index,
            round_player_index=round_player_index,
            hand_index=hand_index,
            bankroll=self.players[permanent_player_index].bankroll,
            bankrolls=bankrolls,
            current_bet=current_bet,
            player_bets=player_bets,
            hand_total=hand_total,
            hand_is_soft=hand_is_soft,
            hand_card_values=hand_card_values,
            hand_came_from_split=hand_came_from_split,
            dealer_upcard_value=dealer_upcard_value,
            legal_actions=ordered_legal_actions,
            betting_order=permanent_betting_order,
            active_players=active_players,
            hit_soft_17=self.hit_soft_17,
            **count_information
        )


    def get_count_information(self):
        cards_remaining = self.shoe.cards_remaining()

        decks_remaining = cards_remaining / 52

        total_shoe_cards = self.decks * 52

        shoe_penetration = 1 - (cards_remaining / total_shoe_cards)

        if decks_remaining > 0:
            true_count = self.card_counter.get_true_count(decks_remaining)
        else:
            true_count = 0

        return {
            "card_value_counts": (
                self.card_counter.get_counts()
            ),
            "cards_seen": (
                self.card_counter.cards_seen
            ),
            "running_count": (
                self.card_counter.running_count
            ),
            "true_count": float(true_count),
            "cards_remaining": cards_remaining,
            "decks_remaining": float(
                decks_remaining
            ),
            "shoe_penetration": float(
                shoe_penetration
            ),
        }

    def build_power_chip_observation(self, round_player_index, hand_index, action):
        table_round = self.current_table_round

        if not table_round:
            raise ValueError("No table round exists")

        current_player_index = table_round.get_current_player_index()

        if round_player_index != current_player_index:
            raise ValueError("It is not this player's turn")

        legal_targets = table_round.get_legal_power_chip_targets(
            round_player_index,
            hand_index,
            action,
        )

        if not legal_targets:
            raise ValueError("No legal power chip targets")

        permanent_player_index = self.active_player_indices[round_player_index]

        player = table_round.get_player(round_player_index)

        player_hand = player.get_hand(hand_index)

        bankrolls = tuple(player.bankroll for player in self.players)

        power_chips = tuple(player.power_chips.remaining() for player in self.players)

        current_bet = player_hand.bet

        hand_total = player_hand.hand.get_total()

        hand_is_soft = player_hand.hand.is_soft()

        hand_card_values = tuple(
            card.get_value()
            for card in player_hand.hand.cards
        )

        dealer_upcard_value = table_round.dealer.hand.cards[0].get_value()

        return PowerChipObservation(
            round_number=self.current_round_number,
            total_rounds=self.number_of_rounds,
            rounds_remaining=(
                self.number_of_rounds - self.current_round_number
            ),
            player_index=permanent_player_index,
            round_player_index=round_player_index,
            hand_index=hand_index,
            bankroll=self.players[permanent_player_index].bankroll,
            bankrolls=bankrolls,
            current_bet=current_bet,
            hand_total=hand_total,
            hand_is_soft=hand_is_soft,
            hand_card_values=hand_card_values,
            dealer_upcard_value=dealer_upcard_value,
            power_chip_counts=power_chips,
            action=action,
            legal_targets=legal_targets,
        )
