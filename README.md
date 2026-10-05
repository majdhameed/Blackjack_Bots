# Blackjack Bots

Blackjack Bots is a Python project for modeling blackjack rules, running games, and comparing betting strategies over repeated tournaments. The current focus is a reusable game engine plus command-line simulations. A future visualization layer can turn simulation results into charts that make strategy performance easier to compare.

## Project scope

The project currently covers three related use cases:

- Play blackjack interactively in a terminal.
- Simulate a basic-strategy player over many rounds.
- Run tournaments in which several bots use the same card-play strategy but different betting systems.

The engine handles normal blackjack actions and round settlement, including hits, stands, doubles, splits, surrender, dealer blackjack, natural blackjack, pushes, configurable soft-17 behavior, multiple decks, a maximum number of split hands, and optional power chips.

This project is intended for simulation and software experimentation. It is not gambling advice, and past simulated performance does not predict real-world results.

## Betting strategies

The tournament simulator currently compares seven strategies:

| Strategy | Wager |
| --- | --- |
| Minimum | The table minimum, or the remaining bankroll when it is lower |
| 10% | 10% of the current bankroll, subject to the table minimum |
| 20% | 20% of the current bankroll, subject to the table minimum |
| 30% | 30% of the current bankroll, subject to the table minimum |
| 40% | 40% of the current bankroll, subject to the table minimum |
| 50% | 50% of the current bankroll, subject to the table minimum |
| All-in | The entire current bankroll |

Every included bot inherits the same basic-strategy card decisions. This isolates bet sizing as the main difference between competitors.

All non-all-in wagers use table-minimum chip increments. With a `$100` minimum, legal strategy outputs are `$100`, `$200`, `$300`, and so on; a short remaining bankroll may still be wagered all-in.

## Power chips

Each player may start with a shared inventory of optional power chips. A chip can replace either opening card after the dealer-blackjack check (`REPLACE`), or replace the most recent hit card before play continues (`REHIT`). On split hands, only the newly dealt card can be replaced. Every replacement draws the next card from the shoe and consumes one chip.

Bots receive an immutable `PowerChipObservation` and may return a legal card index or `None`. The included policies can save busting hits, improve stiff hands against strong dealer cards, combine tactical rules, and reserve chips for large wagers or late rounds. Agents decline power chips by default unless a policy is attached with `set_power_chip_policy(...)`.

The neural-betting trainer also uses state-aware and deliberately unpredictable opponents:

| Strategy | Behavior |
| --- | --- |
| Chaser | Bets the minimum early, then increases risk when outside the top two |
| Early lead | Includes measured 25%, half-bankroll, and opening-all-in variants that attack briefly, then protect a safe lead |
| Lead protector | Builds a small early lead with 20% wagers, then covers visible threats and chases late when behind |
| Controlled lead Martingale | Starts at 5%, doubles after losses up to 20%, and drops to minimum after gaining a $100 lead |
| Human behavior | Reacts to wins and losses, presses streaks, copies visible wagers, protects leads, and occasionally bets impulsively |
| Count aware | Raises its wager only when the Hi-Lo true count is favorable |
| Unpredictable | Changes among minimum, random-percentage, catch-up, copied, and impulsive wagers each round |

Strategies are randomly assigned to seats for every tournament to reduce seat-order bias. A tournament ends when it reaches the requested round limit or no more than one funded player remains.

## Project structure

```text
blackjack_Bots/
|-- agents/                 Betting and playing strategies
|-- blackjack/              Cards, hands, players, rules, and settlement
|-- ml/                     Placeholder for future model feature encoding
|-- tournament/             Multi-player rounds and tournament coordination
|-- tests/                  Automated rule and tournament tests
|-- play_console.py         Interactive terminal game
|-- simulate.py             Single-player simulation driver
`-- simulate_tournament.py  Seven-strategy comparison
```

The main layers are:

- `blackjack`: Owns the core game rules and mutable state.
- `tournament`: Coordinates multiple players, betting order, action order, elimination, and rankings.
- `agents`: Selects bets and actions from immutable observations supplied by the tournament.
- Entry-point scripts: Collect terminal input and present simulation results.

The observation objects intentionally keep bots separate from mutable engine objects. This makes strategy code safer and provides a useful boundary for future machine-learning agents.

## Requirements

- Python 3.9 or newer
- `pytest` to run the test suite

The game and simulation code use only the Python standard library.

## Running the project

Run commands from the repository root.

### Interactive blackjack

```powershell
python play_console.py
```

The console prompts for a starting bankroll, deck count, minimum bet, and the dealer's soft-17 rule.

### Strategy tournament

```powershell
python simulate_tournament.py
```

The simulator prompts for:

- Starting bankroll
- Hands (rounds) requested per tournament
- Number of tournaments
- Number of decks
- Minimum bet
- Whether the dealer hits soft 17

It reports first-place frequency, tie-adjusted win rate, average ending bankroll, bankruptcy rate, and hand outcome counts for each strategy.

### Single-player simulation

```powershell
python simulate.py
```

This script is the older single-player simulation path. Its agent calls still need to be aligned fully with the newer observation-based bot interface, so the tournament simulator is currently the primary automated entry point.

### Train the neural betting bot

```powershell
python train_betting.py
```

Use `--power-chip-count N` to give every training-table player `N` chips. The default is zero, preserving the standard blackjack training setup.

The current overnight configuration uses 200 generations, 30 self-play tournaments per network, 30 mixed-opponent tournaments per network, 300 staged benchmark tournaments, and the top 10 benchmark candidates. Runtime depends on the machine; independently verified new best networks are checkpointed throughout the run.

Each generation combines neural self-play with a league mixture: 40% realistic human tables, 25% proven handcrafted strategies, 20% archived neural champions, 10% randomized adversarial strategies, and 5% six-minimum control tables. Before the first champion qualifies, the league portion falls back to the proven-strategy table. Qualifying champions are retained in `models/league` and become opponents in later generations, so a new policy must keep competing with strategies found earlier instead of exploiting only the current population.

Training scenarios are 45% normal tied starts, 30% mid-stage tables with five to eight rounds remaining and the neural player exactly third or fourth, 20% late-stage tables with two or three rounds remaining and the neural player exactly third, and 5% extreme robustness cases. Mid-stage deficits to second grow from roughly 5-10% to 5-20% over the curriculum; late-stage deficits grow from roughly 3-5% to 3-8%. This emphasizes learning how to create an advantage from a real equal start while retaining recovery practice.

The main fitness directly reflects tournament advancement: first place earns `1.05`, second place earns `1.0`, and lower positions earn zero. Ties share the fitness belonging to the positions they occupy. Training tables also award small, capped shaping credit for improving rank, closing the gap to second, creating a top-two safety margin, and establishing a lead. That shaping fades linearly and is completely disabled by generation 40, so the end of training optimizes actual advancement. Fixed benchmarks always remain strict and award no shaping credit.

The ten best training candidates from every generation are each evaluated over 300 staged tournaments while rotating among fixed adaptive, disciplined, human-like, and early risk-taker lineups. Every candidate is evaluated from the same random-number states, reducing advantages from easier seats, scenarios, human impulses, or shoes. The risk-taker lineup contains two half-bankroll lead builders and one opening-all-in player. The disciplined lineup includes a minimum-bet control so a weak policy cannot look strong merely because aggressive opponents eliminate one another. The benchmark is split into normal, mid-stage, and late-stage tables, weighted 40%, 35%, and 25%.

Each candidate also plays a separate disciplined full-tournament holdout and a six-minimum control holdout. The robust selection score is 45% staged benchmark, 45% disciplined full-tournament holdout, and 10% six-minimum holdout. This prevents one lucky staged result or unusually easy minimum table from dominating selection. Candidates and saved generations with measurable late-stage advancement are preferred over those with none. The final output reports every component.

Whenever a generation appears to establish a new overall best robust score, the challenger and saved incumbent are re-evaluated on the same independent 1,000-tournament staged batch plus their holdouts. The challenger is atomically checkpointed to `models/best_betting_network.npz` only if it still wins that comparison. An interrupted overnight run therefore retains the strongest verified generation completed so far.

The neural encoder supplies explicit tournament features for current rank, distance from the leader, distance from second place, current top-two status, largest visible wager, final-round status, previous wager, previous bankroll change, previous result, and consecutive losses. This history makes controlled loss progressions representable by the feed-forward network.

The neural policy has a separate action head that chooses among legacy fractional betting, minimum, 5% baseline, controlled loss recovery, taking second, taking first, covering visible bets, half-bankroll, all-in, and lead protection. New populations seed all of these actions as well as distinct wager fractions. Every generation also injects one fresh network for each action, preventing the population from permanently collapsing into minimum bettors while still allowing evolution to learn when to switch actions from the encoded tournament state.

The saved network can be checked against the fixed seven-strategy benchmark with:

```powershell
python evaluate_betting_network.py
```

The evaluation table includes a minimum-bet control alongside the neural and adaptive strategies, making it clear whether the learned policy actually outperforms minimum betting under the same cards-and-opponents environment.

Evaluation also runs a separate early risk-taker table containing two half-bankroll lead builders and one opening-all-in leader. When archived champions exist, it then runs a champion-league table so the final policy's performance against earlier winners is explicit, followed by the six-minimum control table.

## Running tests

```powershell
python -m pytest -q
```

The tests cover cards, hand totals, basic strategy, bankroll operations, legal actions, splitting aces, settlement, table rounds, card counting, and tournament behavior.

## Current limitations

- Results are printed to the terminal rather than saved as structured data.
- Simulations do not currently accept a fixed random seed from the command line.
- The single-player simulator predates the observation-based agent API.
- Training and evaluation still use nondeterministic card shuffles.
- There is no graphical interface, report export, or chart generation yet.

## Roadmap

The next reporting milestone is to store simulation results and visualize strategy behavior. Useful additions include:

1. Export each run to CSV or JSON.
2. Add reproducible random seeds and configuration metadata.
3. Plot average ending bankroll by strategy.
4. Compare first-place and bankruptcy rates.
5. Chart bankroll distributions instead of averages alone.
6. Show bankroll trajectories over the course of a tournament.
7. Add confidence intervals as the number of simulations grows.
8. Connect the card counter and future learned agents to the observation API.

Separating result collection from chart rendering will make it possible to support notebooks, static image reports, or a small dashboard without changing the blackjack engine.
