# Chess Blunder Prediction and Time Pressure

## Data

Source: [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games)
on Hugging Face (CC0), hive-partitioned by `year=/month=`. `src/ingest.py` reads it directly from
the HF `hf://` filesystem via DuckDB, filtering with pushed-down predicates rather than downloading
whole months.

### Train/val vs. test months

Training/validation data comes from one month (`MONTH_A`), the held-out test set from a second
month at least three months later (`MONTH_B`). Both are pulled directly from the source rather than
split from a single month. This is deliberate: a random split within one month only tests whether
the model generalises across *games*, since it would still see the same population of players,
engine versions, opening trends, and rating dynamics that produced the training data. Testing on a
separate, later month checks whether it generalises across *time* as well -- the harder and more
realistic bar for a model meant to be useful going forward.

### Filters applied (`src/ingest.py`)

A game is kept only if, in order:

1. `movetext` contains `[%eval` (engine evaluations present).
2. `movetext` also contains `[%clk` (per-move clock data present).
3. `TimeControl` falls in one of the candidate increment-pair buckets.
4. Neither player is a `BOT`.
5. Both `WhiteElo` and `BlackElo` are present and in `[800, 2600]`.
6. `Termination` is `"Normal"` or `"Time forfeit"` (drops abandoned/rules-infraction games; keeps
   time forfeits, since Set 2 needs them for the "final move of a game lost on time" exclusion).
7. At least 20 plies (approximated by counting `[%clk` occurrences in `movetext`, since every
   annotated ply carries exactly one clock comment once step 2 has passed).

`[%eval]` coverage is rare (~6% of all games), and checking for it requires decoding the
`movetext` column for every row in the month regardless of what happens next -- that IO cost is
unavoidable. `ingest.py` pays it exactly once per month: a single pass filters on `[%eval` alone
and writes survivors (with boolean flags for every later filter) to a local intermediate parquet;
every subsequent count and the final column selection run locally against that file, not against
the remote dataset again.

Time-control buckets are tried in order `180+0/180+2`, then `300+0/300+3`, then `600+0/600+5` --
the first pair where both buckets have at least ~15,000 surviving games in *both* months is used
for both months' final output, so train/val and test share a time control. All three pairs' counts
are printed for both months.

### Output

`data/raw/games_{year}-{month}.parquet`, columns: `game_id`, `white`, `black`, `white_elo`,
`black_elo`, `time_control`, `result`, `termination`, `utc_date`, `opening`, `movetext`.

### Requirements

A Hugging Face access token is required (set `HF_TOKEN` in the environment before running) --
anonymous requests to the dataset get rate-limited (HTTP 429) once DuckDB opens more than a
handful of concurrent shard connections, which happens quickly at ~400 shards/month.

## Move-level parsing and labelling (`src/parse.py`)

Each game is replayed with `python-chess` into one row per ply (move exclusions in the code
docstring). The annotation comment attached to the move played at ply `t` describes the position
*after* that move, so for the row at ply `t`: `eval_after` is the eval on ply `t`, `eval_before` is
the eval on ply `t-1`. Clocks only change on their own side's move, so the mover's clock before
their move is their own last update, on ply `t-2`, not `t-1` (that's the opponent's clock).

Evals are converted from centipawns to win probability with the Lichess formula before
thresholding, from the mover's point of view. This has a useful side effect worth calling out: the
sigmoid saturates near 0 and 100, so a position that is already close to lost or won can barely
move win probability further even on a large centipawn swing -- meaningful `wp_loss` is only
possible in positions that are still contested. That's the correct behaviour for a blunder metric
(a move that seals an already-decided game shouldn't count the same as one that throws away a
level position), and it means no separate "skip lost positions" rule is needed.

`blunder = wp_loss >= 20` by default; `wp_loss` itself is stored as a continuous column so Set 4
can re-threshold at 10/15/20/30 without re-parsing.

## Features and splits (`src/features.py`, `src/splits.py`)

### Leakage rules

Every feature must be computable strictly *before* the move is played:

- No feature may use `eval_after`, `cp_after`, `wp_after`, or `wp_loss` -- these describe the
  position after the move, which requires already knowing what was played.
- No feature may use the game `result` or `termination`.
- No feature may use aggregate game statistics (e.g. total accuracy, total blunder count) --
  those are only known once the whole game is over.
- Rolling features (`wp_volatility_3`, `wp_swing_last`, `time_spent_prev`) may only look
  backwards at prior plies of the same game; the current ply's own after-values are never
  part of the window.

`features.py` asserts `BANNED_COLUMNS.isdisjoint(FEATURE_COLUMNS)` at build time and prints the
result, rather than relying on code review to catch a leaked column.

### Splits (`src/splits.py`)

Train/val are both drawn from `MONTH_A`, split 80/20 by a hash of `game_id` -- never by
individual move, since moves within one game are heavily correlated (splitting by move would
leak the rest of that game's context between train and val). Test is all of `MONTH_B`.

`test_unseen_players` is the subset of test whose mover never appears (as a mover) anywhere in
`MONTH_A`. Comparing metrics on test vs. `test_unseen_players` (done in `evaluate.py`) directly
measures how much of the model's performance comes from having seen a given player's tendencies
before, versus generalising from board/clock/eval state alone.

## Models and evaluation (`src/models.py`, `src/evaluate.py`)

### No resampling

Every model is trained on the natural class distribution -- no SMOTE, no undersampling, no
oversampling. Blunders are genuinely rare (see Set 2's verification output), and resampling
changes the base rate the model learns from, which distorts its predicted probabilities away
from the true blunder frequency. Since the whole point of this project is a *calibrated*
probability, not just a ranking, that trade-off isn't acceptable here -- a resampled model might
score similarly on ranking metrics like PR-AUC while being badly miscalibrated.

### Model ladder

Fit in order, all reported side by side: base rate, Elo-only logistic, Elo+position logistic
(`mover_elo`, `wp_before`, `legal_move_count`, `material_total`), full logistic on every feature
(standardised, L2-penalised -- kept deliberately simple and linear so its coefficients stay
interpretable for Set 6), and LightGBM on every feature. LightGBM's `num_leaves`,
`min_child_samples`, and `learning_rate` are grid-searched (`LGB_PARAM_GRID` in `models.py`) with
early stopping on validation PR-AUC via a custom `feval`; `n_estimators` isn't a separate grid
dimension since early stopping already picks the effective number of trees per configuration.
The grid is intentionally small -- this isn't a Kaggle leaderboard exercise, and `evaluate.py`
reports whichever model wins honestly rather than searching until LightGBM comes out on top (see
Verification in the code output).

Rolling-history features are null for a mover's first tracked move in a game (see Set 3). The
logistic models fill those with 0; LightGBM is left the native nulls, since it splits on
missingness directly and doesn't need imputation.

### Calibration

Isotonic regression is fit on the validation set's LightGBM predictions and applied to the test
set's LightGBM predictions (LightGBM is the most flexible model in the ladder, so it's the one
whose calibration is most worth checking). Brier score is reported before and after, and the
10-bin reliability curve for both is saved to `outputs/figures/calibration.png`.

### Metrics

Primary metric is PR-AUC, always reported alongside the base rate it's relative to (PR-AUC isn't
comparable across datasets/splits with different base rates on its own). Also reported: ROC-AUC,
Brier score, log loss, and lift in the top decile of predicted risk -- each computed on
validation, the full test month, and `test_unseen_players` separately.

### Threshold sensitivity

The whole ladder is refit at blunder thresholds of 10/15/20/30 win-percentage points (recomputed
from the already-stored continuous `wp_loss` column, no re-parsing needed). LightGBM uses a fixed
default configuration for this table rather than re-running the grid search at every threshold,
to keep the sweep's compute bounded. Results are in `outputs/tables/threshold_sensitivity.csv`;
if any conclusion (e.g. which model wins) flips across thresholds, that's noted here once real
numbers are in.

### Output

`outputs/tables/model_comparison.csv`, `outputs/tables/threshold_sensitivity.csv`,
`outputs/figures/calibration.png`, `outputs/figures/pr_curve.png`, and pickled fitted models in
`data/processed/model_{name}.pkl`.

## Increment as a natural experiment (`src/causal.py`)

Everything so far is predictive, not causal -- LightGBM being good at ranking blunder risk from
clock state says nothing about whether *more time* actually prevents blunders, since players who
choose fast time controls likely differ from players who choose slow ones in ways that also
affect blunder rate. Increment offers a cleaner design: within a matched pair (same base time,
different increment -- the same pairing `ingest.py` used to pick the dataset's time control),
increment is fixed before the game starts and can't respond to any specific position, and its
effect on the clock only compounds as the game goes on. That gives a
difference-in-differences (DiD) structure:

- unit: player-game
- treatment: `increment > 0`
- "time": ply bucket (`9-20`, `21-40`, `41-60`, `61+`)
- outcome: blunder rate within that player-game's moves in that bucket

The regression is `blunder_rate ~ treatment * ply_bucket` with **player** fixed effects (not
player-game), so a switcher's own games still vary in treatment status and the treatment main
effect stays identified rather than being absorbed. Fixed effects are implemented by demeaning
(the within estimator) rather than a dummy per player, since the player count makes a dummy
design matrix impractical; standard errors are cluster-robust by player. Caveat: plain OLS on
demeaned data doesn't reduce residual degrees of freedom for the number of player means
absorbed, so reported SEs are a close approximation rather than textbook-exact -- acceptable
here since player-clustering is what matters most for validity, and that part is done properly.

Sample is restricted to **switchers**: players who appear in both increment groups (pooling both
months). This removes the selection problem of who chooses which time control -- every
comparison is within-player.

### Required checks

- **Parallel trends**: the earliest ply bucket's blunder rate should be similar across increment
  groups, since the clocks have barely diverged yet. Plotted in
  `outputs/figures/parallel_trends.png`. If the lines start apart, the design is compromised and
  that gets said plainly in the code's printed output, not smoothed over.
- **Placebo**: the treatment x ply_bucket interaction is zero by construction in the omitted
  (earliest) bucket; `causal.py` additionally confirms this empirically by printing the raw,
  non-regression gap between groups in that bucket.
- **Balance table**: mean/SD of mover Elo, opponent Elo, and game length (max observed `ply` in
  the retained move rows -- a lower bound on true game length, since Set 2 drops some plies)
  across increment groups, restricted to the switcher sample, in
  `outputs/tables/increment_balance.csv`.

### Output

`outputs/tables/increment_balance.csv`, `outputs/tables/increment_did_results.csv`,
`outputs/tables/increment_sensitivity.csv`, `outputs/figures/parallel_trends.png`.

## Sensitivity analysis

### E-value

`causal.py` reports an [E-value](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5771830/) (VanderWeele
& Ding, 2017) for the total treatment effect in the latest ply bucket (main effect + interaction
-- the bucket where the design predicts the compounding effect is largest). The linear-probability
effect is converted to an approximate risk ratio using the control group's raw blunder rate in that
bucket as the reference risk, and the E-value is reported for both the point estimate and the
confidence interval bound closest to the null. It answers one specific question: how strongly would
an unmeasured confounder need to be associated with *both* increment and blunder risk, above and
beyond player fixed effects, to fully explain away the estimate?

### What would make this credible, and why my data doesn't get me there

Player fixed effects and the switcher restriction rule out one specific, narrow kind of
confounding: stable differences between people who always play with increment and people who
never do. They rule out nothing else. Three residual threats keep this from being a credible
causal estimate on their own:

**Unobserved position difficulty within a ply bucket.** A ply bucket only fixes how far into the
game a move is, not how sharp, forcing, or theoretically well-trodden the resulting position is.
If increment and non-increment games systematically differ in the kinds of positions reached at a
given depth -- say, because faster time controls encourage different opening choices, or more
forcing tactical middlegames -- the DiD estimate is partly picking up position difficulty, not
time pressure, and ply bucket does nothing to separate the two.

**Differential opponent behaviour under increment.** The design treats the opponent as a fixed
part of the environment, but opponents are not blind to the time control. A player facing someone
with increment may play more patiently, or be more willing to grind out a long ending, than a
player who knows both clocks are about to run out. That shift in opponent behaviour is itself a
function of increment and is folded into the outcome without being separately identified from the
mover's own time pressure.

**Switchers may choose increment based on how they expect to play that day.** This is the most
serious one. Player fixed effects remove stable, between-player selection; they cannot remove
day-to-day, within-player selection. A player who feels sharp, rested, or focused might be more
likely to pick a faster time control precisely because they expect to need less thinking time that
day -- inducing a spurious link between increment and blunder rate that has nothing to do with the
clock itself, and that no amount of player-level fixed effects can absorb.

The E-value puts a number on how strong a confounder matching any of these three stories would
need to be to erase the estimate. A risk ratio in the 2-3 range is not an exotic magnitude for
something like within-player day-to-day form or a systematic difference in position sharpness --
it's entirely plausible that any one of the three threats above clears that bar on its own. This
result is best read as suggestive of a real time-pressure effect on blunder rate, not as a
demonstrated one, and that gap is the honest conclusion of this section.

## Player report (`src/report.py`)

For any username with at least 200 moves in `data/processed/test.parquet` (deliberately
test-only, all of `MONTH_B`: the model was trained on `MONTH_A`, so every row here is
out-of-sample for it regardless of whether the player also appears in `MONTH_A`, which is what
makes their individual calibration check meaningful rather than circular), `report.py` produces
one figure with four panels:

1. **Blunder rate by clock decile** against a baseline of every other player in the same
   100-Elo-point band, using decile edges drawn from the baseline's own clock distribution so
   both curves are binned identically.
2. **Individual calibration**: this player's moves scored by the fitted LightGBM model, binned by
   predicted probability, predicted vs. observed -- the same kind of reliability check as Set 4's
   calibration, but for one person instead of the whole test set.
3. **Time allocation profile**: mean seconds spent per move by ply bucket, against the same
   rating-matched baseline.
4. **A counterfactual**, printed as text on the figure: applying the Step 5 middlegame DiD
   estimate, what happens to this player's expected middlegame blunder count if they shifted 20%
   of their opening time into the middlegame. The conversion from the DiD's blunder-rate effect to
   a per-second rate uses the empirical extra seconds-per-move increment buys in that bucket
   (`causal.py`'s pickled `bucket_time_diff`) as the bridge. This rests on three assumptions,
   stated in the code and worth repeating here: the increment effect is treated as scaling
   linearly with seconds available (an extrapolation -- the original estimate came from a fixed
   per-move bonus compounding over a whole game, not a one-off reallocation), reducing opening
   time is assumed not to raise opening blunder risk (justified by the placebo/parallel-trends
   result: clock differences haven't yet mattered that early), and the player's own move counts
   per bucket are held fixed. If the empirical seconds-per-move gap in the middlegame bucket is
   too small (< 1 second) to divide by reliably, the counterfactual is reported as unavailable
   for that player rather than as a number that looks precise but isn't.

Run with `python src/report.py <username>`. Output: `outputs/figures/player_report_{username}.png`
plus the four panels' underlying tables printed to the console.
