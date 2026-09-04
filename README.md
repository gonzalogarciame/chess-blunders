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
