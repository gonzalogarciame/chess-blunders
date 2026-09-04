"""
Set 1 -- pull a filtered game-level table from Lichess/standard-chess-games (HF, hive-partitioned
by year/month) for two temporally separated months, and write data/raw/games_{month}.parquet.

Strategy: [%eval] is a hard requirement and only ~6% of games carry it, so the movetext column
must be decoded for every row in a month regardless -- that network cost is unavoidable. To pay
it exactly once per month, pass 1 scans the remote parquet shards with a single WHERE (movetext
contains '[%eval') and writes the survivors, plus boolean flags for every later filter step, to
local intermediate parquet files. All sequential filter counts and the time-control bucket counts
are then computed locally from those files (fast, no network), and a second, narrow query slices
out the final columns for the chosen time-control pair.

Pass 1 processes shards in small batches (rather than one glob covering the whole month) so
progress is visible and a crash or a Hugging Face rate-limit stall only costs one batch, not the
whole month: each batch is written to its own file and already-completed batches are skipped on
a re-run.
"""

import json
import re
import time
import urllib.request
from pathlib import Path

import duckdb

HF_BASE = "hf://datasets/Lichess/standard-chess-games/data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

MONTH_A = (2024, 3)  # train/val
MONTH_B = (2024, 9)  # held-out test, 6 months later

TC_PAIRS = [("180+0", "180+2"), ("300+0", "300+3"), ("600+0", "600+5")]
TC_CANDIDATES = [tc for pair in TC_PAIRS for tc in pair]
MIN_BUCKET_GAMES = 15_000

ELO_LO, ELO_HI = 800, 2600
MIN_PLIES = 20

BATCH_SIZE = 15
BATCH_RETRIES = 3

STEP_NAMES = [
    "movetext contains [%eval]",
    "movetext also contains [%clk]",
    "TimeControl in a candidate bucket",
    "neither player is a BOT",
    "both Elo present, 800-2600",
    "Termination is Normal or Time forfeit",
    f"at least {MIN_PLIES} plies",
]


def list_shards(year: int, month: int) -> list[str]:
    url = "https://huggingface.co/api/datasets/Lichess/standard-chess-games"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    prefix = f"data/year={year}/month={month:02d}/train-"
    shards = sorted(s["rfilename"] for s in data["siblings"] if s["rfilename"].startswith(prefix))
    if not shards:
        raise ValueError(f"no shards found for {year}-{month:02d}")
    return [f"{HF_BASE}/year={year}/month={month:02d}/{Path(s).name}" for s in shards]


def month_glob(year: int, month: int) -> str:
    shards = list_shards(year, month)
    total = re.search(r"-of-(\d+)\.parquet$", shards[0]).group(1)
    return f"{HF_BASE}/year={year}/month={month:02d}/train-*-of-{total}.parquet"


def connect() -> duckdb.DuckDBPyConnection:
    # Benchmarked threads=4/8/16 head-to-head: 4 was uniformly slow (~59s/shard), 16 was
    # fast but wildly inconsistent batch-to-batch (rate-limit backoff), 8 was both fast and
    # the most consistent (~33.5s/shard).
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET threads=8;")
    con.execute("SET http_retries=10;")
    con.execute("SET http_retry_wait_ms=2000;")
    con.execute("SET http_retry_backoff=2;")
    return con


def total_scanned(con: duckdb.DuckDBPyConnection, glob: str) -> int:
    return con.execute(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0]


def run_pass1_batch(con: duckdb.DuckDBPyConnection, files: list[str], out_path: Path) -> None:
    flist = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    tc_list = ", ".join(f"'{tc}'" for tc in TC_CANDIDATES)
    tmp_path = out_path.with_suffix(".tmp.parquet")
    query = f"""
    COPY (
        SELECT
            string_split(Site, '/')[-1] AS game_id,
            White AS white,
            Black AS black,
            WhiteElo AS white_elo,
            BlackElo AS black_elo,
            TimeControl AS time_control,
            Result AS result,
            Termination AS termination,
            UTCDate AS utc_date,
            Opening AS opening,
            movetext,
            contains(movetext, '[%clk') AS clk_ok,
            TimeControl IN ({tc_list}) AS tc_candidate_ok,
            (WhiteTitle IS NULL OR WhiteTitle != 'BOT')
                AND (BlackTitle IS NULL OR BlackTitle != 'BOT') AS bot_ok,
            (WhiteElo IS NOT NULL AND BlackElo IS NOT NULL
                AND WhiteElo BETWEEN {ELO_LO} AND {ELO_HI}
                AND BlackElo BETWEEN {ELO_LO} AND {ELO_HI}) AS elo_ok,
            (Termination IN ('Normal', 'Time forfeit')) AS term_ok,
            (len(string_split(movetext, '[%clk')) - 1) AS ply_count
        FROM read_parquet({flist})
        WHERE contains(movetext, '[%eval')
    ) TO '{tmp_path}' (FORMAT PARQUET);
    """
    con.execute(query)
    tmp_path.replace(out_path)  # atomic-ish: a killed/crashed run never leaves a half-written batch


def run_pass1(con: duckdb.DuckDBPyConnection, files: list[str], batch_dir: Path) -> Path:
    batch_dir.mkdir(parents=True, exist_ok=True)
    batches = [files[i:i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    n_batches = len(batches)

    for i, batch_files in enumerate(batches):
        out_path = batch_dir / f"batch_{i:04d}.parquet"
        if out_path.exists():
            print(f"  batch {i + 1}/{n_batches}: already done, skipping", flush=True)
            continue

        t0 = time.time()
        for attempt in range(1, BATCH_RETRIES + 1):
            try:
                run_pass1_batch(con, batch_files, out_path)
                break
            except Exception as e:
                print(f"  batch {i + 1}/{n_batches}: attempt {attempt} failed ({e!r})", flush=True)
                if attempt == BATCH_RETRIES:
                    raise

        rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
        print(f"  batch {i + 1}/{n_batches}: {len(batch_files)} shards, {rows:,} rows, "
              f"{time.time() - t0:.0f}s", flush=True)

    return batch_dir


def sequential_counts(con: duckdb.DuckDBPyConnection, batch_dir: Path) -> list[int]:
    src = f"read_parquet('{batch_dir}/*.parquet')"
    counts = []
    counts.append(con.execute(f"SELECT count(*) FROM {src}").fetchone()[0])  # eval_ok (WHERE already applied)
    counts.append(con.execute(f"SELECT count(*) FROM {src} WHERE clk_ok").fetchone()[0])
    counts.append(con.execute(f"SELECT count(*) FROM {src} WHERE clk_ok AND tc_candidate_ok").fetchone()[0])
    counts.append(con.execute(
        f"SELECT count(*) FROM {src} WHERE clk_ok AND tc_candidate_ok AND bot_ok").fetchone()[0])
    counts.append(con.execute(
        f"SELECT count(*) FROM {src} WHERE clk_ok AND tc_candidate_ok AND bot_ok AND elo_ok").fetchone()[0])
    counts.append(con.execute(
        f"SELECT count(*) FROM {src} WHERE clk_ok AND tc_candidate_ok AND bot_ok AND elo_ok AND term_ok"
    ).fetchone()[0])
    counts.append(con.execute(
        f"SELECT count(*) FROM {src} WHERE clk_ok AND tc_candidate_ok AND bot_ok AND elo_ok AND term_ok "
        f"AND ply_count >= {MIN_PLIES}"
    ).fetchone()[0])
    return counts


def bucket_counts(con: duckdb.DuckDBPyConnection, batch_dir: Path) -> dict[str, int]:
    src = f"read_parquet('{batch_dir}/*.parquet')"
    rows = con.execute(f"""
        SELECT time_control, count(*) FROM {src}
        WHERE clk_ok AND tc_candidate_ok AND bot_ok AND elo_ok AND term_ok AND ply_count >= {MIN_PLIES}
        GROUP BY time_control
    """).fetchall()
    return {tc: n for tc, n in rows}


def choose_tc_pair(bucket_a: dict[str, int], bucket_b: dict[str, int]) -> tuple[str, str]:
    for lo, hi in TC_PAIRS:
        min_count = min(
            bucket_a.get(lo, 0), bucket_a.get(hi, 0),
            bucket_b.get(lo, 0), bucket_b.get(hi, 0),
        )
        if min_count >= MIN_BUCKET_GAMES:
            return lo, hi
    print("WARNING: no candidate pair reached the 15,000-game threshold in both months; "
          "falling back to the slowest pair (600+0/600+5) regardless.")
    return TC_PAIRS[-1]


def write_final(con: duckdb.DuckDBPyConnection, batch_dir: Path, chosen_pair: tuple[str, str],
                 out_path: Path) -> None:
    src = f"read_parquet('{batch_dir}/*.parquet')"
    lo, hi = chosen_pair
    con.execute(f"""
        COPY (
            SELECT game_id, white, black, white_elo, black_elo, time_control, result,
                   termination, utc_date, opening, movetext
            FROM {src}
            WHERE clk_ok AND bot_ok AND elo_ok AND term_ok AND ply_count >= {MIN_PLIES}
                AND time_control IN ('{lo}', '{hi}')
        ) TO '{out_path}' (FORMAT PARQUET);
    """)


def process_month(con: duckdb.DuckDBPyConnection, year: int, month: int) -> dict:
    label = f"{year}-{month:02d}"
    print(f"\n=== {label} ===", flush=True)

    files = list_shards(year, month)
    print(f"{len(files)} shards found", flush=True)

    t0 = time.time()
    glob = month_glob(year, month)
    scanned = total_scanned(con, glob)
    print(f"total games scanned: {scanned:,}  ({time.time() - t0:.0f}s)", flush=True)

    batch_dir = RAW_DIR / f"_batches_{label}"
    t0 = time.time()
    run_pass1(con, files, batch_dir)
    print(f"pass 1 (eval-filter + flags) complete  ({time.time() - t0:.0f}s)", flush=True)

    counts = sequential_counts(con, batch_dir)
    print("survivors by filter step:")
    running = scanned
    for name, c in zip(STEP_NAMES, counts):
        print(f"  scanned {running:,} -> {name}: {c:,}")
        running = c
    print(f"final count (all filters, any candidate TimeControl): {counts[-1]:,}")

    buckets = bucket_counts(con, batch_dir)
    print("candidate TimeControl bucket counts (post all other filters):")
    for lo, hi in TC_PAIRS:
        print(f"  {lo}: {buckets.get(lo, 0):,}   {hi}: {buckets.get(hi, 0):,}")

    return {"label": label, "scanned": scanned, "counts": counts, "buckets": buckets,
            "batch_dir": batch_dir}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()

    result_a = process_month(con, *MONTH_A)
    result_b = process_month(con, *MONTH_B)

    chosen_pair = choose_tc_pair(result_a["buckets"], result_b["buckets"])
    print(f"\nchosen TimeControl pair: {chosen_pair[0]} / {chosen_pair[1]}")

    for result in (result_a, result_b):
        out_path = RAW_DIR / f"games_{result['label']}.parquet"
        write_final(con, result["batch_dir"], chosen_pair, out_path)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
        print(f"\n{result['label']}: wrote {n:,} games to {out_path}")

        samples = con.execute(
            f"SELECT movetext FROM read_parquet('{out_path}') LIMIT 3"
        ).fetchall()
        for i, (mv,) in enumerate(samples):
            print(f"  sample {i}: {mv[:400]}")

        for f in result["batch_dir"].glob("*.parquet"):
            f.unlink()
        result["batch_dir"].rmdir()


if __name__ == "__main__":
    main()
