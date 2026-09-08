"""
Set 8 (part 2) -- build the interactive blunder trainer from motifs.py's output.

Takes data/processed/blunder_motifs_gonzalopelotas.json (one leak-category blunder per row,
already tagged with a motif group and the engine's "acceptable" replies) plus the raw game
movetext, and writes a single self-contained HTML page to outputs/trainer/. Each puzzle:
step through the lead-up, then try to find the move on the board -- wrong tries just say "not
that one, try again"; the answer is only shown once you find it or ask for it. On reveal you
see the move you actually played, the engine's top lines, your clock at the time, and a link
to the real game on chess.com. Puzzles are grouped by motif so you drill one pattern at a
time.

The page is written as an Artifact-ready fragment (starts at <title>, no <html>/<head>/<body>
wrapper). It loads only chess.js (move legality) from cdnjs; the board, the Cburnett piece
set (vendored under src/assets/cburnett/, inlined as <symbol>s), and all puzzle data ship in
the file. Progress is kept per-device in localStorage.
"""

import io
import json
import re
import sys
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT / "data" / "processed"
RAW_DIR = PROJECT / "data" / "raw"
OUT_DIR = PROJECT / "outputs" / "trainer"
ASSETS = Path(__file__).resolve().parent / "assets" / "cburnett"

USERNAME = "gonzalopelotas"
LEAD_IN_PLIES = 6

# Order the motif groups by how much they're worth working on: forcing tactical patterns
# first (most fixable by a concrete habit), then material drops, then quiet slips.
GROUP_ORDER = [
    "Allowed forced mate", "Back-rank tactic", "Missed a fork",
    "Hung the queen", "Hung a rook", "Hung a piece",
    "Lost the exchange", "Dropped a pawn", "Positional slip (no material)",
]

# symbol id -> Cburnett filename stem (l=light/white, d=dark/black, t=the "t" variant)
PIECE_FILES = {
    "wk": "klt", "wq": "qlt", "wr": "rlt", "wb": "blt", "wn": "nlt", "wp": "plt",
    "bk": "kdt", "bq": "qdt", "br": "rdt", "bb": "bdt", "bn": "ndt", "bp": "pdt",
}


def load_motifs() -> list[dict]:
    path = PROCESSED_DIR / f"blunder_motifs_{USERNAME}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python src/motifs.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def game_movetext_lookup() -> dict[str, str]:
    g = pd.read_parquet(RAW_DIR / "games_gonzalopelotas.parquet")
    return dict(zip(g["game_id"].astype(str), g["movetext"]))


def piece_defs() -> str:
    """The 12 Cburnett SVGs, inner markup only, as <symbol>s in one hidden <svg>."""
    syms = []
    for sid, stem in PIECE_FILES.items():
        raw = (ASSETS / f"{stem}.svg").read_text(encoding="utf-8")
        inner = re.search(r"<svg[^>]*>(.*)</svg>", raw, re.S).group(1).strip()
        syms.append(f'<symbol id="pc-{sid}" viewBox="0 0 45 45">{inner}</symbol>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" style="position:absolute;width:0;height:0;'
            'overflow:hidden" aria-hidden="true"><defs>' + "".join(syms) + "</defs></svg>")


def lead_in(movetext: str, ply: int, n: int) -> list[dict]:
    """The last `n` moves played before the puzzle position, each with the resulting FEN.
    The final entry's FEN is the puzzle position itself."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return []
    board = game.board()
    hist = []
    for i, node in enumerate(game.mainline(), start=1):
        san = board.san(node.move)
        board.push(node.move)
        hist.append({"san": san, "fen": board.fen(), "ply": i})
        if i >= ply - 1:
            break
    return hist[-n:]


def fmt_clock(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def build_puzzle(rec: dict, movetext: str | None) -> dict:
    game_id = str(rec["game_id"])
    return {
        "id": f"{game_id}-{rec['ply']}",
        "group": rec["motif_group"],
        "fen": rec["fen_before"],
        "orientation": rec["mover_color"],
        "acceptable": rec["acceptable_moves"],
        "played": rec["played"],
        "bestLineSan": rec["best_line_san"],
        "solutionFens": rec["best_line_fens"],
        "topLines": rec["top_lines"],
        "leadIn": lead_in(movetext, int(rec["ply"]), LEAD_IN_PLIES) if movetext else [],
        "meta": {
            "wpLoss": round(rec["wp_loss"]),
            "clock": fmt_clock(rec["clock_before"]),
            "increment": rec["increment"],
            "date": str(rec["utc_date"])[:10],
            "opening": rec["opening"] or "Unknown opening",
            "phase": rec["phase"],
            "advantage": rec["advantage_state"],
            "material": rec["material_label"],
            "refutation": rec["refutation_type"].replace("_", " "),
            "yourColor": rec["mover_color"],
            "gameUrl": f"https://www.chess.com/game/live/{game_id}",
            "leakTags": [t.replace("_", " ") for t in rec["leak_tags"]],
        },
    }


def group_summary(puzzles: list[dict]) -> list[dict]:
    by_group: dict[str, int] = {}
    for p in puzzles:
        by_group[p["group"]] = by_group.get(p["group"], 0) + 1
    ordered = [g for g in GROUP_ORDER if g in by_group]
    ordered += [g for g in by_group if g not in GROUP_ORDER]
    return [{"name": g, "count": by_group[g]} for g in ordered]


def build_page(puzzles: list[dict], groups: list[dict]) -> str:
    blob = json.dumps({"puzzles": puzzles, "groups": groups}, separators=(",", ":"))
    return (_TEMPLATE
            .replace("<!--__PIECES__-->", piece_defs())
            .replace("/*__DATA__*/", "window.__TRAINER__ = " + blob + ";"))


def main() -> None:
    motifs = load_motifs()
    movetexts = game_movetext_lookup()
    puzzles = [build_puzzle(r, movetexts.get(str(r["game_id"]))) for r in motifs]
    groups = group_summary(puzzles)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "blunder_trainer.html"
    out_path.write_text(build_page(puzzles, groups), encoding="utf-8")

    print(f"wrote {out_path}  ({len(puzzles)} puzzles across {len(groups)} motif groups)")
    for g in groups:
        print(f"  {g['name']:<32} {g['count']}")
    print("\nnext: publish outputs/trainer/blunder_trainer.html as an Artifact.")


_TEMPLATE = r"""<title>Blunder Trainer</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  :root {
    --bg: #eceef0;
    --surface: #ffffff;
    --surface-2: #f3f6f5;
    --ink: #1a1f1d;
    --ink-soft: #55605b;
    --ink-faint: #8a938f;
    --line: #d7ddda;
    --accent: #2f7d63;
    --accent-ink: #1c5344;
    --solved: #2f7d63;
    --missed: #bb4632;
    --retry: #b9852b;
    --move-hi: #e6c15a;
    --sq-light: #e3e7e2;
    --sq-dark: #6e9184;
    --shadow: 0 1px 2px rgba(20, 35, 30, .05), 0 8px 26px rgba(20, 35, 30, .08);
  }
  :root:not([data-theme="light"]) { color-scheme: light dark; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #111513; --surface: #191f1c; --surface-2: #202722;
      --ink: #e7ebe8; --ink-soft: #a2aca7; --ink-faint: #6e7a75; --line: #2c3531;
      --accent: #5cbf9f; --accent-ink: #93dcc5;
      --solved: #5cbf9f; --missed: #e0715a; --retry: #d7a24e; --move-hi: #c99f3f;
      --sq-light: #a9b7b0; --sq-dark: #47635a;
      --shadow: 0 1px 2px rgba(0, 0, 0, .3), 0 12px 34px rgba(0, 0, 0, .4);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #111513; --surface: #191f1c; --surface-2: #202722;
    --ink: #e7ebe8; --ink-soft: #a2aca7; --ink-faint: #6e7a75; --line: #2c3531;
    --accent: #5cbf9f; --accent-ink: #93dcc5;
    --solved: #5cbf9f; --missed: #e0715a; --retry: #d7a24e; --move-hi: #c99f3f;
    --sq-light: #a9b7b0; --sq-dark: #47635a;
    --shadow: 0 1px 2px rgba(0, 0, 0, .3), 0 12px 34px rgba(0, 0, 0, .4);
  }

  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
    line-height: 1.5; -webkit-font-smoothing: antialiased;
  }
  h1, h2, h3 { font-family: "Fraunces", Georgia, serif; font-weight: 600; margin: 0; text-wrap: balance; }
  a { color: var(--accent-ink); }
  button {
    font: inherit; cursor: pointer; border: 1px solid var(--line);
    background: var(--surface); color: var(--ink); border-radius: 8px;
    padding: 8px 14px; transition: border-color .15s, background .15s, color .15s;
  }
  button:hover { border-color: var(--accent); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .primary { border-color: var(--accent); background: var(--accent); color: #fff; }
  .primary:hover { filter: brightness(1.06); border-color: var(--accent); }

  .app { max-width: 1200px; margin: 0 auto; padding: 24px clamp(12px, 3vw, 32px) 64px; }

  .topbar {
    display: flex; align-items: flex-end; justify-content: space-between;
    flex-wrap: wrap; gap: 14px; padding-bottom: 18px; border-bottom: 1px solid var(--line);
  }
  .brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .brand .mark { width: 30px; height: 30px; align-self: center; }
  .brand h1 { font-size: clamp(22px, 3.4vw, 30px); letter-spacing: -.01em; }
  .brand .sub { margin: 0; color: var(--ink-soft); font-size: 13.5px; flex-basis: 100%; }
  .scorebox { display: flex; align-items: center; gap: 14px; }
  .scorebox .tally { font-family: "IBM Plex Mono", monospace; font-size: 15px; color: var(--ink-soft); }
  .scorebox .tally b { color: var(--ink); font-weight: 500; }

  .body { display: grid; grid-template-columns: 250px 1fr; gap: 24px; margin-top: 22px; }
  @media (max-width: 900px) { .body { grid-template-columns: 1fr; } }

  .rail { display: flex; flex-direction: column; gap: 6px; align-content: start; }
  @media (max-width: 900px) {
    .rail { flex-direction: row; overflow-x: auto; padding-bottom: 6px; }
    .rail .group { min-width: 194px; flex: 0 0 auto; }
    .rail h2 { display: none; }
  }
  .rail h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: .09em; color: var(--ink-faint);
    font-family: "IBM Plex Sans", sans-serif; font-weight: 600; margin: 2px 4px 6px;
  }
  .group {
    text-align: left; border: 1px solid var(--line); border-radius: 10px;
    padding: 10px 12px; display: grid; gap: 7px; background: var(--surface);
  }
  .group[aria-current="true"] { border-color: var(--accent); background: var(--surface-2); }
  .group .g-top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .group .g-name { font-weight: 500; font-size: 13.5px; }
  .group .g-count { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-faint); white-space: nowrap; }
  .meter { height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
  .meter > i { display: block; height: 100%; background: var(--solved); width: 0; transition: width .3s; }

  .stage { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
  .puzzle {
    display: grid; grid-template-columns: auto 1fr; gap: 26px;
    background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
    padding: 22px; box-shadow: var(--shadow);
  }
  @media (max-width: 760px) { .puzzle { grid-template-columns: 1fr; gap: 18px; } }

  .board-wrap { position: relative; width: min(72vw, 496px); aspect-ratio: 1; align-self: start; }
  @media (max-width: 760px) { .board-wrap { width: 100%; max-width: 460px; margin: 0 auto; } }
  .board {
    position: absolute; inset: 0;
    display: grid; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr);
    border-radius: 7px; overflow: hidden; border: 1px solid var(--line); user-select: none;
  }
  .board.locked { cursor: default; }
  .sq { position: relative; display: flex; align-items: center; justify-content: center; }
  .sq.l { background: var(--sq-light); }
  .sq.d { background: var(--sq-dark); }
  .sq .coord {
    position: absolute; font-size: 9.5px; font-family: "IBM Plex Mono", monospace;
    font-weight: 500; opacity: .5;
  }
  .sq.l .coord { color: #46554e; }
  .sq.d .coord { color: #eef2ef; }
  .sq .coord.f { right: 3px; bottom: 1px; }
  .sq .coord.r { left: 3px; top: 1px; }
  .sq.sel { box-shadow: inset 0 0 0 4px var(--accent); }
  .sq.hi { box-shadow: inset 0 0 0 4px var(--move-hi); }
  .sq.bad { box-shadow: inset 0 0 0 4px var(--missed); }
  .sq.target::before {
    content: ""; position: absolute; width: 30%; height: 30%; border-radius: 50%;
    background: var(--accent); opacity: .45;
  }
  .sq.target.cap::before {
    width: 84%; height: 84%; background: transparent;
    border: 4px solid var(--accent); opacity: .5;
  }
  .pc {
    width: 92%; height: 92%; position: relative; z-index: 1; pointer-events: none;
    filter: drop-shadow(0 1.5px 1.5px rgba(0, 0, 0, .32));
  }
  .board-overlay { position: absolute; inset: 0; pointer-events: none; z-index: 3; }
  #overlay line { stroke: var(--move-hi); }
  #overlay marker path { fill: var(--move-hi); }

  .panel { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .prompt { font-family: "Fraunces", Georgia, serif; font-size: 18px; line-height: 1.4; }
  .prompt .side {
    display: inline-flex; align-items: center; gap: 6px; color: var(--accent-ink); font-weight: 600;
  }
  .prompt .side::before {
    content: ""; width: 12px; height: 12px; border-radius: 50%;
    background: var(--dot, #fff); border: 1px solid var(--ink-faint);
  }
  .prompt.black .side { --dot: #1c211f; }

  .feedback {
    border-radius: 11px; padding: 13px 14px; border: 1px solid var(--line);
    background: var(--surface-2); font-size: 13.5px; display: grid; gap: 9px;
  }
  .feedback[data-kind="solved"] { border-color: var(--solved); }
  .feedback[data-kind="revealed"] { border-color: var(--missed); }
  .feedback[data-kind="retry"] { border-color: var(--retry); }
  .feedback .verdict { font-weight: 600; }
  .feedback[data-kind="solved"] .verdict { color: var(--solved); }
  .feedback[data-kind="revealed"] .verdict { color: var(--missed); }
  .feedback[data-kind="retry"] .verdict { color: var(--retry); }
  .feedback .lines {
    font-family: "IBM Plex Mono", monospace; font-size: 12px; display: grid; gap: 3px;
    color: var(--ink-soft); overflow-x: auto;
  }
  .feedback .lines b { color: var(--ink); font-weight: 500; }
  .feedback .note { color: var(--ink-faint); }

  .controls { display: flex; flex-wrap: wrap; gap: 8px; }

  .meta { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 5px 14px; font-size: 12.5px; }
  .meta dt { color: var(--ink-faint); }
  .meta dd { margin: 0; font-family: "IBM Plex Mono", monospace; }
  .meta .tags { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
  .meta .tags span {
    font-family: "IBM Plex Sans", sans-serif; font-size: 11px; border: 1px solid var(--line);
    border-radius: 999px; padding: 1px 8px; color: var(--ink-soft);
  }

  .gamelink {
    font-size: 13px; text-decoration: none; border-bottom: 1px solid currentColor;
    align-self: start; padding-bottom: 1px;
  }
  .gamelink:hover { color: var(--accent); }

  .progress {
    display: flex; align-items: center; gap: 12px; font-size: 12.5px; color: var(--ink-soft);
    font-family: "IBM Plex Mono", monospace;
  }
  .progress .track { flex: 1; height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; }
  .progress .track > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .3s; }

  .empty { padding: 46px 20px; text-align: center; color: var(--ink-soft); }

  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<!--__PIECES__-->

<div class="app">
  <header class="topbar">
    <div class="brand">
      <svg class="mark" viewBox="0 0 45 45" aria-hidden="true"><use href="#pc-bn"></use></svg>
      <h1>Blunder Trainer</h1>
      <p class="sub">gonzalopelotas &middot; your own leak-category mistakes, one motif at a time</p>
    </div>
    <div class="scorebox">
      <span class="tally"><b id="solvedCount">0</b> / <span id="totalCount">0</span> solved</span>
      <button id="resetBtn" type="button">Reset progress</button>
    </div>
  </header>

  <div class="body">
    <nav class="rail" id="rail" aria-label="Motif groups">
      <h2>Motif groups</h2>
    </nav>

    <div class="stage">
      <section class="puzzle" id="puzzle">
        <div class="board-wrap">
          <div class="board" id="board"></div>
          <svg class="board-overlay" id="overlay" viewBox="0 0 80 80" aria-hidden="true"></svg>
        </div>
        <div class="panel">
          <div class="prompt" id="prompt"></div>
          <div class="feedback" id="feedback" hidden></div>
          <div class="controls">
            <button id="leadinBtn" type="button">Show the lead-up</button>
            <button id="hintBtn" type="button" hidden>Hint</button>
            <button id="revealBtn" type="button">Show answer</button>
            <button id="lineBtn" type="button" hidden>Step through the line</button>
            <button id="skipBtn" type="button">Skip</button>
            <button id="nextBtn" class="primary" type="button" hidden>Next &rarr;</button>
          </div>
          <dl class="meta" id="meta"></dl>
          <a class="gamelink" id="gameLink" target="_blank" rel="noopener">Replay this game on chess.com &#8599;</a>
        </div>
      </section>
      <div class="progress">
        <div class="track"><i id="progressFill"></i></div>
        <span id="progressText"></span>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.12.1/chess.min.js"></script>
<script>
/*__DATA__*/
(function () {
  "use strict";
  var DATA = window.__TRAINER__ || { puzzles: [], groups: [] };
  var HAS_CHESS = typeof Chess !== "undefined";  // chess.js from cdnjs; degrade gently if absent
  var STORE_KEY = "blunder-trainer-v1";
  var PIECE_NAME = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };

  var progress = loadProgress();
  var byGroup = {};
  DATA.puzzles.forEach(function (p) { (byGroup[p.group] = byGroup[p.group] || []).push(p); });

  var state = {
    group: null, queue: [], current: null,
    selected: null, phase: "solving", tries: 0, leadStep: null, lineStep: 0
  };

  var el = {};
  ["rail", "board", "overlay", "prompt", "feedback", "meta", "gameLink", "leadinBtn",
   "hintBtn", "revealBtn", "lineBtn", "skipBtn", "nextBtn", "solvedCount", "totalCount",
   "resetBtn", "progressFill", "progressText"].forEach(function (id) {
    el[id] = document.getElementById(id);
  });

  function loadProgress() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch (e) { return {}; }
  }
  function saveProgress() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(progress)); } catch (e) {}
  }
  function rec(id) { return progress[id] || { solved: false, seen: false, attempts: 0 }; }
  function putRec(id, r) { progress[id] = r; saveProgress(); }

  function groupStats(name) {
    var list = byGroup[name] || [];
    return {
      total: list.length,
      solved: list.filter(function (p) { return rec(p.id).solved; }).length
    };
  }

  // ---- rail --------------------------------------------------------------
  function renderRail() {
    el.rail.querySelectorAll(".group").forEach(function (n) { n.remove(); });
    DATA.groups.forEach(function (g) {
      var s = groupStats(g.name);
      var btn = document.createElement("button");
      btn.className = "group";
      btn.type = "button";
      btn.setAttribute("aria-current", state.group === g.name ? "true" : "false");
      btn.innerHTML =
        '<span class="g-top"><span class="g-name"></span>' +
        '<span class="g-count">' + s.solved + '/' + s.total + '</span></span>' +
        '<span class="meter"><i style="width:' + (s.total ? s.solved / s.total * 100 : 0) + '%"></i></span>';
      btn.querySelector(".g-name").textContent = g.name;
      btn.addEventListener("click", function () { selectGroup(g.name); });
      el.rail.appendChild(btn);
    });
    el.solvedCount.textContent = DATA.puzzles.filter(function (p) { return rec(p.id).solved; }).length;
    el.totalCount.textContent = DATA.puzzles.length;
  }

  function selectGroup(name) {
    state.group = name;
    var list = (byGroup[name] || []).slice();
    var unsolved = list.filter(function (p) { return !rec(p.id).solved; });
    var solved = list.filter(function (p) { return rec(p.id).solved; });
    state.queue = unsolved.concat(solved);   // data is pre-sorted by win-prob lost
    renderRail();
    nextPuzzle();
  }

  function nextPuzzle() {
    if (!state.queue.length) { showDone(); return; }
    state.current = state.queue.shift();
    startPuzzle(state.current);
    updateProgress();
  }

  function showDone() {
    state.current = null;
    el.board.innerHTML = '<div class="empty">Every puzzle in this group is solved.<br>Pick another group, or Reset progress to run them again.</div>';
    el.overlay.innerHTML = "";
    el.prompt.textContent = "";
    el.feedback.hidden = true;
    el.meta.innerHTML = "";
    ["gameLink", "leadinBtn", "hintBtn", "revealBtn", "lineBtn", "skipBtn", "nextBtn"]
      .forEach(function (k) { el[k].hidden = true; });
  }

  // ---- board rendering --------------------------------------------------
  function files(o) { return o === "white" ? "abcdefgh" : "hgfedcba"; }
  function ranks(o) { return o === "white" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8]; }

  function parseFen(fen) {
    var rows = fen.split(" ")[0].split("/"), map = {};
    for (var r = 0; r < 8; r++) {
      var rank = 8 - r, fileIdx = 0;
      for (var c = 0; c < rows[r].length; c++) {
        var ch = rows[r][c];
        if (/\d/.test(ch)) { fileIdx += +ch; continue; }
        map["abcdefgh"[fileIdx] + rank] =
          { color: ch === ch.toUpperCase() ? "w" : "b", type: ch.toLowerCase() };
        fileIdx++;
      }
    }
    return map;
  }

  function drawBoard(fen, orient, highlights) {
    var map = parseFen(fen), ff = files(orient), rr = ranks(orient);
    el.board.innerHTML = "";
    for (var ri = 0; ri < 8; ri++) {
      for (var fi = 0; fi < 8; fi++) {
        var sq = ff[fi] + rr[ri];
        var d = document.createElement("div");
        var dark = ("abcdefgh".indexOf(ff[fi]) + rr[ri]) % 2 === 1;  // a1 dark: 0 + 1 is odd
        d.className = "sq " + (dark ? "d" : "l");
        if (highlights && highlights[sq]) d.className += " " + highlights[sq];
        d.dataset.square = sq;
        if (fi === 0) d.insertAdjacentHTML("beforeend", '<span class="coord r">' + rr[ri] + '</span>');
        if (ri === 7) d.insertAdjacentHTML("beforeend", '<span class="coord f">' + ff[fi] + '</span>');
        var pc = map[sq];
        if (pc) {
          d.insertAdjacentHTML("beforeend",
            '<svg class="pc" viewBox="0 0 45 45"><use href="#pc-' + pc.color + pc.type + '"></use></svg>');
        }
        d.addEventListener("click", onSquareClick);
        el.board.appendChild(d);
      }
    }
    el.board.classList.toggle("locked", state.phase !== "solving");
  }

  // ---- puzzle lifecycle ----------------------------------------------
  function startPuzzle(p) {
    state.phase = "solving";
    state.tries = 0;
    state.selected = null;
    state.leadStep = null;
    state.lineStep = 0;

    drawBoard(p.fen, p.orientation);
    el.overlay.innerHTML = "";

    var black = p.orientation === "black";
    el.prompt.className = "prompt" + (black ? " black" : "");
    el.prompt.innerHTML = '<span class="side">' + (black ? "Black" : "White") +
      ' to move</span> &mdash; you blundered here. Find the move you missed.';

    el.feedback.hidden = true;
    el.gameLink.hidden = false;
    el.gameLink.href = p.meta.gameUrl;
    el.leadinBtn.hidden = !(p.leadIn && p.leadIn.length);
    el.leadinBtn.textContent = "Show the lead-up";
    el.hintBtn.hidden = true;
    el.revealBtn.hidden = false;
    el.lineBtn.hidden = true;
    el.skipBtn.hidden = false;
    el.nextBtn.hidden = true;

    renderMeta(p, false);
  }

  function renderMeta(p, revealed) {
    var m = p.meta;
    var rows = [
      ["Your clock", m.clock + (m.increment ? " (+" + m.increment + ")" : "")],
      ["Game phase", m.phase],
      ["Win prob. lost", "~" + m.wpLoss + " pts"],
      ["Date", m.date],
      ["Opening", m.opening]
    ];
    if (revealed) {
      rows.push(["What happened", m.advantage]);
      if (m.material !== "positional") rows.push(["Material", m.material]);
      rows.push(["Refutation", m.refutation]);
    }
    var html = rows.map(function (r) {
      return "<dt>" + r[0] + "</dt><dd>" + escapeHtml(r[1]) + "</dd>";
    }).join("");
    if (m.leakTags && m.leakTags.length) {
      html += '<div class="tags">' + m.leakTags.map(function (t) {
        return "<span>" + escapeHtml(t) + "</span>";
      }).join("") + "</div>";
    }
    el.meta.innerHTML = html;
  }

  // ---- interaction ---------------------------------------------------
  function onSquareClick(e) {
    if (!state.current || state.phase !== "solving") return;
    var sq = e.currentTarget.dataset.square;
    var p = state.current;
    var turn = p.orientation === "white" ? "w" : "b";

    if (state.selected && state.selected !== sq) {
      var from = state.selected;
      clearMarks();
      state.selected = null;
      attemptMove(from, sq);
      return;
    }
    clearMarks();
    state.selected = null;

    var occ = HAS_CHESS ? new Chess(p.fen).get(sq) : parseFen(p.fen)[sq];
    if (occ && occ.color === turn) {
      state.selected = sq;
      markSquare(sq, "sel");
      if (HAS_CHESS) {
        new Chess(p.fen).moves({ square: sq, verbose: true }).forEach(function (mv) {
          var t = el.board.querySelector('[data-square="' + mv.to + '"]');
          if (!t) return;
          t.classList.add("target");
          if (mv.flags.indexOf("c") > -1 || mv.flags.indexOf("e") > -1) t.classList.add("cap");
        });
      }
    }
  }

  function markSquare(sq, cls) {
    var n = el.board.querySelector('[data-square="' + sq + '"]');
    if (n) n.classList.add(cls);
  }
  function clearMarks() {
    el.board.querySelectorAll(".sel,.target,.cap,.bad").forEach(function (n) {
      n.classList.remove("sel", "target", "cap", "bad");
    });
  }

  function attemptMove(from, to) {
    var p = state.current;
    var mv;
    if (HAS_CHESS) {
      mv = new Chess(p.fen).move({ from: from, to: to, promotion: "q" });
      if (!mv) return;  // not a legal move -- ignore
    } else {
      mv = { san: from + to };
    }
    var isBest = p.acceptable.some(function (a) { return a.from === from && a.to === to; });
    if (isBest) { solved(mv); return; }

    // wrong -- but don't give it away; let them keep trying
    state.tries += 1;
    var r = rec(p.id); r.attempts += 1; putRec(p.id, r);
    markSquare(to, "bad");
    var wasPlayed = p.played.from === from && p.played.to === to;
    el.feedback.hidden = false;
    el.feedback.dataset.kind = "retry";
    el.feedback.innerHTML =
      '<span class="verdict">' + escapeHtml(mv.san) + " &mdash; not the one" +
        (wasPlayed ? " (that's the move you played in the game)" : "") + ".</span>" +
      "<div>Keep looking, or use <b>Hint</b> / <b>Show answer</b>.</div>";
    if (state.tries >= 2) el.hintBtn.hidden = false;
  }

  function solved(mv) {
    var p = state.current;
    var r = rec(p.id); r.solved = true; putRec(p.id, r);
    reveal("solved", mv);
  }

  el.revealBtn.addEventListener("click", function () {
    if (!state.current || state.phase !== "solving") return;
    var p = state.current;
    var r = rec(p.id); r.seen = true; putRec(p.id, r);
    reveal("revealed", null);
  });

  function reveal(kind, mv) {
    var p = state.current;
    state.phase = "revealed";
    state.selected = null;
    state.lineStep = Math.min(1, (p.solutionFens || []).length - 1);

    drawBoard(solutionFen(state.lineStep), p.orientation, arrowHighlights());
    drawArrow(p.acceptable[0]);
    renderMeta(p, true);

    var best = p.acceptable.map(function (a) { return a.san; }).join(" or ");
    var head = kind === "solved"
      ? '<span class="verdict">&#10003; ' + escapeHtml(mv.san) + " &mdash; that's a top engine move.</span>"
      : '<span class="verdict">The move: <b>' + escapeHtml(best) + "</b>.</span>";
    var lines = p.topLines.map(function (l) {
      return "<div><b>" + escapeHtml(l.eval) + "</b>&nbsp;&nbsp;" + escapeHtml(l.san) + "</div>";
    }).join("");
    el.feedback.hidden = false;
    el.feedback.dataset.kind = kind;
    el.feedback.innerHTML = head +
      "<div>Best line: <b>" + escapeHtml(p.bestLineSan) + "</b></div>" +
      '<div class="lines">' + lines + "</div>" +
      '<div class="note">In the game you played ' + escapeHtml(p.played.san) +
        ", losing about " + p.meta.wpLoss + " points of win probability.</div>";

    el.leadinBtn.hidden = true;
    el.hintBtn.hidden = true;
    el.revealBtn.hidden = true;
    el.skipBtn.hidden = true;
    el.lineBtn.hidden = !(p.solutionFens && p.solutionFens.length > 2);
    el.lineBtn.textContent = "Step through the line";
    el.nextBtn.hidden = false;
    el.nextBtn.focus();

    if (kind !== "solved") {
      var at = Math.min(4, state.queue.length);
      state.queue.splice(at, 0, p);  // see it again later
    }
    renderRail();
    updateProgress();
  }

  function solutionFen(i) {
    var f = state.current.solutionFens;
    if (!f || !f.length) return state.current.fen;
    return f[Math.max(0, Math.min(i, f.length - 1))];
  }
  function arrowHighlights() {
    var a = state.current.acceptable[0];
    if (!a) return null;
    var h = {}; h[a.from] = "hi"; h[a.to] = "hi"; return h;
  }

  // ---- hint --------------------------------------------------------
  el.hintBtn.addEventListener("click", function () {
    var p = state.current;
    if (!p || state.phase !== "solving") return;
    var a = p.acceptable[0];
    var occ = (HAS_CHESS ? new Chess(p.fen).get(a.from) : parseFen(p.fen)[a.from]) || { type: "p" };
    clearMarks();
    markSquare(a.from, "sel");
    el.feedback.hidden = false;
    el.feedback.dataset.kind = "retry";
    el.feedback.innerHTML =
      '<span class="verdict">Hint</span><div>Move your <b>' +
      PIECE_NAME[occ.type] + "</b> from <b>" + a.from + "</b>.</div>";
  });

  // ---- solution line stepper --------------------------------------
  el.lineBtn.addEventListener("click", function () {
    var p = state.current;
    var fens = p.solutionFens || [];
    var sans = (p.bestLineSan || "").split(/\s+/).filter(Boolean);
    state.lineStep += 1;
    if (state.lineStep >= fens.length) state.lineStep = 0;
    drawBoard(solutionFen(state.lineStep), p.orientation,
              state.lineStep === 0 ? arrowHighlights() : null);
    el.overlay.innerHTML = "";
    if (state.lineStep === 0) {
      drawArrow(p.acceptable[0]);
      el.lineBtn.textContent = "Step through the line";
    } else {
      el.lineBtn.textContent = "Next (" + state.lineStep + "/" + (fens.length - 1) + ")  " +
        (sans[state.lineStep - 1] || "");
    }
  });

  // ---- solution arrow -------------------------------------------
  function centre(square, orient) {
    var ff = files(orient), rr = ranks(orient);
    return { x: ff.indexOf(square[0]) * 10 + 5, y: rr.indexOf(+square[1]) * 10 + 5 };
  }
  function drawArrow(move) {
    if (!move) { el.overlay.innerHTML = ""; return; }
    var o = state.current.orientation;
    var a = centre(move.from, o), b = centre(move.to, o);
    el.overlay.innerHTML =
      '<defs><marker id="ah" markerWidth="4.5" markerHeight="4.5" refX="2.6" refY="2.25" orient="auto">' +
      '<path d="M0,0 L4.5,2.25 L0,4.5 Z"></path></marker></defs>' +
      '<line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y +
      '" stroke-width="2" stroke-linecap="round" marker-end="url(#ah)" opacity="0.85"></line>';
  }

  // ---- lead-up stepper -----------------------------------------
  el.leadinBtn.addEventListener("click", function () {
    var p = state.current;
    if (!p || state.phase !== "solving" || !p.leadIn || !p.leadIn.length) return;
    state.leadStep = state.leadStep === null ? 0 : state.leadStep + 1;
    if (state.leadStep >= p.leadIn.length) {
      state.leadStep = null;
      drawBoard(p.fen, p.orientation);
      el.leadinBtn.textContent = "Show the lead-up";
      return;
    }
    var step = p.leadIn[state.leadStep];
    drawBoard(step.fen, p.orientation);
    el.leadinBtn.textContent =
      "Lead-up " + (state.leadStep + 1) + "/" + p.leadIn.length + "  —  " + step.san;
  });

  el.skipBtn.addEventListener("click", function () {
    if (state.current) state.queue.push(state.current);
    nextPuzzle();
  });
  el.nextBtn.addEventListener("click", nextPuzzle);

  el.resetBtn.addEventListener("click", function () {
    if (!confirm("Clear solved / attempted progress on this device?")) return;
    progress = {};
    saveProgress();
    if (state.group) selectGroup(state.group); else renderRail();
  });

  function updateProgress() {
    if (!state.group) return;
    var s = groupStats(state.group);
    el.progressFill.style.width = (s.total ? s.solved / s.total * 100 : 0) + "%";
    el.progressText.textContent = state.group + " — " + s.solved + " / " + s.total + " solved";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // ---- boot -------------------------------------------------
  if (!DATA.puzzles.length) {
    el.board.innerHTML = '<div class="empty">No puzzles found. Run motifs.py then trainer.py.</div>';
    return;
  }
  renderRail();
  selectGroup(DATA.groups[0].name);
})();
</script>
"""

if __name__ == "__main__":
    main()
