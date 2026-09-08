"""
Set 8 (part 2) -- build the interactive blunder trainer from motifs.py's output.

Takes data/processed/blunder_motifs_gonzalopelotas.json (one leak-category blunder per row,
already tagged with a motif group and the engine's "acceptable" replies) plus the raw game
movetext, and writes a single self-contained HTML page to outputs/trainer/. Each puzzle lets
you step through the lead-up, try to find the move on the board, then see the move you
actually played, the engine's lines, your clock at the time, and a link to the real game on
chess.com. Puzzles are grouped by motif so you drill one pattern at a time.

The page is written as an Artifact-ready fragment (starts at <title>, no <html>/<head>/<body>
wrapper) so it can be published with the Artifact tool directly; it also opens fine as a
local file. It loads only chess.js (move legality) from cdnjs -- everything else, pieces
included, ships inline. Progress is kept per-device in localStorage.
"""

import io
import json
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

USERNAME = "gonzalopelotas"
LEAD_IN_PLIES = 6

# Order the motif groups by how much they're worth working on: forcing tactical patterns
# first (most fixable by a concrete habit), then material drops, then quiet slips.
GROUP_ORDER = [
    "Allowed forced mate", "Back-rank tactic", "Missed a fork",
    "Hung the queen", "Hung a rook", "Hung a piece",
    "Lost the exchange", "Dropped a pawn", "Positional slip (no material)",
]


def load_motifs() -> list[dict]:
    path = PROCESSED_DIR / f"blunder_motifs_{USERNAME}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python src/motifs.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def game_movetext_lookup() -> dict[str, str]:
    g = pd.read_parquet(RAW_DIR / "games_gonzalopelotas.parquet")
    return dict(zip(g["game_id"].astype(str), g["movetext"]))


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
    return _TEMPLATE.replace("/*__DATA__*/", "window.__TRAINER__ = " + blob + ";")


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
    --surface-2: #f4f6f6;
    --ink: #1a1f1d;
    --ink-soft: #56605c;
    --ink-faint: #8a938f;
    --line: #d9dedb;
    --accent: #2f7d63;
    --accent-ink: #1c5344;
    --solved: #2f7d63;
    --missed: #bb4632;
    --your-move: #cf9640;
    --sq-light: #dde3df;
    --sq-dark: #6d8d81;
    --sq-sel: #cfa53f;
    --piece-white: #f7f6f2;
    --piece-black: #1c211f;
    --piece-outline: #22312c;
    --shadow: 0 1px 2px rgba(20, 35, 30, .06), 0 6px 20px rgba(20, 35, 30, .07);
  }
  :root:not([data-theme="light"]) {
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #121614;
      --surface: #1a201d;
      --surface-2: #212824;
      --ink: #e7ebe8;
      --ink-soft: #a3ada8;
      --ink-faint: #6f7b76;
      --line: #2e3733;
      --accent: #5bbf9f;
      --accent-ink: #8fd8c1;
      --solved: #5bbf9f;
      --missed: #e0715a;
      --your-move: #e0b463;
      --sq-light: #b7c3bd;
      --sq-dark: #4b665d;
      --sq-sel: #cf9640;
      --piece-white: #f2f1ec;
      --piece-black: #14100e;
      --piece-outline: #05100c;
      --shadow: 0 1px 2px rgba(0, 0, 0, .3), 0 10px 30px rgba(0, 0, 0, .35);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #121614;
    --surface: #1a201d;
    --surface-2: #212824;
    --ink: #e7ebe8;
    --ink-soft: #a3ada8;
    --ink-faint: #6f7b76;
    --line: #2e3733;
    --accent: #5bbf9f;
    --accent-ink: #8fd8c1;
    --solved: #5bbf9f;
    --missed: #e0715a;
    --your-move: #e0b463;
    --sq-light: #b7c3bd;
    --sq-dark: #4b665d;
    --sq-sel: #cf9640;
    --piece-white: #f2f1ec;
    --piece-black: #14100e;
    --piece-outline: #05100c;
    --shadow: 0 1px 2px rgba(0, 0, 0, .3), 0 10px 30px rgba(0, 0, 0, .35);
  }

  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  h1, h2, h3 { font-family: "Fraunces", Georgia, serif; font-weight: 600; margin: 0; text-wrap: balance; }
  a { color: var(--accent-ink); }
  button {
    font: inherit;
    cursor: pointer;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--ink);
    border-radius: 7px;
    padding: 7px 13px;
    transition: border-color .15s, background .15s;
  }
  button:hover { border-color: var(--accent); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  button[hidden] { display: none; }

  .app { max-width: 1180px; margin: 0 auto; padding: 22px clamp(12px, 3vw, 30px) 60px; }

  .topbar {
    display: flex; align-items: flex-end; justify-content: space-between;
    flex-wrap: wrap; gap: 14px;
    padding-bottom: 18px; border-bottom: 1px solid var(--line);
  }
  .brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .brand .mark { font-size: 30px; color: var(--accent); line-height: 1; }
  .brand h1 { font-size: clamp(22px, 3.4vw, 30px); letter-spacing: -.01em; }
  .brand .sub { margin: 0; color: var(--ink-soft); font-size: 13.5px; flex-basis: 100%; }
  .scorebox { text-align: right; display: flex; align-items: center; gap: 14px; }
  .scorebox .tally { font-family: "IBM Plex Mono", monospace; font-size: 15px; color: var(--ink-soft); }
  .scorebox .tally b { color: var(--ink); font-weight: 500; }

  .body { display: grid; grid-template-columns: 248px 1fr; gap: 22px; margin-top: 22px; }
  @media (max-width: 880px) { .body { grid-template-columns: 1fr; } }

  .rail { display: flex; flex-direction: column; gap: 6px; align-content: start; }
  @media (max-width: 880px) {
    .rail { flex-direction: row; overflow-x: auto; padding-bottom: 6px; }
    .rail .group { min-width: 190px; flex: 0 0 auto; }
  }
  .rail h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .09em; color: var(--ink-faint); font-family: "IBM Plex Sans", sans-serif; font-weight: 600; margin: 2px 4px 6px; }
  .group {
    text-align: left; border: 1px solid var(--line); border-radius: 9px;
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
    display: grid; grid-template-columns: auto 1fr; gap: 22px;
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
    padding: 20px; box-shadow: var(--shadow);
  }
  @media (max-width: 720px) { .puzzle { grid-template-columns: 1fr; } }

  .board-wrap { position: relative; width: min(62vw, 424px); aspect-ratio: 1; align-self: start; }
  @media (max-width: 720px) { .board-wrap { width: min(86vw, 420px); margin: 0 auto; } }
  .board {
    position: absolute; inset: 0;
    display: grid; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr);
    border-radius: 6px; overflow: hidden; border: 1px solid var(--line);
    user-select: none;
  }
  .sq { position: relative; display: flex; align-items: center; justify-content: center; }
  .sq.l { background: var(--sq-light); }
  .sq.d { background: var(--sq-dark); }
  .sq .coord { position: absolute; font-size: 9px; font-family: "IBM Plex Mono", monospace; color: rgba(0,0,0,.34); }
  .sq .coord.f { right: 3px; bottom: 2px; }
  .sq .coord.r { left: 3px; top: 2px; }
  .sq.sel::after {
    content: ""; position: absolute; inset: 0; background: var(--sq-sel); opacity: .5;
  }
  .sq.target::before {
    content: ""; position: absolute; width: 30%; height: 30%; border-radius: 50%;
    background: rgba(0,0,0,.22);
  }
  .sq.target.cap::before {
    width: 82%; height: 82%; background: transparent; border: 5px solid rgba(0,0,0,.22);
  }
  .pc {
    font-size: 82%; line-height: 1; position: relative; z-index: 1;
    paint-order: stroke fill;   /* stroke behind the fill so the glyph body stays solid */
    filter: drop-shadow(0 1.5px 1px rgba(0, 0, 0, .3));
  }
  .pc.w { color: var(--piece-white); -webkit-text-stroke: 2.4px var(--piece-outline); }
  .pc.b { color: var(--piece-black); -webkit-text-stroke: 1px rgba(255, 255, 255, .28); }
  .board-overlay { position: absolute; inset: 0; pointer-events: none; z-index: 3; }
  #overlay line { stroke: var(--your-move); }
  #overlay marker path { fill: var(--your-move); }

  .panel { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .prompt { font-family: "Fraunces", Georgia, serif; font-size: 17px; }
  .prompt .side { color: var(--accent-ink); }

  .feedback { border-radius: 10px; padding: 12px 13px; border: 1px solid var(--line); background: var(--surface-2); font-size: 13.5px; display: grid; gap: 9px; }
  .feedback[data-kind="solved"] { border-color: var(--solved); }
  .feedback[data-kind="missed"] { border-color: var(--missed); }
  .feedback .verdict { font-weight: 600; display: flex; align-items: center; gap: 7px; }
  .feedback[data-kind="solved"] .verdict { color: var(--solved); }
  .feedback[data-kind="missed"] .verdict { color: var(--missed); }
  .feedback .lines { font-family: "IBM Plex Mono", monospace; font-size: 12px; display: grid; gap: 3px; color: var(--ink-soft); }
  .feedback .lines b { color: var(--ink); font-weight: 500; }
  .feedback .replay-note { color: var(--ink-faint); }

  .controls { display: flex; flex-wrap: wrap; gap: 8px; }
  .controls .primary { border-color: var(--accent); background: var(--accent); color: #fff; }
  .controls .primary:hover { filter: brightness(1.05); }

  .meta { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 12.5px; }
  .meta dt { color: var(--ink-faint); }
  .meta dd { margin: 0; font-family: "IBM Plex Mono", monospace; }
  .meta .tags { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 5px; margin-top: 3px; }
  .meta .tags span { font-family: "IBM Plex Sans", sans-serif; font-size: 11px; border: 1px solid var(--line); border-radius: 999px; padding: 1px 8px; color: var(--ink-soft); }

  .gamelink { font-size: 13px; text-decoration: none; border-bottom: 1px solid currentColor; align-self: start; padding-bottom: 1px; }
  .gamelink:hover { color: var(--accent); }

  .progress { display: flex; align-items: center; gap: 12px; font-size: 12.5px; color: var(--ink-soft); font-family: "IBM Plex Mono", monospace; }
  .progress .track { flex: 1; height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; }
  .progress .track > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .3s; }

  .empty { padding: 40px; text-align: center; color: var(--ink-soft); }

  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="mark">&#9822;</span>
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
  var HAS_CHESS = typeof Chess !== "undefined";  // chess.js from cdnjs; degrade if it didn't load
  var STORE_KEY = "blunder-trainer-v1";
  // filled glyphs for both colours; .pc.w / .pc.b fill + outline do the light/dark distinction
  var FIG = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
  var GLYPH = { w: FIG, b: FIG };

  var progress = loadProgress();
  var byGroup = {};
  DATA.puzzles.forEach(function (p) { (byGroup[p.group] = byGroup[p.group] || []).push(p); });

  var state = { group: null, queue: [], current: null, selected: null, leadStep: null };

  var el = {
    rail: document.getElementById("rail"),
    board: document.getElementById("board"),
    overlay: document.getElementById("overlay"),
    prompt: document.getElementById("prompt"),
    feedback: document.getElementById("feedback"),
    meta: document.getElementById("meta"),
    gameLink: document.getElementById("gameLink"),
    leadinBtn: document.getElementById("leadinBtn"),
    skipBtn: document.getElementById("skipBtn"),
    nextBtn: document.getElementById("nextBtn"),
    solvedCount: document.getElementById("solvedCount"),
    totalCount: document.getElementById("totalCount"),
    resetBtn: document.getElementById("resetBtn"),
    progressFill: document.getElementById("progressFill"),
    progressText: document.getElementById("progressText")
  };

  function loadProgress() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; }
    catch (e) { return {}; }
  }
  function saveProgress() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(progress)); } catch (e) {}
  }
  function rec(id) { return progress[id] || { solved: false, attempts: 0 }; }

  function groupStats(name) {
    var list = byGroup[name] || [];
    var solved = list.filter(function (p) { return rec(p.id).solved; }).length;
    return { total: list.length, solved: solved };
  }

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
        '<span class="meter"><i style="width:' + (s.total ? (s.solved / s.total * 100) : 0) + '%"></i></span>';
      btn.querySelector(".g-name").textContent = g.name;
      btn.addEventListener("click", function () { selectGroup(g.name); });
      el.rail.appendChild(btn);
    });
    var totalSolved = DATA.puzzles.filter(function (p) { return rec(p.id).solved; }).length;
    el.solvedCount.textContent = totalSolved;
    el.totalCount.textContent = DATA.puzzles.length;
  }

  function selectGroup(name) {
    state.group = name;
    var list = (byGroup[name] || []).slice();
    // unsolved first, in severity order (data is pre-sorted by win-prob lost); solved appended
    var unsolved = list.filter(function (p) { return !rec(p.id).solved; });
    var solved = list.filter(function (p) { return rec(p.id).solved; });
    state.queue = unsolved.concat(solved);
    renderRail();
    nextPuzzle();
  }

  function nextPuzzle() {
    el.feedback.hidden = true;
    el.nextBtn.hidden = true;
    el.skipBtn.hidden = false;
    state.selected = null;
    state.leadStep = null;
    if (!state.queue.length) { showDone(); return; }
    state.current = state.queue.shift();
    drawPuzzle(state.current);
    updateProgress();
  }

  function showDone() {
    state.current = null;
    el.board.innerHTML = '<div class="empty">Every puzzle in this group is solved. Pick another group, or Reset progress to run them again.</div>';
    el.overlay.innerHTML = "";
    el.prompt.textContent = "";
    el.meta.innerHTML = "";
    el.gameLink.hidden = true;
    el.leadinBtn.hidden = true;
    el.skipBtn.hidden = true;
  }

  // ---- board rendering ----------------------------------------------------
  function files(orient) { return orient === "white" ? "abcdefgh" : "hgfedcba"; }
  function ranks(orient) { return orient === "white" ? [8,7,6,5,4,3,2,1] : [1,2,3,4,5,6,7,8]; }

  function parseFen(fen) {
    var rows = fen.split(" ")[0].split("/");
    var map = {};
    for (var r = 0; r < 8; r++) {
      var rank = 8 - r, fileIdx = 0;
      for (var c = 0; c < rows[r].length; c++) {
        var ch = rows[r][c];
        if (/\d/.test(ch)) { fileIdx += parseInt(ch, 10); continue; }
        var sq = "abcdefgh"[fileIdx] + rank;
        map[sq] = { color: ch === ch.toUpperCase() ? "w" : "b", type: ch.toLowerCase() };
        fileIdx++;
      }
    }
    return map;
  }

  function drawBoard(fen, orient) {
    var map = parseFen(fen);
    var ff = files(orient), rr = ranks(orient);
    el.board.innerHTML = "";
    for (var ri = 0; ri < 8; ri++) {
      for (var fi = 0; fi < 8; fi++) {
        var sq = ff[fi] + rr[ri];
        var d = document.createElement("div");
        var dark = ("abcdefgh".indexOf(ff[fi]) + rr[ri]) % 2 === 1;  // a1 dark: (0 + 1) odd
        d.className = "sq " + (dark ? "d" : "l");
        d.dataset.square = sq;
        if (fi === 0) d.innerHTML += '<span class="coord r">' + rr[ri] + '</span>';
        if (ri === 7) d.innerHTML += '<span class="coord f">' + ff[fi] + '</span>';
        var pc = map[sq];
        if (pc) {
          var s = document.createElement("span");
          s.className = "pc " + pc.color;
          s.textContent = GLYPH[pc.color][pc.type];
          d.appendChild(s);
        }
        d.addEventListener("click", onSquareClick);
        el.board.appendChild(d);
      }
    }
  }

  function drawPuzzle(p) {
    el.gameLink.hidden = false;
    el.leadinBtn.hidden = !(p.leadIn && p.leadIn.length);
    el.leadinBtn.textContent = "Show the lead-up";
    drawBoard(p.fen, p.orientation);
    el.overlay.innerHTML = "";
    var side = p.orientation === "white" ? "White" : "Black";
    el.prompt.innerHTML = '<span class="side">' + side + ' to move.</span> You blundered here — find the move you missed.';
    el.gameLink.href = p.meta.gameUrl;
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

  // ---- interaction ------------------------------------------------------
  function onSquareClick(e) {
    if (!state.current) return;
    var sq = e.currentTarget.dataset.square;
    var p = state.current;
    if (isResolved()) return;

    if (state.selected && state.selected !== sq) {
      attemptMove(state.selected, sq);
      clearSelection();
      return;
    }
    clearSelection();
    var turn = p.orientation === "white" ? "w" : "b";
    if (!HAS_CHESS) {
      var occ = parseFen(p.fen)[sq];
      if (occ && occ.color === turn) { state.selected = sq; e.currentTarget.classList.add("sel"); }
      return;
    }
    var game = new Chess(currentFen());
    var piece = game.get(sq);
    if (piece && piece.color === turn) {
      state.selected = sq;
      e.currentTarget.classList.add("sel");
      game.moves({ square: sq, verbose: true }).forEach(function (mv) {
        var t = el.board.querySelector('[data-square="' + mv.to + '"]');
        if (t) { t.classList.add("target"); if (mv.flags.indexOf("c") > -1 || mv.flags.indexOf("e") > -1) t.classList.add("cap"); }
      });
    }
  }

  function currentFen() { return state.current.fen; }
  function isResolved() { return !el.nextBtn.hidden; }

  function clearSelection() {
    state.selected = null;
    el.board.querySelectorAll(".sel,.target,.cap").forEach(function (n) {
      n.classList.remove("sel", "target", "cap");
    });
  }

  function attemptMove(from, to) {
    var p = state.current;
    var mv;
    if (HAS_CHESS) {
      mv = new Chess(p.fen).move({ from: from, to: to, promotion: "q" });
      if (!mv) return; // not a legal move -- ignore the click
    } else {
      mv = { san: from + "-" + to };  // no legality check available
    }
    var isBest = p.acceptable.some(function (a) { return a.from === from && a.to === to; });
    var isPlayed = p.played.from === from && p.played.to === to;
    resolve(isBest ? "solved" : "missed", mv, isPlayed);
  }

  function resolve(kind, mv, isPlayed) {
    var p = state.current;
    var r = rec(p.id);
    r.attempts += 1;
    if (kind === "solved") r.solved = true;
    progress[p.id] = r;
    saveProgress();

    drawBoard(p.solutionFens && p.solutionFens.length ? p.solutionFens[Math.min(1, p.solutionFens.length - 1)] : p.fen, p.orientation);
    drawArrow(p.acceptable[0]);
    renderMeta(p, true);

    el.feedback.hidden = false;
    el.feedback.dataset.kind = kind;
    var best = p.acceptable.map(function (a) { return a.san; }).join(" or ");
    var head = kind === "solved"
      ? '<span class="verdict">&#10003; ' + escapeHtml(mv.san) + " — a top engine move.</span>"
      : '<span class="verdict">&#10007; ' + escapeHtml(mv.san) + " isn’t best.</span>";
    var body = "";
    if (kind === "missed") {
      body += "<div>" + (isPlayed
        ? "That’s the move you actually played in the game."
        : "The move that holds: <b>" + escapeHtml(best) + "</b>.") + "</div>";
      if (isPlayed) body += "<div>The move that holds: <b>" + escapeHtml(best) + "</b>.</div>";
    } else {
      body += "<div>Best line: <b>" + escapeHtml(p.bestLineSan) + "</b></div>";
    }
    var lines = p.topLines.map(function (l) {
      return "<div><b>" + escapeHtml(l.eval) + "</b>  " + escapeHtml(l.san) + "</div>";
    }).join("");
    el.feedback.innerHTML = head +
      '<div class="body">' + body + "</div>" +
      '<div class="lines">' + lines + "</div>" +
      '<div class="replay-note">You played ' + escapeHtml(p.played.san) +
        " here, losing about " + p.meta.wpLoss + " points of win probability.</div>";

    el.nextBtn.hidden = false;
    el.skipBtn.hidden = true;
    el.nextBtn.focus();

    if (kind === "missed") {
      // requeue this one a few positions later for another attempt
      var at = Math.min(4, state.queue.length);
      state.queue.splice(at, 0, p);
    }
    renderRail();
    updateProgress();
  }

  // ---- solution arrow --------------------------------------------------
  function centre(square, orient) {
    var ff = files(orient), rr = ranks(orient);
    var x = ff.indexOf(square[0]);
    var y = rr.indexOf(parseInt(square[1], 10));
    return { x: x * 10 + 5, y: y * 10 + 5 };
  }
  function drawArrow(move) {
    if (!move) return;
    var p = state.current;
    var a = centre(move.from, p.orientation), b = centre(move.to, p.orientation);
    el.overlay.innerHTML =
      '<defs><marker id="ah" markerWidth="4" markerHeight="4" refX="2.4" refY="2" orient="auto">' +
      '<path d="M0,0 L4,2 L0,4 Z"/></marker></defs>' +
      '<line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y +
      '" stroke-width="2.4" stroke-linecap="round" marker-end="url(#ah)" opacity="0.9"/>';
  }

  // ---- lead-up stepper ------------------------------------------------
  el.leadinBtn.addEventListener("click", function () {
    var p = state.current;
    if (!p || !p.leadIn || !p.leadIn.length) return;
    if (state.leadStep === null) state.leadStep = 0; else state.leadStep += 1;
    if (state.leadStep >= p.leadIn.length) {
      state.leadStep = null;
      drawBoard(p.fen, p.orientation);
      el.leadinBtn.textContent = "Show the lead-up";
      return;
    }
    var step = p.leadIn[state.leadStep];
    drawBoard(step.fen, p.orientation);
    el.leadinBtn.textContent = "Next lead-up move (" + (state.leadStep + 1) + "/" + p.leadIn.length + ")  —  " + step.san;
  });

  el.skipBtn.addEventListener("click", function () {
    if (state.current) state.queue.push(state.current);
    nextPuzzle();
  });
  el.nextBtn.addEventListener("click", nextPuzzle);

  el.resetBtn.addEventListener("click", function () {
    if (!confirm("Clear solved/attempted progress on this device?")) return;
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

  // ---- boot ----------------------------------------------------------
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
