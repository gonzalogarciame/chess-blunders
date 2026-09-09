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
wrapper) and ships every dependency inline: chess.js (move legality, vendored under
src/assets/vendor/) and the Cburnett piece set (src/assets/cburnett/, inlined as <symbol>s),
plus all puzzle data. No external network requests at all -- it opens straight from file://
and hosts on any static server. If explain.py has been run, each puzzle also carries a short
"why the engine move is best" / "why your move loses" note (see load_explanations). Progress
is kept per-device in localStorage. The look is deliberately plain -- print chess annotation
and a one-engineer analysis tool, not a dashboard.

main() writes two identical copies: outputs/trainer/blunder_trainer.html (publish as an
Artifact) and docs/trainer/index.html (served by GitHub Pages).
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
PAGES_DIR = PROJECT / "docs" / "trainer"
ASSETS = Path(__file__).resolve().parent / "assets" / "cburnett"
CHESS_JS = Path(__file__).resolve().parent / "assets" / "vendor" / "chess.min.js"

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


def load_explanations() -> dict[str, dict]:
    """explain.py's sidecar: id -> {why_best, why_blunder, source}. Optional -- the page just
    omits the notes if it hasn't been run."""
    path = PROCESSED_DIR / f"blunder_explanations_{USERNAME}.json"
    if not path.exists():
        print(f"note: {path.name} not found -- puzzles will have no why-notes "
              "(run `python src/explain.py` to add them).")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def game_movetext_lookup() -> dict[str, str]:
    g = pd.read_parquet(RAW_DIR / "games_gonzalopelotas.parquet")
    return dict(zip(g["game_id"].astype(str), g["movetext"]))


def chess_js() -> str:
    """The vendored chess.js, wrapped in a <script> tag -- inlined so the page needs no CDN."""
    return "<script>/* chess.js 0.12.1 -- BSD-2-Clause, github.com/jhlywa/chess.js */\n" + \
        CHESS_JS.read_text(encoding="utf-8") + "\n</script>"


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


def build_puzzle(rec: dict, movetext: str | None, explanations: dict | None = None) -> dict:
    game_id = str(rec["game_id"])
    pid = f"{game_id}-{rec['ply']}"
    ex = (explanations or {}).get(pid, {})
    return {
        "id": pid,
        "group": rec["motif_group"],
        "fen": rec["fen_before"],
        "orientation": rec["mover_color"],
        "acceptable": rec["acceptable_moves"],
        "played": rec["played"],
        "bestLineSan": rec["best_line_san"],
        "solutionFens": rec["best_line_fens"],
        "topLines": rec["top_lines"],
        "refutationSan": rec.get("refutation_line_san", ""),
        "refutationFens": rec.get("refutation_line_fens", []),
        "whyBest": ex.get("why_best", ""),
        "whyBlunder": ex.get("why_blunder", ""),
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
            .replace("<!--__CHESSJS__-->", chess_js())
            .replace("/*__DATA__*/", "window.__TRAINER__ = " + blob + ";"))


def main() -> None:
    motifs = load_motifs()
    movetexts = game_movetext_lookup()
    explanations = load_explanations()
    puzzles = [build_puzzle(r, movetexts.get(str(r["game_id"])), explanations) for r in motifs]
    groups = group_summary(puzzles)
    page = build_page(puzzles, groups)

    for out_dir, name in [(OUT_DIR, "blunder_trainer.html"), (PAGES_DIR, "index.html")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text(page, encoding="utf-8")

    n_explained = sum(1 for p in puzzles if p["whyBest"])
    print(f"wrote {OUT_DIR / 'blunder_trainer.html'} and {PAGES_DIR / 'index.html'}  "
          f"({len(puzzles)} puzzles across {len(groups)} motif groups, {n_explained} with why-notes)")
    for g in groups:
        print(f"  {g['name']:<32} {g['count']}")
    print("\nnext: commit docs/trainer/index.html (GitHub Pages) and re-publish "
          "outputs/trainer/blunder_trainer.html as the claude.ai artifact.")


_TEMPLATE = r"""<title>Blunder Trainer</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {
    --paper: #ecebe5;
    --paper-2: #e3e2da;
    --ink: #1e1f1b;
    --ink-2: #5b5d54;
    --ink-3: #8c8e83;
    --rule: #ccccc1;
    --accent: #2f6a55;
    --accent-2: #234f3d;
    --warn: #8f5f27;
    --bad: #97402a;
    --sq-l: #e7e2d3;
    --sq-d: #6d8a7c;
    --frame: #3f4a44;
  }
  :root:not([data-theme="light"]) { color-scheme: light dark; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper: #1a1b16; --paper-2: #222319; --ink: #e8e7dd; --ink-2: #a1a297;
      --ink-3: #737567; --rule: #363730; --accent: #74b199; --accent-2: #90c7b2;
      --warn: #cb9659; --bad: #d27658; --sq-l: #aeaa99; --sq-d: #476055; --frame: #5b665f;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --paper: #1a1b16; --paper-2: #222319; --ink: #e8e7dd; --ink-2: #a1a297;
    --ink-3: #737567; --rule: #363730; --accent: #74b199; --accent-2: #90c7b2;
    --warn: #cb9659; --bad: #d27658; --sq-l: #aeaa99; --sq-d: #476055; --frame: #5b665f;
  }

  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body {
    margin: 0; background: var(--paper); color: var(--ink);
    font-family: "Spectral", Georgia, "Times New Roman", serif;
    font-size: 15px; line-height: 1.55;
    -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
  }
  .mono { font-family: "IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace; }
  a { color: var(--accent-2); text-underline-offset: 2px; }

  .wrap { max-width: 1120px; margin: 0 auto; padding: 26px clamp(14px, 4vw, 40px) 72px; }

  /* masthead ------------------------------------------------------------- */
  .mast { display: flex; align-items: baseline; justify-content: space-between; gap: 16px 24px; flex-wrap: wrap; }
  .mast .title { display: flex; align-items: baseline; gap: 10px; }
  .mast .knight { width: 19px; height: 19px; align-self: center; opacity: .82; }
  .mast h1 { margin: 0; font-weight: 600; font-size: 25px; letter-spacing: .01em; }
  .mast .who {
    font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .02em;
    color: var(--ink-2); text-transform: uppercase;
  }
  .mast .score { font-family: "IBM Plex Mono", monospace; font-size: 12.5px; color: var(--ink-2); }
  .mast .score b { color: var(--ink); font-weight: 600; }
  .mast .score .reset {
    background: none; border: 0; padding: 0 0 0 12px; margin-left: 10px; cursor: pointer;
    font: inherit; color: var(--ink-2); text-decoration: underline; text-underline-offset: 2px;
    border-left: 1px solid var(--rule);
  }
  .mast .score .reset:hover { color: var(--ink); }
  .rule { height: 1px; background: var(--rule); margin: 14px 0 0; }

  /* layout -------------------------------------------------------------- */
  .cols { display: grid; grid-template-columns: 186px 1fr; gap: 34px; margin-top: 26px; }
  @media (max-width: 860px) { .cols { grid-template-columns: 1fr; gap: 20px; } }

  .index .lbl {
    font-family: "IBM Plex Mono", monospace; font-size: 10.5px; letter-spacing: .13em;
    text-transform: uppercase; color: var(--ink-3); margin-bottom: 8px;
  }
  @media (max-width: 860px) {
    .index { overflow-x: auto; }
    .index .grps { display: flex; gap: 0; }
    .index .grp { flex: 0 0 auto; border-bottom: 0; border-right: 1px solid var(--rule); padding: 4px 14px 4px 0; margin-right: 14px; }
    .index .grp[aria-current="true"] { box-shadow: none; }
  }
  .grp {
    display: flex; justify-content: space-between; align-items: baseline; gap: 10px; width: 100%;
    background: none; border: 0; border-bottom: 1px solid var(--rule);
    padding: 6px 0 6px 10px; margin: 0; cursor: pointer; text-align: left; color: var(--ink-2);
    font-family: inherit; font-size: 13.5px; line-height: 1.3;
  }
  .grp:hover { color: var(--ink); }
  .grp[aria-current="true"] { color: var(--ink); box-shadow: inset 2px 0 0 var(--accent); font-weight: 600; }
  .grp .n { font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--ink-3); white-space: nowrap; }
  .grp[aria-current="true"] .n { color: var(--ink-2); }

  .main { min-width: 0; }
  .work { display: grid; grid-template-columns: auto 1fr; gap: 30px; }
  @media (max-width: 720px) { .work { grid-template-columns: 1fr; gap: 20px; } }

  /* board ------------------------------------------------------------- */
  .board-wrap { position: relative; width: min(62vw, 432px); aspect-ratio: 1; align-self: start; }
  @media (max-width: 720px) { .board-wrap { width: 100%; max-width: 440px; } }
  .board {
    position: absolute; inset: 0; display: grid;
    grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr);
    border: 1px solid var(--frame); user-select: none;
  }
  .sq { position: relative; display: flex; align-items: center; justify-content: center; }
  .sq.l { background: var(--sq-l); }
  .sq.d { background: var(--sq-d); }
  .sq .coord {
    position: absolute; font-family: "IBM Plex Mono", monospace; font-size: 9px;
    font-weight: 500; opacity: .55;
  }
  .sq.l .coord { color: #4a564e; }
  .sq.d .coord { color: #edefe8; }
  .sq .coord.f { right: 2.5px; bottom: 0.5px; }
  .sq .coord.r { left: 2.5px; top: 0.5px; }
  .sq.sel { box-shadow: inset 0 0 0 3px var(--accent); }
  .sq.hi { box-shadow: inset 0 0 0 3px var(--warn); }
  .sq.bad { box-shadow: inset 0 0 0 3px var(--bad); }
  .sq.target::before {
    content: ""; position: absolute; width: 26%; height: 26%; border-radius: 50%;
    background: currentColor; color: var(--frame); opacity: .32;
  }
  .sq.target.cap::before {
    width: 86%; height: 86%; background: none; border: 3px solid var(--frame); opacity: .3;
  }
  .pc { width: 90%; height: 90%; position: relative; z-index: 1; pointer-events: none; }
  .board-overlay { position: absolute; inset: 0; pointer-events: none; z-index: 3; }
  #overlay line { stroke: var(--warn); }
  #overlay marker path { fill: var(--warn); }

  /* analysis column ------------------------------------------------- */
  .analysis { display: flex; flex-direction: column; gap: 15px; min-width: 0; font-size: 14.5px; }
  .ask { line-height: 1.5; }
  .ask .stm {
    display: inline-block; width: 10px; height: 10px; margin-right: 7px; vertical-align: 1px;
    border: 1.5px solid var(--ink); background: var(--paper);
  }
  .ask.black .stm { background: var(--ink); }

  .note-block {
    border-left: 2px solid var(--rule); padding: 1px 0 1px 13px;
    display: flex; flex-direction: column; gap: 7px; font-size: 13.5px;
  }
  .note-block[data-kind="solved"] { border-color: var(--accent); }
  .note-block[data-kind="revealed"] { border-color: var(--bad); }
  .note-block[data-kind="retry"] { border-color: var(--warn); }
  .note-block .head { font-weight: 600; }
  .note-block[data-kind="solved"] .head { color: var(--accent-2); }
  .note-block[data-kind="revealed"] .head { color: var(--bad); }
  .note-block[data-kind="retry"] .head { color: var(--warn); }
  .note-block .head .mv { font-family: "IBM Plex Mono", monospace; }
  .note-block .var {
    font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.6;
    color: var(--ink-2); overflow-x: auto; white-space: pre-wrap;
  }
  .note-block .var b { color: var(--ink); font-weight: 600; }
  .note-block .aside { color: var(--ink-3); font-size: 12.5px; }
  .note-block .why {
    margin: 0; font-size: 13px; line-height: 1.5; color: var(--ink);
  }
  .note-block .why.bad { color: var(--ink-2); }
  .note-block .why::before {
    content: "▸ "; color: var(--accent); font-size: 11px;
  }
  .note-block .why.bad::before { content: "▸ "; color: var(--bad); }

  .btns { display: flex; flex-wrap: wrap; gap: 7px; }
  .btns button {
    font-family: "IBM Plex Mono", monospace; font-size: 12px; cursor: pointer;
    background: none; color: var(--ink-2); border: 1px solid var(--rule);
    padding: 5px 10px; border-radius: 1px;
  }
  .btns button:hover { border-color: var(--ink-2); color: var(--ink); }
  .btns button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  .btns .go { background: var(--ink); color: var(--paper); border-color: var(--ink); }
  .btns .go:hover { background: var(--accent-2); border-color: var(--accent-2); color: #fff; }

  .facts { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: 3px 16px; font-size: 12.5px; }
  .facts dt { color: var(--ink-3); }
  .facts dd { margin: 0; font-family: "IBM Plex Mono", monospace; color: var(--ink); }
  .facts .tags { grid-column: 1 / -1; margin-top: 5px; color: var(--ink-3); font-family: "IBM Plex Mono", monospace; font-size: 11px; }

  .replay { font-size: 12.5px; align-self: start; }
  .replay a { color: var(--ink-2); }
  .replay a:hover { color: var(--accent-2); }

  .foot {
    margin-top: 26px; display: flex; align-items: baseline; gap: 14px;
    font-family: "IBM Plex Mono", monospace; font-size: 11.5px; color: var(--ink-3);
  }
  .foot .bar { flex: 1; height: 2px; background: var(--rule); position: relative; }
  .foot .bar > i { position: absolute; left: 0; top: 0; height: 100%; background: var(--accent); }

  .empty { padding: 40px 4px; color: var(--ink-2); font-size: 14px; }

  @media (prefers-reduced-motion: no-preference) {
    .foot .bar > i { transition: width .25s ease; }
  }
</style>

<!--__PIECES__-->

<div class="wrap">
  <header class="mast">
    <div class="title">
      <svg class="knight" viewBox="0 0 45 45" aria-hidden="true"><use href="#pc-bn"></use></svg>
      <h1>Blunder Trainer</h1>
      <span class="who">gonzalopelotas</span>
    </div>
    <div class="score">
      <b id="solvedCount">0</b> / <span id="totalCount">0</span> solved
      <button id="resetBtn" class="reset" type="button">reset</button>
    </div>
  </header>
  <div class="rule"></div>

  <div class="cols">
    <nav class="index" id="rail" aria-label="Motif groups">
      <div class="lbl">Mistake patterns</div>
      <div class="grps" id="grps"></div>
    </nav>

    <div class="main">
      <div class="work">
        <div class="board-wrap">
          <div class="board" id="board"></div>
          <svg class="board-overlay" id="overlay" viewBox="0 0 80 80" aria-hidden="true"></svg>
        </div>
        <div class="analysis">
          <p class="ask" id="prompt"></p>
          <div class="note-block" id="feedback" hidden></div>
          <div class="btns">
            <button id="leadinBtn" type="button">Lead-up</button>
            <button id="hintBtn" type="button" hidden>Hint</button>
            <button id="revealBtn" type="button">Show answer</button>
            <button id="lineBtn" type="button" hidden>Play the line</button>
            <button id="punishBtn" type="button" hidden>Show the punishment</button>
            <button id="skipBtn" type="button">Skip</button>
            <button id="nextBtn" class="go" type="button" hidden>Next &rarr;</button>
          </div>
          <dl class="facts" id="meta"></dl>
          <p class="replay"><a id="gameLink" target="_blank" rel="noopener">See the game on chess.com &#8599;</a></p>
        </div>
      </div>
      <div class="foot">
        <span id="progressText"></span>
        <span class="bar"><i id="progressFill"></i></span>
      </div>
    </div>
  </div>
</div>

<!--__CHESSJS__-->
<script>
/*__DATA__*/
(function () {
  "use strict";
  var DATA = window.__TRAINER__ || { puzzles: [], groups: [] };
  var HAS_CHESS = typeof Chess !== "undefined";  // vendored chess.js; degrade gently if absent
  var STORE_KEY = "blunder-trainer-v1";
  var PIECE_NAME = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };

  var progress = loadProgress();
  var byGroup = {};
  DATA.puzzles.forEach(function (p) { (byGroup[p.group] = byGroup[p.group] || []).push(p); });

  var state = {
    group: null, queue: [], current: null,
    selected: null, phase: "solving", tries: 0, leadStep: null, lineStep: 0, punishStep: 0
  };

  var el = {};
  ["rail", "grps", "board", "overlay", "prompt", "feedback", "meta", "gameLink", "leadinBtn",
   "hintBtn", "revealBtn", "lineBtn", "punishBtn", "skipBtn", "nextBtn", "solvedCount",
   "totalCount", "resetBtn", "progressFill", "progressText"].forEach(function (id) {
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

  // ---- index -----------------------------------------------------------
  function renderRail() {
    el.grps.innerHTML = "";
    DATA.groups.forEach(function (g) {
      var s = groupStats(g.name);
      var btn = document.createElement("button");
      btn.className = "grp";
      btn.type = "button";
      btn.setAttribute("aria-current", state.group === g.name ? "true" : "false");
      btn.innerHTML = '<span class="nm"></span><span class="n">' + s.solved + " / " + s.total + "</span>";
      btn.querySelector(".nm").textContent = g.name;
      btn.addEventListener("click", function () { selectGroup(g.name); });
      el.grps.appendChild(btn);
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
    el.board.innerHTML = '<div class="empty">Every puzzle in this group is solved. Pick another pattern on the left, or reset progress to run them again.</div>';
    el.overlay.innerHTML = "";
    el.prompt.textContent = "";
    el.feedback.hidden = true;
    el.meta.innerHTML = "";
    ["gameLink", "leadinBtn", "hintBtn", "revealBtn", "lineBtn", "punishBtn", "skipBtn", "nextBtn"]
      .forEach(function (k) { el[k].hidden = true; });
  }

  // ---- board ---------------------------------------------------------
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
        if (fi === 0) d.insertAdjacentHTML("beforeend", '<span class="coord r">' + rr[ri] + "</span>");
        if (ri === 7) d.insertAdjacentHTML("beforeend", '<span class="coord f">' + ff[fi] + "</span>");
        var pc = map[sq];
        if (pc) {
          d.insertAdjacentHTML("beforeend",
            '<svg class="pc" viewBox="0 0 45 45"><use href="#pc-' + pc.color + pc.type + '"></use></svg>');
        }
        d.addEventListener("click", onSquareClick);
        el.board.appendChild(d);
      }
    }
  }

  // ---- puzzle lifecycle -------------------------------------------
  function startPuzzle(p) {
    state.phase = "solving";
    state.tries = 0;
    state.selected = null;
    state.leadStep = null;
    state.lineStep = 0;
    state.punishStep = 0;

    drawBoard(p.fen, p.orientation);
    el.overlay.innerHTML = "";

    var black = p.orientation === "black";
    el.prompt.className = "ask" + (black ? " black" : "");
    el.prompt.innerHTML = '<span class="stm"></span>' + (black ? "Black" : "White") +
      " to play. You went wrong here — find the move you missed.";

    el.feedback.hidden = true;
    el.gameLink.hidden = false;
    el.gameLink.href = p.meta.gameUrl;
    el.leadinBtn.hidden = !(p.leadIn && p.leadIn.length);
    el.leadinBtn.textContent = "Lead-up";
    el.hintBtn.hidden = true;
    el.revealBtn.hidden = false;
    el.lineBtn.hidden = true;
    el.punishBtn.hidden = true;
    el.skipBtn.hidden = false;
    el.nextBtn.hidden = true;

    renderMeta(p, false);
  }

  function renderMeta(p, revealed) {
    var m = p.meta;
    var rows = [
      ["clock", m.clock + (m.increment ? " (+" + m.increment + ")" : "")],
      ["phase", m.phase],
      ["win% lost", "~" + m.wpLoss],
      ["date", m.date],
      ["opening", m.opening]
    ];
    if (revealed) {
      rows.push(["situation", m.advantage]);
      if (m.material !== "positional") rows.push(["material", m.material]);
      rows.push(["refutation", m.refutation]);
    }
    var html = rows.map(function (r) {
      return "<dt>" + r[0] + "</dt><dd>" + escapeHtml(r[1]) + "</dd>";
    }).join("");
    if (m.leakTags && m.leakTags.length) {
      html += '<div class="tags">' + m.leakTags.map(escapeHtml).join(" &nbsp;·&nbsp; ") + "</div>";
    }
    el.meta.innerHTML = html;
  }

  // ---- interaction ----------------------------------------------
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
    // Playing your own game move is your worst try -- so it's fair to say why it loses right
    // here, without giving away the move you should have found.
    var whyBlunder = (wasPlayed && p.whyBlunder)
      ? '<p class="why bad">' + escapeHtml(p.whyBlunder) + "</p>" : "";
    el.feedback.hidden = false;
    el.feedback.dataset.kind = "retry";
    el.feedback.innerHTML =
      '<div class="head"><span class="mv">' + escapeHtml(mv.san) + (wasPlayed ? " ??" : " ?!") +
        "</span> — not it" + (wasPlayed ? ", and it's what you played in the game" : "") + ".</div>" +
      whyBlunder +
      "<div class=\"aside\">Keep looking" + (state.tries >= 2 ? "" : ", or take a hint after another try") + ".</div>";
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
    state.punishStep = 0;

    drawBoard(solutionFen(state.lineStep), p.orientation, arrowHighlights());
    drawArrow(p.acceptable[0]);
    renderMeta(p, true);

    var best = p.acceptable.map(function (a) { return a.san; }).join(" / ");
    var head = kind === "solved"
      ? '<div class="head"><span class="mv">' + escapeHtml(mv.san) + " !</span> — a top engine move.</div>"
      : '<div class="head">The move was <span class="mv">' + escapeHtml(best) + " !</span></div>";
    var lines = p.topLines.map(function (l) {
      return "<div><b>" + escapeHtml(l.eval) + "</b>   " + escapeHtml(l.san) + "</div>";
    }).join("");
    var whyBest = p.whyBest ? '<p class="why">' + escapeHtml(p.whyBest) + "</p>" : "";
    var whyBlunder = p.whyBlunder ? '<p class="why bad">' + escapeHtml(p.whyBlunder) + "</p>" : "";
    el.feedback.hidden = false;
    el.feedback.dataset.kind = kind;
    el.feedback.innerHTML = head + whyBest +
      '<div class="var">' + lines + "</div>" +
      '<div class="aside">You played <span class="mono">' + escapeHtml(p.played.san) +
        " ??</span> here — about " + p.meta.wpLoss + " win% gone.</div>" + whyBlunder;

    el.leadinBtn.hidden = true;
    el.hintBtn.hidden = true;
    el.revealBtn.hidden = true;
    el.skipBtn.hidden = true;
    el.lineBtn.hidden = !(p.solutionFens && p.solutionFens.length > 2);
    el.lineBtn.textContent = "Play the line";
    el.punishBtn.hidden = !(p.refutationFens && p.refutationFens.length > 1);
    el.punishBtn.textContent = "Show the punishment";
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

  // ---- hint ----------------------------------------------------
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
      '<div class="head">Hint</div><div class="aside">It’s a <b>' +
      PIECE_NAME[occ.type] + "</b> move, from <span class=\"mono\">" + a.from + "</span>.</div>";
  });

  // ---- solution line stepper ---------------------------------
  el.lineBtn.addEventListener("click", function () {
    var p = state.current;
    var fens = p.solutionFens || [];
    var sans = (p.bestLineSan || "").split(/\s+/).filter(Boolean);
    state.punishStep = 0;
    el.punishBtn.textContent = "Show the punishment";
    state.lineStep += 1;
    if (state.lineStep >= fens.length) state.lineStep = 0;
    drawBoard(solutionFen(state.lineStep), p.orientation,
              state.lineStep === 0 ? arrowHighlights() : null);
    el.overlay.innerHTML = "";
    if (state.lineStep === 0) {
      drawArrow(p.acceptable[0]);
      el.lineBtn.textContent = "Play the line";
    } else {
      el.lineBtn.textContent = sans[state.lineStep - 1] + "  (" + state.lineStep + "/" + (fens.length - 1) + ")";
    }
  });

  // ---- refutation (punishment) stepper: the move you played, then how it's refuted -------
  el.punishBtn.addEventListener("click", function () {
    var p = state.current;
    var fens = p.refutationFens || [];
    var sans = (p.refutationSan || "").split(/\s+/).filter(Boolean);
    if (fens.length < 2) return;
    state.lineStep = 0;
    state.punishStep += 1;
    if (state.punishStep >= fens.length) state.punishStep = 0;
    el.overlay.innerHTML = "";
    el.lineBtn.textContent = "Play the line";
    if (state.punishStep === 0) {
      drawBoard(solutionFen(0), p.orientation, arrowHighlights());
      drawArrow(p.acceptable[0]);
      el.punishBtn.textContent = "Show the punishment";
    } else {
      drawBoard(fens[state.punishStep], p.orientation);
      el.punishBtn.textContent = sans[state.punishStep - 1] + "  (" + state.punishStep + "/" + (fens.length - 1) + ")";
    }
  });

  // ---- arrow -------------------------------------------------
  function centre(square, orient) {
    var ff = files(orient), rr = ranks(orient);
    return { x: ff.indexOf(square[0]) * 10 + 5, y: rr.indexOf(+square[1]) * 10 + 5 };
  }
  function drawArrow(move) {
    if (!move) { el.overlay.innerHTML = ""; return; }
    var o = state.current.orientation;
    var a = centre(move.from, o), b = centre(move.to, o);
    el.overlay.innerHTML =
      '<defs><marker id="ah" markerWidth="3.6" markerHeight="3.6" refX="2.2" refY="1.8" orient="auto">' +
      '<path d="M0,0 L3.6,1.8 L0,3.6 Z"></path></marker></defs>' +
      '<line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y +
      '" stroke-width="1.25" stroke-linecap="round" marker-end="url(#ah)" opacity="0.8"></line>';
  }

  // ---- lead-up stepper -------------------------------------
  el.leadinBtn.addEventListener("click", function () {
    var p = state.current;
    if (!p || state.phase !== "solving" || !p.leadIn || !p.leadIn.length) return;
    state.leadStep = state.leadStep === null ? 0 : state.leadStep + 1;
    if (state.leadStep >= p.leadIn.length) {
      state.leadStep = null;
      drawBoard(p.fen, p.orientation);
      el.leadinBtn.textContent = "Lead-up";
      return;
    }
    var step = p.leadIn[state.leadStep];
    drawBoard(step.fen, p.orientation);
    el.leadinBtn.textContent = step.san + "  (" + (state.leadStep + 1) + "/" + p.leadIn.length + ")";
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
    el.progressText.textContent = state.group + "  ·  " + s.solved + " of " + s.total;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // ---- boot -----------------------------------------------
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
