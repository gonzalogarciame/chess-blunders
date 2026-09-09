# Vendored front-end libraries

## `chess.min.js`

`chess.js` **0.12.1** by Jeff Hlywa (<https://github.com/jhlywa/chess.js>), the minified UMD
build as published on cdnjs. Used only for move legality on the board. **BSD-2-Clause** --
full text in `chess.js.LICENSE`.

`src/trainer.py` inlines this file into the generated trainer HTML (inside a `<script>` tag),
the same way the Cburnett piece SVGs are inlined, so the page has **zero** external network
dependencies -- it opens straight from `file://` and hosts on any static server (GitHub
Pages, a claude.ai artifact, ...).
