# README assets

READMEs embed the survivorship chart, `cne-demo.png`, the clearly labelled
illustrative `cne-serve-hero-demo.png`, and the
`architecture-diagram-v3.png` architecture diagram. The factual dashboard
capture remains available as `cne-serve-hero.png` for documentation and QA.
The previous v2 source and compatibility export remain as
`architecture-diagram.svg` and `architecture-diagram-v2.png`; the illustrative
dashboard source is `serve-hero-demo.html`.
Other PNGs below are for docs / social / re-exports.
PyPI uses the short `README.pypi.md`, which points at absolute
`raw.githubusercontent.com` URLs for the one demo screenshot it embeds.

## Social preview

Set in **Settings → Social preview**, not embedded in either README — GitHub
reads `og:image` from the repo setting, and a 1MB marketing card above the fold
would only push the actual content down.

| File | Size | Use |
|------|------|-----|
| `social-preview-cn.png` | 1774×887 | Chinese social preview to upload. |
| `social-preview-en.png` | 1774×887 | English social preview to upload. |
| `social-preview-bilingual.png` | 1774×887 | Combined Chinese/English preview for review or re-export. |
| `og-image-brand.png` | 1280×640 | Compact branded fallback card. GitHub caps the social preview at 1MB. |
| `og-image.png` | 1280×640 | Compact fallback export. |
| `social-preview.png` | 1774×887 | Legacy generic export retained for compatibility. |
| `og-image.html` | — | Source the PNGs are rendered from. |

## Charts

| File | Shows |
|------|--------|
| `survivorship-gap.svg` | Survivorship bias, English labels — embedded in `README.en.md` |
| `survivorship-gap.zh.svg` | The same numbers with Chinese labels — embedded in `README.md` |

Same geometry and same measurement; only the string table differs. Re-render
both after a backfill changes the numbers:

```bash
python scripts/survivorship_gap.py --lang en --svg docs/assets/survivorship-gap.svg
python scripts/survivorship_gap.py --lang zh --svg docs/assets/survivorship-gap.zh.svg
```

## Architecture

`architecture-overview.png` is a compatibility export and is no longer
embedded. The current diagram is `architecture-diagram-v3.png` in this
directory and is embedded in both `README.md` and `README.en.md`; update both
references when storage layers, orchestration, quality, query, operations,
Serve, or MCP boundaries change.

## Terminal screenshots

| File | Shows |
|------|--------|
| `cne-demo.png` | `cne demo` phased progress + sample bars — embedded in both READMEs |
| `cne-query.png` | `cne query` SQL result with `source` (kept for re-exports) |
| `cne-load.png` | Python `load()` REPL (kept for re-exports) |

```bash
.venv/bin/python scripts/render_readme_screenshots.py
```

Banner copy should track `cne demo` (no mootdx). Sample bar numbers may be
from an older live run; re-render after UX copy changes.

## Dashboard screenshots

These are real captures, not rendered text, so they need a running server and a
lake with something in it.

| File | Shows |
|------|--------|
| `cne-serve-hero-demo.png` | Synthetic README illustration: a clearly labelled full-coverage heatmap |
| `cne-serve-hero.png` | 1440×820 factual current overview: health, 42 datasets, KPIs, coverage heatmap and action state |
| `cne-serve.png` | 1440px-wide full-page overview (source / docs) |
| `cne-serve-dataset.png` | `trade_ticks` metadata tab (for docs; not in README) |

```bash
cne stats rebuild
cne serve --config configs/cnequity.toml --port 8791
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --hide-scrollbars \
  --window-size=1440,820 --virtual-time-budget=9000 \
  --screenshot=docs/assets/cne-serve-hero.png \
  "http://127.0.0.1:8791/"
```

Capture `cne-serve.png` separately as a full-page screenshot at the same
1440px viewport width. Before saving either image, confirm the page shows
“运行正常”; “度量表过期” is a transient stats state and should be cleared with
`cne stats rebuild` rather than advertised in the README.
