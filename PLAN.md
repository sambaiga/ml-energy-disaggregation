# Book Plan: Machine Learning for Smart Meter Data

- **Title:** *Machine Learning for Smart Meter Data*
- **Subtitle:** *A Structured Path from Meter Signal to Business Insight*
- **Scope:** the full smart-meter-data value chain: disaggregation, forecasting,
  anomaly detection, and grid-edge value, for both consumer and utility/business
  audiences. Not NILM-only, not the full "energy transition" breadth (forecasting +
  disaggregation + reliability + physics-informed ML); that broader scope is parked
  as a possible future **Volume 2**, not folded into this book.

## Differentiation

- Every chapter pairs a narrative page with a companion notebook, run top to
  bottom against the shared `ark/` pipeline, not just read.
- One pipeline, one narrative arc across all four parts, not a stitched-together
  multi-author reference.
- Part 4's LV/DER-under-DER-growth angle is the author's own research strength;
  it has no book-length competition and is the flagship differentiator, not a
  minor use case.

## Structure: four parts

Colors reuse the `act-1`..`act-4` tokens already defined in
`book/_quarto/brand.scss` (blue / violet / green / amber).

### Part 1: Foundations (blue)

General, reusable signal-processing content, not yet written. Scoped from the
material also feeding Part 2 (see `resources/NILM_survey/revised_structure_nilm.tex`):

- Reading a meter signal: acquisition, sampling rates (high vs. low frequency,
  smart meter vs. custom power meter).
- General event/change detection in a time series (the concept, not the
  NILM-specific application; this gets reused again in Part 4's anomaly
  detection).
- General feature-engineering concepts for power signals (steady-state vs.
  transient signatures), taught generically here; the advanced, NILM-specific
  versions (V-I trajectories, graph-based features) live in Part 2.

### Part 2: Disaggregation / NILM (violet)

Source material, verified by reading the actual papers (not just titles):

- `resources/NILM_survey/revised_structure_nilm.tex`: a draft survey, rough
  but a solid field-level taxonomy.
- `resources/PhD_thesis___AF/Chapters/chapter-nilm.tex`: the author's own
  original research (Weighted Recurrence Graphs, Adaptive WRG, phase
  imbalance, multi-label disaggregation).
- `resources/Multilabel_VI/paper_v3.tex`: multi-label appliance recognition
  from high-frequency current using Fryze current decomposition (active/
  non-active components) and a Euclidean-distance-similarity image fed to a
  softmax CNN; evaluated on PLAID (30 kHz).
- `resources/papers/wrg.pdf` / `AWRG.pdf`: the (Adaptive) Weighted Recurrence
  Graph feature methods.
- `resources/papers/FTNILM.pdf` (same paper as `IndustrialNILM.pdf`):
  "Applying Symmetrical Component Transform for Industrial Appliance
  Classification in NILM". The Fortesque Transform decomposes an unbalanced
  three-phase current into positive/negative/zero-sequence components; the
  zero-sequence specifically captures a single-phase appliance's signature
  under phase imbalance. Evaluated on LILACD (50 kHz, three-phase).
- `resources/papers/DeepNLMTK.pdf`: not a disaggregation method itself, but
  Deep-NILMTK, an open-source, framework-agnostic toolkit and benchmark
  (cross-validation, Optuna hyperparameter search, MLflow tracking) for deep
  NILM models (Seq2Point, Seq2Seq, DAE, RNN, GRU, SAED, BERT4NILM,
  UNet-NILM) on UK-DALE at 8s sampling.
- `resources/papers/wattshome.pdf`: Völker et al. (already cited in the preface).
- `resources/nilm-code/`: the author's own code for these papers, cloned
  locally (gitignored; `AWRGNILM`, `MLCFCD`, `End2EndNILM`, `UNETNiLM`,
  `Deep-NILMtk`), several with real preprocessed PLAID/LILAC data already
  sitting in them. See `.claude/skills/nilm-code-reference/SKILL.md` before
  writing any Part 2 notebook: real data and a working reference
  implementation usually already exist there, rather than needing synthetic
  data or a from-scratch implementation.

**The key organizing fact, confirmed across every one of these papers:** NILM
splits cleanly along the high-frequency/low-frequency line this book's
Chapter 1 introduces. WRG/AWRG, the multi-label V-I work, and the phase-
imbalance work are all high-frequency (kHz-range), event-based, load-
*identification* methods built on the same graph/distance-matrix
representation idea. Deep-NILMTK's benchmark is low-frequency (UK-DALE, 8s),
non-event-based, energy-*estimation* work. Earlier drafts of this plan
conflated the two (grouping UNet-NILM's quantile regression with the
high-frequency multi-label chapter); the table below fixes that.

Four chapters, moving from the problem statement through the high-frequency
thread this author's own research actually built, then back out to
low-frequency deep learning:

| # | Chapter | Track | Primary source | Status |
| --- | --- | --- | --- | --- |
| 1 | What Is Non-Intrusive Load Monitoring | Both | Survey's problem formulation + 4-step pipeline | **Done**: `book/02-disaggregation/01-what-is-nilm.qmd` |
| 2 | Feature Engineering for Appliance Recognition (WRG, AWRG, phase imbalance) | High-frequency | `chapter-nilm.tex` WRG/AWRG/FT sections + `wrg.pdf`/`AWRG.pdf`/`FTNILM.pdf`; code and real PLAID/LILACD data from `resources/nilm-code/AWRGNILM` | **Done**: `book/02-disaggregation/02-feature-engineering.qmd` |
| 3 | Multi-Appliance Recognition from High-Frequency Signals | High-frequency | `Final_version.tex` (Fryze decomposition + multi-label softmax CNN, PLAID); code and real multi-label PLAID data from `resources/nilm-code/MLCFCD` | **Done**: `book/02-disaggregation/03-multi-appliance-recognition.qmd` |
| 4 | Deep Learning and Reproducible Benchmarking at Low Frequency | Low-frequency | `DeepNLMTK.pdf` (direct code reuse, same pattern as Twiga in Part 3; UK-DALE) | Not started |

### Part 3: Forecasting (green)

Household load and PV production forecasting, built on
[Twiga](https://github.com/sambaiga/twiga) (point + probabilistic: conformal
prediction, Bayesian NNs, quantile regression) rather than reimplemented from
scratch. Evaluated with proper probabilistic metrics (CRPS, PICP, Winkler score),
not just MAE/RMSE.

### Part 4: Grid-Edge Value (amber, flagship)

- Customer/feeder clustering and segmentation from meter data
- Anomaly and electricity-theft detection
- LV network hosting-capacity / DER-stress estimation under DER growth

Case-study data: real public smart-meter load shapes replayed through an
OpenDSS-simulated LV feeder (via `OpenDSSDirect.py`), not private substation
datasets. This is fully reproducible by readers, with controllable
DER-penetration scenarios (e.g. the same feeder at 0% vs. 40% PV penetration).

## Chapter authoring format: paired .qmd + notebook

Each chapter is a `.qmd` (narrative) plus a companion `.ipynb` (all the code). The
notebook is the single source of truth for code; the `.qmd` pulls specific cells
into the prose via Quarto's `{{< embed >}}` shortcode and never duplicates code.

- **The `.qmd` and its companion notebook must use different filename stems**
  (e.g. `01-intro.qmd` + `01-intro-code.ipynb`); same-stem pairs are silently
  dropped by Quarto instead of being treated as paired.
- `{{< embed 01-intro-code.ipynb#cell-tag >}}` pulls in that cell's output only
  (not source), and auto-links to a full notebook-preview page with a
  "Download Notebook" button.
- The notebook must already contain saved outputs at render time. Outputs are
  committed to git (`nbstripout --keep-output` is already configured for this).
- `freeze: auto` (already set project-wide) avoids re-executing unrelated
  chapters on unrelated renders.
- A machine can have a global `*.ipynb filter=nbstripout` registered (from an
  unrelated project's `nbstripout --install --global`, e.g. in
  `~/.config/git/attributes`) that silently strips outputs and regenerates
  cell ids on every `git add`, with none of this project's flags. This repo's
  own `.gitattributes` explicitly unsets that filter (`*.ipynb -filter`) so
  only this project's own pre-commit/pre-push hooks
  (`nbstripout --keep-output --keep-id`) touch notebooks. If a commit ever
  shows notebook cell ids collapsed to sequential integers ("0", "1", "2",
  ...), this is why: check `git check-attr -a <path>.ipynb` for a stray
  `filter: nbstripout` first.

### Figure sizing convention

Keep figure dimensions consistent within each role rather than picking
`ggsize`/`viewBox` numbers per figure. What matters for on-page consistency
is the *aspect ratio*, since Quarto renders at a consistent container width
and scales height to match:

| Role | Size (px) | Used for |
| --- | --- | --- |
| Full-width single chart | `ggsize(650, 320)` | One line/bar chart on its own (Chapter 1's aggregate-signal plot) |
| Two-panel faceted comparison | `ggsize(560, 320)` | `facet_wrap` comparisons like RG-vs-WRG, V-I-vs-WRG |
| Paired square panels | `ggsize(330, 330)` each, combined via `gggrid([...], ncol=2)` | Two *different* plot objects shown side by side from one notebook cell (waveform + V-I trajectory) |
| Standalone SVG diagram, single concept | `viewBox="0 0 520 240"` | One-row hand-authored diagrams (frequency split, transient/steady-state, WRG pipeline, activation extraction) |
| Standalone SVG diagram, two-row/denser | width `520`, height whatever the content needs (e.g. `278.57` for the single-vs-multi-appliance diagram) | Diagrams with multiple stacked sections |

Layout mechanics, two different mechanisms depending on figure type:

- **Hand-authored SVGs** (`![caption](path.svg){#fig-x}`): Quarto reads
  sizing from the figure's own attributes (e.g. `width="70%"`), not the
  SVG's `viewBox`. For side by side, wrap two figures in
  `::: {layout-ncol="2"} ... :::`. For top and bottom, just don't wrap them;
  plain sequential figures stack vertically by default.
- **Lets-plot figures via `{{< embed >}}`**: Quarto's `layout-ncol` div does
  *not* reliably arrange these, since embed just injects whatever HTML the
  notebook already produced. Set sizing at generation time with `ggsize()`,
  and for two different plot objects that need to sit side by side, combine
  them with lets-plot's own `gggrid([plot_a, plot_b], ncol=2)` inside the
  notebook into one image, rather than trying to lay out two separate
  embeds in the qmd.

## Dependencies

- `twiga[nn]>=1.0.0` added to the `modelling` extra in `pyproject.toml` for Part 3.
  Verified: `uv lock`, `uv sync --extra modelling --extra dev --extra test`, and
  `import twiga` all succeed in this project's venv.
- `torch` (already in the `modelling` extra for Twiga/Part 3) is now also
  needed to render Part 2, Chapter 2's own notebook: it trains a real small
  CNN on real LILACD data. The `modelling` extra is no longer Part-3-only.
- `scipy` moved from a transitive dependency to an explicit one (core
  `dependencies`, not an extra): Chapter 2's notebook imports
  `scipy.signal.medfilt` directly.
- `OpenDSSDirect.py`: needed for Part 4, not yet added.

## Known limitation: citation links don't scope to a chapter

References render per chapter (each chapter ends with its own `## References`
list), but Quarto/pandoc has no way to scope an in-text citation's *link
target* to "the chapter it appears in" across a multi-chapter book: the
generated href always pointed at whichever chapter happened to be first in
the `chapters:` list with a populated `#refs` div, regardless of which
chapter the citation was actually in (confirmed via quarto-dev/quarto-cli
discussion #11873; no fix exists yet, and a per-file `bibliography:` field
does not change this). Rather than ship a citation link that silently jumps
to the wrong chapter, `link-citations: false` is set project-wide in
`_quarto.yml`: citations render as plain readable text with no link, and the
correct full reference is a scroll away in the same chapter's own References
section.

**A second, worse bug, confirmed by direct reproduction in an isolated
sandbox project (not just the GitHub discussion above):** any chapter with
its own `bibliography:` + `::: {#refs} :::` that sits directly in the
top-level `chapters:` list (not nested under a `part:`) gets its reference
list silently emptied on a full `quarto render` of the book. Its citations
still render as inline text, but the formatted reference list at the end of
the chapter disappears (empty div, no `id="ref-..."` entries). This
currently affects `preface.qmd` (its one reference, Völker et al. 2021, has
no formatted entry at the end of the preface).

**Chapters nested under a `part:` are not affected the same way**, verified
with two sibling chapters both under "Part 2: Disaggregation," each with its
own citations: both correctly keep their own, distinct reference list.
Quarto does still hide each one with an inline `style="display: none"`
(patched project-wide with `#refs.csl-bib-body { display: block !important;
}` in both `brand.scss` and `brand-dark.scss`), but the underlying content is
right. So Chapter 1 and Chapter 2 both render correct reference lists today;
only front-matter pages living outside any `part:` (currently just
`preface.qmd`) are missing theirs. If a future top-level page (not nested in
a part) needs its own bibliography, nest it under a trivial `part:` as the
workaround, or switch to a single consolidated references page at the end of
the book (the standard, unaffected Quarto book pattern) if this keeps being
worth working around.

## Open items

- [ ] Build out the remaining `book/` chapter content for each part (see the
      Part 2 table above for the seven-chapter breakdown and sourcing).
- [ ] Decide on renaming the GitHub repo (`ml-energy-disaggregation` to, say,
      `ml-smart-meter-data`). Separate, harder-to-reverse decision: breaks old
      links, requires a GitHub Pages settings update and a `site-url` change.
- [ ] Fix the color-token collision: `ark-activity` and `ark-example` resolve to the
      identical accent color in dark mode (`#34D399` for both) and are
      near-identical in light mode (`#009E73` vs `#059669`). Proposed: shift
      `ark-example` to a rose accent (`#BE185D` light / `#F472B6` dark). Needs
      updating `brand.scss`, `brand-dark.scss`, and `ark/plot/tokens.py` together,
      plus the mirror in `sambaiga.github.io`'s `_defaults.scss`.
- [ ] Design a social-preview (Open Graph) image; none exists yet.
      `open-graph`/`twitter-card` are enabled in `_quarto.yml` but render without a
      picture until a 1200x630 image is added.
- [ ] Add `OpenDSSDirect.py` as a dependency; pick/build the base LV feeder
      model(s) and DER-penetration scenarios for Part 4.

## Already applied to the repo

- Fixed the duplicate title block on `index.qmd` (`title-block-banner: false` +
  `body:has(.cover-hero) #title-block-header { display: none; }` in both SCSS files).
- Added `description` (book-level and on `index.qmd`) and enabled
  `open-graph`/`twitter-card` in `format.html`.
- Added `twiga[nn]>=1.0.0` to the `modelling` extra.
- Renamed `tutorials/` to `book/` throughout the project and CI.
- Wired up shared acronyms (`acronyms.yml`, the `acronyms` Quarto extension) and
  bibliography (`references.bib`), with references rendered per chapter (each
  chapter ends with its own `## References` section) rather than consolidated.
- Redesigned the cover (`index.qmd`): full-width hero via Quarto's `column-page`,
  a real background/foreground redesign (dropped the invented scan-line texture,
  anchored on the site's actual `$dark-body-bg`, added `$ai-accent`/`$energy-accent`
  glows), a Meter Signal -> Data & ML -> Business Value diagram built from real
  Bootstrap icons, and the 4-part roadmap grid.
- Rewrote `preface.qmd` and `about.qmd` for voice and accuracy; added
  `.claude/skills/book-writing-craft/SKILL.md` as the standing writing-craft
  reference (Pressfield, Zinsser, Dicks, the author's own thesis voice, and a
  hard no-em-dash rule).
- Added `how-to-use.qmd` (moved off the cover page, which was competing with
  the sell) and extended `check-qmd.yml`'s lint checks to cover root-level
  `.qmd` files, not just `book/**/*.qmd`.
- Wrote Part 2, Chapter 1 ("What Is Non-Intrusive Load Monitoring"):
  `book/02-disaggregation/01-what-is-nilm.qmd` + companion
  `01-what-is-nilm-code.ipynb`, sourced from `resources/NILM_survey/` and
  `resources/papers/`. Added `hart1992nonintrusive`, `kelly2015ukdale`,
  `zoha2012nilm`, `batra2016gemello` to `references.bib`.
- Added two more Chapter 1 diagrams (transient-to-steady-state waveform,
  single-appliance vs. multi-label model) and the NILM use-cases beyond the
  household bill, with 10 new citations in `references.bib`.
- Switched to IEEE citation style (`book/_quarto/ieee.csl`, `csl:` in
  `_quarto.yml`) and fixed the `#refs` visibility bug above with a CSS
  override in both SCSS files.
- Wrote Part 2, Chapter 2 ("Feature Engineering for Appliance Recognition"):
  `book/02-disaggregation/02-feature-engineering.qmd` + companion
  `02-feature-engineering-code.ipynb`, covering activation-window extraction,
  the V-I trajectory, WRG, AWRG, and the Fortescue transform for phase
  imbalance. Code and real PLAID/LILACD data reused from
  `resources/nilm-code/AWRGNILM` per `.claude/skills/nilm-code-reference/SKILL.md`.
  Added `faustine2020wrg`, `faustine2021awrg`, `faustine2023ftnilm`,
  `marwan2007recurrence`, and 9 dataset/baseline citations to
  `references.bib`. Added `assets/nilm-wrg-pipeline.svg` and
  `assets/nilm-activation-extraction.svg`.
- Added a real "Training and evaluating the classifier" section to Chapter
  2: a small CNN (reusing the architecture from `AWRGNILM/src/net/model.py`)
  trained end to end on the full, real LILACD dataset (all 16 appliance
  types, three phases stacked as channels), comparing V-I, WRG, and WRG+FT
  live rather than only citing the papers' numbers. Confirms the same
  ranking as the published results at a smaller scale (40 iterations, one
  75/25 split, not 600 iterations across 4-fold CV).
- Applied the visualization best-practices checklist (see the auto-memory
  entry `reference-dataviz-best-practices`) to Chapter 2's plots: value
  labels on the confusion matrix (was unreadable with no legend and no
  labels), muted the two losing bars in the F1 comparison chart so the
  accent color highlights only the winning method, and rewrote several
  chart titles to state the finding instead of just describing the axes.
  Also standardized figure sizing across both chapters per the "Figure
  sizing convention" above, resizing Chapter 1's three SVGs and its
  notebook chart to match Chapter 2's established dimensions.
- Every function in both chapters' notebooks now has a proper docstring,
  and each notebook has short markdown section headers so it reads
  coherently on its own (e.g. via the "Download Notebook" link), without
  duplicating the chapter's full narrative.
