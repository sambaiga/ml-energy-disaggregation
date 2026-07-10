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
| 4 | Multi-Task Deep Learning at Low Frequency | Low-frequency | `inilm-multitask.pdf` (UNet-NILM paper) + the author's own `UNETNiLM` repo for the model; `DeepNLMTK.pdf` for benchmarking methodology only, since its own UNet code turned out single-task-only; UK-DALE house 1 | **Done**: `book/02-disaggregation/04-multi-task-low-frequency.qmd` |

### Part 3: Forecasting (green)

Household load and PV production forecasting, built on
[Twiga](https://github.com/sambaiga/twiga) (point + probabilistic: conformal
prediction, Bayesian NNs, quantile regression) rather than reimplemented from
scratch. Evaluated with proper probabilistic metrics (CRPS, PICP, Winkler score),
not just MAE/RMSE.

**Data**: the "Medium-Low Voltage Energy Distribution Network and Smart Meter
Dataset" (Maree et al., Mendeley Data, published March 2025,
[data.mendeley.com/datasets/jv3rz8k35r/1](https://data.mendeley.com/datasets/jv3rz8k35r/1)),
verified directly. A real Norwegian DSO network (Lede AS, Porsgrunn) with
~6,809 customers, hourly AMI active/reactive power from 2022-01-01 to
2024-09-30 (~2.75 years, enough real seasonal structure for a forecasting
chapter), plus regional weather and electricity spot prices, network topology
already in pandapower/CGMES format. Freely downloadable, no signup gate.

The dataset has no PV/EV sub-metering, only net active/reactive power per
meter. This is a deliberate framing choice, not a gap to patch around: the
chapter forecasts *net* load with PV and EV as invisible confounders, the
realistic problem a utility actually faces, since most real smart meters
never expose behind-the-meter generation or EV charging separately. Pecan
Street's Dataport has real combined load + PV + EV readings from the same
households, but its genuinely free tier is capped at 75 homes (25 each in
Austin, New York, and California) and restricted to "academic research
purposes, no funded research" (verified against
[pecanstreet.org/dataport/licenses](https://www.pecanstreet.org/dataport/licenses/);
full access is $10k-15k/year). Useful as a small, optional illustrative
example of what's hiding inside a net-load signal, not as the chapter's
backbone dataset.

### Part 4: Grid-Edge Value (amber, flagship)

Four threads, ordered so each builds on the last. Threads 2-4 are still
planned around the same SMART-DS feeder (see the Open Items entry below
for the exact real, tutorial-sized subfolder identified:
`AUS/P1U/.../p1uhs0_1247--p1udt12703/`, ~2.4 MB, 8 `.dss` files); thread 1
was revised after Chapters 1-2 shipped to reuse the real 31-customer
AusNet feeder those chapters already built and vendored, rather than
introduce a new dataset for one thread only.

1. **Phase detection and topology identification** (Chapter 3, branch
   `part4-ch3-phase-identification`): which phase ($A$, $B$, or $C$) each
   meter is connected to, purely from voltage correlation, no field audit.
   `Eneida.io`'s phase-clustering work is not usable as a data source,
   checked directly: it is a commercial LV-analytics vendor showcasing a
   CIRED 2025 paper, not a public dataset, and this book only uses
   reproducible public data. Revised plan, decided after a literature
   search turned up a direct precedent: reuse the same real 31-customer
   AusNet feeder and smart-meter data Chapters 1-2 already vendored
   (`resources/cvar_flexibility/data/timeseries-lv/`, via
   `scripts/fetch_part4_ausnet_data.py`, no new data-fetch work needed),
   compute voltage correlations from real customer LoadShapes, and cluster
   them (correlation matrices, then PCA + k-means) to recover phase
   connectivity, checked against the real answer already sitting in that
   feeder's own `LVcircuit-loads.txt` (each load's phase is explicit
   ground truth in the model). A fully reproducible, self-checking
   exercise real unlabeled utility data could never offer. Five real,
   verified reference papers (checked directly, not assumed):
   - Short, T.A. (2013), "Advanced Metering for Phase Identification,
     Transformer Identification, and Secondary Modeling," *IEEE Trans.
     Smart Grid*, 4(2):651-658. The foundational correlation-based method.
   - Simonovska, A. & Ochoa, L.F. (2021), "Phase Grouping in PV-Rich LV
     Feeders: Smart Meter Data and Unconstrained k-Means," *IEEE Madrid
     PowerTech*. PCA + k-means, tested on this book's own 31-customer
     AusNet feeder under different PV penetrations, same author lineage
     as Team-Nando's own DER-hosting-capacity tutorials (L.F. Ochoa is
     "Nando" Ochoa). The direct precedent the revised plan is built on.
   - Blakely, L. & Reno, M.J. (2020), "Phase Identification Using
     Co-Association Matrix Ensemble Clustering," *IET Smart Grid*. Sandia
     National Labs; ensemble spectral clustering that needs no existing
     phase labels.
   - Hoogsteyn, A., Vanin, M., Koirala, A., Van Hertem, D. (2022), "Low
     Voltage Customer Phase Identification Methods Based on Smart Meter
     Data," *Electric Power Systems Research*, 212. Benchmarks multiple
     methods across meter accuracy and deployment density, the basis for
     a planned sensitivity exercise.
   - Hangawatta, D.C., Gargoom, A., Kouzani, A. (2025),
     "Machine-Learning-Driven Identification of Electrical Phases in
     Low-Sampling-Rate Consumer Data," *Energies*, 18(1):128. A neural-
     network approach for low-frequency data, closer to this book's own
     30-minute AusNet resolution than most of the above.

   Planned structure: stakes (a wrong phase record silently poisons every
   hosting-capacity/balance calculation Chapter 2 already ran) -> the
   idea (phase-mates share correlated voltage fluctuations from common
   upstream impedance and simultaneous demand) -> a ground-truth check
   against the feeder's own known answer (mirrors Chapter 1's own
   verification habit) -> method (correlation matrix first, Short-style
   and transparent, then PCA + k-means, matching Simonovska & Ochoa's real
   published benchmark on this exact feeder) -> a real result with an
   honest failure mode, not glossed over -> a sensitivity "Do it
   yourself" (accuracy vs. fewer meters/shorter windows/lower resolution,
   per Hoogsteyn et al.'s own finding) -> payoff (corrected phase
   labels feed back into Chapter 2's own hosting-capacity machinery)
   -> a pointer to thread 2. Notebook first, matching Chapter 2's own
   build discipline; `.qmd` narrative after the notebook is reviewed.
2. **Customer/feeder clustering and segmentation from meter data**: segment
   SMART-DS's synthetic smart meters into consumer archetypes (residential,
   commercial, industrial). `AUS/P1U` (mixed commercial/residential) is the
   confirmed fit, real load-curve-shape diversity to cluster.
3. **Voltage violation / power-quality anomaly detection**: catching
   sustained over/undervoltage events in real time. The single biggest real
   constraint DSOs actually hit today, since PV back-feed pushing feeder
   voltage out of statutory limits is what caps hosting capacity in
   practice, not a hypothetical. Reuses the same SMART-DS feeder and
   OpenDSS-simulated bus-voltage timeseries as thread 1, this time treated
   as a live monitoring stream (control limits, isolation forest, or
   autoencoder reconstruction error) rather than a phase-clustering input.
   Paired with thread 4 below: this thread catches a violation as it
   happens, thread 4 forecasts whether one is coming. A short companion
   section on meter/sensor data-quality anomalies (stuck readings,
   communication dropouts, calibration drift), using Maree's real data,
   closes the thread: not a separate chapter, but the cross-cutting practice
   every earlier forecasting/clustering chapter quietly assumed away.
   Electricity-theft detection was considered and dropped: real,
   publicly-labeled theft data essentially does not exist (utilities will
   not release it), so almost every academic theft-detection paper works by
   synthetically injecting fake tampering patterns onto real load data, a
   step away from this book's real-verified-numbers discipline.
4. **Smart grid analytics and network management**: evaluate grid health
   (voltage violations, thermal overloading) under *forecasted* DER growth
   scenarios, then test mitigation strategies, the scenario-based
   counterpart to thread 3's real-time detection. This is the chapter where
   Part 3 stops being standalone: its forecasts (from the Maree dataset)
   become the DER-penetration scenario driver here, not just a chapter that
   happens to come before this one. "Mitigation strategies" maps directly to
   Team-Nando's Operating-Envelope/ANM/Volt-Watt-control tutorial code
   (BSD-3, reused as methodology per the `nilm-code-reference`-skill pattern
   of reusing algorithms, not raw data or figures).

Case-study data: real public smart-meter load shapes and a real (or, for
phase ID, ground-truth-bearing synthetic) network topology replayed through
an OpenDSS-simulated LV feeder (via `OpenDSSDirect.py`/`dss_python`), not
private substation datasets. Fully reproducible by readers, with
controllable DER-penetration scenarios (SMART-DS already ships a real
13-scenario solar x battery penetration matrix per feeder, no need to
synthesize penetration levels from scratch).

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
- `OpenDSSDirect.py>=0.9.4`: added as the `grid` extra for Part 4. Verified
  cross-platform directly this session (installs and solves correctly on
  macOS arm64, unlike `py_dss_interface`, which crashes on instantiation
  there, confirmed both from its source, no macOS branch in `DSS.py`, and
  by reproducing the crash live). `uv lock` + `uv sync --extra grid`
  succeed.
- `ark/dss/`: a new sibling to `ark/plot/` (`ark/__init__.py` is empty, so
  each domain gets its own self-contained sub-package). Two files:
  - `results.py`: module-level functions reading a solved `opendssdirect`
    module's state into tidy pandas DataFrames (`bus_voltages`,
    `line_currents`, `line_losses`, `element_powers`, `circuit_summary`).
  - `circuit.py`: a pythonic `Circuit` class wrapping OpenDSSDirect.py's
    COM-style API (manual `First()`/`Next()` iteration over a shared
    "active element" pointer) behind ordinary Python iteration and small
    frozen dataclasses (`Line`, `Load`, `Transformer`), e.g.
    `for line in circuit.lines:` instead of a manual loop. `Circuit.load()`
    compiles + solves a circuit; `circuit.command(...)` sends a raw OpenDSS
    command and returns its text result (reads `Text.Result()`, since
    `opendssdirect.Command()` itself always returns `None`, a real bug
    caught by executing the ported notebook end-to-end, not just unit
    checks); `circuit.dss` exposes the raw `opendssdirect` module as an
    explicit escape hatch for anything not wrapped (monitors, meters,
    per-terminal currents). `results.py`'s functions are reused as
    `Circuit`'s DataFrame-returning methods, not duplicated.
  - Deliberately minimal: no visualization (routes through `ark.plot`
    instead, this project's existing house charting library) and no
    scenario-sweep or workflow-class abstractions yet (`py-dss-toolkit` has
    these, but depends on `py_dss_interface` and inherits its macOS
    breakage; wait for a real Part 4 chapter to reveal the actual repeated
    pattern before adding them, the same notebook-first-then-promote
    discipline as everything else in this project).
  - Verified against the author's own OpenDSS tutorial
    (`resources/cvar_flexibility/notebook/Opendss_basic.ipynb`)'s simple
    3-house LV network, both directly and via the full ported notebook
    executing end-to-end with 0 errors (see the Open Items entry below):
    `circuit_summary` matches the known circuit (6 buses, 4 lines, 3 loads,
    1 transformer, 0.455 kW total line losses) and `bus_voltages`
    reproduces the known per-phase spread at buses C/D/E
    (0.977/0.983/0.972 pu).
- `ark/plot/icons.py`: a new, general-purpose (not OpenDSS-specific)
  addition to `ark.plot`, for diagrams that need a real pictographic icon
  at a data point, not just a marker shape. Vendors
  `ark/plot/assets/bootstrap-icons.ttf`, converted once from the `.woff`
  Quarto already bundles into `_book/site_libs/bootstrap/` at render time
  (this book's own site icon set, MIT licensed, see the `bi bi-*` classes
  in `_quarto.yml`): matplotlib's bundled FreeType cannot open `.woff`
  directly (`FT_Open_Face ... broken table`, verified live), only
  `.ttf`/`.otf`, so `fontTools` (already a transitive dependency) converts
  it once. No new runtime dependency either way. `ICONS` maps icon name to
  its Bootstrap Icons v1.13.1 codepoint, `icon_font()` returns a
  `FontProperties` for drawing a glyph via `ax.text(...)`; only the
  handful of icons `book/04-grid-edge/01-opendss-lv-modeling-code.ipynb`
  actually needed (house/sun/lightning-bolt/network-hub/a ring-in-circle
  for a plain bus junction, `record-circle-fill`, picked over
  `node-plus-fill`/`diagram-3-fill`/`broadcast-pin`/`share-fill` after
  rendering all of them side by side, since a plain colored dot for
  non-load buses read as an unfinished node next to the pictographic ones)
  are mapped so far. Considered and rejected: Plotly's `layout.images` can also place
  per-point icon images, but it's a new dependency this book doesn't use
  anywhere else, and its interactive HTML output is heavier (CDN
  dependency or ~3.5MB inline JS bundle per chart) for no benefit here
  over matplotlib, which was already a hard dependency.

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
- [x] Add `OpenDSSDirect.py` as a dependency (`grid` extra) and build
      `ark/dss/`, see the Dependencies section above.
- [x] Port the author's own OpenDSS-LV-modeling tutorial
      (`resources/cvar_flexibility/notebook/Opendss_basic.ipynb`, originally
      `py_dss_interface`) to `ark.dss.Circuit` and place it at
      `book/04-grid-edge/01-opendss-lv-modeling-code.ipynb` (notebook only,
      no `.qmd` narrative yet). Preserves the original's section-by-section
      structure; the five circuit-topology figures were copied alongside it.
      Executes end-to-end with 0 errors.
      - Adds a "Visualizing the network" section: `topology_layout` (a
        plain BFS hop-distance layout, since this hand-built circuit has no
        bus coordinate data), `bus_roles`, `voltage_status`, and
        `plot_topology`, all notebook-local functions (not yet promoted to
        `ark.dss`, same notebook-first-then-promote discipline), reused
        across all three networks in the notebook. Renders in matplotlib,
        not Lets-Plot (every other chart in the book), specifically because
        Lets-Plot has no per-point pictographic marker (`geom_raster` is a
        continuous field, `geom_point`'s `shape` aesthetic is built-in
        shapes only); matplotlib's font manager can place real icon glyphs
        via `ark.plot.icons` (see the Dependencies entry below), styled
        through `ark.plot.matplot_theme.configure_matplotlib_style()` for
        brand-consistent fonts. Nodes are colored by a fixed,
        threshold-anchored status scale (green/amber/red at 0.94/1.10 pu),
        not a continuous gradient normalized to the current data's own
        min/max, a real design bug caught by looking at the rendered
        output: the first version painted a perfectly healthy 0.972-1.000
        pu feeder end-to-end red-to-blue, which reads as "half this network
        is in trouble" when none of it is. All non-node sizes (transformer/
        PV badges, icon glyphs, label spacing) are a fixed ratio of the one
        `node_radius` parameter rather than independently hardcoded, so the
        diagram stays proportional if a caller changes it; `low`/`high`/
        `near_margin` are parameters too, threaded through to
        `voltage_status`. The legend is a real `ax.legend()` (a custom
        `_IconHandler(HandlerBase)` teaches it to draw a Bootstrap Icon
        glyph as a swatch), not a second independent set of hand-placed
        `ax.text()` calls at hardcoded axes-fraction coordinates, an
        earlier version did that and needed 10+ magic numbers tuned by
        hand; the real legend inherits frame/font-size/spacing from
        `configure_matplotlib_style()`'s rcParams for free. Also fixed: the
        function used to return a Figure that was still open, so Jupyter's
        inline backend auto-displayed it *and* separately reprs the
        returned object, rendering every diagram twice; `plt.close(fig)`
        right before `return fig` leaves only the one correct render. Every
        bus gets an icon now, not just source/load/PV buses: a plain
        junction previously rendered as a bare colored disc (looked like
        an unfinished node next to the pictographic ones), now uses
        `record-circle-fill`, picked by rendering seven Bootstrap Icons
        candidates side by side (`node-plus-fill`, `diagram-3-fill`,
        `broadcast-pin`, `share-fill`, ... ) and comparing legibility at
        small node sizes and semantic fit. `plot_topology(...,
        orientation="horizontal")` grows the tree left-to-right instead of
        top-to-bottom; `topology_layout` just swaps which axis carries
        BFS-depth vs. sibling-spread, with sibling spacing widened in
        horizontal mode specifically (labels are always placed "below" a
        node, and "below" is the tight sibling-spacing axis in horizontal
        mode, not the generously-spaced depth axis like in vertical mode,
        so it has to widen or labels collide with the next sibling down).
      - Adds worked solutions for all three "Do It Yourself" exercises
        (the exercise prompts themselves are kept verbatim as the unsolved
        text first): E1 (network expansion + annual-loss/voltage-compliance
        analysis across a load-duration curve), E2 (PV systems on the new
        houses), E3 (an entirely new from-scratch network per Figure 4/5,
        with and without PV). The voltage-compliance charts (still
        Lets-Plot, unlike the topology diagram above; a plain points-plus-
        two-dashed-lines chart doesn't need per-point icons) shade the
        compliant band green (`geom_rect` with finite bounds sized to the
        discrete x-axis's category count, verified live: `-Inf`/`Inf`
        bounds silently fail to render against a discrete axis in
        Lets-Plot) and connect each house's points into a line, so the
        trend across loading levels reads directly. Two real, non-obvious
        OpenDSS bugs were
        caught only by executing the ported solutions, not by writing them:
        `status=fixed` loads ignore `set LoadMult=...` entirely (each
        load's `kW` has to be edited directly per loading level instead),
        and `opendssdirect` is a *singleton engine*, so a `Circuit` object
        built earlier in the notebook silently starts reporting a
        different network's state once a later cell `Clear`s and rebuilds
        (fixed by capturing scalar results like `base_peak_loss_kw` into
        plain variables at computation time, not re-querying a stale
        `Circuit` handle later). Also caught by execution, not review:
        `opendssdirect.Command()` always returns `None` (`Circuit.command`
        reads `Text.Result()` instead, see the Dependencies entry above),
        and lets-plot needs `LetsPlot.setup_html()` called once per
        notebook (matches the convention in every `book/02-disaggregation/`
        notebook) or every chart silently renders as an inert placeholder
        div instead of erroring.
- [x] Write the `.qmd` narrative for `book/04-grid-edge/01-opendss-lv-modeling`
      and wire it into `_quarto.yml` under a new `Part 4: Grid-Edge Value`
      block. Written, then substantially revised after user review flagged
      five gaps: no concrete LV-to-smart-meter-data connection, no LV-
      network/power-grid fundamentals before diving into OpenDSS mechanics,
      thesis diagrams not consulted, and the OpenDSS coverage should extend
      to everything already built in the companion notebook, not just a
      curated four-cell subset. Final structure: opening hook, "What a
      low-voltage network actually is" (HV/MV/LV chain, a new branded
      centralized-vs-decentralized diagram, a real 31-customer AusNet
      feeder photo via Team-Nando as contrast to the toy 3-house example),
      "Why the low-voltage network matters now", "From smart meter to
      network model" (the concrete bridge: a meter reading becomes a
      `LoadShape`, combined with topology no meter reports), "Three ways
      DER strain a feeder", the per-unit-voltage concept box, the OpenDSS
      walkthrough (build/solve/voltages/topology diagram), "Reading what's
      already there" (iteration, the `circuit.command` escape hatch,
      per-element power extraction, transformer currents via `circuit.dss`,
      8 embeds pulled in from notebook sections the first draft skipped),
      the `.ark-mistake` callout (repositioned as setup, not the closing
      beat), "Checking a feeder across a full day" (the notebook's own
      Exercise 1 solution narrated in full: `evaluate_loading_levels`
      against the real load-duration curve and the `plot_voltage_compliance`
      shaded-band chart, explicitly requested), then the closing thread
      preview, now naming Chapter 2 as the immediate next step. Also added
      `ark/plot/diagrams.py::centralized_vs_decentralized_grid_diagram()`
      (new Bootstrap-icon-based concept diagram, curved arrows and dashed
      zone boundaries for a hand-composed feel rather than an auto-generated
      flowchart look, reused `ICONS["building-fill"]`/`"ev-front-fill"`/
      `"arrow-down-up"` added to `ark/plot/icons.py` for it) and 5 new
      `references.bib` entries sourced directly from the thesis's own
      bibliography (`iea2023investment`, `bystrom2022distribution`,
      `ucer2020congestion`, plus `azizivahed2020reconfiguration` and
      `kumari2020substation` for future Part 4 use), and `EV`/`DSO`/`BESS`/
      `HV`/`RES` to `acronyms.yml`. Full book render verified clean twice
      (once per revision pass), `index.qmd`'s Part 4 card updated to link
      Chapter 1 and drop a stale "theft detection" mention left over from
      before that thread was explicitly dropped. One more real bug caught
      only by the pre-commit hook, not review: the notebook-generator
      script's `code()`/`md()` helpers let `nbformat` assign a *random*
      cell id on every run, so any later regeneration (even for an
      unrelated fix elsewhere) silently changed every cell id, breaking
      every `{{< embed notebook.ipynb#cellid >}}` reference in the paired
      qmd. Fixed by hashing each cell's own content into its id instead,
      so an id only changes when that specific cell's content does; a
      one-time migration re-pointed all 17 of this chapter's embeds at
      their new stable ids.
- [x] Write `book/04-grid-edge/02-timeseries-der-modeling`, covering what
      Chapter 1 doesn't: `LoadShape` mechanics, `Set Mode=daily` time-series
      solving, irradiance/temperature-driven PV models, Volt-Watt/Volt-VAr
      control via `InvControl`, the `Storage` element, and a PV hosting-
      capacity study. Source material: `resources/cvar_flexibility/notebook/
      Opendss_der-model.ipynb` and `Oendss_timeseries.ipynb`, both confirmed
      to be local adaptations of
      `Tutorial-DERHostingCapacity-{2-TimeSeries_LV,3-VoltWatt_LV}`. Real
      data vendored via a new `scripts/fetch_part4_ausnet_data.py` (342
      real AusNet customers, 30-min resolution, one year, plus the real
      31-customer network definition, from `Tutorial-DERHostingCapacity-
      {2-TimeSeries_LV,3-VoltWatt_LV}`, both fetched into the gitignored
      `resources/cvar_flexibility/data/` tree). A companion
      `scripts/fetch_part3_mendeley_data.py` vendors the Maree et al.
      dataset for Part 3 the same way (real API endpoint reverse-engineered
      from Mendeley Data's own frontend bundle, since its public REST API
      caps a naive `/files` listing at 100 of 7,372 entries with no working
      pagination; the whole-dataset zip endpoint their own "Download All"
      button calls, `/public-api/zip/{id}/download/{version}`, has none of
      that limit).

      `ark/dss/circuit.py` extended, not just used as-is: `PVSystem`/
      `StorageUnit` dataclasses and `circuit.pvsystems`/`circuit.storage_units`
      iterators (mirroring `Load`/`Line`/`Transformer`), and
      `circuit.solve_daily(steps, stepsize)`, a generator stepping a
      `Set Mode=daily` solve, both genuinely repeated patterns across the
      chapter's six sections, the same notebook-first-then-promote
      discipline as everything else in `ark/`. Caught and fixed a latent
      bug while extending it: `element_powers("pvsystems")` guessed the
      `opendssdirect` module name via `.capitalize()`, which produces
      `"Pvsystems"`, not the real `"PVsystems"` attribute; never exercised
      before since Chapter 1 only ever called it with `"loads"`/
      `"transformers"`. Replaced the guess with an explicit mapping and
      added `"storages"`.

      Real bugs caught only by executing, not by review, same discipline
      as Chapter 1: (1) `PVSystem` defaults `%cutin`/`%cutout` to 20%, not
      the 5% every Team-Nando tutorial actually uses, silently zeroing two
      real hours of PV output at low irradiance, now the chapter's own
      worked example rather than something glossed over. (2) `InvControl`
      needs `Set maxcontroliter=1000`, without it the solve raises "Max
      Control Iterations Exceeded" instead of converging. (3) Building a
      second `Circuit` while holding a reference to a first one (the
      Storage charge/discharge comparison) reproduced the exact singleton-
      engine bug Chapter 1's own `base_peak_loss_kw` note already warned
      about; fixed by reading each result immediately after its own solve.
      (4) A first-draft "voltage over a day" chart let its y-axis
      auto-scale to the data's actual ~0.000003 pu range, drawing a
      dramatic-looking cliff out of what the surrounding text says is
      noise; fixed with a fixed axis spanning the real statutory band
      instead, so the chart's shape actually matches its own caption.
      Real hosting-capacity result found, not fabricated: on the sunniest
      day in the real AusNet data, with realistic 5kVA rooftop systems,
      this feeder stays voltage-compliant through 60% PV penetration and
      breaches the 1.10 pu limit at 80%.

      `ark/plot/diagrams.py::centralized_vs_decentralized_grid_diagram()`'s
      two panels also reworked in this pass: they previously shared one
      geometry (coordinates, box sizes, arrow curvature) recolored per
      side, which read as a template despite the curved-arrow styling;
      each panel now has its own hand-picked numbers. `ieee1547-2018` and
      `IEEE` added to `references.bib`/`acronyms.yml`. Full book render
      verified clean, including a Playwright-rendered visual check of
      every chart (the lets-plot HTML widgets don't show up in the raw
      notebook JSON, so a browser check is the only way to actually see
      them).

      Substantially extended in a second pass, after review flagged the
      first draft as reading like a tour of disconnected OpenDSS features
      (three separate networks: a toy 3-house continuity demo, a synthetic
      PV stress network, and the real feeder only in the final section)
      rather than one throughline. Rebuilt around the same real feeder end
      to end: real load/PV diversity across all 342 customers and 365
      days, a genuine multi-day (week-long) time-series simulation
      matching `Oendss_timeseries.ipynb`'s own `start_day`/`total_days`
      structure, a load-shape verification cross-check (real data vs. live
      solve vs. stored `LoadShape`), a transformer thermal constraint
      alongside the voltage one, Volt-Watt/Volt-VAr applied directly to
      the real 80%-penetration violation (not a fresh demo network) plus
      a full 0-100% sweep under control showing both mitigations raise
      hosting capacity outright, and a "Do it yourself" section built on
      the source tutorial's own seasonal worst-case exercise (using the
      four vendored seasonal PV files), which finds a genuinely different,
      more optimistic answer than the single-sunniest-day sweep, a real
      lesson about how "worst case" gets defined, not a contradiction.

      Added a new `## 9. EV demand` section, the opposite failure mode:
      EV charging adds demand instead of generation, at the evening peak
      instead of midday. Real UK "Electric Nation" trial charging data
      (via Team-Nando's `EV-Demand-Profiles` repo, vendored by a new
      `scripts/fetch_part4_ev_profiles.py`) informs a customized scenario
      rather than being reused verbatim, since that source repo ships no
      LICENSE unlike every other Team-Nando repo this project draws on.
      Real, non-obvious finding: realistic diversified EV charging barely
      moves either constraint even at 100% adoption (diversity across many
      vehicles does real mitigating work on its own); an uncoordinated
      stress test (every charger at full rated power, no smart charging)
      still doesn't violate at 100% adoption; scaling to multiple EVs per
      house finds where it actually breaks, and thermal crosses its limit
      before voltage does, the opposite order from the PV story. Volt-Watt
      and Volt-VAr only control `PVSystem` elements, so they have nothing
      to act on for an EV `Load`, setting up `Storage` as the one lever
      that helps both the midday PV peak and the evening EV peak.

      One more real bug caught while extending `ark/dss/circuit.py` for
      this section: a naive network-wide `bus_voltages()["vmag_pu"].min()`
      silently picked up the source (slack) bus's own fixed per-unit
      setpoint, not a real customer voltage, since Chapter 1 never took
      this kind of blind aggregate (it always filtered to named customer
      buses first). Fixed by adding `Circuit.source_bus` (found via the
      `Vsources` collection, not hardcoded) and excluding it from every
      network-wide voltage aggregate; the bug had been silently flattening
      the chapter's own "voltage envelope" chart to a flat, meaningless
      line at the source's ~1.0 pu setpoint regardless of load.

      `.qmd` narrative written last, after the notebook was fully reviewed
      and approved: a three-act structure (infrastructure, the PV crisis
      and its fix, the EV twist the first fix can't touch), curated to
      roughly 30 embeds out of the notebook's 71 cells rather than
      embedding everything, the same "wall of embeds" lesson Chapter 1's
      own review already surfaced. New
      `ark/plot/diagrams.py::pv_vs_ev_failure_mode_diagram()` (a small
      two-panel diagram contrasting the two failure modes: same feeder,
      opposite flow direction, opposite time of day, opposite binding
      constraint) placed at the exact narrative pivot into the EV
      section. Full book render and a Playwright visual check both clean.

      Final review pass against the book-writing-craft skill found "real"
      overused as a modifier past the point of deliberate emphasis (25+
      instances, trimmed throughout, kept only where it makes an actual
      real-vs-synthetic contrast), two section boundaries with no bridging
      sentence, and a stale "Section 6" cross-reference left over from an
      earlier numbered-heading draft (this chapter's headings were never
      numbered). PR #7 (later renumbered #8 after a rebase) merged to
      `main` 2026-07-10.
- [ ] Write `book/04-grid-edge/03-phase-identification` (branch
      `part4-ch3-phase-identification`), Part 4's thread 1: recovering
      which phase each customer is actually connected to from voltage
      correlation alone, no field audit. Reuses the real 31-customer
      AusNet feeder and smart-meter data Chapters 1-2 already vendored
      (no new data-fetch work), checked against the real per-load phase
      ground truth already sitting in `LVcircuit-loads.txt`. Five real
      reference papers identified (see the thread-1 entry above for full
      citations): Short (2013, the foundational correlation method);
      Simonovska & Ochoa (2021, PCA + k-means, tested on this exact
      AusNet feeder, the direct precedent this plan is built on); Blakely
      & Reno (2020, Sandia, ensemble spectral clustering needing no
      existing labels); Hoogsteyn et al. (2022, a benchmark across meter
      accuracy/density, the basis for a planned sensitivity exercise);
      Hangawatta et al. (2025, a neural-network approach for low-frequency
      data closer to this book's own 30-minute resolution). Notebook
      first, matching Chapter 2's own build discipline; `.qmd` narrative
      after the notebook is reviewed. Not yet started.
- [ ] Pick/build the base LV feeder model(s) and DER-penetration scenarios
      for Part 4 threads 2-4 (SMART-DS candidates already identified
      above; thread 1 no longer needs this, see above).
- [x] Download and vendor the Maree et al. Mendeley dataset for Part 3
      (gitignored, same pattern as `resources/nilm-code/`, not checked into
      git given its size). Vendored via `scripts/fetch_part3_mendeley_data.py`
      into `resources/mendeley-lede-porsgrunn-ami/`: 6,809 real Lede AS/
      Porsgrunn AMI customers (hourly active/reactive power, 2022-01-01 to
      2024-09-30), plus weather, spot-price, and pandapower topology data,
      confirmed by loading a sample file directly (24,030 hourly rows per
      customer, matching the stated date range). Not yet used in a chapter.
- [x] Part 4 network/DER data: evaluate NREL SMART-DS against Team-Nando's
      real AusNet MV-LV topology (real GIS network, BSD-3 OpenDSS/`dss_python`
      tutorial code, reused as methodology regardless of which network data
      is chosen). SMART-DS's real S3 layout, verified directly (the actual
      structure is `SMART-DS/v1.0/2018/<Region>/<category>/scenarios/<scenario>/opendss[_no_loadshapes]/`,
      not the flat `<Region>/<Substation>/<Feeder>` initially assumed):
      - `GSO/urban-suburban` (Greensboro) and `AUS/P1U` (Austin) are whole
        multi-feeder regions, not single feeders: 634-1,630 files, 87-109 MB
        of topology alone (`Buscoords.dss` for `AUS/P1U` is 9 MB by itself,
        thousands of buses). Both already ship a real 13-scenario DER matrix
        (`base_timeseries` + solar penetration extreme/high/medium/low crossed
        with battery penetration high/low/none), exactly the "same feeder at
        different DER penetration" structure Part 4 wants, no synthesis
        needed.
      - A single feeder-transformer circuit nested inside, e.g.
        `AUS/P1U/.../opendss_no_loadshapes/p1uhs0_1247/p1uhs0_1247--p1udt12703/`,
        is ~2.4 MB across 8 `.dss` files, genuinely tutorial-sized. This is
        the right granularity for a book chapter, not the whole region.
      - GSO has no AUS-style `p12u`/`p13u` per-feeder codes at the region
        level (only `urban-suburban`/`rural`/`industrial` categories); AUS's
        `P1U`-`P5U`/`P1R` categories do contain individually-addressable
        `<hs-id>/<hs-id>--<dt-id>/` feeder subfolders. Pick the specific
        feeder subfolder up front rather than downloading a whole category.

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
