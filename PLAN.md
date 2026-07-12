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

Not another single-model forecasting tutorial. The real, hard problem a
{{< acr DSO >}} or utility actually faces is forecasting household-level
load demand, and the generation hiding inside it, at the scale of hundreds
or thousands of individual homes, each with genuinely different
variability, not one national-grid-style curve assumed to represent every
customer. Two explicit asks anchor every chapter below: exploit that
variability directly rather than average it away, and scale to thousands of
real meters rather than one feeder-level model. Generation enters the way a
real utility actually sees it, not as its own separately-metered target
(the vendored dataset has no {{< acr PV >}}/{{< acr EV >}} sub-metering,
confirmed), but as the invisible confounder that makes net load look the
way it does. Built on [Twiga](https://github.com/sambaiga/twiga) (point +
probabilistic: conformal prediction, Bayesian NNs, quantile regression)
rather than reimplemented from scratch, but every model below is a tool
inside a real question, never a chapter's own subject, the same discipline
already applied to Part 4 Chapter 6's own "why bother" section. Evaluated
with proper probabilistic metrics (CRPS, PICP, Winkler score), computed
directly via `twiga.core.metrics.evaluate_forecast` (confirmed installed),
not hand-rolled.

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

The vendored copy at `resources/mendeley-lede-porsgrunn-ami/` confirms and
sharpens the numbers above: 6,809 real per-customer AMI parquet files
(`data/ami/`), hourly `dateTime`/`activePowerIn`/`activePowerOut`/
`reactivePowerIn`/`reactivePowerOut` columns, real regional weather and
price data (`data/aux/`), and 185 real LV topology models
(`data/pandapower/topology/`), not just one region-wide network. The real
per-customer parquet layout is itself the reason "hundreds of smart
meters" is achievable directly, no synthetic scaling needed.

Four chapters, organized around real questions rather than model
architectures, revised into this shape after the user rejected an earlier,
architecture-first draft directly: "This plan is focused on ML model
instead of problem and value it add on smart meters data, customers and
utility. I would not want to focus much on my model but use them as a tool
to address the critical challenge and value they add." Checked against
Twiga's own installed package (`twiga[nn]==1.0.0`, already a
`pyproject.toml` dependency for Part 3) directly, module by module, the
author's own private `twiga-forecast` dev repo (not part of this book's
git tree), the author's own real, verified publication list, two local
worked notebooks, and real published literature on the same premise, not
assumed from any of these sources alone. Ordered so each chapter's own real
finding sets up the next one's real question, the same "does complexity
earn its keep" discipline Parts 2 and 4 already run throughout, not a
features list. A fifth angle originally considered for this part, using
these forecasts for {{< acr LV >}} network state estimation, moved to Part
4 Thread 4 instead (see that section below): producing forecasts is Part
3's job, using them for network state estimation is a grid-analytics
question alongside Thread 4's other already-planned angle.

1. **Characterizing home load and PV, and what that says about which
   approach to use.** Combines forecastability diagnosis with honest
   baseline-benchmarking, since both are really the same question: what
   does this signal's own behavior tell us before reaching for anything
   complicated? Grounded in real published literature on this exact
   premise, not just this book's own tooling repeating itself: Peng, Wang,
   Lu, Li, Shi, Wang & Li, "Short-term Load Forecasting at Different
   Aggregation Levels with Predictability Analysis" ({{< acr ISGT >}} Asia,
   2019), using approximate entropy to measure predictability across
   aggregation levels on a real Irish smart-meter dataset, finding
   individual loads "typically more volatile and much more challenging to
   forecast" than substation- or city-level loads, exactly the premise this
   chapter checks directly on the real Lede/Porsgrunn pool; and Fias,
   Hashmi & Deconinck, "Uncertainty Quantification in Load Profiles with
   Rising {{< acr EV >}} and {{< acr PV >}} Adoption" (arXiv, December
   2025), which finds a genuinely counter-intuitive result worth checking
   directly rather than assumed: joint {{< acr EV >}}+{{< acr PV >}}
   adoption can *reduce* net-load uncertainty through a compensatory
   daytime-charging/daytime-generation effect, not the simple "more
   {{< acr DER >}} means more volatility" story a reader might otherwise
   assume. Two more real, current papers ground this chapter's own
   methodology directly, not just its premise: Wang, Klee & Roos, "Time
   Series Forecastability Measures" (arXiv, July 2025), proposing two
   metrics to assess forecastability *before* model development, a
   Spectral Predictability Score (strength/regularity of frequency
   components) and the Largest Lyapunov Exponent (chaos/stability of the
   generating system), reporting strong correlation with actual forecast
   performance on the M5 competition data and a concrete practical
   recommendation this chapter directly adopts: focus modeling effort on
   the customers forecastability already flags as tractable, set different
   expectations for the rest, rather than running the same intensive
   pipeline on every customer regardless. Worth checking directly whether
   these two metrics add real signal beyond twiga's own entropy/Hurst
   battery below, not assumed redundant. And De Bortoli, Ferrari, Ravazzolo
   & Rossini, "Model Selection Confidence Sets for Time Series Models with
   Applications to Electricity Load Data" (arXiv, February 2026), real and
   current, applying a Model Selection Confidence Set method to real
   Italian hourly electricity load data: rather than picking one "best"
   model, it identifies the *set* of models statistically indistinguishable
   from the true data-generating process at a given confidence level,
   finding noisier customers produce larger, more contested sets and
   cleaner ones narrow sharply, and names intraday lags, temperature,
   calendar effects, and solar generation as the real load drivers behind
   that uncertainty. Directly shapes this chapter's own closing "decision
   framework" bullet below: report a real confidence set of viable
   approaches per forecastability regime, not a single falsely-confident
   recommendation, the same honesty this book already applies to "does
   complexity earn its keep" everywhere else. On top of that literature
   base, Twiga's own `twiga.core.stats`
   module (confirmed installed): `get_permutation_entropy`,
   `get_sample_entropy`, `get_hurst_exponent`, `get_dfa_exponent`
   (`entropy.py`), `get_acf_values`/`get_pacf_values`
   (`seasonality.py`/`autocorr.py`), `adf_test`/`kpss_test`
   (`stationarity.py`), `compute_xicorr` (`xicorr.py`),
   `compute_crosslag_association`/`recommend_predictive_lags`
   (`crosslag.py`), plus two more real diagnostics this session found and
   the original plan never cited, `ppscore.py` (predictive power score) and
   `residual.py` (Cook's distance/leverage, for flagging customers whose
   own forecastability read is distorted by a few outlier readings).
   Reused directly from
   [`02-forecastability-analysis`](https://sambaiga.github.io/twiga-docs/tutorials/notebooks/02-forecastability-analysis.html),
   the author's own tutorial (real worked numbers on its own MLVS-PT
   example: Hurst exponent 0.737, {{< acr ACF >}} peaks at lag 48 and 336,
   an estimated {{< acr AR >}} order of 8, trend-stationary). This book's
   own real contribution: running the same diagnostic across hundreds of
   real Lede/Porsgrunn customers individually, not one aggregate signal,
   a real, checkable spread reported honestly. Honest baselines next, same
   signal, reused from
   [`15-baseline-benchmarking`](https://sambaiga.github.io/twiga-docs/tutorials/notebooks/15-baseline-benchmarking.html):
   naive, seasonal-naive, window-average, drift, and context-parrot
   (`twiga.models.baseline`), checked against LightGBM
   (`twiga.models.ml`). The tutorial's own real finding, verified directly:
   seasonal-naive wins outright (MAE 5.32 kW, RMSE 7.08 kW) and LightGBM
   does not beat it either, "if LightGBM cannot beat Seasonal Naive on MAE,
   the problem is in your data or features, not the model architecture,"
   checked per customer across the real Lede pool, not assumed to hold
   uniformly. The vendored dataset has no {{< acr PV >}}/{{< acr EV >}}
   sub-metering, so this chapter cannot characterize {{< acr PV >}}
   generation as its own signal at full scale; its own
   `reactivePowerIn`/`reactivePowerOut` columns (a real, if indirect,
   signature of inverter/{{< acr DER >}} behavior) and Pecan Street's
   small {{< acr PV >}}-separated sample stand in for it instead, checked
   directly against the Fias/Hashmi/Deconinck finding above rather than
   assumed. Closes with a decision framework, not just a diagnostic table,
   framed as a real confidence set of viable approaches per forecastability
   regime rather than one falsely-confident recommendation, per De Bortoli
   et al. above: a mapping from a customer's measured forecastability
   profile to which later chapter's approach is actually worth reaching
   for, e.g. a near-constant, low-entropy customer may never need more
   than seasonal-naive. One more real, checked wrinkle worth flagging here: the
   author's own `MLGAF_HYPERPARAMETERS.md` ablation found the {{< acr UK >}}
   and Portuguese datasets need genuinely different architectures (2 vs. 4
   layers, batch size 64 vs. 8) for the same model family, a sign that
   "which approach" may need re-asking per real dataset/region too, not
   assumed to transfer unchanged.
2. **Scaling to hundreds of meters, does profiling help?** The chapter's
   real question is scale, not architecture. Workhorse models:
   `twiga.models.ml` (`lightgbm_model.py`, `xgboost_model.py`,
   `catboost_model.py`, `randomforest_model.py`, plus the `qrlightgbm`/
   `qrxgboost`/`qrcatboost`/`qrrandomforest`/`ngboost*` variants as
   candidates if chapter 1 flags customers where a single point estimate
   genuinely isn't enough, all confirmed installed). Profiling, the
   chapter's real new idea: reuses Part 4 Chapter 4's own customer-archetype
   clustering (`ark.cluster.idec.fit_idec`, the {{< acr IDEC >}} autoencoder
   already built and validated for this book's Lede-parallel dataset,
   AusNet), run on the Lede/Porsgrunn pool instead of reimplementing a new
   clustering approach. Three real options compared honestly: one model per
   customer (most accurate, least scalable), one shared model per archetype
   (the profiling answer to this chapter's question), and one global model
   (most scalable, least tailored). `twiga.forecaster.ensemble`/
   `twiga.pipeline` (confirmed installed) is the real mechanism for running
   this comparison at pool scale, not a bespoke script. Real deployment
   question, not just accuracy: `twiga.mlops`/`twiga.serve`/
   `twiga.tracking`/`twiga.experiment` (confirmed installed) for what
   training and serving "hundreds of models" actually costs a utility in
   practice, reused rather than building bespoke {{< acr MLOps >}} tooling.
   Sets up chapter 3 directly: whichever scaling tier chapter 2 finds
   actually earns its cost becomes the backbone chapter 3 wraps in
   uncertainty.
3. **What is a probabilistic forecast actually worth, and which kind
   should you trust?** Four real {{< acr UQ >}} paradigms, compared by the
   real decision each one supports, not by which architecture backs them.
   **Parametric distributional heads** (`twiga.models.nn.mlpf_model` and
   its Normal/Student-t/Laplace/LogNormal/Gamma/Beta variants, from the
   author's own published "Efficiency through Simplicity: {{< acr MLP >}}-
   based Approach for Net-Load Forecasting with Uncertainty Estimates in
   Low-Voltage Distribution Networks," IEEE Transactions on Power Systems,
   40(1), 2025): a full density, the only family that directly answers
   "what's the probability my load exceeds X kW," a real risk-of-exceedance
   question, at the cost of assuming a fixed parametric family.
   **Quantile regression / {{< acr FPQR >}}** (`mlpfqr_model.py` and
   `mlpffpqr_model.py`, the {{< acr FPSeq2Q >}} architecture from
   "{{< acr FPSeq2Q >}}: Fully Parameterized Sequence to Quantile
   Regression for Net-Load Forecasting With Uncertainty Estimates," IEEE
   Transactions on Smart Grid, 13(3), 2022, plus chapter 2's own `qr*` tree
   models as a cheaper backbone): no distributional assumption, a direct
   answer to a capacity/sizing decision ("what's the {{< acr P95 >}} load a
   transformer needs to survive"). A newer, in-preparation successor worth
   checking directly if chapter 1 finds heavy-tailed residuals, not
   assumed necessary: FPQR_Kuma/FPQR_V2 (two overlapping drafts, a
   Kumaraswamy tail-adaptive quantile head, reporting 12-18% forecast-skill
   gains on heavy-tailed distributions in the author's own ablations).
   **Conformal prediction** (`twiga.distributions.conformal`, two real
   families: `crc.py`, residual-based, from the author's own published
   "Conformal Multilayer Perceptron-Based Probabilistic Net-Load
   Forecasting for Low-Voltage Distribution Systems with {{< acr PV >}}
   Generation," IEEE SmartGridComm, Oslo, 2024, detailed in the author's
   own PhD thesis Chapter 7 (`resources/PhD_thesis___AF/Chapters/
   Chapter-7.tex`), real reported {{< acr PICP >}} 0.85 vs 0.79 and
   {{< acr NMPI >}} 0.29 vs 0.23 for an absolute-vs-signed non-conformity
   score on {{< acr MLVS-PT >}}; and `cqr.py`, Conformalized Quantile
   Regression, pairing naturally with chapter 2's own `qr*` tree models). A
   newer, in-preparation follow-up, MLPF_CRF ("Adaptive Conformal
   Prediction with an MLPGAM Architecture"), reports a stronger real result
   on the same problem (CWE 0.815, 9.6% narrower intervals, 22.2%
   Winkler-score reduction vs standard split-conformal), and the author's
   own `CRC_ABLATION_ANALYSIS.md` (an already-run, 87,624-eval ablation)
   both supports and corrects it: found and fixed a metric-artifact bug in
   the paper's own {{< acr CWE >}} score, corrected result `conf_90_sign`
   beating standard residual conformal in 8 of 10 folds. Conformal
   prediction is the only one of the four families with an actual
   distribution-free coverage *guarantee*, the most defensible answer for a
   compliance or regulatory risk statement, at the cost that a guarantee on
   coverage says nothing about sharpness. The chapter's own genuinely new
   contribution stays here: cluster-conditional conformal calibration
   (`resources/Adaptive-conformal-PV-Forecasting.ipynb`'s real working
   code, `cal_UMAPF`, `gmm.predict`, per-cluster
   `calculate_conformal_value`), reusing chapter 2's own archetypes instead
   of a fresh {{< acr UMAP >}}/{{< acr GMM >}} clustering pass, one global
   conformal threshold vs one per archetype, checked honestly, not assumed
   sharper. Also worth checking directly since it changes what decision the
   forecast supports: the real dataset's own `reactivePowerIn`/
   `reactivePowerOut` columns enable a real, checkable version of a third
   published paper, "Enhancing {{< acr LV >}} system resilience through
   probabilistic forecasting of interdependent variables: voltage, reactive
   and active power" ({{< acr CIRED >}} Chicago Workshop 2024), jointly
   forecasting active and reactive power, a genuinely different decision
   from any single-target family above (reactive-power intervals feed a
   voltage-support/{{< acr DER >}}-dispatch decision, not a capacity-sizing
   or exceedance one). **Bayesian forecasting**
   ([`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast),
   a real, mature, {{< acr PyPI >}}-published package, {{< acr SVI >}} and
   {{< acr NUTS >}} inference, rolling-window backtesting, real
   {{< acr CRPS >}} evaluation built in, a new project dependency): a
   genuinely different question from the other three, not "what's the
   range of outcomes" but "how confident is the model itself in this
   forecast," separating epistemic (model) uncertainty from aleatoric
   (irreducible) noise, at the cost of heavier compute and no formal
   coverage guarantee. The chapter's real deliverable: one honest
   comparison table across all four families on real coverage
   ({{< acr PICP >}}), sharpness ({{< acr CRPS >}}/Winkler score, via
   `twiga.core.metrics.evaluate_forecast`), and, matching Chapter 6's own
   {{< acr AUC-ROC >}} decision, which real decision each family is
   actually suited to, not a single "winner." Directly reusable for the
   conformal half of this comparison: twiga's own
   `10-conformal-prediction.ipynb` tutorial. A model-architecture
   side-note, checked but not the chapter's focus: whether a group-additive
   backbone (GANF/MLPGAF, in preparation, `twiga.models.nn.mlpgaf_model`,
   real ablation numbers: gating alone +0.5% MAE, loss design +3.3%, warmup
   scheduling avoids a -1.88% degradation without it) changes which
   {{< acr UQ >}} family wins, or is orthogonal to the question, one
   paragraph, not a chapter. Twiga's own model zoo has several more real
   backbone families beyond MLPF and MLPGAF (`mlpgam`, `nhits`, `rnn`, each
   with 8-10 distributional/quantile variants), available as a stronger
   backbone if this chapter's own honest comparison calls for one, not
   surveyed for their own sake.
4. **Do foundation models actually help, and do they fail like everything
   else does?** Keeps `twiga.models.foundational` (`moirai_model.py`,
   `timesfm_model.py`, `chronos2_model.py`, `lag_llama_model.py`,
   `tabicl_model.py`, confirmed installed, plus the
   `18-foundational-models.ipynb` tutorial), benchmarked zero-shot against
   chapter 2's own per-customer/per-archetype/global comparison and chapter
   1's baselines, per `15-baseline-benchmarking`'s own established pattern.
   The real question is not "do foundation models scale" (chapter 2's job)
   but **do foundation models fail the same way other models do, or
   differently**: cross-referencing every model's real errors against
   chapter 1's own per-customer forecastability profile, checking directly
   whether a foundation model's failures cluster on the same hard customers
   classical/{{< acr NN >}} models already struggle with, or on a
   genuinely different subset (e.g. {{< acr PV >}}-heavy, volatile
   customers foundation models were never pretrained to expect). A real,
   practically useful finding either way: if failure modes overlap, a
   foundation model is a substitute; if they diverge, it's a real candidate
   for an ensemble/fallback role instead, a more useful recommendation than
   a single accuracy leaderboard.

Data source, scope, and sequencing above are the real, checked state as of
this session; detailed per-chapter notebook design (exact cells, exact
real numbers) is not started, the same "confirmed rather than assumed"
discipline the rest of this plan already applies, deferred until Part 3
work actually begins, starting with Chapter 1 since every later chapter
depends on its forecastability/decision-framework output.

### Part 4: Grid-Edge Value (amber, flagship)

Five threads. The original four are ordered so each builds on the last;
thread 5 was built ahead of threads 3-4 by explicit decision since it
depends on neither, and is now built and merged. Threads 1, 2, and 5 are
all built and merged. Thread 3's own data source was revised this session
after checking directly (an Explore agent confirmed Chapter 4's own
SMART-DS usage never solves a live circuit, shape-only), the same
discipline threads 1-2 already applied when they moved off SMART-DS onto
AusNet's own real 31-customer feeder; thread 3 now reuses AusNet as
primary and Thread 5's own UK network (`deakinmt/uk-mvlv-models`) as
secondary, not SMART-DS. Thread 4 is blocked, not just unstarted: it
depends on Part 3's forecasts, and Part 3 does not exist yet
(`book/03-forecasting/` is empty, `twiga` is an unused dependency pin),
confirmed directly rather than assumed, so thread 4 is deliberately not
planned until Part 3 ships a real forecast to design against.

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
2. **Customer/feeder clustering and segmentation from meter data**
   (Chapter 4, branch `part4-ch4-customer-feeder-clustering`): grouping
   customers and feeders by how they actually behave, not how they're
   labeled. Revised plan, decided after checking data directly rather
   than assuming SMART-DS was still the right fit: AusNet's own
   342-customer real smart-meter pool (Chapters 1-3's dataset, no new
   fetch step, genuinely real household readings, not synthetic) carries
   the chapter's core customer-archetype clustering, a direct payoff of
   Chapter 2's own "Real customers, real diversity" finding ("no two
   profiles agree"). SMART-DS `AUS/P1U` (~8-10 substations, ~24-30 LV
   feeders, CC-BY-4.0, confirmed reachable) supports a smaller, secondary
   feeder-level clustering section, since AusNet's single 31-customer
   feeder can't support feeder-level clustering on its own. Ten real
   reference papers (five foundational 2014-2017, five current 2020-2026,
   full citations in `references.bib` once added): McLoughlin et al.
   (2015), Kwac et al. (2014), Haben et al. (2016), Li et al. (2015),
   Guo et al. (2017, IDEC); Rajabi et al. (2020), Michalakopoulos et al.
   (2024), Kumar & Mallipeddi (2024), Yerbury et al. (2026, CROCS), Ledva
   & Mathieu (2020). Structure: hook (Chapter 2's diversity chart) ->
   theory (clustering vs. classification, shape not magnitude) -> data ->
   baseline k-means with a real optimal-k check -> a deep-clustering
   (IDEC) contrast, genuinely tested not assumed to win -> feeder-level
   clustering -> a novel contribution none of the 10 papers test, are
   archetypes stable across a year or a snapshot artifact, reusing
   `adjusted_rand_score` (Chapter 3's own tool) to measure stability
   instead of correctness -> a real archetype-vs-DER-risk cross-check
   using Chapter 2's own `run_penetration(seed=42)` per-customer subset
   (exactly reproducible) against this chapter's archetype labels -> a
   sensitivity "Do it yourself" -> a "Why bother" section (matching Part
   2's own closing pattern) tied explicitly to Chapter 1's three DER
   strain modes and Chapter 2's PV-vs-EV constraint finding, with both
   utility-side (proactive connection triage, targeted mitigation) and
   customer-side (faster approval, fairer tariffs) value made concrete by
   the risk cross-check and stability finding, not asserted narratively.
   Reusable assets found in `resources/profiling 3/src/`: a real,
   MIT-licensed IDEC implementation (`net/idec.py`, matches
   `dawnranger/IDEC-pytorch`) to adapt rather than build from scratch, and
   `utils/prepare_feature.py`'s `create_seasonal_features` for the
   stability contribution. Notebook first, matching Chapters 1-3's own
   build discipline; `.qmd` narrative after the notebook is reviewed.
   Built and merged.
3. **Anomaly detection combining smart-meter data and LV network topology**
   (Chapter 3, branch `part4-ch3-anomaly-detection`), reframed this session
   after direct research (two codebase-exploration agents plus verified
   literature search) away from PLAN.md's original "run isolation forest
   on a voltage timeseries" sketch. Every anomaly-detection paper found,
   including the most current ones (2025-2026), still frames detection as
   one meter, one baseline, decide if today's reading deviates from it, an
   assumption that breaks down as LV networks evolve: DER adoption is a
   permanent re-baselining, not a one-time event (Chapter 4's own 0.06-0.34
   cross-quarter archetype-drift finding), and DER adoption clusters and
   synchronizes, so a real anomaly can live in how several individually
   normal customers coincide, not in any one customer's own meter, a
   structural blind spot no per-customer detector in the literature found
   this session can see by construction. Three chained questions: (1) does
   today's per-customer detection paradigm even work, tested honestly with
   a real feature representation (a rolling-window/{{< acr FFT >}}
   representation plus Chapters 4-5's own shape embedding, not raw
   per-step readings, echoing Part 2's own feature-engineering lesson for
   a different signal), a parametric-and-non-parametric detector ensemble
   (`sklearn`'s `EllipticEnvelope`/`KernelDensity`/`IsolationForest`/
   `LocalOutlierFactor`, plus a hand-rolled ECOD score, Li et al. 2023, no
   new dependency), and the threshold set by a fourth application of this
   book's own split-conformal calibration rather than a heuristic
   percentile (Hennhöfer, Kirsch & Preisach 2026 is the real precedent for
   treating anomaly thresholds as a conformal-calibration problem); (2) the
   chapter's first real contribution, an anomaly no single meter can see,
   a real coincident-EV-charging scenario every Section-1 detector misses
   by construction, caught instead by a topology-weighted coincidence
   check, tested on AusNet (primary) and extended across Thread 5's own
   414-feeder UK network for statistical weight (a GNN enhancement was
   checked for feasibility and dropped before being built: AusNet's own
   feeder is a single-transformer star, not a multi-hop graph, so a GNN's
   real strength has nothing to work with here, a specific reason, not a
   generic deferral); (3) the chapter's second real contribution, is the
   detector itself still trustworthy as the network keeps changing, a
   simulated multi-stage DER-adoption trajectory tracking what fraction of
   the population falls outside Chapter 5's own conformal retrieval-
   confidence threshold over time, a real, checkable, forward-looking
   early-warning KPI for exactly when a {{< acr DSO >}}'s own archetype
   and retrieval model needs refreshing, built entirely from machinery
   Chapters 3-5 already validated, no new algorithm. An energy-balance /
   non-technical-loss conservation check (sum of every customer's real
   metered consumption against the feeder head's own real power flow) is
   folded into Section 1 as a third detection type, answering PLAN.md's
   own dropped electricity-theft idea honestly: no theft label needed,
   this is a physics-based conservation check, not a classifier trained on
   labeled fraud, sidestepping the exact reason theft detection was
   dropped in the first place. Not yet started; full research, reference
   list, and section-by-section design in the approved plan this session
   produced (superseding the paragraph this replaces).

   **Feature representation, tried and dropped, a real, checked
   answer**: Section 1's own shape/rolling-stat/{{< acr FFT >}} features
   are a low-frequency (30-minute AMI) representation, chosen for a
   low-frequency signal, the same discipline that made Part 2's own
   high-frequency WRG/AWRG distance-matrix representation (Chapter 2,
   real PLAID/LILAC current/voltage data, `resources/nilm-code/AWRGNILM`'s
   `get_distance_measure()`/`get_img_from_VI()`) the right choice there.
   Built and checked directly rather than assumed: a WRG-style
   distance-matrix embedding of a customer's own rolling window trailed
   the existing shape/FFT features on this book's own AMI-rate data, both
   on recall and on threshold-independent AUC-ROC, while taking
   meaningfully longer to compute, a real complexity-versus-benefit
   question answered rather than deferred. Confirms the same lesson Part
   2 taught for a different signal does not transfer automatically to a
   different frequency; not added to the shipped chapter.

   **Root-cause explanation, built this session, a real and honestly mixed
   finding**: SHAP's `TreeExplainer` on the ensemble's own `isolation_forest`
   component (the one detector here with a fast, exact Shapley explainer)
   gives a real, mathematically exact attribution of that detector's own
   score back to individual input features. Checked directly across several
   real customer-day pairs, not assumed to work cleanly: the top-attributed
   features do not reliably line up with the exact clock hours where an
   injected stuck-reading fault sits, since `IsolationForest`'s own random
   splits have no built-in preference for temporal locality. What it does
   deliver honestly is a coarser, still useful signal, which *kind* of
   feature (a specific hour's shape, the day's own mean/spread, or a
   frequency-domain component) drove a flagged case, a real first step
   toward triage, not a guarantee of pinpointing the hour a field crew
   should check first.

   **Anomaly persistence/recurrence tracking, worth adding as a follow-up,
   not yet built**: distinguishing a one-off blip from a developing fault
   by checking whether the same real customer's own detector flags the
   *same* failure mode across several consecutive real days, not just one
   isolated injected day the way the current recall test already checks.
   Reuses the exact same per-customer self-baseline pipeline Section 1
   already built, no new algorithm, just a different question asked of it:
   run the ensemble across a real consecutive run of days with a repeated
   injected fault and report how many of those days it actually catches in
   a row, a real, checkable "is this getting worse" signal a DSO can act
   on differently than a single flagged day.

   **DER event/behavior-based anomaly, a genuinely different framing,
   worth its own follow-up notebook, planned and executed with the same
   care as this chapter rather than bolted onto it**: every detector in
   this chapter, per-
   customer and topology-aware alike, asks whether a *signal value*
   (voltage, current, power magnitude) looks unusual. A real, different
   question: is a customer's own *DER behavior*, when they export, how
   often, for how long, unusual, independent of whether any single
   reading's magnitude crosses a threshold. A PV inverter exporting at an
   odd hour, or a real export/charging pattern whose own frequency or
   timing has shifted, can be a real, meaningful signal (a fault, a
   metering irregularity, a changed household routine) even when every
   individual reading stays inside a normal magnitude range, the entire
   point of asking a behavioral question instead of a magnitude one. Real,
   buildable groundwork already exists in this book's own data: Section
   2's own net-load construction (`load - PV_KVA * pv_data`) already turns
   raw readings into real export/import events, so a first version could
   define a real "export event" as any half-hour a PV-equipped customer's
   own net load goes negative, then ask whether the count, timing, or
   duration of those events, not their magnitude, looks unusual against
   that customer's own real history, split-conformal calibrated the same
   way as every other threshold in this chapter. AusNet's own real data
   has no separate {{< acr EV >}} sub-metering (confirmed, matches Part 3's
   own dataset finding), so an equivalent EV-charging-event version is not
   directly buildable on this data without a real proxy signal, worth
   checking directly rather than assumed once this angle is built.

   **COPOD, worth adding as a follow-up detector, not yet built**: Li,
   Zhao and colleagues' 2020 copula-based outlier detector (IEEE ICDM),
   ECOD's own older sibling from the same authors. Checked directly
   against PyOD's own source rather than assumed: like ECOD, it is
   parameter-free and works from a per-dimension empirical CDF, but
   weights each dimension's tail by that dimension's own skewness before
   combining, roughly 40 lines of real logic once class boilerplate is
   stripped, the same "simple enough to hand-roll" bar ECOD was held to.
   A real, natural sixth addition to `ark.anomaly.detectors`, no new
   dependency, worth checking whether it adds real detection signal
   beyond ECOD on this chapter's own injected-anomaly test before adding
   it to the ensemble, not assumed to help just because it is newer.

   **Detector selection guidance instead of ensembling by default, a
   genuinely different pedagogical framing, worth a dedicated future
   chapter or section, not a bolt-on to this one**: this chapter builds
   an ensemble first and asks afterward whether the ensemble beats a
   single family (Section 1's own parametric-vs-non-parametric and
   combination-rule comparisons). A different, arguably more useful
   teaching order: start by guiding the reader through *choosing* an
   appropriate detector for their own data's real characteristics
   (dimensionality, whether a real covariance structure exists, expected
   anomaly locality, contamination rate, whether interpretability
   matters more than recall) before ever reaching for an ensemble, the
   same diagnose-first instinct this book already applies elsewhere
   (Chapter 2's raw-current-vs-feature-representation check, Section 1's
   own feature-representation-vs-raw-reading test above). This is a
   real, separate initiative, not a Chapter 6 addition: it reframes how
   the whole chapter teaches detector choice, not just adds a detector,
   so it belongs in its own future plan, not scoped or built now.
4. **Smart grid analytics and network management** (blocked, not planned in
   detail yet, confirmed rather than assumed): evaluate grid health
   (voltage violations, thermal overloading) under *forecasted* DER growth
   scenarios, then test mitigation strategies, the scenario-based
   counterpart to thread 3's real-time detection. This is the chapter where
   Part 3 stops being standalone: its forecasts become the DER-penetration
   scenario driver here, not just a chapter that happens to come before
   this one. Checked directly this session: `book/03-forecasting/` does
   not exist, `_quarto.yml` has no Part 3 entries, and `twiga` is an unused
   `pyproject.toml` pin, so there is no forecast artifact to design this
   thread against yet. The Maree dataset (Lede AS/Porsgrunn, real, already
   vendored) is net active/reactive power only, no {{< acr PV >}}/
   {{< acr EV >}} sub-metering, so even once Part 3 exists, this thread's
   DER-growth driver needs to come from a proxy derived from net-load
   forecasts, not a direct DER forecast, a real design question to answer
   once Part 3 actually ships. "Mitigation strategies" still maps to
   Team-Nando's Operating-Envelope/ANM/Volt-Watt-control tutorial code
   (BSD-3, reused as methodology per the `nilm-code-reference`-skill pattern
   of reusing algorithms, not raw data or figures), confirmed to exist
   (`Team-Nando/OEs-The-Basics`, `Team-Nando/ANM-of-LV-Networks-Rule-Based-
   Approach`) but not yet vendored locally. Do not plan this thread in
   detail until Part 3 exists: build Part 3 first.

   Two angles to fold in once Part 3 ships a real forecast to design
   against, recorded here so they are not lost before detailed planning
   starts:
   - **What-if scenario analysis**, combining Part 3's own net-load
     forecast with the LV network model Part 4 already built (Chapters 1-2's
     `ark/dss/` power-flow machinery): run a forecasted future load/DER
     trajectory through the real network model to see which real
     constraints (voltage, thermal) a DSO would face under that scenario,
     before it happens, not just under today's snapshot. This is the same
     "scenario-based counterpart to thread 3's real-time detection" framing
     already above, made explicit as a named capability rather than left
     implicit in "evaluate grid health under forecasted DER growth."
   - **State estimation of the LV network itself from smart-meter data**,
     complementing {{< acr AC >}} power flow: a different question from
     forecasting one customer's own future load, and the user's own explicit
     ask this session to fold into this thread rather than Part 3, since
     producing forecasts is Part 3's job and using them for network state
     estimation is a grid-analytics question. Real LV networks are
     chronically under-monitored below the substation, sparse or no
     real-time measurement at most buses, so a {{< acr DSO >}} often does
     not know its own network's current voltage profile or loading state
     directly. Distribution System State Estimation ({{< acr DSSE >}}) is
     the established real technique for inferring a network's full state
     from sparse measurements plus a network model; the real, checkable
     question is whether smart-meter readings, paired with the forecasts
     and calibrated intervals Part 3 will have already produced by the time
     this thread starts, can serve as real pseudo-measurements that improve
     a classical {{< acr AC >}} power-flow-based state estimate at the
     buses this book's real networks (AusNet, the UK {{< acr MV >}}/
     {{< acr LV >}} network, Part 4's own) don't directly meter. Open,
     unverified as of this session, flagged honestly rather than assumed:
     `pandapower` has a real, established {{< acr WLS >}} state-estimation
     module (`pandapower.estimation`), not currently a dependency of this
     project (confirmed: `pandapower` is not installed here). This book's
     own established power-flow engine is OpenDSSDirect.py instead (used
     throughout Part 4, confirmed via `ark/dss/circuit.py`'s own imports).
     Before this thread's notebook design starts, a real, direct decision
     is needed: adopt `pandapower` as a second power-flow dependency for
     this one thread (its own state-estimation module is mature and would
     be reused, not reimplemented, matching this book's "borrow a real
     package" precedent), or keep everything on OpenDSSDirect and implement
     a smaller, from-scratch {{< acr WLS >}} estimator against Part 4's own
     AusNet/UK network models instead, avoiding a second power-flow
     dependency, at the cost of building the estimator by hand. Not
     resolved here, a real open question for whoever starts this thread's
     work, not papered over, the same "confirmed rather than assumed"
     discipline the rest of this thread already applies.
5. **Ranking and recommendation for LV management under DER** (new thread,
   Chapter 5, branch `part4-ch5-ranking-recommendation`, built ahead of
   threads 3-4 by explicit decision, it needs neither): applies
   established machine-learning ranking and recommendation techniques,
   borrowed from other domains, not invented for this book, to real LV
   problems under DER. Three chained sub-problems, all grounded in what
   Chapters 1-4 already built: retrieval-based cold-start prediction (a
   new/unmonitored customer's DER risk, predicted by retrieving similar
   customers in Chapter 4's own embedding space and using their
   already-simulated risk, extending Chapter 3/4's split-conformal
   machinery a third time into a retrieval-confidence set); priority
   ranking at two granularities (customers, for connection-queue triage;
   real bus positions, for network-reinforcement priority, grounded in a
   real transformer-restoration-priority precedent) plus a real-time
   violation-severity ranking borrowed directly from industrial
   alarm-flood management (a different established discipline than the
   other two); and mitigation-lever recommendation (Volt-Watt, Volt-VAr,
   storage, no action), a case-based recommender checked against real
   simulated before/after fixes, not a rule of thumb. Thirteen real
   reference papers, split between general-domain foundations (Aamodt &
   Plaza 1994 on case-based reasoning, Adomavicius & Tuzhilin 2005 on
   recommender-system taxonomy, Liu 2009 on learning to rank, Bernardi et
   al. 2015 on the e-commerce cold-start problem, Parvez et al. 2022 on
   alarm-flood ranking, Liao et al. 2026 on conformal ranking, He et al.
   2020's LightGCN considered but not committed) and energy-domain
   precedent showing where this has and has not started (Luo et al. 2017,
   Qammar et al. 2024, Andagoya-Alba et al. 2024, Gangwar & Shaik 2023,
   Duran & Monti 2025, Hardowar et al. 2017). Data: AusNet stays primary
   (the one network with a genuine real feeder-to-customer pairing); a
   real UK network, `deakinmt/uk-mvlv-models` (Deakin et al. 2021, real
   LVNS circuits on a real UKGDS backbone, 112,887 buses, 19,072 loads,
   414 real feeders), checked directly (downloaded, unzipped, solved) and
   verified clean, becomes the secondary source, AusNet's real customer
   shapes borrowed onto its real topology. Two other real candidates
   (CRE21, a real Australian network already vendored locally; the
   Koirala et al. 2020 non-synthetic European system) were checked and
   logged, not used: CRE21 has an unresolved structural voltage-collapse
   defect that doesn't scale away with load, and the European system
   ships only raw GIS/MATLAB conversion scripts, no ready OpenDSS model,
   both real, disclosed reasons rather than silently dropped. "Why
   bother" ties back to Chapter 1's three DER strain modes explicitly,
   utility- and customer-side value backed only by analysis already shown
   earlier in the chapter, the same discipline Chapter 4's own "why
   bother" followed. Notebook first, matching Chapters 1-4's own build
   discipline; `.qmd` narrative after the notebook is reviewed. Built and
   merged (PR #11).

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
- [x] Write `book/04-grid-edge/03-phase-identification` (branch
      `part4-ch3-phase-identification`), Part 4's thread 1: recovering
      which phase each customer is actually connected to from voltage
      correlation alone, no field audit. Reuses the real 31-customer
      AusNet feeder and smart-meter data Chapters 1-2 already vendored
      (no new data-fetch work), checked against the real per-load phase
      ground truth (extracted from each service line's own source-side
      bus suffix, not the load's own `bus1`, which is uniformly `.1`).
      Five real reference papers cited: Short (2013, the foundational
      correlation method); Simonovska & Ochoa (2021, PCA + k-means,
      tested on this exact AusNet feeder, the direct precedent this
      chapter is built on); Blakely & Reno (2020, Sandia, ensemble
      spectral clustering needing no existing labels); Hoogsteyn et al.
      (2022, a benchmark across meter accuracy/density, the basis for
      this chapter's own sensitivity section); Hangawatta et al. (2025,
      a neural-network approach for low-frequency data closer to this
      book's own 30-minute resolution). Core methodology: naive PCA on
      raw voltage underperforms (ARI 0.24), PCA on the correlation
      matrix recovers all 31 real phases exactly (ARI 1.0), matching
      Simonovska & Ochoa's own published result. Sensitivity: reliable
      down to 12 hours of data and 7 meters, degrading unpredictably
      below either threshold. Extended beyond the five cited papers with
      two genuinely novel contributions: minimum-anchor labeling (how
      many known-phase customers are needed to name, not just group, the
      clusters; reliable at 8-12 anchors) and split conformal prediction
      sets built on those same anchors, giving each customer a
      statistically calibrated confidence set rather than a bare point
      estimate, validated on a degraded window where it correctly flags
      the customer the point estimate actually misclassifies. Also
      compared six pairwise association measures (Pearson, Spearman,
      Kendall, xicor, mutual information, PPS) under a severe 2-hour
      window: linear/rank-based measures degrade together, mutual
      information and PPS hold up. A planned 200-customer scale-up story
      was investigated and abandoned: three independent candidate
      networks (SMART-DS, a locally vendored four-feeders network, and
      the real Australian CRE21 network) were tested directly and none
      of them produce a phase-differentiated voltage signal, confirmed
      with a deterministic 80x demand-imbalance test on each, because all
      three model a customer's final connection as a bundled multi-phase
      conductor rather than AusNet's genuinely separate single-phase
      service drop. Notebook and `.qmd` both complete, rendered, and
      committed on branch `part4-ch3-phase-identification`; not yet
      merged.
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
