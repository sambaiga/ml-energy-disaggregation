# Book restructuring: one Part per ML-technique family + real new content

## Context

The author received an external write-up proposing a large slate of new topics for this book ("Machine Learning for Smart Meter Data": phase identification, non-technical-loss/theft detection, predictive maintenance, outage localization, dynamic operating envelopes, federated learning, cyber-physical spoofing detection, an IR/RecSys reframing of Part 4 Chapter 5, and physics-informed ML). Rather than accept it wholesale, the author asked for a critical review grounded in what this book's own prior work actually supports, then — mid-session — escalated the ask into a full structural reorganization: instead of one bundled "Part 4: Grid-Edge Value" covering network modeling, phase ID, clustering, ranking, and anomaly detection together, each ML-technique family becomes its own top-level Part, each exploring that technique's application to LV distribution + smart-meter data specifically. Every part/chapter should go beyond basic technique demonstration to include real theoretical foundations, a literature-grounded best-practices review, and at least one genuinely novel angle — the book should read like a reference for graduate students, DSOs, data scientists, and researchers, not an applied-ML tutorial.

This plan is the product of three parallel Explore agents (full book/PLAN.md structure catalog; a deep technical read of the three most-relevant existing chapters against every proposed topic; an exhaustive `resources/` search for real prior-research anchors) plus one Plan agent that pressure-tested the proposed restructuring specifically for mechanical/narrative risk. The author has since confirmed two open placement questions raised by that pressure-test (below).

## Authoring principle for every Part from 3 onward

Every Part after Part 2 presents, in order: (1) the domain-specific theoretical foundation for that Part's own topic, (2) a state-of-the-art/best-practices literature review, (3) real use-cases and applications. Part 2 holds all *generic* ML theory (what k-means minimizes, what conformal prediction guarantees, what gradient boosting derives from) exactly once; no later Part re-derives it. A later Part's own "theoretical foundation" section is domain-specific only — e.g., Part 5 (Clustering)'s own theory section is about *what makes smart-meter load shapes a genuinely hard clustering problem* (multi-scale periodicity, behavioral drift, magnitude confound), not a re-explanation of what k-means is, which lives in Part 2 Ch.06. Likewise Part 4 (Network Foundations) holds its own domain-specific theory (per-unit voltage, Kirchhoff's/Ohm's laws, the AC power-flow equations) directly, since that's electrical-engineering domain knowledge, not generic ML — it does not belong in Part 2 either.

## Part 1 and Part 2: tentative chapter outlines

Both parts already have a real, previously-drafted outline in `PLAN.md` (lines 61-256), built well before this session's restructuring — genuinely well-scoped and, notably, *already designed around the exact no-repeat principle above* (its own stated purpose: "Part 2 is that missing foundation... not a survey of ML for its own sake"). Adopting it as the tentative plan rather than re-deriving from scratch, with one real gap identified and one mechanical note.

**Part 1: Signal Foundations** (unchanged from `PLAN.md`, 3 chapters) — general, reusable signal-processing content every later Part's *signal* (not analysis technique) sits on top of:
1. Reading a meter signal (acquisition, sampling rates, high-frequency current/voltage vs. low-frequency AMI, where every later Part sits on that same frequency spectrum).
2. Detecting change and events in a time series (classical, model-free only — sliding-window edge detection, CUSUM; the *learned* detectors are explicitly a Part 2 topic, not this one).
3. Feature engineering for power signals: steady-state vs. transient (generic time/frequency-domain representations; the NILM-specific advanced versions stay in Part 3).

`PLAN.md`'s own disclosure carried forward, not silently resolved: "not yet verified against the source papers... this is the proposed scope... not re-derived from scratch this session."

**Part 2: ML Foundations** (`PLAN.md`'s existing 7 chapters, **plus one new 8th chapter this restructuring requires**):
1. What Is Machine Learning: Taxonomy and Learning Theory (ERM, bias-variance, no-free-lunch, generalization bounds).
2. The ML Workflow, Evaluation, and Diagnosis (metric derivations, proper scoring rules, exchangeability).
3. Optimization and Neural Network Foundations (SGD, backprop, universal approximation, regularization as Bayesian prior).
4. Classical Model Families: Linear Models and Tree Ensembles (MLE justification, boosting derivation, SHAP).
5. Deep Architectures for Structured and Sequential Data (CNN, RNN/LSTM/GRU, attention, autoencoders, U-Net, multi-task learning).
6. Unsupervised Learning: Clustering, Dimensionality Reduction, and Retrieval (PCA, k-means, IDEC, UMAP, embedding retrieval/case-based reasoning) — the direct theoretical grounding for Part 5 and Part 6.
7. Anomaly Detection and Probabilistic Machine Learning (density/distance/isolation detectors; quantile loss, conformal prediction theory, Bayesian NNs, CRPS) — the direct theoretical grounding for Part 7 and Part 8.
8. **[NEW] Optimization and Decision-Making Under Uncertainty** — a real, confirmed gap: none of the original 7 chapters cover convex optimization, risk measures, or control theory, because Part 9 (Optimization for Decision-Making) did not exist when this outline was drafted. `PLAN.md` itself already anticipated needing this (its own line 1108: the "Optimization and Decision-Making AI" tier that Part 2 Ch.01's own taxonomy names but never built out). Real, standard content: convex sets/functions and why convexity guarantees a global optimum, Lagrangian duality and KKT conditions; Value-at-Risk vs. Conditional Value-at-Risk (Rockafellar & Uryasev 2000's real CVaR-as-linear-program formulation — the actual theoretical foundation Part 9 Ch.02 currently lacks anywhere in the book); receding-horizon/Model Predictive Control (the real finite-horizon-solved-repeatedly formulation, stochastic MPC as its uncertainty-aware extension) — the direct grounding for Part 9 Ch.03; multi-criteria decision theory (when a simple weighted/worst-of score suffices vs. when true Pareto-optimality is needed — grounds both Part 6's own "transparent score, not a learned ranker" choice and Part 9's rule-based-vs-CVaR-optimal comparison). Candidate references, needing the same citation-accuracy check `PLAN.md` already runs on every other chapter before use: Boyd & Vandenberghe, *Convex Optimization*; Rockafellar & Uryasev (2000), *Journal of Risk*; Rawlings, Mayne & Diehl, *Model Predictive Control: Theory, Computation, and Design*.

**Mechanical note, real and confirmed**: `PLAN.md`'s own Part 1/2 draft text contains 12+ literal "Part 4 Thread N"/"Part 5" cross-references to the *old* pre-restructuring numbering (e.g. "the exact architecture Part 4 Thread 2 already reuses," now Part 5). These need the same hand-verified remapping sweep as every rendered chapter in Phase 0 — `PLAN.md`'s own planning prose is not exempt from the renumbering cost already documented above.

## Real network models to supplement AusNet (checked directly)

- **Team-Nando/MV-LV-Networks is a major, confirmed find**: it is *more real AusNet Services (Victoria, Australia) data* — the same utility as this book's existing 342-customer feeder, but four different, larger, real networks: two rural SWER (single-wire earth return) networks, **SMR8** (3,608 customers, 704 transformers) and **KLO14** (4,691 customers, 700 transformers), and two urban networks, **HPK11** (5,274 customers, 44 transformers) and **CRE21** (3,374 customers, 79 transformers). Real anonymised 30-min residential demand profiles are paired with each network. OpenDSS via `dss_python`, BSD-3-Clause.
- **`CRE21` is already vendored in this repo**, at `resources/cvar_flexibility/simulation/Network-CRE21/` — confirmed by matching its real `basekv=66`/50Hz circuit definition against the Team-Nando repo's own description, not previously recognized as this specific dataset. Essentially free to formalize into use.
- **Recommendation**: fetch **SMR8 or KLO14** specifically for the network-foundations and outage-localization chapters. SWER is a genuinely different real LV architecture never touched in this book (long, single-wire, high-impedance radial structure with materially different fault/outage propagation behavior than the urban feeder everywhere else in the book) — and same-utility generalization (AusNet-to-AusNet) is a cleaner comparison than switching countries, since it isolates real feeder-structure effects from confounding utility-specific practices. Larger customer counts (3,600-5,300 vs. the current 342) are also useful for the outage-localization spatio-temporal work specifically.
- **`resources/cvar_flexibility/simulation/feeder37/`** is the genuine **IEEE 37-bus test feeder** (official OpenDSS-format files, already vendored, zero new fetch work) — the right choice whenever a widely-recognized academic benchmark network is wanted instead of a proprietary real one (e.g., if the outage-localization chapter wants a network readers can independently cross-check against other published work).
- **`resources/cvar_flexibility/simulation/four-feeders-network/`** uses the same `network_X_Y_ZZZZ` naming convention as `resources/uk-mvlv/HV_UG_full/lvNetworks/` — likely a hand-picked 4-feeder subset of the same real UK (Deakin et al.) dataset already used in Part 6, not a separate network; **confirm directly before use, don't assume.**
- **`resources/cvar_flexibility/simulation/Test-case/`** (Slovak filenames, smart-inverter Volt-Var control content) appears to be an unrelated third-party testbed, not AusNet or UK data — likely only relevant if a future smart-inverter-control chapter is ever built, not this plan's scope.
- **Northern Powergrid's "Aggregated Smart Metering Data"** (UK, real, free, no login required) is genuinely useful for one specific purpose: a **real** (not synthetic) substation-vs-LV-feeder aggregate comparison for the NTL/theft-detection deepening in Part 7 Ch.01 — it's aggregated-only (not per-customer), so it can't feed archetyping/clustering/ranking work, but it's a real upgrade over a synthetic-only validation for that one chapter specifically.
- **The earlier unconfirmed lead is now fully confirmed, and it's a major find in its own right.** `wzy.ece.iastate.edu` is Zhaoyu Wang's real research group at Iowa State ECE. Their "Iowa Distribution Test Systems" page provides two real, free, OpenDSS-format US utility networks, each with **a full real year of real smart-meter data already paired**: a **240-bus** and a **296-bus** system (Bu, Yuan, Wang, Dehghanpour & Kimber, NAPS 2019, PDF at `wzy.ece.iastate.edu/CV/Test_system.pdf`, also mirrored on Dropbox). <!-- codespell:ignore, "Bu" is the author's real surname --> This is a genuinely strong candidate to supplement AusNet with a real, different country/utility (US, not Australian) network-plus-meter pairing — a cleaner cross-utility generalization check than AusNet's own SMR8/KLO14 (same utility) for anything specifically checking whether a finding holds across genuinely different grid codes/customer behavior, not just different feeder structure.
- **The same research group has real, peer-reviewed papers directly on behind-the-meter PV disaggregation** — exactly the second angle already planned for the new Part 3 Ch.06: "A Data-Driven Game-Theoretic Approach for Behind-the-Meter PV Generation Disaggregation" (`arxiv.org/pdf/1907.06747`, also an OSTI-indexed journal version) and "A Two-layer Approach for Estimating Behind-the-Meter PV Generation Using Smart Meter Data" (`arxiv.org/pdf/2110.07697`). Also found: "Enhancing the Spatio-temporal Observability of Grid-Edge Resources in Distribution Grids" (`arxiv.org/pdf/2102.07801`), directly relevant to Part 4 Ch.04's partial-AMI-coverage state-estimation finding too. These give Part 3 Ch.06 a real, credible, peer-reviewed literature foundation to build its formal theory section on, not an invented method.

## Topic feasibility verdict (condensed)

Checked against this book's own established discipline — every chapter anchors on the author's real prior research or a real public dataset, never a synthetic demo:

- **Already fully covered, no action needed**: phase identification (`04-grid-edge/03-phase-identification.qmd` already exists; a 200-customer scale-up was tried and abandoned for a real, disclosed reason — three candidate networks, none phase-differentiated at that scale).
- **NTL/theft — already substantially covered, one real deepening worth adding**: `06-anomaly-detection.qmd` Section 1 already has a real energy-balance/conservation check, conformal-calibrated, 60/84/80% recall at 30/50/70% under-reporting. A dedicated "unmetered bypass" notebook section (a customer whose real physical load stays present downstream but whose meter reads reduced/zero — a distinct scenario from the generic under-reporting severity already tested) is a genuine, low-risk deepening of this existing section, not a new chapter. **Added to Phase 2.**
- **Outage localization / "last gasp" — reclassified from out-of-scope to buildable.** Checked directly: real, standard IEEE PES test feeders (13/34/37/123/8500-bus) are freely downloadable as ready OpenDSS scripts from OpenDSS's own documentation site; "last gasp" (a capacitor/battery-powered final transmission before a meter loses power) and AMI message-collision-during-mass-outage are both real, documented phenomena. No free, directly-downloadable real last-gasp event dataset exists (only a paid commercial marketplace product), so this needs simulation — but that simulation now has a real, standard, honest foundation to build on (a real/standard feeder, not a fabricated one), matching the author's own explicit authorization to simulate from an accepted IEEE standard. Build: open a switch/line in OpenDSS to simulate a real outage on a standard or already-used real feeder, generate synthetic last-gasp timestamps with realistic transmission jitter, then a real spatio-temporal clustering/graph algorithm recovering the true de-energized boundary from alert timing + known topology, explicitly disclosed as simulated. **Added to Phase 2/3, Part 7 (Anomaly Detection).**
- **NILM & hidden-asset detection — buildable now, zero new anchor needed.** Part 3 already has real appliance-level ground truth (UK-DALE, PLAID). This is a genuine reframing/extension of already-validated models around a "DER blind-spot" narrative (can this NILM model detect an EV charger or heat pump the utility has no record of), the lowest-risk of every topic checked this session. **Added to Phase 2, Part 3.**
- **Predictive maintenance / asset health — reclassified from zero-anchor to real, with a genuine reframing.** A real, public, labeled dataset exists: the VSB Power Line Fault Detection dataset (real partial-discharge signals from real medium-voltage lines, ENET Centre/VŠB, via Kaggle; a newer peer-reviewed terabyte-scale successor published in *Scientific Data*). This is a different sensor modality from every other dataset in this book (40MHz waveform sampling vs. 30-min/hourly consumption), so it does not reuse AusNet/OpenDSS infrastructure — but the author's own reframing resolves this cleanly: position it as **the high-frequency end of the high-frequency/low-frequency NILM split Part 3 Chapter 1 already establishes**, taken to its logical extreme — the same kind of high-frequency electrical-signature classification Part 3 Ch.2-3 already do for appliance recognition, applied to a different target (asset health, not appliance identity) using its own real dataset, the same way each existing Part 3 chapter already uses its own separate real public dataset. **Added to Phase 2/3 as a new Part 3 chapter.**
- **Explicitly considered and rejected already, with real stated reasons** — Learning-to-Rank and Collaborative Filtering, both named and rejected in `05-ranking-recommendation.qmd` ("little complexity to earn back" on ~342 real customers; "no sparse matrix here"). Re-adding either needs a genuinely different problem framing, not a re-litigation of the same rejected fit.
- **LTR for demand-response dispatch prioritization, checked directly (not assumed)**: the "probability of responding" and "historical fatigue score" components have **zero real data anchor anywhere** — exhaustively grepped every dataset and `resources/` directory; every "demand response" hit is bibliography-abstract metadata for cited papers, not real program data. The third component, "available shiftable load," *does* have a real anchor (Part 3's own NILM appliance-level disaggregation; Part 4's own real EV-charging diversified-demand profiles). This is not a zero-anchor topic like federated learning, though — Ch.6 (Anomaly Detection) already establishes a real precedent for exactly this situation: inject a synthetic-but-realistic, **explicitly disclosed as synthetic** signal (there, fault injection; here, response-probability/fatigue) rather than pretending it's real behavioral data, then honestly test whether a learned ranker beats the existing simple score on this fresh target. Reclassified from Phase 3 to **Phase 2** — genuinely buildable now, not blocked on a new dataset, so long as the synthetic components are disclosed with the same honesty Ch.6's fault injection already models.
- **Zero real code/data anchor anywhere in `resources/`, recommend NOT building as full chapters**: federated learning, peer-to-peer trading, RC-thermal/Kirchhoff-constrained PIML as a distinct implemented method. Building any of these would mean an entirely synthetic demo, a real departure from how every other chapter in this book was built. **Out of scope for this plan.**
- **Zero `resources/` anchor but a real, cheap anchor exists elsewhere in the book**: cyber-physical bad-data/false-data-injection detection — Part 5's own already-shipped WLS state estimator is the anchor; injecting synthetic bad data and running a classical largest-normalized-residual test is a small, well-scoped extension, not a from-scratch demo.
- **Genuinely new, real low-risk anchor (same AusNet population already used everywhere)**: DTW/shapelet-based query-by-example load-curve retrieval.
- **Partial anchor, worth building modestly**: Dynamic Operating Envelopes, scoped as a **supervised proxy-model** (train a regressor on existing OpenDSS + PV-sweep data to predict safe injection limits), not the real-time RL/PINN system the original proposal implied — that oversells a genuinely unsolved, safety-critical research problem.
- **Two real, entirely unvendored optimization codebases found**, more than the original research surfaced: `resources/cvar_flexibility/` (CVaR-based customer-flexibility dispatch, already disclosed in the book's own text as a real open item) and `resources/stochastic-mpc/` (a second, undocumented, real MPC/genetic-algorithm codebase — `optimizer.py`, `heuristic.py`, `genetic_algo.py` — not mentioned in the original proposal at all).
- **This session's own already-executed clustering-generalization work** (CROCS-inspired RLS+WSMD, Tucker tensor decomposition, Chronos-2 zero-shot embeddings, diffusion maps, all on GoiEner; plus a third population, London LCL, from earlier PRs #30/#31) is real, already-PR'd content (#32, #34, #35, #36) ready to become the new Clustering part's second chapter.

## Final structure

```
Part 1: Signal Foundations                [unbuilt; tentative 3-chapter outline above]
Part 2: ML Foundations                    [unbuilt; tentative 8-chapter outline above]
Part 3: Disaggregation                    [4 existing chapters, + 3 new below]
  05. [NEW — Phase 2] Hidden Assets: Detecting Unregistered EVs and Heat Pumps from the
      Aggregate Signal (zero new anchor — reframes Part 3's own already-validated NILM
      models around a DER-blind-spot narrative, using the same real UK-DALE/PLAID data)
  06. [NEW — Phase 2, real anchor already cited but unused] Disaggregation Beyond the Meter:
      Recovering DER and Customer-Class Contributions from Feeder-Head Signals — the same
      NILM question (what's inside this aggregate signal) asked one level up the network,
      given only a feeder-head or substation aggregate, not the household's own smart meter.
      Real anchor already sitting in this book's own bibliography, cited but never acted on:
      Ledva & Mathieu's "Separating Feeder Demand Into Components Using Substation, Feeder,
      and Smart Meter Measurements" (`ledva2020feederdemand`, currently only cited in Part 5
      Ch.01 as motivation for feeder-*clustering*, not as a method for feeder-*disaggregation*
      the way its own real title describes). Two real, buildable angles on the same existing
      AusNet feeder, zero new data: (1) partial-AMI feeder decomposition — given only a subset
      of real smart meters (echoing Part 4 Ch.04's own partial-AMI-coverage finding), separate
      the feeder-head aggregate into metered-customer contributions plus an unmetered
      residual; (2) behind-the-meter PV disaggregation — estimate gross real PV generation
      hidden inside the net feeder-head signal from weather/irradiance covariates alone,
      validated directly against this book's own real, already-known PV ground truth. Real,
      peer-reviewed literature foundation for angle (2), confirmed directly, not invented:
      Zhaoyu Wang's group at Iowa State (game-theoretic and two-layer BTM-PV-disaggregation
      methods, arXiv 1907.06747 and 2110.07697); the same group's real 240-bus/296-bus Iowa
      test systems (free, OpenDSS, a full real year of real smart-meter data already paired)
      are also a strong candidate to supplement AusNet with a genuinely different real
      country/utility, complementing SMR8/KLO14's same-utility comparison above.
  07. [NEW — Phase 2/3] Asset Health from High-Frequency Signatures: Partial Discharge and
      Predictive Maintenance (the high-frequency end of Ch.01's own high-frequency/
      low-frequency NILM split, taken to its logical extreme — same electrical-signature-
      classification technique family, applied to asset health instead of appliance ID;
      real, public, labeled dataset: VSB Power Line Fault Detection, a different sensor
      modality from every other book dataset, its own self-contained real data source the
      same way each existing Part 3 chapter already uses its own separate real dataset)

Part 4: LV Network and Topology Foundations
  01. Modeling the Low-Voltage Network              (was 04-grid-edge/01-opendss-lv-modeling)
  02. Time-Series Simulation and DER Modeling        (was 04-grid-edge/02-timeseries-der-modeling)
  03. Phase Identification from Smart-Meter Data     (was 04-grid-edge/03-phase-identification)
  04. Can Smart-Meter Readings Fill In What the Grid Cannot See (LV state estimation)
      — MOVED from 05-forecasting/06-lv-state-estimation: recovering network state is a
        prerequisite for the network this Part builds, belongs here in read order, not at
        the end of Forecasting.

Part 5: Customer and Feeder Clustering
  01. [NEW — built this session, precedes everything else in this Part per the
      author's own explicit call: "this should be the first chapter before any
      other clustering chapter"] Clustering Foundations — the Part's own
      domain-specific theoretical foundation (four clustering-objective families:
      partition, hierarchical, density-connectivity, graph/spectral; DTW and
      other representation choices for a load curve specifically; constrained
      clustering/must-link-cannot-link, grounding the eventual Ch.06 below), a
      state-of-the-art review of the energy-domain clustering literature (Chicco
      through Yerbury et al.'s CROCS), and best practices (evaluating a
      clustering, choosing K, checking stability via ensembles, the curse of
      dimensionality). Phase 0 directory sweep now complete:
      `book/05-clustering/01-clustering-foundations.qmd`.
  02. Customer and Feeder Clustering                 (was 04-grid-edge/04-customer-feeder-clustering)
  03. [NEW — Phase 1, already executed] Generalizing Beyond One Utility: London, GoiEner,
      Tucker Decomposition, Chronos-2 Embeddings, and Diffusion Maps
  04. [NEW — Phase 2, real anchor already cited but unused] Substation-Level Clustering for
      Network Planning — the same "cluster whole feeders" question (Ch.02's own SMART-DS
      section) asked one level up again, at the substation rather than the feeder. Real
      anchor already sitting in this book's bibliography, cited but never built:
      `li2015substation`, "Development of Low Voltage Network Templates, Part I: Substation
      Clustering and Classification," currently only cited (in the same sentence as
      `ledva2020feederdemand`) as motivation for the existing feeder-clustering section, not
      built as its own method — the identical unused-citation pattern just fixed for Part 3
      Ch.06. Real data: SMART-DS's own multiple substations behind its 33 feeders, or the
      Team-Nando/AusNet networks (SMR8/KLO14/HPK11/CRE21) already recommended for outage
      localization and multi-level forecasting — fetching that one dataset now serves three
      new chapters across three different Parts.
  05. [NEW — Phase 2, real anchor already built] EV Charging-Session Clustering — a genuinely
      different clustering granularity from every other chapter in this Part: charging
      *sessions*, not whole-household daily loads, grouped into behavior archetypes (home,
      workplace, opportunistic). Real anchor: the Norway EV charging fetch/explore scripts
      already built earlier this session (real per-session kWh, plug-in/plug-out timestamps).
  06. [NEW — Phase 3, longer horizon, no existing anchor] Kirchhoff-Constrained Clustering
      — Ch.01's own constrained-clustering vocabulary (must-link/cannot-link, COP-KMeans)
      is the direct theoretical grounding this chapter puts to real use.

Further clustering ideas surfaced but not scoped into this plan, named here for traceability
rather than silently dropped: DER-adoption-propensity clustering (reframing the earlier-
rejected "collaborative filtering for LCT adoption" as clustering instead, which sidesteps the
original "no sparse matrix" rejection reasoning — real, but not yet prioritized); vulnerability/
fuel-poverty segmentation (a real DSO/regulator use case, but ethically sensitive — correlation-
based judgments about vulnerable households need careful, disclosed framing before any build,
not a quick add); topology-based "transfer feasibility" clustering (which feeders are
structurally similar enough that a phase-ID or hosting-capacity finding transfers without
re-simulating); and hosting-capacity-risk clustering bridging into Part 6/Part 9's own ranking
and optimization work. None of these has a concrete anchor checked yet the way the two Phase 2
additions above do.

Part 6: Retrieval, Ranking, and Recommendation
  01. Ranking and Recommendation under DER            (was 04-grid-edge/05-ranking-recommendation;
      mitigation-*lever-selection* logic moves to Part 9, retrieval/ranking machinery stays)
  02. [NEW — Phase 2] Elastic Similarity for Load-Curve Retrieval: DTW / Shapelet Query-by-Example
  03. [NEW — Phase 2, reclassified from Phase 3] When Learning-to-Rank Actually Earns Its
      Complexity: A Demand-Response Dispatch Worked Example (real shiftable-load estimates
      from Part 3/Part 4 + explicitly-disclosed synthetic response-probability/fatigue signal,
      following Part 7 Ch.01's own synthetic-fault-injection precedent; tests whether a learned
      ranker beats the existing simple score on this fresh target, not a re-litigation of the
      DER-risk case Ch.05 Ch.01 already settled). Collaborative-filtering theory treatment can
      still follow in Phase 3 if a genuinely new target for it ever surfaces.

Part 7: Anomaly Detection
  01. Anomaly Detection Beyond the Meter               (was 04-grid-edge/06-anomaly-detection;
      gains a dedicated "unmetered bypass" notebook section deepening its existing
      energy-balance/NTL check, real, low-risk, Phase 2)
  02. [NEW — Phase 2] Bad-Data and False-Data-Injection Detection via Classical Residual Tests
      (extends Part 4 Ch.4's WLS state estimator)
  03. [NEW — Phase 2/3] Automated Outage Localization from Simulated Last-Gasp Alerts
      (recommended network: AusNet's own real SMR8 or KLO14 SWER feeder from
      Team-Nando/MV-LV-Networks, same utility as every other AusNet chapter but a
      genuinely different real rural topology; IEEE 37-bus already vendored in
      `resources/cvar_flexibility/simulation/feeder37/` as the alternative if a
      widely-recognized academic benchmark is preferred over a proprietary real one;
      simulated outage event, explicitly disclosed as simulated; spatio-temporal
      clustering/graph recovery of the true de-energized boundary from alert timing +
      known topology)

Part 8: Forecasting
  01. Household Smart-Meter Forecastability
  02. Scaling to Hundreds of Meters
  03. What a Probabilistic Forecast Is Actually Worth
  04. Do Foundation Models Fail Like Everything Else Does?
  05. [NEW — Phase 2, real gap confirmed] Does Forecasting Get Easier as You Aggregate?
      Household, Feeder, and Substation-Level Forecasts Compared — checked directly: every
      existing Forecasting chapter is customer-level only, zero feeder- or substation-level
      forecasting anywhere in the current 6 chapters. Real, checkable question: fit the same
      established forecasting approach (LightGBM/Chronos-2 zero-shot, already built in Ch.2/
      Ch.4) at three real aggregation levels and test the classical "aggregation smooths
      variance" hypothesis directly rather than assume it, formalized via real hierarchical-
      forecasting-reconciliation theory (top-down/bottom-up/MinT). Real anchor: aggregate the
      existing 342-customer AusNet feeder up to its own feeder head (already done for hosting-
      capacity studies in Part 4 Ch.2); for genuine substation-level structure (multiple real
      feeders under one real substation), SMR8/KLO14/HPK11 (Team-Nando/MV-LV-Networks, see
      above) give a real multi-feeder-per-substation hierarchy this book has never had before,
      the same dataset already recommended for the outage-localization chapter.
  06. [NEW — Phase 2, reuses more existing infrastructure than almost anything else in this
      plan] Trusting a Forecast Before You Know If It Was Right — the real, named gap: split-
      conformal guarantees (already used throughout this book, e.g. Ch.03) are *marginal*
      ("90% of intervals contain the truth, averaged across many forecasts"), which says
      nothing about whether *this specific forecast, right now* is trustworthy — a real,
      important distinction worth formalizing, not glossed over. Three complementary,
      real-time signals, each catching a different real failure mode, not one silver bullet:
      (1) input distribution shift — is the situation being forecast actually similar to what
      the model was calibrated on, checked by reusing Part 7's own anomaly-detection toolkit
      (Isolation Forest/LOF/ECOD) on the forecasting *input* space rather than the load signal;
      (2) cross-model disagreement — Ch.04 already compares a trained LightGBM model against
      Chronos-2 and TimesFM zero-shot, so this disagreement data already exists; whether it
      actually correlates with real historical error is a genuinely checkable claim, not an
      assumption; (3) Adaptive Conformal Inference (Gibbs & Candès, real, published, NeurIPS
      2021, candidate citation needing the same accuracy check as every other reference in
      this book) — continuously adjusting the trust/coverage level using a lagged window of
      ground truth as it eventually arrives, rather than a fixed calibration set. This is the
      one genuinely new method; the other two reuse infrastructure this book already built.
  (loses its former Ch.5/Ch.6 capstones to Parts 9/4 but gains these two — needs a real new
   closing beat regardless, see Phase 0)

Part 9: Optimization for Decision-Making
  01. What-if Scenario Analysis Under a Zero-Shot Forecast   (MOVED from 05-forecasting/05:
      "does this change what a DSO does" is a decision question, not a forecasting-accuracy one)
  02. [NEW — Phase 2, needs discovery work first] CVaR-Based Optimal Customer-Flexibility
      Dispatch (real, unvendored `resources/cvar_flexibility/`; absorbs the rule-based
      mitigation-lever-selection logic moved from Part 6 as its own honest baseline —
      "the rule this Part's CVaR-optimal dispatch has to beat")
  03. [NEW — Phase 3, longer horizon] Stochastic MPC for DER Dispatch (real, unvendored,
      entirely undocumented `resources/stochastic-mpc/` codebase)
  04. [Optional — Phase 2/3] Dynamic Operating Envelopes as a supervised power-flow proxy
      (placement here rather than Part 4, since computing a safe operating limit is itself
      a decision/constraint-computation problem — confirm this placement before building)
```

Both contested placements from the pressure-test are resolved: Ch.5/Ch.6 both move out of Forecasting (per author decision), and Anomaly Detection and Optimization for Decision-Making both stand up as top-level Parts immediately in Phase 0, not folded in as interim homes (per author decision) — accepting that Part 9 ships with one real chapter (the moved what-if-scenario chapter) until Phase 2/3 lands.

## Physics-Informed ML (PIML): a fourth named cross-cutting thread, not a blanket mandate

Checked directly against the same discipline that already rejected LTR, collaborative filtering, and GNNs elsewhere in this book ("don't add complexity a technique doesn't need to earn") — a blanket PIML pass across most chapters would break that pattern. DTW/shapelet retrieval, the LTR-for-demand-response worked example, Chronos-2 embeddings, diffusion maps, and outage localization from last-gasp timing are all genuinely useful as pure ML/statistics, with no real physical constraint to check them against; forcing one on would be unearned complexity, not rigor.

But a real, coherent thread already runs through five specific chapters in this plan, currently implicit rather than named:

1. **Mass-balance NILM** (Part 3, Phase 2) — energy conservation as an explicit loss-function constraint.
2. **Behind-the-meter PV disaggregation** (Part 3 Ch.06) — Iowa State's own real methods use weather/irradiance physics, not just curve-fitting.
3. **WLS state estimation** (Part 4 Ch.04) — its own measurement function *is* real AC power-flow physics (Ohm's/Kirchhoff's laws), not a generic regression target.
4. **Dynamic Operating Envelope supervised proxy** (Part 9 Ch.04) — an ML model standing in for a real power-flow solve, checked against real physics, not assumed to approximate it.
5. **Kirchhoff-constrained clustering** (Part 5 Ch.06, Phase 3) — genuinely novel, no existing anchor, the most explicitly "PIML" of the five.

Recommendation: introduce PIML once, explicitly, the first time it appears (most likely the mass-balance NILM chapter, being both earliest in the book and conceptually simplest), then flag it consistently in these five specific chapters — matching the book's own existing pedagogical-marker convention (Key Concept, Activity, Common Mistake, Pro Tip) with a recurring callout, e.g. "Physics Check," rather than inventing a new device from scratch. Explicitly skip it everywhere else, rather than let it become a fourth diluting cross-cutting narrative alongside split-conformal calibration, the AusNet feeder case study, and partial-AMI coverage (below) — three is already a lot for a reader to track across nine Parts; a fourth needs to earn its place as narrowly as this one does.

## Real mechanical constraints, confirmed directly in this repo (not assumptions)

- **`number-sections: false`** is set project-wide — nothing auto-renumbers. Every "Part N"/"Chapter N" reference is hand-typed prose: **73 "Part N" + 197 "Chapter N" literal occurrences** across current Part 3-5 alone.
- **This exact kind of renumbering has already been botched once and shipped**: several chapters still say "Part 2" for what `_quarto.yml` now calls Part 3, left over from before Parts 1-2 were inserted. Any renumbering sweep must be verified by hand per-occurrence, never blind sed.
- Every chapter ends with a hand-written `## Where this leaves Part N` closing section (11 found) that both names the part and previews the next chapter — needs a genuine rewrite per moved/split chapter, not a relabel. Forecasting's own closing beat, in particular, needs new prose once it loses Ch.5/Ch.6.
- **`.github/workflows/check-notebooks.yml`'s `SKIP_LIST` hardcodes exact notebook paths** (e.g. the OpenDSS-heavy ranking-recommendation notebook, the ~14-minute foundation-models notebook) with prose comments explaining why. Any move must update this list in lockstep, or CI either silently stops skipping an expensive notebook (timeout risk) or the skip entry orphans and does nothing.
- **Do not rename `fetch_part4_*`/`fetch_part5_*` scripts** to match new part numbers — they're referenced by literal name across 12+ files; decouple script naming from part numbers so future reshuffles don't touch them again.
- Companion notebooks assume `book/<dir>/<file>.ipynb` sits exactly two directories below repo root (`Path("../../resources/...")`); new part directories must stay at that same single level of nesting, never nested deeper.
- References/acronyms are low-risk: one shared `references.bib` and `acronyms.yml`, no per-part scoping, unaffected by chapter moves.
- The cross-cutting split-conformal-calibration thread (introduced in Phase ID, reused in Clustering/Ranking/Forecasting) now threads through four separate Parts instead of two — the load-bearing methodological narrative to watch most carefully during the prose pass, more so than any single "reuses Chapter X's archetypes" callout.
- Part 4's current "four threads" narrative device (the thing that gives the bundled part its felt cohesion today) disappears once the bundle splits — plan for a replacement continuity device (e.g. a recurring "this Part continues the AusNet feeder case study" callout) rather than letting the throughline go implicit.
- "Optimization for Decision-Making" spans three-to-four genuinely different paradigms (convex risk-measure optimization, receding-horizon MPC, rule-based heuristics, and possibly supervised-proxy envelope estimation) — budget for separate real theory treatments per chapter, not one shared framing.

## Phased execution plan

**Phase 0 — pure restructuring, zero content changes.** `git mv` every file into its new part directory; update `_quarto.yml`'s `part:` blocks; fix every "Part N"/"Chapter N" reference (hand-verified against the real inventory, not blind sed); rewrite every `## Where this leaves Part N` closing section, including a new one for Forecasting; update `check-notebooks.yml`'s `SKIP_LIST` paths; update `index.qmd`'s roadmap and add new `act-N`/`pipeline-dot-N` color tokens in `brand.scss`/`brand-dark.scss` for the new part count; update `PLAN.md`'s own structure section. Extract the mitigation-lever-*selection* content out of `05-ranking-recommendation.qmd` with a clear forward-reference stub to its new Part 9 home (full rewrite happens in Phase 2 alongside the CVaR chapter). Exit criterion: full `quarto render` succeeds, zero narrative content changed beyond structural relabeling, Playwright-checked nav reflects the new part structure. Do not mix any new content into this phase.

**Phase 1 — fold in already-executed, already-PR'd work.** Merge/build out PRs #30-32, #34-36 (London + GoiEner clustering replicas, multi-resolution-features caution, CROCS-inspired RLS+WSMD, Tucker/Chronos-2/diffusion-maps) into Part 5's new Ch.03. This is real content-writing work in its own right — these currently exist as bare notebooks with no paired `.qmd`; turning five exploratory notebooks into one coherent chapter at this book's established depth bar (Key Concept boxes, formal math, diagrams, real embed cells) is a substantial task, not a merge-and-done.

**Phase 2 — real new content, priority order matching real anchors:**
1. NILM hidden-asset detection (Part 3 Ch.05) — lowest risk of everything in this plan, zero new anchor, reuses existing real UK-DALE/PLAID data and already-validated models.
2. Feeder-head disaggregation (Part 3 Ch.06) — zero new data, reuses the existing AusNet feeder; activates a citation already sitting unused in `references.bib` (`ledva2020feederdemand`), currently only cited for feeder-*clustering* motivation, never built as the feeder-*disaggregation* method its own title describes.
3. DTW/shapelet retrieval (Part 6 Ch.02) — lowest risk, same AusNet population, no new dependency.
4. Bad-data/FDI residual-test extension (Part 7 Ch.02) — reuses Part 4's now-relocated state-estimation chapter, no new anchor needed.
5. NTL "unmetered bypass" deepening (Part 7 Ch.01) — real, low-risk, deepens an already-existing section.
6. Mass-balance (energy-conservation) constraint on an existing Part 3 NILM model — cheap, real, no new anchor needed.
7. LTR-for-demand-response worked example (Part 6 Ch.03) — real shiftable-load anchor (Part 3/Part 4) + disclosed synthetic response/fatigue signal, following Part 7 Ch.01's own fault-injection precedent.
8. Outage localization (Part 7 Ch.03) — recommended network: AusNet's own real SMR8/KLO14 SWER feeder (Team-Nando/MV-LV-Networks); IEEE 37-bus (`resources/cvar_flexibility/simulation/feeder37/`, already vendored) as the alternative.
9. Substation-level clustering (Part 5 Ch.04) — real anchor already cited but unused (`li2015substation`), same SMART-DS/Team-Nando substation-scale data as items 8 and 10.
10. Multi-level forecasting: household vs. feeder vs. substation (Part 8 Ch.05) — real gap confirmed (every existing Forecasting chapter is customer-level only); reuses this book's own established forecasting models, plus the same SMR8/KLO14/HPK11 real multi-feeder-per-substation structure recommended for outage localization and substation clustering above — fetching that one dataset now serves **three** new chapters across three Parts. Also a natural sibling to Part 3 Ch.06 — both ask "what can you learn about a feeder's real composition from an aggregate signal," one at a point in time, one over time.
11. EV charging-session clustering (Part 5 Ch.05) — real anchor already built (the Norway EV fetch/explore scripts from earlier this session); a genuinely different clustering granularity (sessions, not households) from every other chapter in the Part.
12. Predictive maintenance / asset health (Part 3 Ch.07) — real VSB Power Line Fault Detection dataset; bigger lift than the above since it's a standalone real dataset with no reuse of existing book infrastructure, but no discovery work blocking it.
13. CVaR-based flexibility dispatch (Part 9 Ch.02) — needs real discovery work first: the actual `optimalcvar` formulation is still "not yet found or verified" in the vendored `resources/cvar_flexibility/src/` copy per the book's own prior disclosure; don't schedule with false confidence it's ready.
14. Dynamic Operating Envelope supervised proxy (Part 9 Ch.04, placement to confirm) — scope as "build the ML proxy on top of the real static hosting-capacity sweeps already in `resources/cvar_flexibility/data/{timeseries-lv,voltwatt-lv}/Tutorial-DERHC-*.ipynb`," not as an existing chapter waiting to be vendored.

**Phase 3 — longer-horizon, bigger-lift items:**
- Stochastic MPC (Part 9 Ch.03) — real, unvendored, completely undocumented codebase; needs its own discovery pass before scoping.
- A formal literature-grounding pass across every relocated/new chapter, accelerated by PLAN.md's own already-curated (but unbuilt) Part 2 reference list — it already maps candidate citations (IDEC, UMAP, Isolation Forest, conformal prediction, etc.) to exactly the topics these Parts need.
- Kirchhoff-constrained clustering (Part 5 Ch.06) — genuinely novel research, no existing anchor; scope and timeline separately from everything else.

## Explicitly out of scope for this plan

Federated learning, peer-to-peer energy trading, RC-thermal/Kirchhoff-constrained PIML as a distinct method, session-based NILM-as-sequence, and multi-armed/contextual bandits for tariffs. None has a real data/code anchor anywhere in this author's prior work, nor a viable simulation/public-dataset path the way outage localization and predictive maintenance turned out to have; building any would mean an from-scratch synthetic demo, breaking this book's established discipline. Not mentioned in the book at all per the author's chosen scope (no "further directions" placeholder either).

## Verification (every phase)

- `uv run ruff check` / `ruff format --check` on every touched file.
- `uv run pytest tests/` (existing suite).
- Full `quarto render` (after `rm -rf .quarto/idx`), zero warnings, for every affected page.
- Playwright visual + 375px mobile-viewport check of the new nav structure and every changed/new section.
- `pytest --nbmake` real execution for every new/moved notebook not already in `check-notebooks.yml`'s `SKIP_LIST`.
- Em-dash check on every new/touched `.qmd`.
- Git history preserved via `git mv`, never delete+recreate.
