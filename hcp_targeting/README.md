# HCP targeting — incremental-response prototype

## Run

Python 3.10+. From this directory:

```bash
python -m pip install -r requirements.txt
python run.py
python verify.py
```

Open `outputs/rep_view.html` directly in a browser. It is self-contained, needs no web server, and supports territory/access filters, evidence audits, filtered CSV export, and local rep feedback.

## Submitted artifacts

- `generate_data.py` generates exactly 4,000 synthetic HCPs.
- `run.py` reconstructs a predictive baseline, adds an all-universe predictive diagnostic, trains a two-model uplift learner, and evaluates held-out results.
- `outputs/hcp_data.csv` contains pre-decision features, randomized synthetic visit assignment, four-week outcome, and HCP-disjoint split.
- `outputs/simulation_truth.csv` holds oracle potential-outcome probabilities. These NEVER enter training or feature selection.
- `outputs/validation_universe.csv`, `list_overlap.csv`, `metrics.json`, and `uplift_curve.png` document results.
- `outputs/ranked_list.csv`, `ranked_list.json`, and `reason_audit.json` support the 400-person call list.
- `reason_templates.json` records the LLM prompt and ChatGPT-authored factual language templates.
- `llm_reasons.py` optionally calls an LLM for each HCP in batches and validates exact approved factual sentences. It uses `OPENAI_API_KEY` and an explicitly set `OPENAI_MODEL`. It was not API-executed in this submission. Do not put a secret in a file or browser HTML.
- `verify.py` independently checks each selected reason fact against the raw HCP record and rejects a fabricated claim for every HCP.
- `DECISIONS.md` contains the diagnosis, assumptions, metrics and rollout proposal.

## What this demonstrates

The case's AUC of 0.86 and top-decile response of 71% are supplied client exhibit metrics. They are not fabricated as reproduced results. This simulation's fitted results are reported separately. Outcome definition is assumed to be at least one new prescription within four weeks, because the exhibit does not specify it. Clarify whether “71% showed the predicted score” actually means top-decile outcome rate; a probability score is not itself an outcome.

The exhibit states only the validation population has 3+ prior visits. This prototype ALSO restricts baseline training to that population as an explicit reconstruction assumption. We cannot know the client's actual algorithm, training population, target, leakage, or feature list from the exhibit. The broad predictive model is included to show why broader coverage alone does not change the objective from response to incrementality.

No real patient, HCP identifier, or CMS record is used. Synthetic IDs are not NPIs. Lognormal prescribing counts and specialty mix are illustrative assumptions; the simulation is schema-informed, not empirically calibrated to CMS. Do not claim CMS proves these distributions or visit effects.

## Causal assumptions and limits

Treatment means one rep visit at the decision point. Historical visit counts are pre-decision covariates. Current treatment is randomized 50/50 in the synthetic generator. The outcome is collected after four weeks. This randomized assignment is an experiment assumption, not the client's actual rep behavior. Actual observational logs require measured confounders, overlap diagnostics, propensity estimation and a justified doubly robust / causal approach; missing confounders can still invalidate causality. Recommend a randomized client pilot before any causal business claim.

The generator deliberately contains accessible, less-visited incremental responders and saturated high-propensity prescribers. This demonstrates the requested mechanism. Oracle effects are valid only within the designed simulation and do not establish real commercial lift. The supplied brand launched nine months ago; the 12-month calls are general rep activity, and the 90-day brand history is post-launch. Current weekly refresh is a rescoring schedule, not evidence that outcome labels mature weekly.

Train/validation/test split is 60/20/20 by HCP, one record each. All preprocessing references use training rows. Test outcomes are not used for fitting or tuning. This is a cross-sectional reconstruction, not a temporal backtest. For production, implement as-of snapshots and forward, HCP/territory-aware evaluation with delayed labels. Validation is reserved and unused for tuning here.

## Reproducible result and honest interpretation (seed 42)

Current restricted model test AUC is 0.776 in familiar HCPs versus 0.593 in 0–2 visit HCPs. Its held-out top-decile list has 95% familiar HCPs and 70% visited in the previous 30 days. This is familiarity overlap, not measured overlap with an actual future rep plan, which has not been provided.

At equal top-decile capacity (80 held-out HCPs), observed treated-minus-control response difference is 5.6 percentage points for predictive targeting versus 13.3 for uplift. Arm counts are small. Unnormalized propensity-weighted Qini area is 0.00144 for current targeting versus -0.00049 for uplift. The paired bootstrap 95% interval for the Qini difference is about [-0.0617, 0.0539]. These observations do NOT establish overall uplift-ranking superiority. Known simulated mean effects are 2.1 pp versus 29.4 pp in the respective top deciles. Oracle diagnostics support the synthetic mechanism; they do not repair uncertainty in observed outcomes. Keep these conclusions separate.

Qini definition in this code: sort by score; accumulate the inverse-propensity transformed outcome `Y*(T/0.5 - (1-T)/0.5)`; divide cumulative totals by N; integrate the difference from the straight random-ranking line. It is an unnormalized Horvitz–Thompson cumulative gain area. Do not compare its scale with normalized library Qini coefficients. The bootstrap fixes the fitted models, resamples held-out HCPs, and re-ranks each resample. It omits model-fit uncertainty.

## LLM and evidence grounding

This ChatGPT conversation authored the feature-specific templates. Each flagged HCP's two largest absolute model sensitivity drivers are converted into literal source facts using those templates. This is a constrained LLM-assisted renderer, not a claim of 400 separate generation calls. Optional live per-HCP calls can order approved sentences, but cannot invent wording or facts. API failures or invalid responses keep the already verified reason. The optional path is deliberately restrictive so every displayed claim is auditable. `llm_call_audit.json` appears only if it is run. Default reason quality has been checked locally; API/network behavior has not been verified.

Local sensitivity replaces one feature with its training-set median and recomputes predicted uplift. It can be positive or negative; both signs are retained in the audit. It is model sensitivity, not SHAP, a clinical mechanism, or proof that changing that feature changes response. Median replacement can produce unrealistic feature combinations; validate explanations with users. Action text is a proposed review, not guaranteed clinical or commercial benefit.

## Public reference sources

- CMS Part D by Provider landing page: https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider
- CMS data dictionary: https://data.cms.gov/sites/default/files/2025-04/MUP_DPR_RY25_20250401_DD_PRV_508.pdf
- CMS methodology: https://data.cms.gov/sites/default/files/2026-05/MUP_DPR_RY26_20260421_Methodology_508.pdf
- Qini reference: https://www.uplift-modeling.com/en/stable/api/metrics/qini_auc_score.html
- Structured output reference: https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat

CMS supplies prescribing counts and specialty fields for Medicare Part D, not rep visits, access, brand-new-launch four-week effects, or the entire dermatology market. Small-count suppression and payer/population coverage constrain calibration. A future extension should sample specialty-specific actual volume quantiles, document year/filtering/suppression, and avoid transferring Medicare patterns uncritically.

## Verification status

The full Python pipeline ran successfully, Python files compile, JavaScript passes `node --check`, and all 400 reasons passed source-data checks plus fabricated-claim rejection. A browser launch was attempted but this environment lacks the Chromium executable, so visual layout and live browser interactions have not been verified here. The optional live LLM API path also remains unexecuted.
