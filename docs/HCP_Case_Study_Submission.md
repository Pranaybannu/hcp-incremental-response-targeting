# HCP targeting case study submission

Sambodhu Pranay Raj | 3 November 2026

Sambodhu Pranay Raj | CustomerInsights AI FDE assessment | 3 November 2026

## 1 Problem understanding and main result

I understood the problem as a mismatch between strong prescribing-prediction metrics and recommendations that add little to representatives’ existing call plans. I built a reproducible 4,000-HCP simulation, a predictive baseline, a broader predictive comparator, a treatment-effect model and a territory-filtered HTML call list with audited reasons. In the held-out top decile, 95% of baseline recommendations had at least three prior visits, compared with 5% for uplift targeting, while observed visit-versus-no-visit response differences were 5.6 and 13.3 percentage points respectively. However, the overall Qini comparison was inconclusive, so I recommend a controlled pilot rather than claim proven client impact.

## 2 Approach and methodology

### Define the decision and investigate usage

The decision is how to allocate limited rep calls to HCPs whose response could improve after a visit. The logged contact fractions are 4.12%, 3.28% and 4.55% despite list openings. I would check CRM attribution, actual overlap with the pre-existing call plan, territory ownership, access and rep capacity before attributing low adoption entirely to the model.

### Generate a reproducible universe

I generated prescribing history, previous call activity, specialty, territory, access and engagement. Features precede the simulated visit decision. One outcome is observed four weeks later. A fixed seed and separate simulation-truth file make results reproducible and prevent oracle probabilities from entering training.

### Reconstruct the predictive objective

I used histogram gradient boosting to predict a four-week prescribing outcome. The reconstructed baseline trains on HCPs with at least three prior visits and is evaluated separately on familiar and less-visited HCPs. A second predictive model trains across the full universe to distinguish broader population coverage from changing the prediction objective.

### Estimate incremental response and explain recommendations

A T-learner fits separate outcome models for the randomized visit and no-visit arms. Their probability difference ranks incremental-response opportunities. I compare equal top-decile capacity and select each recommended HCP’s two largest absolute local sensitivity drivers, render source-backed factual reasons, and expose territory filters, source values and rep feedback in the HTML.

## 3 How I arrived at the solution

I considered retaining the predictive model and only improving the interface. This might reduce workflow friction, but it would leave the propensity objective unchanged. I also fitted an all-universe predictive model, which improves coverage but still estimates the likelihood of prescribing rather than the additional benefit of a visit. I chose the T-learner for its understandable two-model structure under randomized synthetic treatment. A propensity-adjusted or doubly robust estimator is a future option for observational client logs, subject to adequate overlap and measured confounding.

I rejected treating the supplied AUC of 0.86 as evidence of incremental commercial impact. I also rejected reporting known synthetic effects as proof of real-world performance or hiding the inconclusive observed Qini comparison.

| Assumption or clarification | Status and consequence |
| --- | --- |
| Outcome and time window | Assumed at least one new brand prescription within four weeks. The client target must be confirmed. |
| Baseline reconstruction | The exhibit restricts validation to 3+ visits. Restricting training too is my explicit assumption, not a supplied fact. |
| 71% top-decile figure | Provisionally interpreted as the positive-outcome rate in the highest-scored 10%. Exact wording needs confirmation. |
| Historical calls and launch | 12-month calls represent general rep activity. Brand prescribing features cover the recent 90 days after launch. |
| Clarification desk | No clarification-desk answers were supplied in this conversation. No answer has been invented or relied on. |

## 4 Data and ground truth

The generator creates exactly 4,000 synthetic IDs across Dermatology and Primary care and five territories. Category prescription counts follow an illustrative clipped lognormal distribution. Brand counts depend on synthetic adoption, and historical calls preferentially follow visible prescribing. Access and engagement are simulated. The specialty mix, distribution parameters and response effects are modelling assumptions.

The generator defines no-visit and visit outcome probabilities for every HCP. It deliberately includes accessible, less-visited HCPs with positive incremental response and familiar high-propensity HCPs with little remaining benefit. A 50/50 randomized visit indicator determines which Bernoulli outcome is observed. The true probability difference is stored only for simulation diagnostics. This causal structure is designed, not estimated from public data.

CMS Medicare Part D by Provider documentation informed the choice of prescribing-count and specialty fields. No CMS records were downloaded or used to calibrate parameters. CMS does not provide rep visits or causal visit effects, and Medicare coverage and small-count suppression limit its use as a representation of the complete topical-dermatology market.

## 5 Evaluation results and uncertainty

I split HCPs 60% training, 20% validation and 20% test, leaving 800 HCPs for reported evaluation. The validation split is reserved and unused for tuning. This is a cross-sectional demonstration with one record per HCP. Client deployment needs forward evaluation using weekly as-of snapshots and mature outcome labels.

| Held-out measure | Predictive baseline | Uplift T-learner |
| --- | --- | --- |
| AUC in HCPs with 3+ visits | 0.776 | Not the uplift objective |
| AUC in HCPs with 0–2 visits | 0.593 | Not the uplift objective |
| Top-decile prior-visit overlap | 95.0% | 5.0% |
| Top-decile visited in last 30 days | 70.0% | 0.0% |
| Observed top-decile response difference | 5.6 percentage points | 13.3 percentage points |
| Top-decile visit / no-visit counts | 44 / 36 | 39 / 41 |
| HT cumulative gain area | 0.04769 | 0.04576 |
| Qini area above random ranking | 0.00144 | −0.00049 |
| Oracle mean effect in top decile | 2.1 percentage points | 29.4 percentage points |

The all-universe predictive comparator reaches AUC 0.856 across the test universe and Qini area 0.01091. These are separate simulated results, not a reproduction of the client’s AUC 0.86 or top-decile 71% exhibit. Familiarity overlap uses prior-call proxies because the real future call plan is unavailable.

The cumulative gain curve ranks held-out HCPs and sums Y × [T/0.5 − (1−T)/0.5], divided by test size. Qini is the unnormalized area above the random-ranking line. It should not be compared numerically with a library-normalized Qini coefficient. I also calculated average precision and Brier score for predictive diagnostics.

Across 400 paired held-out bootstrap resamples, the 95% interval for uplift-minus-baseline Qini is approximately −0.0617 to 0.0539. It crosses zero and excludes model-retraining uncertainty. The observed data therefore do not establish overall ranking superiority. Oracle results confirm the intended synthetic mechanism but cannot replace observed-outcome evaluation.

No mid-session client update was supplied, so there was no update-driven model change to report. During development, the actual Qini result changed my interpretation from an expected demonstration of superiority to an inconclusive empirical comparison requiring further experimental evidence.

## 6 Technology stack

| Component | Implementation |
| --- | --- |
| Data and computation | Python, NumPy and pandas with a fixed random seed |
| Models and predictive evaluation | scikit-learn HistGradientBoostingClassifier and metric functions |
| Uplift evaluation | Custom inverse-propensity cumulative gain and Qini area; NumPy bootstrap |
| Chart | Matplotlib uplift_curve.png and the underlying CSV |
| Representative view | Self-contained HTML, CSS and JavaScript; CSV export and local browser feedback |
| Verification | Python source-value checks, fabricated-claim rejection, compilation and JavaScript syntax checks |
| Optional LLM calls | Python standard-library HTTPS client, JSON responses and fail-closed validation |

## 7 Use of LLMs and grounding

I used ChatGPT to help structure the solution, draft code, explain assumptions and author plain-language templates for the selected model drivers. The business purpose of the explanation component is to let a rep inspect why an unfamiliar HCP appears on the call list and decide whether a call is feasible.

The executed default pipeline uses ChatGPT-authored templates and deterministically inserts each HCP’s verified values. This is LLM-assisted template generation. It does not claim that a separate generative call was executed for each HCP. The optional llm_reasons.py path sends each flagged HCP’s selected facts to an API in batches, lets the model order approved sentences, and validates exact sentence membership. That API path was implemented but was not executed or network-tested.

The sensitivity measure replaces one feature with its training median and recomputes predicted uplift. The two largest absolute changes select the facts shown, with signs and magnitudes preserved in the audit. These are model sensitivities, not causal driver explanations. The proposed call-review action is separate from the factual evidence.

For each of the 400 recommendations, verify.py independently checks driver values against the raw HCP record and the approved sentence. It also substitutes an unsupported prescribing guarantee and checks that validation rejects it. All 400 source checks and 400 rejection checks passed. Exact factual checking limits unsupported text, but it does not establish that the explanation is useful to reps; that needs user review.

### Verification status

The full data and model pipeline ran successfully. Python compilation, JavaScript syntax and embedded-data checks passed. Browser visual testing was attempted but Chromium was unavailable, so the HTML layout and live interactions remain unverified in this environment. Local feedback is demonstration-only and is not a shared CRM workflow.

## 8 Limitations and next steps

### Client data and experimental design

Confirm the actual target, prescribing window, access rules and meaning of a completed list contact. Obtain longitudinal prescribing and as-of CRM events, rep plans and capacity, territory ownership and missingness indicators. Check whether visited and unvisited HCPs have adequate overlap. Historical calls are observational, so unmeasured access or rep selection can bias treatment-effect estimates. Run a randomized test before claiming causality.

Randomize matched rep-territory clusters to the current or new list policy with equal call budgets. Avoid HCP overlap across arms and record overrides. Measure intention-to-treat prescribing outcomes for all eligible HCPs, with cluster-aware uncertainty and a pre-agreed minimum useful effect. The available cluster count and variance determine power.

### Adoption and impact monitoring

Track displayed, reviewed, accepted, scheduled and completed recommendations. Deduplicate against pre-existing plans to measure additional calls per rep-week. Monitor dismiss reasons, repeated suggestions, access failures, displaced calls, territory coverage and workload. Track mature four-week outcomes, calibration, data freshness, feature drift, list stability and uncertainty in treatment-effect rankings.

Use week 1 for data and rep-workflow audits, week 2 for shadow lists, and weeks 3–6 for the pilot. At renewal, distinguish early adoption evidence from mature prescribing outcomes. Later pilot cohorts need follow-up beyond week 6.

### Security and deployment

Use approved HCP identifiers and minimum necessary fields, role-based territory access, encryption in transit and at rest, managed secrets, audit logs and explicit retention controls. Keep identifiers and commercially sensitive prescribing data out of unapproved LLM endpoints. Use an approved provider contract and deployment boundary, validate all generated output, and retain a verified fallback. Connect rep feedback to an authenticated CRM service and version data, models and reason policies for rollback.

## 9 Submission index

The runnable collateral is supplied in HCP_Targeting_Submission.zip. Paths below are relative to its hcp_targeting folder. The companion Markdown reproduces this report for review and reuse.

| File or directory | Purpose |
| --- | --- |
| generate_data.py and run.py | Data generation, model training, evaluation and view generation |
| requirements.txt and README.md | Dependencies, run instructions, assumptions and reproducibility |
| reason_templates.json and llm_reasons.py | LLM prompt, grounded language and optional per-HCP API pass |
| verify.py and DECISIONS.md | Evidence verification and rollout decisions |
| outputs/hcp_data.csv and simulation_truth.csv | Observed synthetic dataset and separate oracle probabilities |
| outputs/metrics.json and validation_universe.csv | Uplift and predictive results |
| outputs/list_overlap.csv and uplift_curve.* | Overlap diagnostics and cumulative gain chart/data |
| outputs/rep_view.html and ranked_list.* | Rep interface and ranked recommendations |
| outputs/reason_audit.json and all_scores.csv | Driver evidence and all-universe scores |

Run the following commands from the extracted hcp_targeting directory. The generated HTML opens directly in a browser. No API credentials are required for the default demonstration.

```bash
python -m pip install -r requirements.txt
python run.py
python verify.py
```

### Public documentation consulted

CMS Medicare Part D Prescribers by Provider
https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider

CMS provider data dictionary
https://data.cms.gov/sites/default/files/2025-04/MUP_DPR_RY25_20250401_DD_PRV_508.pdf

CMS methodological overview
https://data.cms.gov/sites/default/files/2026-05/MUP_DPR_RY26_20260421_Methodology_508.pdf

Qini metric documentation
https://www.uplift-modeling.com/en/stable/api/metrics/qini_auc_score.html

OpenAI structured output documentation
https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat

CMS documentation informed schema choices only. No public HCP microdata were downloaded, and no public dataset establishes the simulated treatment effects. The delivered metrics and limitations refer to the executed fixed-seed prototype.
