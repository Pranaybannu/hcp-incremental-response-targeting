# Decision brief

## Diagnosis

The model may rank “likely prescribers” rather than “HCPs whose response a new visit changes.” Familiar-HCP validation selects an already-served population. A high AUC on that subset can coexist with negligible list novelty. The trainer's report makes this a hypothesis to test, not a proven root cause.

Logged list-contact rates are 14/340 = 4.12%, 11/335 = 3.28%, and 16/352 = 4.55%. An opened list is not an accepted recommendation. Clarify whether “from list” measures CRM attribution or all calls to flagged HCPs; under-recording could explain apparent non-adoption. These are week snapshots, not enough to reconstruct every week of five months.

Audit actual overlap with reps' pre-list plans, accessible HCP coverage, territory ownership, capacity, duplicates, novelty across weekly refreshes, outcome horizon and data latency. Interview representatives on the top ten suggestions and rejected reasons. Weekly refreshing can repeatedly return the same high-propensity people without adding value. Avoid assuming the algorithm alone explains adoption.

## Build choices

1. A 4,000-HCP synthetic universe with dermatology/primary care, five territories, right-skewed category volume, brand history, prior calls, recency, access and engagement.
2. Restricted predictive baseline and broad predictive comparator, evaluated by familiar versus low-call segments.
3. T learner: fit separate outcome models for randomized synthetic visit and no-visit arms; score the probability difference. Match top-decile capacity across methods.
4. Hold out HCPs, evaluate AUC/AP/Brier for prediction and HT gain/Qini plus uplift-at-capacity for incrementality. Show oracle effects only as labelled simulation diagnostics. The observed Qini comparison is inconclusive.
5. Rank full universe for a rep demo, filter by territory/access, show source-backed reasons, record acceptance/dismissal. Full-universe descriptive rankings include training rows; evaluation does not.

A 400-person list is the explicit 10% of 4,000 demo capacity. The client's 335–352 flagged counts need clarification on eligibility, threshold or territory filtering. Territory capacity and access constraints should be applied before production assignment. Do not assume all positive uplift HCPs can actually be visited.

## Rollout metrics

Adoption funnel: eligible flagged → displayed → reviewed → accepted → scheduled → completed. Primary adoption rate is completed additional calls / eligible recommendations, deduplicated against the pre-existing call plan. Also track net-new calls per rep-week, recurring suggestions, dismiss reasons, time to schedule, and list churn.

Impact: intention-to-treat difference in four-week new prescribing per eligible HCP between list-policy arms, with fixed exposure windows and an equal call budget. Record prescription volume as a secondary outcome if the operational objective is volume rather than first/new prescription. Actual objective must be agreed with the client before rebuilding. Report effect sizes, cluster-aware uncertainty and cost per incremental response; no ROI claim without value/cost inputs.

Guardrails: displaced previously planned calls, access failures, rep workload, territory coverage, over-contact, data staleness, and adverse feedback. Rep explanations must avoid unsupported patient/clinical details.

## Six-week experiment

Week 1: confirm target, as-of features, attribution definition and familiar-HCP coverage; inspect plan overlap and interview reps.
Week 2: run shadow lists, check data freshness and reasons, establish call capacity and pre-register outcomes / minimum useful effect.
Weeks 3–6: randomize matched rep-territory clusters to current versus new lists, balance capacity and baseline rates; retain intention-to-treat assignment despite overrides. Prevent cross-arm HCP overlap and record spillover.
At renewal: report completed-call adoption and mature four-week outcome cohorts only. Later cohorts require follow-up after renewal. Client cluster count and outcome variance determine power; don't promise statistical resolution within six weeks.

## What I accepted and challenged from AI

Accepted: distinguish predictive propensity from incremental treatment response; use equal-capacity held-out comparisons; create restricted factual reason templates.
Modified: label baseline training restriction as an assumption because only validation restriction is given; define outcome provisionally; use honest synthetic schema rather than imply downloaded CMS calibration.
Rejected: equating AUC with business impact, claiming synthetic true effects prove client lift, or hiding the nonpositive Qini result. The held-out observed evidence calls for a randomized pilot.
