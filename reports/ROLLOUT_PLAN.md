# After-rollout measurement

## Separate adoption from impact

Adoption measures whether reps use recommendations. It cannot establish whether visits cause extra prescribing.

Track a weekly snapshot ID, model version, HCP ID, territory, rep ID, publication time, recommendation rank and whether the HCP was already on the frozen rep call plan. Freeze the plan before model-list delivery to avoid measuring a plan already changed by the list.

For every event store timestamp and context: list available, opened, HCP reviewed, accepted/planned, contacted, deferred and reason. Measure:
- List availability before the agreed weekly planning deadline.
- Unique rep open rate: reps opening / reps assigned a list.
- Recommended-HCP contact rate: distinct flagged HCPs contacted during a defined window / distinct eligible flagged HCPs.
- New-contact rate: contacted recommendations absent from the pre-existing plan / eligible recommendations absent from that plan.
- Acceptance and deferral reasons, segmented by territory, stage and rep seniority.
- Grounding defect rate, rep-rated explanation usefulness, and time spent reviewing.

Deduplicate within model/snapshot/HCP and distinguish planned from completed visits. Do not count repeated clicks as additional adoption. The app collects only a local subset of this schema and has no authenticated rep identity.

## Fair old-vs-new policy pilot

Randomise sufficiently many reps or territories to current probability-ranked lists versus uplift-ranked lists. Choose the unit to reduce spillover; stratify on baseline volume, territory and activity. Keep list size, visit capacity, timing, UI, training and follow-up window equivalent so the targeting policy is the principal difference. Do not show both lists to the same trial rep.

Define the primary outcome and minimum worthwhile effect before launch. Match the model target with HCP-level any-brand prescribing within 28 days; separately measure prescription volume, contribution net of visit cost, rep workload and access/compliance guardrails. Volume and profit require data not currently simulated in the binary target.

Estimate the intention-to-treat effect of assigning the new policy, including recommendations reps did not follow. Use baseline adjustment and confidence intervals that account for the randomisation unit. Plan sample size using cluster effects and a realistic minimum detectable improvement; allow the complete 28-day outcome lag before analysis. Avoid early conclusions from partial follow-up.

Randomising lists establishes the value of the targeting policy. Validating visit-effect estimates themselves requires randomised additional visits or another credible causal design. An optional within-policy randomised visit holdout may help, subject to operational feasibility.

## Monitoring

Monitor input freshness/missingness, territory coverage, score and stage distributions, recurring HCPs, list stability, drift, probability calibration and explanation grounding. Compare outcomes after labels mature. Log model/prompt versions, source features and validation results for every reason; fall back to a verified template on failure.
Respect CRM permissions and minimise sensitive rep notes. Real client deployment requires approved access, retention, security and content review.
