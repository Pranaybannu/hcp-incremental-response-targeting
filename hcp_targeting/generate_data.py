"""Entirely synthetic HCPs. Oracle probabilities are for evaluation only."""
import numpy as np
import pandas as pd


def generate(n=4000, seed=42):
    r = np.random.default_rng(seed)
    specialty = r.choice(['Dermatology', 'Primary care'], n, p=[.4, .6])
    category = np.clip(r.lognormal(3.8 + .5*(specialty=='Dermatology'), .9, n), 2, 1000).astype(int)
    adoption = r.beta(1.2, 5, n)
    brand = r.binomial(category, adoption)
    # Rep selection follows already-visible prescribing and familiarity.
    visits = r.poisson(np.clip(.4 + .13*brand, .4, 14))
    days = np.maximum(7, (180/(1+visits) + r.normal(0, 12, n))).astype(int)
    access = r.binomial(1, .75, n)
    engagement = np.round(r.beta(2, 2, n), 3)
    territory = r.choice(['North', 'South', 'East', 'West', 'Central'], n)
    # Outcome: at least one new brand prescription in next four weeks.
    logit = -3.2 + 1.6*np.log1p(brand) + .25*(specialty=='Dermatology') + .2*engagement
    p0 = 1/(1+np.exp(-logit))
    # Genuine heterogeneous effects. Low-call, high-category, accessible HCPs
    # contain incremental opportunities. Saturated HCPs gain little.
    tau = (.025 + .34*((visits<3)&(category>=35))*access
           + .09*engagement*access - .035*(visits>=6))
    p1 = np.clip(p0+tau, .001, .995)
    # Deliberate randomized experiment assumption, NOT inferred from rep history.
    treatment = r.binomial(1, .5, n)
    outcome = r.binomial(1, np.where(treatment==1, p1, p0))
    frame = pd.DataFrame(dict(hcp_id=[f'SYN-{i:04d}' for i in range(n)],
        specialty=specialty, territory=territory, category_rx_90d=category,
        brand_rx_90d=brand, visits_12m=visits, days_since_visit=days,
        access_confirmed=access, engagement_score=engagement,
        treatment=treatment, outcome_4w=outcome))
    oracle = pd.DataFrame(dict(hcp_id=frame.hcp_id, p_without_visit=p0,
        p_with_visit=p1, true_uplift=p1-p0))
    return frame, oracle

if __name__ == '__main__':
    from pathlib import Path
    out = Path('outputs'); out.mkdir(exist_ok=True)
    data, truth = generate(); data.to_csv(out/'hcp_data.csv', index=False)
    truth.to_csv(out/'simulation_truth.csv', index=False)
