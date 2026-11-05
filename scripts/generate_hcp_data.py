"""Generate historical HCP features and separate synthetic future visit outcomes.

Run: python generate_hcp_data.py
Dependencies: numpy, pandas
No model is trained. All data and response effects are illustrative assumptions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n_hcps=4000, seed=42, cutoff='2026-11-01'):
    if n_hcps < 1:
        raise ValueError('n_hcps must be positive')
    rng = np.random.default_rng(seed)
    cutoff_day = pd.Timestamp(cutoff).normalize()
    dates = pd.date_range(end=cutoff_day, periods=365, freq='D')
    records, truth, visit_records = [], [], []

    for i in range(n_hcps):
        hcp_id = f'HCP{i + 1:04d}'
        specialty = rng.choice(['Dermatology', 'Primary care'], p=[0.4, 0.6])
        territory = rng.choice(['North', 'South', 'East', 'West', 'Central'])

        # Right-skewed volume is illustrative, not calibrated to public records.
        category_rate = float(np.clip(rng.lognormal(0.25, 0.85), 0.05, 20))
        category_rate *= 1.4 if specialty == 'Dermatology' else 1
        category_daily = rng.poisson(category_rate, 365)
        brand_share = float(rng.beta(1.0, 5.0))
        never_adopted = rng.random() < 0.30
        if never_adopted:
            brand_share = 0.0
        trend = float(rng.uniform(0.65, 1.45))
        daily_share = np.full(365, brand_share)
        # Brand launched nine months ago, approximated as 270 days.
        daily_share[:95] = 0
        daily_share[-90:] = np.clip(brand_share * trend, 0, 0.95)
        brand_daily = rng.binomial(category_daily, daily_share)
        category_recent = int(category_daily[-90:].sum())
        brand_recent = int(brand_daily[-90:].sum())
        brand_previous = int(brand_daily[-180:-90].sum())

        # Historical reps tend to visit established prescribers more often.
        expected_calls = float(np.clip(0.7 + brand_recent / 18, 0.7, 14))
        call_count = min(int(rng.poisson(expected_calls)), 24)
        # Last possible offset leaves enough raw history for a 7-day pre-visit count.
        offsets = sorted(rng.choice(358, size=call_count, replace=False), reverse=True)
        visits = [cutoff_day - pd.Timedelta(days=int(day))
                  + pd.Timedelta(hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
                  for day in offsets]
        samples = rng.binomial(1, 0.6, call_count)
        calls90 = sum(v.normalize() > cutoff_day - pd.Timedelta(days=90) for v in visits)
        samples_previous = int(samples[:-1].sum()) if call_count else 0
        ever_prescribed = bool(brand_daily.sum() > 0)
        if ever_prescribed:
            stage = 'Already prescribing'
        elif samples.sum() > 0:
            stage = 'Testing phase'
        else:
            stage = 'Never accepted a sample'

        latest = visits[-1] if visits else None
        gap = (visits[-1].normalize() - visits[-2].normalize()).days if call_count >= 2 else None
        weekly_count = None
        review = None
        review_timestamp = None
        if latest is not None:
            day = latest.normalize()
            window = (dates >= day - pd.Timedelta(days=7)) & (dates < day)
            weekly_count = int(brand_daily[window].sum())
            if samples[-1]:
                observation = 'Rep observation: accepted a sample and requested product information.'
            else:
                observation = 'Rep observation: declined the sample during this visit.'
            share_text = f'{100 * brand_recent / category_recent:.1f}%' if category_recent else 'unavailable'
            interpretation = (
                f'Data interpretation at cutoff: {category_recent} category prescriptions and '
                f'{brand_recent} client-brand prescriptions in the recent 90 days '
                f'(brand share {share_text}); {brand_previous} brand prescriptions in the preceding 90 days.'
            )
            review = observation + ' ' + interpretation
            review_timestamp = (latest + pd.Timedelta(hours=2)).isoformat(sep=' ')
        # For unvisited HCPs no rep review is fabricated. This is structural missingness.
        for j, visit in enumerate(visits):
            visit_records.append(dict(hcp_id=hcp_id, visit_id=f'{hcp_id}-V{j+1:02d}',
                visit_timestamp=visit.isoformat(sep=' '),
                review_timestamp=(visit + pd.Timedelta(hours=2)).isoformat(sep=' '),
                rep_review=('Accepted a sample and requested product information.' if samples[j]
                            else 'Declined the sample during this visit.')))

        # Future intervention is randomized, unlike historical call selection.
        # Simplification: an assigned visit is completed on cutoff + 1 day.
        assigned = int(rng.binomial(1, 0.5))
        p_without = float(0.02 + 0.90 * (1 - np.exp(-brand_recent / 20)))
        current_share = brand_recent / category_recent if category_recent else 0
        category_potential = category_recent / (category_recent + 100)
        effect = (0.02 + 0.35 * category_potential * (1 - current_share)
                  * np.exp(-calls90 / 4))
        effect *= float(rng.uniform(0.6, 1.4))
        if rng.random() < 0.20:
            effect = 0.0
        p_with = float(min(0.98, p_without + effect))
        outcome = int(rng.binomial(1, p_with if assigned else p_without))
        records.append(dict(
            hcp_id=hcp_id, feature_cutoff_date=cutoff_day.date().isoformat(),
            most_recent_visit_timestamp=latest.isoformat(sep=' ') if latest else None,
            prescriptions_week_before_visit=weekly_count,
            rep_review_with_interpretation=review,
            rep_review_timestamp=review_timestamp,
            interpretation_as_of_date=cutoff_day.date().isoformat() if latest else None,
            hcp_stage=stage, calls_last_90d=int(calls90),
            visits_last_12m=call_count, territory=territory, hcp_specialty=specialty,
            category_prescriptions_before_90d=category_recent,
            brand_prescriptions_before_90d=brand_recent,
            brand_prescriptions_previous_90d=brand_previous,
            days_since_previous_visit=gap,
            samples_accepted_previous_visits=samples_previous,
            visit_assigned=assigned, prescribed_next_28d=outcome))
        truth.append(dict(hcp_id=hcp_id, probability_without_visit=p_without,
            probability_with_visit=p_with, true_visit_effect=p_with-p_without))

    data = pd.DataFrame(records)
    for column in ['prescriptions_week_before_visit', 'days_since_previous_visit']:
        data[column] = data[column].astype('Int64')
    oracle = pd.DataFrame(truth)
    history = pd.DataFrame(visit_records, columns=[
        'hcp_id','visit_id','visit_timestamp','review_timestamp','rep_review'])
    validate(data, oracle, history)
    return data, oracle, history


def validate(data, truth, history):
    assert data.hcp_id.is_unique
    assert set(data.visit_assigned.unique()) <= {0, 1}
    assert set(data.prescribed_next_28d.unique()) <= {0, 1}
    assert (data.brand_prescriptions_before_90d <= data.category_prescriptions_before_90d).all()
    assert (data.calls_last_90d <= data.visits_last_12m).all()
    assert (data.samples_accepted_previous_visits <= (data.visits_last_12m-1).clip(lower=0)).all()
    visited = data.visits_last_12m > 0
    assert data.loc[visited, 'rep_review_with_interpretation'].notna().all()
    assert data.loc[~visited, 'most_recent_visit_timestamp'].isna().all()
    assert data.loc[data.visits_last_12m < 2, 'days_since_previous_visit'].isna().all()
    assert (truth.probability_without_visit.between(0, 1)).all()
    assert (truth.probability_with_visit.between(0, 1)).all()
    if len(history):
        cutoff = pd.to_datetime(data.feature_cutoff_date.iloc[0]) + pd.Timedelta(days=1)
        assert (pd.to_datetime(history.review_timestamp) < cutoff).all()
        actual_calls = history.groupby('hcp_id').size().reindex(data.hcp_id, fill_value=0).to_numpy()
        assert np.array_equal(actual_calls, data.visits_last_12m)
    never = data.hcp_stage == 'Never accepted a sample'
    assert (data.loc[never, 'brand_prescriptions_before_90d'] == 0).all()
    assert 'sample_accepted' not in data.columns
    assert not any('tier' in c or 'after' in c for c in data.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-hcps', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cutoff', default='2026-11-01')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent.parent/'data/raw')
    args = parser.parse_args()
    data, truth, history = generate(args.n_hcps, args.seed, args.cutoff)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_dir/'hcp_dataset_4000.csv', index=False)
    # Deployment scoring inputs contain no future assignment or label.
    data.drop(columns=['visit_assigned','prescribed_next_28d']).to_csv(
        args.out_dir/'historical_features_only.csv', index=False)
    truth.to_csv(args.out_dir/'simulation_truth_DO_NOT_TRAIN.csv', index=False)
    history.to_csv(args.out_dir/'historical_visit_log.csv', index=False)
    summary = dict(rows=len(data), columns=len(data.columns), seed=args.seed,
        cutoff=args.cutoff, stage_counts=data.hcp_stage.value_counts().to_dict(),
        visit_assigned_counts=data.visit_assigned.value_counts().to_dict(),
        target_counts=data.prescribed_next_28d.value_counts().to_dict(),
        zero_previous_visits=int((data.visits_last_12m==0).sum()),
        three_or_more_previous_visits=int((data.visits_last_12m>=3).sum()),
        historical_visit_rows=len(history))
    (args.out_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
