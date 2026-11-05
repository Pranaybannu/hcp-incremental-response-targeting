"""Engineer historical features and create a jointly stratified 60/20/20 split.

Default VS Code usage: python feature_engineering_split.py
Place this script beside 02_hcp_encoded.csv, or pass --input explicitly.
Requires pandas, numpy and scikit-learn. No model training or scaling occurs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_FEATURES = [
    'prescriptions_week_before_visit', 'calls_last_90d', 'visits_last_12m',
    'category_prescriptions_before_90d', 'brand_prescriptions_before_90d',
    'brand_prescriptions_previous_90d', 'days_since_previous_visit',
    'samples_accepted_previous_visits', 'has_previous_visit', 'has_two_or_more_visits',
    'territory_central', 'territory_east', 'territory_north', 'territory_south', 'territory_west',
    'hcp_specialty_dermatology', 'hcp_specialty_primary_care',
    'hcp_stage_never_accepted_a_sample', 'hcp_stage_testing_phase', 'hcp_stage_already_prescribing',
]
ENGINEERED_FEATURES = [
    'days_since_most_recent_visit', 'brand_share_90d', 'brand_rx_change_90d',
]


def engineer_historical_features(df):
    required = set(BASE_FEATURES + ['hcp_id', 'feature_cutoff_date',
        'most_recent_visit_timestamp', 'visit_assigned', 'prescribed_next_28d'])
    missing = required-set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.hcp_id.isna().any() or not df.hcp_id.is_unique:
        raise ValueError('Expected one populated, unique HCP ID per row')
    if df[BASE_FEATURES].isna().any().any():
        raise ValueError('Run the missing-value preprocessing step first')
    for field in ['visit_assigned','prescribed_next_28d']:
        if not df[field].isin([0,1]).all():
            raise ValueError(f'{field} must contain only 0 and 1')
    result = df.copy()
    cutoff = pd.to_datetime(df.feature_cutoff_date, errors='raise').dt.normalize()
    latest = pd.to_datetime(df.most_recent_visit_timestamp, errors='raise').dt.normalize()
    if not latest.notna().equals(df.has_previous_visit.eq(1)):
        raise ValueError('Visit timestamp availability conflicts with history indicator')
    recency = (cutoff-latest).dt.days
    if recency.dropna().lt(0).any():
        raise ValueError('A historical visit occurs after the feature cutoff')
    # Zero is a computational placeholder for never visited; the existing
    # has_previous_visit feature distinguishes it from a real same-day visit.
    result['days_since_most_recent_visit'] = recency.fillna(0).astype(int)
    category = df.category_prescriptions_before_90d
    recent = df.brand_prescriptions_before_90d
    previous = df.brand_prescriptions_previous_90d
    if (recent > category).any() or (df[BASE_FEATURES[:8]] < 0).any().any():
        raise ValueError('Invalid historical counts')
    # A zero-category denominator uses a zero placeholder; the original
    # category count in X identifies that undefined-share case.
    result['brand_share_90d'] = recent.div(category.replace(0,np.nan)).fillna(0)
    result['brand_rx_change_90d'] = recent-previous
    return result


def split_and_save(df, output_dir, seed=42):
    engineered = engineer_historical_features(df)
    # Explicit allowlist keeps IDs, review text, timestamps, treatment,
    # outcome and any accidental extra/oracle columns outside model inputs.
    x = engineered[BASE_FEATURES + ENGINEERED_FEATURES].copy()
    y = engineered['prescribed_next_28d'].astype(int)
    treatment = engineered['visit_assigned'].astype(int)
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Model features contain missing or infinite values')
    strata = y.astype(str)+'_'+treatment.astype(str)
    if strata.value_counts().min() < 10:
        raise ValueError('Too few records in a target/treatment group for this stratified split')
    indices = np.arange(len(df))
    train, remaining = train_test_split(indices, test_size=0.4,
        random_state=seed, stratify=strata)
    validation, test = train_test_split(remaining, test_size=0.5,
        random_state=seed, stratify=strata.iloc[remaining])
    splits = {'train':train, 'validation':validation, 'test':test}
    assert not (set(train)&set(validation) or set(train)&set(test) or set(validation)&set(test))
    assert set(np.concatenate(list(splits.values()))) == set(indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(output_dir/'hcp_engineered.csv',index=False)
    manifest = pd.DataFrame({'hcp_id':df.hcp_id, 'split':''})
    summary = {}
    for name, ix in splits.items():
        x.iloc[ix].to_csv(output_dir/f'X_{name}.csv', index=False)
        y.iloc[ix].to_csv(output_dir/f'y_{name}.csv', index=False)
        treatment.iloc[ix].to_csv(output_dir/f'T_{name}.csv', index=False)
        df[['hcp_id']].iloc[ix].to_csv(output_dir/f'ids_{name}.csv', index=False)
        manifest.loc[ix,'split'] = name
        summary[name] = dict(rows=len(ix),
            prescribing_rate=float(y.iloc[ix].mean()),
            visit_assignment_rate=float(treatment.iloc[ix].mean()),
            joint_counts=strata.iloc[ix].value_counts().sort_index().to_dict())
    manifest.to_csv(output_dir/'split_manifest.csv',index=False)
    report = dict(seed=seed, features=x.columns.tolist(), feature_count=x.shape[1],
        split_ratio='60% train / 20% validation / 20% test',
        stratification='Joint prescribed_next_28d and visit_assigned; keys are target_treatment',
        splits=summary, leakage_checks=dict(disjoint_hcp_sets=True,
            target_excluded_from_X=True,treatment_excluded_from_X=True,
            all_features_historical=True,no_learned_preprocessing=True),
        limitation='One common synthetic cutoff. Random stratification is not a forward temporal backtest.')
    (output_dir/'split_report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=root/'data/processed/02_hcp_encoded.csv')
    parser.add_argument('--output-dir',type=Path,default=root/'data/splits')
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    print(json.dumps(split_and_save(pd.read_csv(args.input),args.output_dir,args.seed),indent=2))


if __name__ == '__main__':
    main()
