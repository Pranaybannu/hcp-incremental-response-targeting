"""Impute structural missingness, then one-hot encode agreed categories.

Run with Python 3.10+, NumPy and pandas installed:
python preprocess_hcp.py --input /path/to/hcp_dataset_4000.csv

These operations use fixed rules, not population statistics. No model is trained.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

CATEGORIES = {
    'territory': ['Central', 'East', 'North', 'South', 'West'],
    'hcp_specialty': ['Dermatology', 'Primary care'],
    'hcp_stage': ['Never accepted a sample', 'Testing phase', 'Already prescribing'],
}
NUMERIC_FEATURES = [
    'prescriptions_week_before_visit', 'calls_last_90d', 'visits_last_12m',
    'category_prescriptions_before_90d', 'brand_prescriptions_before_90d',
    'brand_prescriptions_previous_90d', 'days_since_previous_visit',
    'samples_accepted_previous_visits', 'has_previous_visit', 'has_two_or_more_visits',
]
DATE_METADATA = ['most_recent_visit_timestamp', 'rep_review_timestamp', 'interpretation_as_of_date']
OUTCOME_COLUMNS = ['visit_assigned', 'prescribed_next_28d']


def impute(raw):
    """Preserve meaningful no-history states and reject unexpected missingness."""
    df = raw.copy()
    required = set(NUMERIC_FEATURES[:-2] + list(CATEGORIES) + DATE_METADATA +
                   ['hcp_id', 'feature_cutoff_date', 'rep_review_with_interpretation'])
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f'Missing required columns: {sorted(absent)}')
    if df.hcp_id.isna().any() or not df.hcp_id.is_unique:
        raise ValueError('HCP IDs must be populated and unique')
    visits = pd.to_numeric(df.visits_last_12m, errors='raise')
    if visits.isna().any() or (visits < 0).any() or (visits % 1 != 0).any():
        raise ValueError('Historical visit counts must be nonnegative integers')
    visited, two_visits = visits.gt(0), visits.ge(2)
    expected_missing = {
        'prescriptions_week_before_visit': ~visited,
        'days_since_previous_visit': ~two_visits,
        'rep_review_with_interpretation': ~visited,
        **{column: ~visited for column in DATE_METADATA},
    }
    for column, expected in expected_missing.items():
        if not df[column].isna().equals(expected):
            raise ValueError(f'{column}: missingness differs from the agreed visit-history rule; investigate it first')
    allowed_missing = set(expected_missing)
    unexpected = [c for c in df if c not in allowed_missing and df[c].isna().any()]
    if unexpected:
        raise ValueError(f'Unexpected missing values in {unexpected}; no automatic imputation policy agreed')
    df['has_previous_visit'] = visited.astype('int8')
    df['has_two_or_more_visits'] = two_visits.astype('int8')
    for column in ['prescriptions_week_before_visit', 'days_since_previous_visit']:
        df[column] = df[column].fillna(0).astype('int64')
    df['rep_review_with_interpretation'] = df.rep_review_with_interpretation.fillna('No previous visit review')
    # Timestamp metadata stays missing for never-visited HCPs. It is not fed to
    # the numeric feature matrix. Never replace it with an invented date.
    return df


def encode_categories(imputed):
    """Fixed, explicitly named one-hot indicators; no arbitrary numerical order."""
    df = imputed.copy()
    mapping = {}
    for column, values in CATEGORIES.items():
        unknown = set(df[column].dropna().unique()) - set(values)
        if df[column].isna().any() or unknown:
            raise ValueError(f'{column}: missing/unknown category {unknown}; update schema explicitly')
        encoded_columns = []
        for category in values:
            suffix = category.lower().replace(' ', '_')
            name = f'{column}_{suffix}'
            df[name] = df[column].eq(category).astype('int8')
            encoded_columns.append(name)
        if not df[encoded_columns].sum(axis=1).eq(1).all():
            raise ValueError(f'Invalid one-hot encoding for {column}')
        mapping[column] = dict(zip(values, encoded_columns))
        df = df.drop(columns=column)
    encoded_names = [name for group in mapping.values() for name in group.values()]
    x = df[NUMERIC_FEATURES + encoded_names].copy()
    if x.isna().any().any():
        raise ValueError('Numeric feature matrix contains missing values')
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
        raise ValueError('Numeric feature matrix contains nonnumeric columns')
    return df, x, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent.parent/'data/raw/hcp_dataset_4000.csv')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent.parent/'data/processed')
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    imputed = impute(raw)
    encoded, x, mapping = encode_categories(imputed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imputed.to_csv(args.output_dir/'01_hcp_imputed.csv', index=False)
    encoded.to_csv(args.output_dir/'02_hcp_encoded.csv', index=False)
    x.to_csv(args.output_dir/'03_model_features_X.csv', index=False)
    # Same row order as X; this table is a key/treatment/target sidecar, not X.
    sidecar = raw[['hcp_id'] + [c for c in OUTCOME_COLUMNS if c in raw]].copy()
    sidecar.to_csv(args.output_dir/'04_hcp_ids_treatment_target.csv', index=False)
    report = {
        'rows': len(raw), 'input_columns': len(raw.columns),
        'model_feature_columns': len(x.columns),
        'missing_before': {c: int(n) for c,n in raw.isna().sum().items() if n},
        'missing_after_in_imputed_data': {c: int(n) for c,n in imputed.isna().sum().items() if n},
        'missing_in_model_features': int(x.isna().sum().sum()),
        'categorical_mapping': mapping,
        'imputation': {
            'prescriptions_week_before_visit': '0 with has_previous_visit indicator',
            'days_since_previous_visit': '0 with has_two_or_more_visits indicator',
            'rep_review_with_interpretation': 'No previous visit review',
            'timestamp_metadata': 'Retained missing and excluded from X',
        },
        'treatment_target_preserved': all(raw[c].equals(encoded[c]) for c in OUTCOME_COLUMNS if c in raw),
        'feature_policy': 'No IDs, timestamps, raw review text, treatment, target, tiers or oracle probabilities in X.',
        'statistical_fitting': 'None. All transformations use fixed rules and declared category vocabulary.',
    }
    (args.output_dir/'preprocessing_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
