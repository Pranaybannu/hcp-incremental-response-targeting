"""Create clearly labelled synthetic rep plans and >=3-visit subsets.
Run from the project workspace. This script uses no outcomes to build rep plans.
"""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from train_classifiers import FEATURES

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, help='Raw hcp_dataset_4000.csv')
parser.add_argument('--data-dir', type=Path, help='Feature engineering outputs folder')
parser.add_argument('--model', type=Path, help='Trusted refitted logistic regression joblib')
parser.add_argument('--output-dir', type=Path, default=root/'results/ranking')
args = parser.parse_args()

def locate(explicit, candidates, label, option):
    if explicit is not None:
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f'{label} not found: {explicit.resolve()}')
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = '\n'.join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f'{label} not found. Use {option} to specify its path.\nChecked:\n{tried}')

raw_path = locate(args.input, [root/'data/raw/hcp_dataset_4000.csv',
    root/'data/raw/hcp_dataset_4000.csv', root/'original_hcp_data.csv',
    root.parent/'hcp_data_fresh/data/hcp_dataset_4000.csv'], 'Raw dataset', '--input')
folders = [args.data_dir] if args.data_dir else [root/'data/splits', root.parent/'hcp_feature_split/outputs']
engineered_path = locate(None, [p/'hcp_engineered.csv' for p in folders],
                         'Engineered dataset', '--data-dir')
manifest_path = engineered_path.parent/'split_manifest.csv'
model_path = locate(args.model, [root/'results/refit/logistic_regression/logistic_regression_final.joblib',
    root/'logistic_regression_final.joblib'], 'Refitted LR model', '--model')
out = args.output_dir
raw = pd.read_csv(raw_path)
manifest = pd.read_csv(manifest_path)
data = raw.merge(manifest, on='hcp_id', validate='one_to_one')
if len(data) != len(raw) or set(raw.hcp_id) != set(manifest.hcp_id):
    raise ValueError('Raw dataset and split manifest must contain the same HCP IDs.')
engineered = pd.read_csv(engineered_path)
if engineered.hcp_id.isna().any() or not engineered.hcp_id.is_unique or set(engineered.hcp_id) != set(data.hcp_id):
    raise ValueError('Engineered data must contain exactly one row per raw HCP ID.')
model = joblib.load(model_path)
out.mkdir(parents=True, exist_ok=True)
eligible = data[data.visits_last_12m >= 3].copy()
eligible.to_csv(out/'eligible_universe_ge3_visits.csv', index=False)
eligible[eligible.split == 'validation'].to_csv(out/'original_validation_subset_ge3_visits.csv', index=False)
eligible[eligible.split == 'test'].to_csv(out/'test_subset_ge3_visits.csv', index=False)

# Illustrative existing call plan: 80 HCPs per territory, selected by historical
# contact frequency, then brand prescribing volume, then ID. Not an actual plan.
plan = (data.sort_values(['calls_last_90d','visits_last_12m',
         'brand_prescriptions_before_90d','hcp_id'], ascending=[False,False,False,True])
         .groupby('territory', sort=False).head(80).copy())
plan['synthetic_rep_priority_in_territory'] = plan.groupby('territory').cumcount()+1
plan['plan_is_synthetic'] = True
plan['planning_assumption'] = '80 HCPs per territory; calls, annual visits, brand Rx descending'
cols = ['hcp_id','territory','synthetic_rep_priority_in_territory','calls_last_90d',
        'visits_last_12m','brand_prescriptions_before_90d','plan_is_synthetic','planning_assumption']
plan[cols].to_csv(out/'synthetic_rep_planned_list.csv', index=False)
ranked = data.copy()
prob = pd.Series(model.predict_proba(engineered[FEATURES])[:,1], index=engineered.hcp_id)
ranked['predicted_prescribing_probability'] = ranked.hcp_id.map(prob)
ranked['on_synthetic_rep_plan'] = ranked.hcp_id.isin(plan.hcp_id)
ranked = ranked.sort_values(['predicted_prescribing_probability','hcp_id'], ascending=[False,True])
ranked['model_rank'] = range(1,len(ranked)+1)
ranked['in_top_10_percent'] = ranked.model_rank <= 400
ranked.to_csv(out/'lr_ranked_hcp_universe.csv', index=False)
ranked.head(400).to_csv(out/'lr_top400_recommendations.csv', index=False)
examples = pd.concat([ranked[ranked.on_synthetic_rep_plan].head(1),
                     ranked[~ranked.on_synthetic_rep_plan].head(1), ranked.tail(1)])
show = ['hcp_id','model_rank','predicted_prescribing_probability','visits_last_12m',
        'calls_last_90d','brand_prescriptions_before_90d','on_synthetic_rep_plan','split']
examples[show].to_csv(out/'three_row_match_example.csv', index=False)
summary = dict(universe=len(data), eligible_ge3=len(eligible),
    original_validation_ge3=int((eligible.split=='validation').sum()),
    test_ge3=int((eligible.split=='test').sum()), synthetic_rep_plan=len(plan),
    model_top400_on_synthetic_plan=int(ranked.head(400).on_synthetic_rep_plan.sum()))
(out/'summary.json').write_text(json.dumps(summary, indent=2))
(out/'README.txt').write_text('All HCP data and rep plans are synthetic. No actual rep plan was provided.\n'
    'Rep plan assumes 80 HCPs per territory selected by calls, annual visits and brand volume.\n'
    'Overlap is partly constructed by this assumption; it is not evidence of real rep behavior.\n'
    'Eligible universe is all HCPs with >=3 annual visits. Original validation subset is their intersection with the existing validation split.\n'
    'Validation rows were included in final refit; that subset is no longer held out.\n'
    'The existing LR model was trained on the full train+validation population, not restricted to >=3 visits.\n'
    'All-universe rankings include training rows and are illustrative scoring, not unbiased model evaluation.\n'
    'Probability ranks descending; HCP ID breaks ties. It predicts prescribing, not incremental visit response.\n'
    'Matching is by hcp_id. Synthetic rep plan is separate from randomized visit_assigned.\n')
print(json.dumps(summary, indent=2))
print(examples[show].to_string(index=False))
print(f'Output folder: {out.resolve()}')
