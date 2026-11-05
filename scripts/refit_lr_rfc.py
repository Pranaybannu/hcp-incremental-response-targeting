"""Refit tuned LR and RFC on train + validation, then evaluate test.

Keep beside train_classifiers.py. Run the grid search first.
python refit_lr_rfc.py --data-dir outputs --models-dir classification_results
Only load trusted joblib files produced by your own training script.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from train_classifiers import load_split, load_ids, evaluate


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root/'data/splits')
    parser.add_argument('--models-dir', type=Path, default=root/'results/classification')
    parser.add_argument('--output-dir', type=Path, default=root/'results/refit')
    args = parser.parse_args()
    splits = {name: load_split(args.data_dir, name)
              for name in ['train', 'validation', 'test']}
    ids = {name: load_ids(args.data_dir, name, len(pair[0]))
           for name, pair in splits.items()}
    if any(value is None for value in ids.values()):
        raise ValueError('Include ids_train.csv, ids_validation.csv and ids_test.csv to verify HCP separation.')
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(ids[left]) & set(ids[right]):
            raise ValueError(f'HCP overlap between {left} and {right}')
    x_full = pd.concat([splits['train'][0], splits['validation'][0]], ignore_index=True)
    y_full = pd.concat([splits['train'][1], splits['validation'][1]], ignore_index=True)
    x_test, y_test = splits['test']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, expected_type in [('logistic_regression', LogisticRegression),
                                 ('random_forest', RandomForestClassifier)]:
        source = args.models_dir/f'{name}_best.joblib'
        if not source.is_file():
            raise FileNotFoundError(f'Missing {source}. Run train_classifiers.py first.')
        selected = joblib.load(source)
        estimator = selected.named_steps.get('model') if hasattr(selected, 'named_steps') else selected
        if not isinstance(estimator, expected_type):
            raise ValueError(f'{source.name} contains an unexpected model type')
        # Cloning clears all fitted state, including LR scaling statistics.
        # Selected hyperparameters remain fixed. Test data never enter fit.
        final = clone(selected)
        final.fit(x_full, y_full)
        probability = final.predict_proba(x_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        metrics = dict(model=name, training_rows=len(splits['train'][0]),
                       validation_rows=len(splits['validation'][0]),
                       refit_rows=len(x_full), test_rows=len(x_test), threshold=0.5,
                       **evaluate(y_test, probability))
        rows.append(metrics)
        folder = args.output_dir/name
        folder.mkdir(exist_ok=True)
        joblib.dump(final, folder/f'{name}_final.joblib')
        (folder/'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        report = classification_report(y_test, predicted, labels=[0, 1], digits=4, zero_division=0)
        (folder/'test_classification_report.txt').write_text(report, encoding='utf-8')
        pd.DataFrame(classification_report(y_test, predicted, labels=[0, 1],
                     output_dict=True, zero_division=0)).T.to_csv(folder/'test_classification_report.csv')
        pd.DataFrame(confusion_matrix(y_test, predicted, labels=[0, 1]),
                     index=['actual_0', 'actual_1'], columns=['predicted_0', 'predicted_1']).to_csv(folder/'confusion_matrix.csv')
        pd.DataFrame(dict(hcp_id=ids['test'], actual=y_test, probability=probability,
                          predicted=predicted)).to_csv(folder/'test_predictions.csv', index=False)
        print(f'\n{name}: refitted on {len(x_full)} HCPs, tested on {len(x_test)}')
        print(report)
        print(json.dumps(metrics, indent=2))
    pd.DataFrame(rows).to_csv(args.output_dir/'test_comparison.csv', index=False)
    print(f'\nSaved to {args.output_dir.resolve()}')
    print('Test results are for reporting, not further tuning. Prior test inspection limits unbiased assessment.')


if __name__ == '__main__':
    main()
