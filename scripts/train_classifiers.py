"""Small GridSearchCV comparison of five HCP classification models.

Install in the VS Code terminal:
    python -m pip install pandas numpy scikit-learn xgboost joblib

Place this script beside the outputs folder from feature_engineering_split.py:
    python train_classifiers.py
Or specify its path:
    python train_classifiers.py --data-dir "C:\\path\\to\\outputs"

Optional final evaluation of the validation-selected winner:
    python train_classifiers.py --evaluate-test

Only structured historical features enter these predictive classifiers.
Reviews, IDs, visit assignment, outcome and oracle probabilities are excluded.
This script estimates prescribing probability, not incremental visit effect.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    'prescriptions_week_before_visit', 'calls_last_90d', 'visits_last_12m',
    'category_prescriptions_before_90d', 'brand_prescriptions_before_90d',
    'brand_prescriptions_previous_90d', 'days_since_previous_visit',
    'samples_accepted_previous_visits', 'has_previous_visit', 'has_two_or_more_visits',
    'territory_central', 'territory_east', 'territory_north', 'territory_south', 'territory_west',
    'hcp_specialty_dermatology', 'hcp_specialty_primary_care',
    'hcp_stage_never_accepted_a_sample', 'hcp_stage_testing_phase', 'hcp_stage_already_prescribing',
    'days_since_most_recent_visit', 'brand_share_90d', 'brand_rx_change_90d',
]
SCALE_FEATURES = [
    'prescriptions_week_before_visit', 'calls_last_90d', 'visits_last_12m',
    'category_prescriptions_before_90d', 'brand_prescriptions_before_90d',
    'brand_prescriptions_previous_90d', 'days_since_previous_visit',
    'samples_accepted_previous_visits', 'days_since_most_recent_visit',
    'brand_share_90d', 'brand_rx_change_90d',
]


def load_split(folder, split):
    """Load aligned X/y files and reject unintended model inputs."""
    x_path, y_path = folder/f'X_{split}.csv', folder/f'y_{split}.csv'
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f'Missing {x_path.name} or {y_path.name} in {folder}. '
            'Use --data-dir to point to the feature-engineering outputs folder.')
    x = pd.read_csv(x_path)
    labels = pd.read_csv(y_path)
    if list(labels.columns) != ['prescribed_next_28d']:
        raise ValueError(f'{y_path.name} must contain only prescribed_next_28d')
    y = labels['prescribed_next_28d']
    if set(x.columns) != set(FEATURES):
        raise ValueError(f'{x_path.name}: expected the agreed 23 features. '
            f'Missing: {sorted(set(FEATURES)-set(x.columns))}. '
            f'Unexpected: {sorted(set(x.columns)-set(FEATURES))}. '
            'Use the updated feature-engineering script.')
    x = x[FEATURES]
    if len(x) != len(y) or len(x) == 0:
        raise ValueError(f'{split}: X and y must have matching, nonempty row counts')
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError(f'{split}: target must contain both classes 0 and 1')
    if not all(pd.api.types.is_numeric_dtype(t) for t in x.dtypes):
        raise ValueError(f'{split}: all X columns must be numeric')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError(f'{split}: missing or infinite inputs; check preprocessing')
    return x, y.astype(int)


def load_ids(folder, split, expected_length):
    path = folder/f'ids_{split}.csv'
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if list(frame.columns) != ['hcp_id']:
        raise ValueError(f'{path.name} must contain only hcp_id')
    ids = frame['hcp_id']
    if len(ids) != expected_length or ids.isna().any() or not ids.is_unique:
        raise ValueError(f'{path.name}: invalid or misaligned IDs')
    return ids


def evaluate(y, probability):
    """Threshold metrics use a fixed 0.5; ROC AUC uses probabilities."""
    predicted = (probability >= 0.5).astype(int)
    return dict(
        roc_auc=float(roc_auc_score(y, probability)),
        average_precision=float(average_precision_score(y, probability)),
        accuracy=float(accuracy_score(y, predicted)),
        precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        f1=float(f1_score(y, predicted, zero_division=0)),
        brier_score=float(brier_score_loss(y, probability)),
        log_loss=float(log_loss(y, probability)),
    )


def model_specs(seed):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError('Install XGBoost in this VS Code environment: '
                          'python -m pip install xgboost') from exc
    scaler = ColumnTransformer(
        [('numeric', StandardScaler(), SCALE_FEATURES)], remainder='passthrough')
    logistic = Pipeline([
        ('preprocessing', scaler),
        ('model', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed)),
    ])
    # Four candidates per model = 20 candidates total.
    return {
        'logistic_regression': (
            logistic,
            {'model__C': [0.1, 1.0], 'model__class_weight': [None, 'balanced']},
        ),
        'random_forest': (
            RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
            {'max_depth': [6, None], 'min_samples_leaf': [3, 10]},
        ),
        'decision_tree': (
            DecisionTreeClassifier(random_state=seed),
            {'max_depth': [3, 6], 'min_samples_leaf': [10, 30]},
        ),
        'xgboost': (
            XGBClassifier(n_estimators=150, objective='binary:logistic',
                eval_metric='logloss', tree_method='hist', n_jobs=1,
                random_state=seed, verbosity=0),
            {'max_depth': [2, 3], 'learning_rate': [0.05, 0.1]},
        ),
        'gradient_boosting': (
            GradientBoostingClassifier(n_estimators=100, random_state=seed),
            {'max_depth': [2, 3], 'learning_rate': [0.05, 0.1]},
        ),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=root/'data/splits')
    parser.add_argument('--results-dir', type=Path, default=root/'results/classification')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=1,
        help='Parallel grid-search workers. Default 1 limits laptop memory use.')
    parser.add_argument('--evaluate-test', action='store_true',
        help='Evaluate only the validation-selected winner on the test set.')
    args = parser.parse_args()
    x_train, y_train = load_split(args.data_dir, 'train')
    x_val, y_val = load_split(args.data_dir, 'validation')
    ids_train = load_ids(args.data_dir, 'train', len(x_train))
    ids_val = load_ids(args.data_dir, 'validation', len(x_val))
    if ids_train is not None and ids_val is not None:
        if set(ids_train) & set(ids_val):
            raise ValueError('HCP overlap between training and validation sets')
    else:
        print('ID sidecars missing: HCP overlap cannot be checked.')
    if y_train.value_counts().min() < 3:
        raise ValueError('Need at least three examples of each class for 3-fold CV')
    specs = model_specs(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    results, fitted_models = [], {}
    val_predictions = pd.DataFrame({'actual': y_val})
    if ids_val is not None:
        val_predictions.insert(0, 'hcp_id', ids_val)
    print(f'Train: {len(x_train)} rows | Validation: {len(x_val)} rows | '
          f'Features: {x_train.shape[1]}', flush=True)
    for name, (estimator, grid) in specs.items():
        print(f'\n{name}: {len(list(ParameterGrid(grid)))} candidates, 3 folds', flush=True)
        start = time.perf_counter()
        search = GridSearchCV(estimator, param_grid=grid, scoring='roc_auc', cv=cv,
            n_jobs=args.n_jobs, pre_dispatch='n_jobs', refit=True,
            return_train_score=True, error_score='raise', verbose=1)
        search.fit(x_train, y_train)
        probability = search.best_estimator_.predict_proba(x_val)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        report = classification_report(
            y_val, predicted, labels=[0, 1],
            target_names=['Will not prescribe (0)', 'Will prescribe (1)'],
            output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(
            args.results_dir/f'{name}_validation_classification_report.csv')
        (args.results_dir/f'{name}_validation_classification_report.txt').write_text(
            classification_report(
                y_val, predicted, labels=[0, 1],
                target_names=['Will not prescribe (0)', 'Will prescribe (1)'],
                digits=4, zero_division=0))
        row = dict(model=name, cv_auc=float(search.best_score_),
            cv_auc_std=float(search.cv_results_['std_test_score'][search.best_index_]),
            **evaluate(y_val, probability),
            best_parameters=json.dumps(search.best_params_),
            elapsed_seconds=round(time.perf_counter()-start, 2))
        results.append(row)
        fitted_models[name] = search.best_estimator_
        val_predictions[name] = probability
        pd.DataFrame(search.cv_results_).to_csv(args.results_dir/f'{name}_grid_results.csv', index=False)
        joblib.dump(search.best_estimator_, args.results_dir/f'{name}_best.joblib')
        print(f"CV AUC {row['cv_auc']:.4f} | Validation AUC {row['roc_auc']:.4f} | "
              f"Validation F1 {row['f1']:.4f}", flush=True)
        print('Best parameters:', search.best_params_, flush=True)
        print('\nValidation classification report (threshold 0.5):', flush=True)
        print(classification_report(
            y_val, predicted, labels=[0, 1],
            target_names=['Will not prescribe (0)', 'Will prescribe (1)'],
            digits=4, zero_division=0), flush=True)
    comparison = pd.DataFrame(results).sort_values('roc_auc', ascending=False, kind='stable')
    comparison.to_csv(args.results_dir/'validation_comparison.csv', index=False)
    val_predictions.to_csv(args.results_dir/'validation_predictions.csv', index=False)
    winner = comparison.iloc[0]['model']
    joblib.dump(fitted_models[winner], args.results_dir/'selected_model.joblib')
    import xgboost
    summary = dict(selected_model=winner, selection_metric='Validation ROC AUC',
        cv='3-fold stratified CV on training rows only; hyperparameters selected by mean ROC AUC',
        training_rows=len(x_train), validation_rows=len(x_val),
        classification_threshold=0.5, feature_names=FEATURES,
        models_refitted_on='Training set only, as part of GridSearchCV',
        test_evaluated=False,
        note='Predictive classification only. No visit-effect or causal claims.',
        versions=dict(python=platform.python_version(), pandas=pd.__version__,
            scikit_learn=sklearn.__version__, xgboost=xgboost.__version__))
    if args.evaluate_test:
        x_test, y_test = load_split(args.data_dir, 'test')
        ids_test = load_ids(args.data_dir, 'test', len(x_test))
        if ids_test is not None:
            for known_ids in [ids_train, ids_val]:
                if known_ids is not None and set(ids_test) & set(known_ids):
                    raise ValueError('Test HCPs overlap with development HCPs')
        probability = fitted_models[winner].predict_proba(x_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        final = dict(model=winner, rows=len(x_test), **evaluate(y_test, probability),
            confusion_matrix=confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
            confusion_matrix_order='Rows actual 0/1; columns predicted 0/1',
            classification_report=classification_report(y_test, predicted, output_dict=True, zero_division=0))
        test_report = classification_report(
            y_test, predicted, labels=[0, 1],
            target_names=['Will not prescribe (0)', 'Will prescribe (1)'],
            output_dict=True, zero_division=0)
        pd.DataFrame(test_report).transpose().to_csv(
            args.results_dir/'selected_model_test_classification_report.csv')
        test_report_text = classification_report(
            y_test, predicted, labels=[0, 1],
            target_names=['Will not prescribe (0)', 'Will prescribe (1)'],
            digits=4, zero_division=0)
        (args.results_dir/'selected_model_test_classification_report.txt').write_text(test_report_text)
        print('\nSelected-model test classification report:', flush=True)
        print(test_report_text, flush=True)
        (args.results_dir/'selected_model_test_metrics.json').write_text(json.dumps(final, indent=2))
        predictions = pd.DataFrame({'actual':y_test, 'probability':probability, 'predicted':predicted})
        if ids_test is not None:
            predictions.insert(0, 'hcp_id', ids_test)
        predictions.to_csv(args.results_dir/'selected_model_test_predictions.csv', index=False)
        summary['test_evaluated'] = True
        print('\nSelected-model test results:', json.dumps(final, indent=2), flush=True)
    (args.results_dir/'run_summary.json').write_text(json.dumps(summary, indent=2))
    print('\nValidation comparison:')
    print(comparison[['model','cv_auc','roc_auc','average_precision','precision','recall','f1']].to_string(index=False))
    print(f'\nSelected by validation ROC AUC: {winner}')
    print(f'Results saved to: {args.results_dir.resolve()}')
    if not args.evaluate_test:
        print('Test set was not loaded. Add --evaluate-test for final evaluation of the winner.')


if __name__ == '__main__':
    main()
