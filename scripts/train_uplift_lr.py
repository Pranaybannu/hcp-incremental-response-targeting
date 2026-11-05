"""Two-model logistic regression uplift on randomized additional-visit data.
Keep beside train_classifiers.py. Run: python train_uplift_lr.py --data-dir outputs
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, brier_score_loss, roc_auc_score
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k, qini_curve, uplift_curve
from train_classifiers import load_split, load_ids, SCALE_FEATURES


def make_model(seed):
    # No balanced class weights: we need outcome probabilities, not reweighted scores.
    return Pipeline([
        ('preprocessing', ColumnTransformer([('numeric', StandardScaler(), SCALE_FEATURES)], remainder='passthrough')),
        ('model', LogisticRegression(C=0.1, class_weight=None, max_iter=2000, random_state=seed))])


def top_effect(y, t, score, fraction):
    ix = np.argsort(-score, kind='stable')[:int(np.ceil(len(y)*fraction))]
    a, b = y[ix][t[ix] == 1], y[ix][t[ix] == 0]
    return float(a.mean()-b.mean()) if len(a) and len(b) else float('nan')


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root/'data/splits')
    parser.add_argument('--output-dir', type=Path, default=root/'results/uplift')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    parts, ids = {}, {}
    for split in ['train', 'validation', 'test']:
        x, y = load_split(args.data_dir, split)
        frame = pd.read_csv(args.data_dir/f'T_{split}.csv')
        if list(frame.columns) != ['visit_assigned'] or len(frame) != len(x) or not frame.visit_assigned.isin([0, 1]).all():
            raise ValueError(f'Invalid T_{split}.csv. Expected aligned binary visit_assigned.')
        t = frame.visit_assigned.astype(int)
        ids[split] = load_ids(args.data_dir, split, len(x))
        if ids[split] is None:
            raise ValueError(f'Missing ids_{split}.csv')
        if any(y[t == arm].nunique() != 2 for arm in [0, 1]):
            raise ValueError(f'{split}: both outcomes required in each treatment arm')
        parts[split] = (x, y, t)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if set(ids[a]) & set(ids[b]):
            raise ValueError(f'HCP overlap: {a}, {b}')
    # Prespecified C=0.1; no hyperparameter search or test-based selection.
    x = pd.concat([parts['train'][0], parts['validation'][0]], ignore_index=True)
    y = pd.concat([parts['train'][1], parts['validation'][1]], ignore_index=True)
    t = pd.concat([parts['train'][2], parts['validation'][2]], ignore_index=True)
    models = {}
    for arm in [0, 1]:
        models[arm] = make_model(args.seed).fit(x[t == arm], y[t == arm])
    baseline = make_model(args.seed).fit(x, y)
    xt, yt, tt = parts['test']
    yv, tv = yt.to_numpy(), tt.to_numpy()
    p0, p1 = [models[arm].predict_proba(xt)[:, 1] for arm in [0, 1]]
    u = p1-p0
    scores = {'uplift_lr': u, 'pooled_lr_prescribing_probability': baseline.predict_proba(xt)[:, 1]}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(args.seed)
    bootstrap = {name: [] for name in scores}
    difference = []
    # Paired HCP bootstrap, holding fitted models fixed. Does not include refit uncertainty.
    for _ in range(500):
        ix = rng.integers(0, len(yv), len(yv))
        vals = [top_effect(yv[ix], tv[ix], score[ix], .2) for score in scores.values()]
        for name, value in zip(scores, vals):
            bootstrap[name].append(value)
        difference.append(vals[0]-vals[1])
    for name, score in scores.items():
        row = dict(ranking=name, normalized_qini_auc=float(qini_auc_score(yv, score, tv)),
                   normalized_uplift_auc=float(uplift_auc_score(yv, score, tv)))
        for k in [.1,.2,.5]:
            row[f'uplift_at_{int(k*100)}pct'] = float(uplift_at_k(yv, score, tv, strategy='overall', k=k))
        row['uplift_at_20pct_ci95_low'], row['uplift_at_20pct_ci95_high'] = np.nanpercentile(bootstrap[name], [2.5,97.5]).tolist()
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out/'ranking_comparison.csv', index=False)
    predictions = pd.DataFrame({'hcp_id':ids['test'], 'actual':yv, 'visit_assigned':tv,
        'p_without_additional_visit':p0, 'p_with_additional_visit':p1, 'predicted_uplift':u,
        'pooled_lr_probability':scores['pooled_lr_prescribing_probability']})
    predictions = predictions.sort_values(['predicted_uplift','hcp_id'], ascending=[False,True])
    predictions.insert(0,'uplift_rank',range(1,len(predictions)+1))
    predictions.to_csv(out/'test_uplift_ranking.csv', index=False)
    bins = []
    for number, indices in enumerate(np.array_split(np.arange(len(predictions)), 5), start=1):
        chunk = predictions.iloc[indices]
        v, c = chunk[chunk.visit_assigned==1],chunk[chunk.visit_assigned==0]
        bins.append(dict(quintile=number, rows=len(chunk), visited=len(v), control=len(c),
            predicted_uplift_mean=float(chunk.predicted_uplift.mean()),
            observed_rate_difference=float(v.actual.mean()-c.actual.mean())))
    pd.DataFrame(bins).to_csv(out/'uplift_by_quintile.csv',index=False)
    arm_metrics=[]
    for arm, prob in [(0,p0),(1,p1)]:
        mask=tv==arm
        arm_metrics.append(dict(arm=arm, test_rows=int(mask.sum()), auc=float(roc_auc_score(yv[mask],prob[mask])),
                                brier=float(brier_score_loss(yv[mask],prob[mask]))))
        (out/f'arm_{arm}_classification_report.txt').write_text(classification_report(
            yv[mask],(prob[mask]>=.5).astype(int),digits=4,zero_division=0),encoding='utf-8')
    joblib.dump(models[0],out/'lr_control.joblib')
    joblib.dump(models[1],out/'lr_additional_visit.joblib')
    joblib.dump(baseline,out/'lr_pooled_baseline.joblib')
    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax, fn, title in [(axes[0],qini_curve,'Qini: cumulative incremental prescribers'),
                           (axes[1],uplift_curve,'Uplift: cumulative gain, population scale')]:
        for name, score in scores.items():
            cx,cy=fn(yv,score,tv)
            ax.plot(cx/len(yv)*100,cy,label=name)
            pd.DataFrame({'fraction_selected':cx/len(yv),'gain':cy}).to_csv(out/f'{name}_{fn.__name__}.csv',index=False)
        ax.plot([0,100],[0,cy[-1]],'--',color='gray',label='Random ordering baseline')
        ax.set(xlabel='Highest-ranked HCPs selected (%)',ylabel='Estimated incremental prescribers',title=title)
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'test_qini_uplift_curves.png',dpi=160);plt.close(fig)
    summary=dict(method='T-learner, two independent logistic regressions',C=.1,class_weight=None,
        hyperparameters='Prespecified; no tuning on test or model selection',refit_rows=len(x),
        control_training_rows=int((t==0).sum()),visited_training_rows=int((t==1).sum()),test_rows=len(yv),
        test_observed_ate=float(yv[tv==1].mean()-yv[tv==0].mean()),
        test_mean_predicted_uplift=float(u.mean()),arm_prediction_metrics=arm_metrics,
        top20_uplift_difference_ci95=np.nanpercentile(difference,[2.5,97.5]).tolist(),
        causal_assumption='Synthetic randomized additional visit; all assignments completed; no interference.',
        limitations='Synthetic evidence only. Previously inspected test set. Bootstrap holds models fixed; no training uncertainty. Tiny selected arms are noisy.',
        metrics_definition='scikit-uplift normalized Qini and uplift AUC; top-k is visited rate minus control rate within overall selected fraction.')
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(comparison.to_string(index=False))
    print(json.dumps(summary,indent=2))
    print(f'Outputs: {out.resolve()}')


if __name__ == '__main__':
    main()
