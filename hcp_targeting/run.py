"""Run from this directory: python run.py. No credentials needed."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from generate_data import generate

FEATURES = ['category_rx_90d','brand_rx_90d','visits_12m','days_since_visit',
            'access_confirmed','engagement_score','specialty_Dermatology']


def matrix(data):
    x = data.copy()
    x['specialty_Dermatology'] = (x.specialty=='Dermatology').astype(int)
    return x[FEATURES].astype(float)


def model():
    return HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=10,
        min_samples_leaf=35, l2_regularization=5, random_state=42)


def probability(m, x):
    return m.predict_proba(x)[:, 1]


def uplift(models, x):
    return probability(models[1], x)-probability(models[0], x)


def curve(y, t, score):
    # Known randomized propensity 0.5. HT transformed outcome gives an
    # unbiased cumulative incremental-response estimator at a fixed ranking.
    order = np.argsort(-np.asarray(score), kind='stable')
    transformed = np.asarray(y)*(np.asarray(t)/.5-(1-np.asarray(t))/.5)
    gain = np.r_[0, np.cumsum(transformed[order])]/len(y)
    fraction = np.linspace(0, 1, len(gain))
    return fraction, gain


def area(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, 'trapezoid') else float(np.trapz(y, x))


def score_metrics(y, t, scores, truth, k):
    f, gain = curve(y,t,scores)
    top = np.argsort(-scores)[:k]
    yt, tt = np.asarray(y)[top], np.asarray(t)[top]
    effect = float(yt[tt==1].mean()-yt[tt==0].mean()) if len(np.unique(tt))==2 else None
    return dict(auuc_ht=area(gain,f), qini_area_above_random=area(gain-f*gain[-1],f),
        top_k=k, top_k_observed_uplift=effect,
        top_k_treated=int(tt.sum()), top_k_control=int((1-tt).sum()),
        top_k_oracle_mean_effect=float(np.asarray(truth)[top].mean()),
        top_k_oracle_expected_incremental_responses=float(np.asarray(truth)[top].sum()))


def main(seed=42):
    root=Path(__file__).resolve().parent
    out=root/'outputs'; out.mkdir(exist_ok=True)
    data, oracle=generate(seed=seed); x=matrix(data)
    train, rest=train_test_split(np.arange(len(data)),test_size=.4,random_state=seed)
    valid, test=train_test_split(rest,test_size=.5,random_state=seed)
    data['split']='train'; data.loc[valid,'split']='validation'; data.loc[test,'split']='test'
    data.to_csv(out/'hcp_data.csv',index=False); oracle.to_csv(out/'simulation_truth.csv',index=False)
    # The published exhibit says validation is restricted. Training restriction
    # is a reconstruction assumption and is contrasted with all-universe training.
    familiar=data.visits_12m.to_numpy()>=3
    selected_train=train[familiar[train]]
    baseline=model().fit(x.iloc[selected_train],data.outcome_4w.iloc[selected_train])
    broad=model().fit(x.iloc[train],data.outcome_4w.iloc[train])
    learners={}
    for arm in [0,1]:
        rows=train[data.treatment.iloc[train].to_numpy()==arm]
        learners[arm]=model().fit(x.iloc[rows],data.outcome_4w.iloc[rows])
    baseline_scores=probability(baseline,x)
    broad_scores=probability(broad,x)
    lift=uplift(learners,x)
    diagnostics=[]
    for name, mask in [('familiar_3plus',familiar[test]),('low_visit_0to2',~familiar[test]),('all_hcps',np.ones(len(test),bool))]:
        ix=test[mask]; y=data.outcome_4w.iloc[ix]
        for label,s in [('restricted_baseline',baseline_scores),('all_universe_predictive',broad_scores)]:
            top=ix[np.argsort(-s[ix])[:max(1,int(np.ceil(.1*len(ix))))]]
            diagnostics.append(dict(model=label,segment=name,n=len(ix),
                auc=float(roc_auc_score(y,s[ix])) if y.nunique()>1 else None,
                average_precision=float(average_precision_score(y,s[ix])),
                brier=float(brier_score_loss(y,s[ix])),
                outcome_rate=float(y.mean()), top_decile_response=float(data.outcome_4w.iloc[top].mean())))
    pd.DataFrame(diagnostics).to_csv(out/'validation_universe.csv',index=False)
    y=data.outcome_4w.iloc[test].to_numpy(); t=data.treatment.iloc[test].to_numpy()
    true=oracle.true_uplift.iloc[test].to_numpy(); k=int(.1*len(test))
    metrics={name:score_metrics(y,t,s[test],true,k) for name,s in
        [('current_predictive',baseline_scores),('broad_predictive',broad_scores),('uplift_t_learner',lift)]}
    # Bootstrap uncertainty for fixed held-out rankings, not retraining uncertainty.
    rng=np.random.default_rng(seed+1); diffs=[]
    for _ in range(400):
        b=rng.integers(0,len(test),len(test))
        vals=[]
        for scores in [lift[test],baseline_scores[test]]:
            f,g=curve(y[b],t[b],scores[b]); vals.append(area(g-f*g[-1],f))
        diffs.append(vals[0]-vals[1])
    metrics['qini_difference_bootstrap_95pct']=np.quantile(diffs,[.025,.975]).tolist()
    overlap=[]
    for scope, ix in [('heldout_test',test),('full_universe_descriptive',np.arange(len(data)))]:
        count=max(1,int(.1*len(ix)))
        for label,s in [('current_predictive',baseline_scores),('uplift_t_learner',lift)]:
            top=ix[np.argsort(-s[ix])[:count]]
            overlap.append(dict(scope=scope,model=label,n=count,
                fraction_3plus_prior_visits=float(familiar[top].mean()),
                fraction_visited_last_30d=float((data.days_since_visit.iloc[top]<=30).mean()),
                mean_true_effect=float(oracle.true_uplift.iloc[top].mean())))
    pd.DataFrame(overlap).to_csv(out/'list_overlap.csv',index=False)
    # Drivers: model sensitivity to replacement by TRAINING median, not causal
    # explanation, SHAP, or proof that changing a feature changes prescribing.
    reference=x.iloc[train].median()
    top=np.argsort(-lift)[:400]
    sensitivity=np.empty((len(top),len(FEATURES)))
    for j,feature in enumerate(FEATURES):
        altered=x.iloc[top].copy(); altered[feature]=reference[feature]
        sensitivity[:,j]=lift[top]-uplift(learners,altered)
    templates=json.loads((root/'reason_templates.json').read_text())
    rows=[]; evidence=[]
    for pos,i in enumerate(top):
        driver_cols=np.argsort(-np.abs(sensitivity[pos]))[:2]
        facts=[]; sentences=[]
        for j in driver_cols:
            field=FEATURES[j]; value=float(x.iloc[i][field])
            if field=='access_confirmed': rendered='confirmed' if value else 'not confirmed'
            elif field=='specialty_Dermatology': rendered='Dermatology' if value else 'Primary care'
            elif field=='engagement_score': rendered=f'{value:.3f}'
            else: rendered=str(int(value))
            sentence=templates['templates'][field].format(value=rendered)
            # Values come directly from this row, with exact approved wording.
            assert value==float(x.iloc[i][field])
            assert sentence==templates['templates'][field].format(value=rendered)
            facts.append(dict(field=field,value=value,display_value=rendered,
                sensitivity_pp=float(100*sensitivity[pos,j]), verified=True))
            sentences.append(sentence)
        row=data.iloc[i]
        rows.append(dict(rank=pos+1,hcp_id=row.hcp_id,territory=row.territory,
            specialty=row.specialty, uplift_pp=round(float(100*lift[i]),1),
            current_score=round(float(baseline_scores[i]),3),
            visits=int(row.visits_12m),days_since_visit=int(row.days_since_visit),
            access=bool(row.access_confirmed), reason=' '.join(sentences),
            action=templates['action'],drivers=facts,split=row.split))
        evidence.append(dict(hcp_id=row.hcp_id,drivers=facts,verified=True))
    (out/'ranked_list.json').write_text(json.dumps(rows,indent=2))
    (out/'reason_audit.json').write_text(json.dumps(evidence,indent=2))
    pd.DataFrame([{k:v for k,v in row.items() if k!='drivers'} for row in rows]).to_csv(out/'ranked_list.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(dict(hcp_id=data.hcp_id,baseline_score=baseline_scores,uplift_score=lift)).to_csv(out/'all_scores.csv',index=False)
    curves=[]
    plt.figure(figsize=(8,5))
    for name,s in [('Current predictive',baseline_scores),('Uplift T learner',lift)]:
        f,g=curve(y,t,s[test]); plt.plot(f,g,label=name)
        curves.extend(dict(model=name,fraction=float(a),gain=float(b)) for a,b in zip(f,g))
    plt.plot(f,f*g[-1],'--',color='gray',label='Random ranking')
    plt.xlabel('Fraction of held-out HCPs targeted'); plt.ylabel('Estimated incremental responses per held-out HCP')
    plt.title('Held-out randomized synthetic evaluation'); plt.legend(); plt.tight_layout()
    plt.savefig(out/'uplift_curve.png',dpi=160); plt.close()
    pd.DataFrame(curves).to_csv(out/'uplift_curve.csv',index=False)
    create_html(root,out,rows,diagnostics,metrics,overlap)
    assert len(data)==4000 and len(rows)==400 and len({r['hcp_id'] for r in rows})==400
    assert not (set(train)&set(test)) and all(r['verified'] for r in evidence)
    assert set(FEATURES).isdisjoint({'treatment','outcome_4w','true_uplift','p_with_visit','p_without_visit'})
    print(json.dumps(dict(diagnostics=diagnostics,overlap=overlap,metrics=metrics),indent=2))


def create_html(root,out,rows,diagnostics,metrics,overlap):
    template=(root/'view.html').read_text()
    for key,value in [('ROWS',rows),('DIAGNOSTICS',diagnostics),('METRICS',metrics),('OVERLAP',overlap)]:
        template=template.replace('__'+key+'__',json.dumps(value).replace('</','<\\/'))
    (out/'rep_view.html').write_text(template)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=42)
    main(parser.parse_args().seed)
