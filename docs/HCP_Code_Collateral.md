# HCP targeting source code collateral

Reproduce with Python 3.10 or later. Create the following files in one hcp_targeting directory, then run the commands below. Results are synthetic. The optional LLM API pass has not been executed.

```bash
python -m pip install -r requirements.txt
python run.py
python verify.py
```

## requirements.txt

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7

```

## generate_data.py

```python
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

```

## run.py

```python
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

```

## reason_templates.json

```json
{
  "provenance": "Language templates authored by ChatGPT for this assessment. Each HCP's selected model-driver facts are rendered into these constrained templates. No separate per-HCP API call is claimed.",
  "prompt": "Turn selected local uplift-model drivers into short factual rep-facing sentences. Use only the supplied feature name and value. Do not promise prescribing changes or invent clinical facts. Return templates with a single value placeholder. Keep the proposed action separate from the evidence.",
  "templates": {
    "category_rx_90d": "Recorded category volume is {value} prescriptions in the past 90 days.",
    "brand_rx_90d": "Recorded brand volume is {value} prescriptions in the past 90 days.",
    "visits_12m": "The HCP has {value} recorded rep visits in the past 12 months.",
    "days_since_visit": "The last recorded rep visit was {value} days ago.",
    "access_confirmed": "Recorded access status is {value}.",
    "engagement_score": "The recorded engagement score is {value} on a 0 to 1 scale.",
    "specialty_Dermatology": "The specialty record indicates {value}."
  },
  "action": "Review for an additional call this week; confirm access and fit with the existing territory plan."
}

```

## llm_reasons.py

```python
"""Optional per-HCP LLM pass. Requires OPENAI_API_KEY and OPENAI_MODEL.
Only approved factual sentences can reach the view. Fail closed to audited text.
Run after run.py: python llm_reasons.py
"""
import json
import os
from pathlib import Path
import urllib.request
from run import create_html
import pandas as pd


def validate(candidate, source):
    if set(candidate)!={'hcp_id','sentences'} or candidate['hcp_id']!=source['hcp_id']:
        return False
    sentences=candidate['sentences']
    approved=source['reason'].split('. ')
    approved=[s if s.endswith('.') else s+'.' for s in approved]
    return (isinstance(sentences,list) and len(sentences)==len(approved)
            and len(set(sentences))==len(approved) and set(sentences)==set(approved))


def main():
    root=Path(__file__).resolve().parent; out=root/'outputs'
    rows=json.loads((out/'ranked_list.json').read_text())
    key=os.environ.get('OPENAI_API_KEY'); model=os.environ.get('OPENAI_MODEL')
    if not key or not model:
        raise SystemExit('Set OPENAI_API_KEY and OPENAI_MODEL to use the optional per-HCP pass. The audited template demo already runs without either.')
    log=[]
    for start in range(0,len(rows),20):
        batch=rows[start:start+20]
        prompt=('Return JSON with key reasons, a list of {hcp_id, sentences}. For each HCP, '
                'turn its selected model-driver facts into a short rep reason by selecting and '
                'ordering ALL of its approved factual sentences. Copy sentences verbatim. '
                'Do not add causal, clinical, patient, or performance claims. The action is '
                'provided separately by the application. Input: '+json.dumps([
                    dict(hcp_id=r['hcp_id'],drivers=r['drivers'],approved_reason=r['reason']) for r in batch]))
        body=json.dumps(dict(model=model,messages=[dict(role='user',content=prompt)],
            response_format={'type':'json_object'})).encode()
        request=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=body,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=90) as response:
                raw=json.load(response)
            content=raw['choices'][0]['message']['content']
            candidates=json.loads(content)['reasons']
            mapping={c['hcp_id']:c for c in candidates}
            if len(mapping)!=len(candidates):
                raise ValueError('Duplicate HCP IDs')
            for source in batch:
                candidate=mapping.get(source['hcp_id'],{})
                accepted=validate(candidate,source)
                if accepted: source['reason']=' '.join(candidate['sentences'])
                log.append(dict(hcp_id=source['hcp_id'],accepted=accepted,
                    fallback=not accepted,candidate=candidate))
        except Exception as exc:
            # No credential or response body in logs.
            log.extend(dict(hcp_id=r['hcp_id'],accepted=False,fallback=True,
                error_type=type(exc).__name__) for r in batch)
    (out/'llm_call_audit.json').write_text(json.dumps(dict(model=model,
        policy='Exact approved sentence membership; invalid output replaced by verified template.',results=log),indent=2))
    (out/'ranked_list.json').write_text(json.dumps(rows,indent=2))
    diagnostics=pd.read_csv(out/'validation_universe.csv').to_dict('records')
    metrics=json.loads((out/'metrics.json').read_text())
    overlaps=pd.read_csv(out/'list_overlap.csv').to_dict('records')
    create_html(root,out,rows,diagnostics,metrics,overlaps)
    html=(out/'rep_view.html').read_text().replace('No separate per-HCP LLM API call was made.',
        'The optional per-HCP LLM pass was attempted. See llm_call_audit.json for accepted responses and verified fallbacks.')
    (out/'rep_view.html').write_text(html)
    print('Audited',len(log),'per-HCP results; accepted',sum(r['accepted'] for r in log))

if __name__=='__main__': main()

```

## verify.py

```python
"""Independent evidence and artifact checks, including a tampered-reason check."""
import copy
import json
from pathlib import Path
import pandas as pd
from llm_reasons import validate

root=Path(__file__).resolve().parent
rows=json.loads((root/'outputs/ranked_list.json').read_text())
data=pd.read_csv(root/'outputs/hcp_data.csv').set_index('hcp_id')
templates=json.loads((root/'reason_templates.json').read_text())['templates']
for row in rows:
    source=data.loc[row['hcp_id']]
    expected=[]
    for fact in row['drivers']:
        field=fact['field']
        value=int(source.specialty=='Dermatology') if field=='specialty_Dermatology' else float(source[field])
        assert value==fact['value'], (row['hcp_id'],field)
        if field=='access_confirmed': text='confirmed' if value else 'not confirmed'
        elif field=='specialty_Dermatology': text='Dermatology' if value else 'Primary care'
        elif field=='engagement_score': text=f'{value:.3f}'
        else: text=str(int(value))
        expected.append(templates[field].format(value=text))
    assert row['reason']==' '.join(expected)
    assert validate({'hcp_id':row['hcp_id'],'sentences':expected},row)
    bad=copy.deepcopy(expected); bad[0]='This HCP will definitely prescribe after a visit.'
    assert not validate({'hcp_id':row['hcp_id'],'sentences':bad},row)
assert len(data)==4000 and len(rows)==400
html=(root/'outputs/rep_view.html').read_text()
assert '__ROWS__' not in html and 'id="territory"' in html
print('PASS: 400 reasons independently checked against source data; 400 unsupported claims rejected; 4,000 HCPs and HTML payload verified.')

```

## view.html

```html
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HCP call opportunities</title><style>
:root{font:15px system-ui;color:#172b3a;background:#f3f6f8}body{margin:0}header{background:#112f42;color:white;padding:28px 5%}h1{margin:8px 0;font-size:30px}.tag{font-size:12px;letter-spacing:2px;color:#8ce4d2}main{max-width:1280px;margin:auto;padding:24px}.note{background:#fff3d8;padding:14px;border-left:4px solid #d8a144;line-height:1.6}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:20px 0}.card{background:white;padding:20px;border-radius:10px}.number{font-size:27px;font-weight:700;color:#087c70}label{display:inline-block;margin:10px 16px 10px 0}select,input{padding:9px;border:1px solid #bbcbd4;border-radius:6px;background:white}button{padding:10px 16px;background:#087c70;color:white;border:0;border-radius:6px;cursor:pointer}.scroll{overflow:auto;background:white;border-radius:10px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:14px;border-bottom:1px solid #e4ebef;vertical-align:top}th{background:#eaf0f3;font-size:12px;text-transform:uppercase}td:nth-child(4){min-width:290px}small{display:block;color:#5e7280;margin-top:7px;line-height:1.5}.pill{background:#dcf3eb;color:#126649;padding:4px 8px;border-radius:15px;white-space:nowrap}details{background:white;padding:18px;margin-top:20px;border-radius:10px}pre{white-space:pre-wrap;font-size:12px}a{color:#087c70}h2{font-size:20px}.muted{color:#637c8b}footer{padding:20px 0;color:#637c8b}@media(max-width:700px){.cards{grid-template-columns:1fr}main{padding:12px}}
</style></head><body><header><div class="tag">SALES OPERATIONS · SYNTHETIC DEMONSTRATION</div><h1>Find calls that could make a difference</h1><p>Ranked by estimated incremental prescribing response to one additional visit.</p></header><main>
<div class="note">All HCPs and outcomes are synthetic. Scores estimate a probability difference, not guaranteed results. Verify access and territory fit before planning a call. Reasons describe recorded facts selected by model sensitivity; they do not establish causal drivers.</div>
<div class="cards"><div class="card"><div class="muted">Current model · familiar-HCP AUC</div><div class="number" id="auc"></div><small>Held-out HCPs with at least 3 visits</small></div><div class="card"><div class="muted">Current top-list familiarity</div><div class="number" id="overlap"></div><small>Top 10% with at least 3 prior visits · held-out</small></div><div class="card"><div class="muted">Uplift ranking · simulated effect</div><div class="number" id="effect"></div><small>Mean true uplift in held-out top 10% · oracle diagnostic only</small></div></div>
<h2>Weekly call opportunities</h2><p class="muted">400 highest-uplift HCPs from the 4,000-person universe. Full-universe scores include training rows; performance metrics below use held-out rows only.</p>
<label>Territory <select id="territory"><option value="">All territories</option></select></label><label>Access <select id="access"><option value="">Any</option><option value="yes">Confirmed</option></select></label><label>HCP ID <input id="search" placeholder="SYN-0123"></label><label>Show <select id="limit"><option>25</option><option>50</option><option>100</option><option>400</option></select></label><button id="export">Export filtered CSV</button><p id="count" class="muted"></p>
<div class="scroll"><table><thead><tr><th>Rank / HCP</th><th>Territory / specialty</th><th>Estimated uplift</th><th>Evidence and next action</th><th>Rep feedback</th></tr></thead><tbody id="body"></tbody></table></div>
<details><summary>How reasons were generated and checked</summary><p>ChatGPT authored a constrained factual language template for each available feature. The pipeline selects each HCP's two largest absolute local sensitivity drivers and inserts that HCP's verified values. Every sentence must match an approved template. This is LLM-assisted template generation, with deterministic per-HCP rendering. No separate per-HCP LLM API call was made.</p><p>A driver measures the change in predicted uplift when one feature is replaced by the training median. It is model sensitivity, not a causal explanation. Each selected fact and sensitivity is available below and in reason_audit.json. The action is a proposed rep review, not a clinical claim.</p></details>
<details><summary>Held-out evaluation and uncertainty</summary><p>The uplift model uses two outcome models, one per randomly assigned visit arm. Predicted uplift = P(outcome | visit, X) − P(outcome | no visit, X). Synthetic assignment is 50/50. The test set contains 800 HCPs, never used for fitting.</p><p>Cumulative gain is the cumulative sum of Y[T/0.5 − (1−T)/0.5], divided by test-set size. Qini area is the area between that curve and random ranking. This is an unnormalized, propensity-weighted area, not a library-normalized Qini score. Top-decile observed uplift is noisy; inspect arm counts and the bootstrap interval.</p><pre id="metrics"></pre><h3>Validation-universe diagnostics</h3><pre id="diag"></pre></details>
<details><summary>Rollout and measurement plan</summary><p>Primary adoption metric: additional completed calls attributable to the recommendation per eligible flagged HCP, with deduplication against the rep's pre-existing plan. Track viewed → accepted → scheduled → completed, dismiss reasons, unavailable access, territory capacity, stale records, and list novelty. Opening the list alone is insufficient.</p><p>Randomize comparable rep-territory clusters to the existing or uplift list, balance baseline prescribing and capacity, freeze assignment, keep call budgets equal, and avoid cross-arm HCPs. Measure intention-to-treat incremental new prescribing per eligible HCP over a fixed four-week window and completed incremental calls per rep-week. Cluster-bootstrap intervals and report crossovers. This tests the new policy, not just those the reps chose to visit.</p><p>Six weeks: week 1 audit data, outcome and overlap, and interview reps; week 2 shadow lists and confirm feasibility; weeks 3–6 run a randomized pilot. Report adoption early and prescribing only for cohorts with mature four-week outcomes. Do not claim all week-6 outcomes are mature at renewal. No power calculation is possible without client variance, cluster counts and target minimum effect.</p></details>
<footer>Generated locally · fixed-seed reproducible · feedback stays in this browser and is not transmitted.</footer></main><script>
const rows=__ROWS__,diagnostics=__DIAGNOSTICS__,metrics=__METRICS__,overlaps=__OVERLAP__;
const el=id=>document.getElementById(id), escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
el('auc').textContent=diagnostics.find(x=>x.model==='restricted_baseline'&&x.segment==='familiar_3plus').auc.toFixed(3);
el('overlap').textContent=(100*overlaps.find(x=>x.scope==='heldout_test'&&x.model==='current_predictive').fraction_3plus_prior_visits).toFixed(1)+'%';
el('effect').textContent=(100*metrics.uplift_t_learner.top_k_oracle_mean_effect).toFixed(1)+' pp';
for(const t of [...new Set(rows.map(x=>x.territory))].sort()){const o=document.createElement('option');o.value=t;o.textContent=t;el('territory').append(o)}
function filtered(){return rows.filter(r=>(!el('territory').value||r.territory===el('territory').value)&&(!el('access').value||r.access)&&r.hcp_id.toLowerCase().includes(el('search').value.toLowerCase()))}
function render(){const chosen=filtered();el('count').textContent=chosen.length+' matching recommendations';el('body').innerHTML=chosen.slice(0,Number(el('limit').value)).map(r=>`<tr><td><b>#${r.rank} · ${escape(r.hcp_id)}</b><small>${r.visits} prior visits · last visit ${r.days_since_visit} days ago</small></td><td>${escape(r.territory)}<small>${escape(r.specialty)} · access ${r.access?'confirmed':'unconfirmed'}</small></td><td><span class="pill">${r.uplift_pp>0?'+':''}${r.uplift_pp} pp</span><small>Current propensity ${r.current_score}</small></td><td>${escape(r.reason)}<small><b>Next action</b> ${escape(r.action)}</small><details><summary>Evidence audit</summary><pre>${escape(JSON.stringify(r.drivers,null,2))}</pre></details></td><td><select data-hcp="${r.hcp_id}"><option>Review</option><option>Accept</option><option>Already planned</option><option>No access</option><option>Wrong territory</option><option>Not relevant</option></select><small>Local demo feedback</small></td></tr>`).join('');for(const s of document.querySelectorAll('[data-hcp]')){try{s.value=localStorage.getItem('hcp-'+s.dataset.hcp)||'Review';s.onchange=()=>localStorage.setItem('hcp-'+s.dataset.hcp,s.value)}catch(e){}}}
for(const id of ['territory','access','search','limit'])el(id).addEventListener('input',render);
el('metrics').textContent=JSON.stringify(metrics,null,2);el('diag').textContent=JSON.stringify(diagnostics,null,2);
el('export').onclick=()=>{const keys=['rank','hcp_id','territory','specialty','uplift_pp','reason','action'];const csv=[keys.join(','),...filtered().map(r=>keys.map(k=>'"'+String(r[k]).replaceAll('"','""')+'"').join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='territory_calls.csv';a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>

```
