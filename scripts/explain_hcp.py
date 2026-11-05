"""Grounded uplift LR explanations. No LLM call is made by this script.
Creates deterministic reasons, model-driver audit, and grounded LLM prompts.
Keep beside train_classifiers.py. Only load trusted fitted joblib files.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train_classifiers import FEATURES, load_split


def contributions(model, x, reference):
    pre, lr = model.named_steps['preprocessing'], model.named_steps['model']
    z, zr = np.asarray(pre.transform(x)), np.asarray(pre.transform(reference))
    b = lr.coef_[0]
    delta = (z-zr)*b
    total = delta.sum(axis=1)
    p, pr = model.predict_proba(x)[:,1], model.predict_proba(reference)[:,1]
    # Exact straight-path integrated gradients for a linear logit + sigmoid.
    scale = np.divide(p-pr, total, out=np.full_like(total, pr[0]*(1-pr[0])), where=np.abs(total)>1e-10)
    return delta*scale[:,None], p, float(pr[0]), pre.get_feature_names_out()


def fact(name, row):
    n = lambda col: f'{float(row[col]):g}'
    if name.startswith('territory_'):
        return f'Territory: {row.territory}'
    if name.startswith('hcp_specialty_'):
        return f'Specialty: {row.hcp_specialty}'
    if name.startswith('hcp_stage_'):
        return f'Recorded stage: {row.hcp_stage}'
    labels = {
        'prescriptions_week_before_visit':'Prescriptions in the week before the most recent historical visit',
        'calls_last_90d':'Calls in the last 90 days',
        'visits_last_12m':'Visits in the last 12 months',
        'category_prescriptions_before_90d':'Category prescriptions in the last 90 days',
        'brand_prescriptions_before_90d':'Client-brand prescriptions in the last 90 days',
        'brand_prescriptions_previous_90d':'Client-brand prescriptions in the preceding 90 days',
        'days_since_previous_visit':'Days between the last two historical visits',
        'samples_accepted_previous_visits':'Historical sample acceptances',
        'has_previous_visit':'Has a recorded historical visit (1=yes, 0=no)',
        'has_two_or_more_visits':'Has two or more historical visits (1=yes, 0=no)',
        'days_since_most_recent_visit':'Days since the most recent visit',
        'brand_rx_change_90d':'Brand prescription change between the two 90-day periods'}
    if name == 'brand_share_90d':
        return f'Client-brand share of category prescriptions: {float(row[name])*100:.1f}%'
    return f'{labels.get(name,name)}: {n(name)}'


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=root/'data/raw/hcp_dataset_4000.csv')
    p.add_argument('--data-dir',type=Path,default=root/'data/splits')
    p.add_argument('--models-dir',type=Path,default=root/'results/uplift')
    p.add_argument('--output-dir',type=Path,default=root/'results/reasons')
    p.add_argument('--top-k',type=int,default=400)
    args=p.parse_args()
    if args.top_k<1: raise ValueError('--top-k must be positive')
    raw=pd.read_csv(args.input)
    engineered=pd.read_csv(args.data_dir/'hcp_engineered.csv')
    if not raw.hcp_id.is_unique or not engineered.hcp_id.is_unique:
        raise ValueError('HCP IDs must be unique')
    if set(raw.hcp_id)!=set(engineered.hcp_id): raise ValueError('HCP universes differ')
    engineered=engineered.set_index('hcp_id').loc[raw.hcp_id].reset_index()
    x=engineered[FEATURES]
    # Actual training HCP as the reference, chosen near the population's training
    # median profile; reference selection excludes test rows and outcomes.
    a,_=load_split(args.data_dir,'train');b,_=load_split(args.data_dir,'validation')
    train=pd.concat([a,b],ignore_index=True)
    spread=train.std().replace(0,1)
    nearest=(((train-train.median())/spread)**2).sum(axis=1).idxmin()
    reference=train.loc[[nearest]]
    models=[joblib.load(args.models_dir/file) for file in ['lr_control.joblib','lr_additional_visit.joblib']]
    c0,p0,r0,names0=contributions(models[0],x,reference)
    c1,p1,r1,names1=contributions(models[1],x,reference)
    if list(names0)!=list(names1): raise ValueError('Preprocessing columns differ')
    d=c1-c0; uplift=p1-p0
    if not np.allclose(d.sum(axis=1)+(r1-r0),uplift,atol=1e-8):
        raise ValueError('Attribution reconstruction failed')
    groups={}
    for i,name in enumerate(names0):
        col=name.split('__',1)[1]
        group=next((g for g in ['territory','hcp_specialty','hcp_stage'] if col.startswith(g+'_')),col)
        groups.setdefault(group,[]).append(i)
    audit=[]; output=[]; prompts=[]
    for idx in np.argsort(-uplift,kind='stable')[:min(args.top_k,len(raw))]:
        row=raw.iloc[idx].copy()
        for col in FEATURES: row[col]=x.iloc[idx][col]
        # Do not turn placeholder zeroes into misleading historical facts.
        excluded=set()
        if not row.has_previous_visit: excluded.update(['prescriptions_week_before_visit','days_since_most_recent_visit'])
        if not row.has_two_or_more_visits: excluded.add('days_since_previous_visit')
        if row.category_prescriptions_before_90d==0: excluded.add('brand_share_90d')
        drivers=sorted([(g,float(d[idx,cols].sum())) for g,cols in groups.items()],key=lambda z:abs(z[1]),reverse=True)
        usable=[(g,v) for g,v in drivers if g not in excluded][:3]
        evidence=[]
        for g,v in usable:
            lookup={'territory':'territory_','hcp_specialty':'hcp_specialty_','hcp_stage':'hcp_stage_'}.get(g,g)
            text=fact(lookup,row)
            audit.append(dict(hcp_id=row.hcp_id,feature=g,actual_fact=text,contribution_pp=v*100,
                              direction='raises estimate vs reference' if v>0 else 'lowers estimate vs reference'))
            evidence.append(dict(feature=g,fact=text,contribution_pp=round(v*100,4)))
        action='Review contact suitability and consider a follow-up discussion.'
        if row.hcp_stage=='Testing phase': action='Follow up on the recorded testing phase and ask about outstanding questions.'
        elif row.hcp_stage=='Never accepted a sample': action='Check willingness to discuss the therapy before arranging a visit.'
        reason=f"Estimated additional-visit benefit: {uplift[idx]*100:+.1f} percentage points. "+'; '.join(e['fact'] for e in evidence[:2])+'. '+action
        output.append(dict(rank=len(output)+1,hcp_id=row.hcp_id,territory=row.territory,
            feature_cutoff_date=row.feature_cutoff_date,p_without_visit=p0[idx],p_with_visit=p1[idx],
            uplift=uplift[idx],reason=reason,rep_review=row.rep_review_with_interpretation,
            rep_review_timestamp=row.rep_review_timestamp,reason_method='Grounded deterministic template; no LLM call',
            requires_review=bool(uplift[idx]<=0)))
        prompts.append(dict(hcp_id=row.hcp_id,instruction='Write at most 45 words for a sales rep. Use only the supplied facts and estimated benefit. Mention one or two driver facts and the suggested action. Call the benefit an estimate; do not claim guaranteed prescribing or that an individual feature causes response. Do not invent clinical claims, access, barriers, dates, preferences or quantities. Return JSON with reason and evidence_features. Driver contributions are relative to the supplied reference, not interventions.',
            estimated_uplift_pp=round(uplift[idx]*100,2),evidence=evidence,suggested_action=action))
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(output).to_csv(out/'flagged_hcp_reasons.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'model_driver_audit.csv',index=False)
    (out/'llm_grounded_prompts.jsonl').write_text('\n'.join(json.dumps(r) for r in prompts),encoding='utf-8')
    (out/'reference_and_checks.json').write_text(json.dumps(dict(reference_features=reference.iloc[0].to_dict(),reference_uplift=r1-r0,
        attribution='Exact straight-path integrated gradients, grouped, relative to one training reference HCP',
        all_attributions_reconstruct_uplift=True,rows=len(output),llm_called=False,
        limitations='Synthetic data. Explanation depends on reference and correlated features. Feature attribution is not causal evidence. All-universe scoring includes training rows, not evaluation.'),indent=2))
    print(pd.DataFrame(output)[['hcp_id','reason']].head(3).to_string(index=False))
    print(f'Saved {len(output)} grounded reasons to {out.resolve()}')


if __name__=='__main__':main()
