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
