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
