"""Rescore human-annotation reference models on the shared pixel support."""
import json
from pathlib import Path
import numpy as np
from attention.data import load_osie
from attention.model import predict
from attention.image_only import pixel_information_gain
from attention.metrics import bootstrap_mean_ci

ROOT=Path(__file__).resolve().parents[1]
scenes,_=load_osie(ROOT/'data/raw')
primary=json.loads((ROOT/'results/primary/coefficients.json').read_text())
split=json.loads((ROOT/'results/primary/splits.json').read_text())
byname={s['name']:s for s in scenes};test=[byname[n] for n in split['test']]
output={'split_seed':20261102,'models':{}}
for name in ('geometry','semantics'):
    ps=predict(test,np.asarray(primary[name]))
    rows=[{'image':s['name'],'ig_uniform_bits':pixel_information_gain(s['counts']/s['counts'].sum(),p,s['area'])} for s,p in zip(test,ps)]
    values=[r['ig_uniform_bits'] for r in rows]
    output['models'][name]={'ig_uniform_bits':float(np.mean(values)),'ci95':bootstrap_mean_ci(values),'per_image':rows}
path=ROOT/'results/image_only/oracle_pixel_reference.json';path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v['ig_uniform_bits'] for k,v in output['models'].items()}),flush=True)
