"""Retrain population models with five observers completely withheld per fold."""
import csv
import json
import time
from pathlib import Path
import numpy as np

from attention.data import load_osie
from attention.image_only import fixation_records, make_scene, pixel_information_gain
from attention.robustness import observer_target
from attention.model import fit, predict
from attention.metrics import scores, bootstrap_mean_ci
from attention.visual import sha256

ROOT = Path(__file__).resolve().parents[1]


def observer_subset(scenes, subjects, columns):
    output = []
    for s in scenes:
        if s['subject_counts'].shape[0] != 15:
            raise ValueError('OSIE subject order contract changed')
        counts = s['subject_counts'][subjects]
        # Counts here are normalized equal-observer targets, deliberately summing to one.
        output.append(dict(s, counts=observer_target(counts), subject_counts=counts,
                           features=s['features'][:, columns].copy()))
    return output


def main():
    started = time.monotonic(); out=ROOT/'results/observer_holdout'; out.mkdir(parents=True,exist_ok=True)
    annotated, _ = load_osie(ROOT/'data/raw')
    fixation = fixation_records(ROOT/'data/raw/fixations.mat')
    folder = ROOT/'data/processed/detections'; detector=json.loads((folder/'config.json').read_text())
    detected=[make_scene(json.loads(p.read_text()), fixation[p.stem+'.jpg'], len(detector['categories']))
              for p in sorted(folder.glob('*.json')) if p.name != 'config.json']
    split=json.loads((ROOT/'results/primary/splits.json').read_text())
    grid=(0.,1e-6,1e-5,1e-4,1e-3,1e-2)
    summary=dict(status='in_progress', weighting='equal observer, then equal image, then equal fold', folds=[],
                 source_sha256={p:sha256(ROOT/p) for p in ['scripts/run_observer_holdout.py','research/observer_holdout_protocol.md',
                 'src/attention/data.py','src/attention/model.py','src/attention/robustness.py','src/attention/image_only.py']},
                 split_sha256=sha256(ROOT/'results/primary/splits.json'))
    rows=[]
    for fold in range(3):
        heldout=list(range(fold*5,(fold+1)*5)); seen=sorted(set(range(15))-set(heldout))
        assert not set(seen)&set(heldout)
        record=dict(fold=fold,train_observers=seen,validation_observers=seen,test_observers=heldout,arms={})
        for arm, scenes in [('annotated',annotated),('detected',detected)]:
            byname={s['name']:s for s in scenes}
            nf=18 if arm=='annotated' else 7+len(detector['categories'])
            base=6 if arm=='annotated' else 7
            result={'models':{},'contrast':{}};test_records={}
            for variant, count in [('geometry',base),('full',nf)]:
                train=observer_subset([byname[n] for n in split['train']],seen,list(range(count)))
                val=observer_subset([byname[n] for n in split['validation']],seen,list(range(count)))
                test=observer_subset([byname[n] for n in split['test']],heldout,list(range(count)))
                candidates={l:fit(train,count,l) for l in grid}
                validation={l:float(np.mean([scores(s['counts'],p)['nll_bits'] for s,p in zip(val,predict(val,t))])) for l,t in candidates.items()}
                selected=min(validation,key=validation.get);theta=candidates[selected]
                rs=[dict(image=s['name'],ig_uniform_bits=pixel_information_gain(s['counts'],p,s['area']),
                         **scores(s['counts'],p)) for s,p in zip(test,predict(test,theta))]
                test_records[variant]=rs
                result['models'][variant]=dict(l2=selected,validation_nll_bits=validation,coefficients=theta.tolist(),
                    test={k:float(np.mean([r[k] for r in rs])) for k in ('nll_bits','brier','ig_uniform_bits')})
                rows.extend(dict(fold=fold,arm=arm,model=variant,**r) for r in rs)
            delta=np.array([b['nll_bits']-a['nll_bits'] for a,b in zip(test_records['full'],test_records['geometry'])])
            result['contrast']=dict(full_over_geometry_bits=float(delta.mean()),image_ci95=bootstrap_mean_ci(delta))
            record['arms'][arm]=result
            print(f'Observer fold {fold} {arm} complete',flush=True)
        summary['folds'].append(record)
    summary['aggregate']={}
    for arm in ('annotated','detected'):
        paired=[]
        for fold in range(3):
            subset={(r['model'],r['image']):r for r in rows if r['fold']==fold and r['arm']==arm}
            paired.append([subset['geometry',n]['nll_bits']-subset['full',n]['nll_bits'] for n in split['test']])
        delta=np.mean(paired,axis=0)
        summary['aggregate'][arm]=dict(full_over_geometry_bits=float(delta.mean()),image_ci95=bootstrap_mean_ci(delta),
            full_ig_bits=float(np.mean([f['arms'][arm]['models']['full']['test']['ig_uniform_bits'] for f in summary['folds']])))
    summary.update(status='complete',elapsed_seconds=time.monotonic()-started)
    with (out/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['aggregate'],indent=2))


if __name__=='__main__':main()
