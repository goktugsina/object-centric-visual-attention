"""Fit image-only object attention and score shared fixation pixels."""
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from attention.data import split_images
from attention.model import fit,predict
from attention.metrics import bootstrap_mean_ci
from attention.robustness import observer_target
from attention.image_only import make_scene,fixation_records,pixel_information_gain,center_information_gain

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/image_only'
GRID=(0.,1e-6,1e-5,1e-4,1e-3,1e-2)
SEEDS=list(range(20261102,20261107))


def evaluate(scenes,ps):
    output=[]
    for s,p in zip(scenes,ps):
        ig=pixel_information_gain(s['counts']/s['counts'].sum(),p,s['area'])
        output.append({'image':s['name'],'ig_uniform_bits':ig,'pixel_nll_bits':float(np.log2(s['pixels'])-ig),
                       'ig_equal_observer_bits':pixel_information_gain(observer_target(s['subject_counts']),p,s['area'])})
    return output


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'data/processed/detections';config=json.loads((folder/'config.json').read_text())
    class_count=len(config['categories'])
    detections=[json.loads(p.read_text()) for p in sorted(folder.glob('*.json')) if p.name!='config.json']
    assert len(detections)==700,len(detections)
    fixation=fixation_records(ROOT/'data/raw/fixations.mat')
    scenes=[make_scene(d,fixation[d['image']],class_count) for d in detections]
    assert len(scenes)==700 and sum(s['counts'].sum() for s in scenes)==87662
    byname={s['name']:s for s in scenes};detbyname={d['image']:d for d in detections}
    variants={'geometry':6,'geometry_confidence':7,'detected_class':7+class_count}
    center={sigma:{s['name']:center_information_gain(fixation[s['name']],800,600,sigma) for s in scenes}
            for sigma in (.1,.15,.2,.3,.4)}
    summary={'status':'in_progress','detector':config,'runs':[],
             'environment': {'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__},
             'metric':'Image-average log2(pixel density / uniform-pixel density). Larger is better.',
             'audit':{'images':len(scenes),'fixations':int(sum(s['counts'].sum() for s in scenes)),
                      'zero_detection_images':sum(s['active_detections']==0 for s in scenes),
                      'mean_foreground_area':float(np.mean([s['foreground_area'] for s in scenes])),
                      'mean_foreground_fixation_fraction':float(np.mean([s['foreground_fixation_fraction'] for s in scenes]))}}
    allrows=[]
    for seed in SEEDS:
        split=split_images(byname,seed);train,val,test=[[byname[n] for n in split[k]] for k in ('train','validation','test')]
        sigma=max(center,key=lambda sig:np.mean([center[sig][s['name']] for s in val]))
        center_test=np.array([center[sigma][s['name']] for s in test])
        run={'seed':seed,'splits':split,'center_bandwidth':sigma,'center_ig_bits':float(center_test.mean()),'models':{},'contrasts':{}}
        records={}
        for name,nf in variants.items():
            models={l2:fit(train,nf,l2) for l2 in GRID}
            selection={l2:float(np.mean([r['ig_uniform_bits'] for r in evaluate(val,predict(val,t))])) for l2,t in models.items()}
            l2=max(selection,key=selection.get);theta=models[l2]
            rows=evaluate(test,predict(test,theta));records[name]=rows
            ig=np.array([r['ig_uniform_bits'] for r in rows]);gain=ig-center_test
            run['models'][name]={'features':nf,'l2':l2,'validation_ig_bits':selection,'coefficients':theta.tolist(),
                                 'ig_uniform_bits':float(ig.mean()),'ig_uniform_ci95':bootstrap_mean_ci(ig),
                                 'ig_over_center_bits':float(gain.mean()),'ig_over_center_ci95':bootstrap_mean_ci(gain),
                                 'ig_equal_observer_bits':float(np.mean([r['ig_equal_observer_bits'] for r in rows]))}
            allrows.extend(dict(seed=seed,model=name,center_ig_bits=float(center_test[i]),**r) for i,r in enumerate(rows))
            print(f'{seed} {name}: IG={ig.mean():.6f}; over center={gain.mean():.6f}',flush=True)
            if seed==SEEDS[0] and name=='detected_class':original_theta,original_test=theta,test
        delta=np.array([a['ig_uniform_bits']-b['ig_uniform_bits'] for a,b in zip(records['detected_class'],records['geometry_confidence'])])
        run['contrasts']['class_over_geometry_confidence']={'improvement_bits':float(delta.mean()),'ci95':bootstrap_mean_ci(delta)}
        summary['runs'].append(run);(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    perturbations=[]
    for label,scale,drop in [('original',1.,0.),('shrink_20pct',.8,0.),('expand_20pct',1.2,0.),('drop_25pct',1.,.25),('drop_50pct',1.,.5),('drop_75pct',1.,.75),('drop_all',1.,1.)]:
        rng=np.random.default_rng(20261301);altered=[]
        for s in original_test:
            det=detbyname[s['name']];keep=rng.random(len(det['boxes']))>=drop
            altered.append(make_scene(det,fixation[s['name']],class_count,scale,keep))
        values=np.array([r['ig_uniform_bits'] for r in evaluate(altered,predict(altered,original_theta))])
        perturbations.append({'condition':label,'scale':scale,'drop_probability':drop,'ig_uniform_bits':float(values.mean()),
                              'ci95':bootstrap_mean_ci(values),'mean_foreground_fixation_fraction':float(np.mean([s['foreground_fixation_fraction'] for s in altered]))})
        print(f'Stress {label}: IG={values.mean():.6f}',flush=True)
    summary.update(status='complete',perturbations=perturbations,elapsed_seconds=time.monotonic()-start)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'model.json').write_text(json.dumps({'detector':config,'coefficients':original_theta.tolist(),'split_seed':SEEDS[0],
                                            'region_density':'uniform_within_predicted_region','features':'geometry6,confidence,COCO_one_hot91'},indent=2)+'\n')
    with (OUT/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    (OUT/'region_audit.json').write_text(json.dumps([{k:s[k] for k in ('name','foreground_area','foreground_fixation_fraction','active_detections')} for s in scenes],indent=2)+'\n')
    print('Completed image-only evaluation',flush=True)


if __name__=='__main__':main()
