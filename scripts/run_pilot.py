"""Run the pre-specified CPU pilot and write all numerical paper inputs."""
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import h5py
from attention.data import load_osie, split_images
from attention.model import fit, predict, fit_temperature, temperature_scale
from attention.metrics import scores, calibration, bootstrap_mean_ci, entropy

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261102


def evaluate(scenes, ps):
    rows = [dict(image=s['name'], fixations=int(s['counts'].sum()),
                 **scores(s['counts']/s['counts'].sum(), p)) for s, p in zip(scenes, ps)]
    summary = {k: float(np.mean([r[k] for r in rows]))
               for k in ('nll_bits', 'brier', 'top1', 'top3', 'predictive_entropy', 'empirical_entropy')}
    summary['nll_ci95'] = bootstrap_mean_ci([r['nll_bits'] for r in rows])
    summary['calibration'] = calibration(rows)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop-first', action='store_true')
    parser.add_argument('--overlap', choices=['smallest', 'nearest'], default='smallest')
    args = parser.parse_args()
    start = time.monotonic()
    run_name = 'drop_first' if args.drop_first else 'primary'
    if args.overlap != 'smallest':
        run_name += '_' + args.overlap
    out = ROOT / 'results' / run_name
    out.mkdir(parents=True, exist_ok=True)
    scenes, audit = load_osie(ROOT/'data/raw', drop_first=args.drop_first, overlap_policy=args.overlap)
    splits = split_images([s['name'] for s in scenes], SEED)
    byname = {s['name']: s for s in scenes}
    subsets = {k: [byname[n] for n in v] for k, v in splits.items()}
    train, val, test = [subsets[k] for k in ('train', 'validation', 'test')]
    (out/'splits.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Data audit:', json.dumps({k:v for k,v in audit.items() if isinstance(v,int)}), flush=True)
    predictions, selection, coefficients = {}, {}, {}
    predictions['uniform_regions'] = [np.full(len(s['area']), 1/len(s['area'])) for s in test]
    predictions['area'] = [s['area'] for s in test]
    center_scores = {str(sigma): evaluate(val, [s['center_priors'][str(sigma)] for s in val])[1]['nll_bits']
                     for sigma in (.10, .15, .20, .30, .40)}
    bandwidth = min(center_scores, key=center_scores.get)
    selection['center'] = dict(validation_nll=center_scores, bandwidth=float(bandwidth))
    predictions['center'] = [s['center_priors'][bandwidth] for s in test]
    selected = {}
    for name, nf in (('geometry', 6), ('semantics', 18)):
        candidates = {}
        models = {}
        for l2 in (.0001, .001, .01):
            theta = fit(train, nf, l2)
            models[l2] = theta
            candidates[str(l2)] = evaluate(val, predict(val, theta))[1]['nll_bits']
        l2 = float(min(candidates, key=candidates.get))
        selected[name] = (models[l2], l2)
        selection[name] = dict(l2=l2, validation_nll=candidates)
        coefficients[name] = models[l2].tolist()
        predictions[name] = predict(test, models[l2])
        print(f'Fitted {name}; selected L2={l2}', flush=True)
    theta, l2 = selected['semantics']
    temperature = fit_temperature(val, predict(val, theta))
    predictions['semantics_calibrated'] = predict(test, theta, temperature)
    selection['semantics_temperature'] = temperature
    rng = np.random.default_rng(SEED + 1)
    members, member_val = [], []
    coefficients['ensemble'] = []
    for member in range(10):
        counts = np.bincount(rng.integers(0, len(train), size=len(train)), minlength=len(train))
        model = fit(train, 18, l2, counts)
        coefficients['ensemble'].append(model.tolist())
        members.append(predict(test, model))
        member_val.append(predict(val, model))
        print(f'Bootstrap fit {member+1}/10', flush=True)
    avg = lambda outputs: [np.mean([member[i] for member in outputs], axis=0) for i in range(len(outputs[0]))]
    predictions['ensemble'] = avg(members)
    ens_temperature = fit_temperature(val, avg(member_val))
    predictions['ensemble_calibrated'] = temperature_scale(predictions['ensemble'], ens_temperature)
    selection['ensemble_temperature'] = ens_temperature
    rows_by_model, summaries, csvrows = {}, {}, []
    for name, ps in predictions.items():
        rows, summary = evaluate(test, ps)
        rows_by_model[name], summaries[name] = rows, summary
        csvrows.extend([dict(model=name, **r) for r in rows])
    paired = {}
    for a, b in [('geometry','semantics'), ('semantics','semantics_calibrated'),
                 ('semantics','ensemble'), ('ensemble','ensemble_calibrated'), ('center','semantics')]:
        delta = np.array([x['nll_bits']-y['nll_bits'] for x,y in zip(rows_by_model[a], rows_by_model[b])])
        paired[f'{b}_over_{a}'] = dict(improvement_bits=float(delta.mean()), ci95=bootstrap_mean_ci(delta))
    uncertainty = []
    for i,s in enumerate(test):
        mean_p = predictions['ensemble'][i]
        disagreement = entropy(mean_p) - np.mean([entropy(m[i]) for m in members])
        subj = s['subject_counts']
        subject_ps = subj[subj.sum(axis=1)>0] / subj[subj.sum(axis=1)>0].sum(axis=1, keepdims=True)
        # Equal-observer empirical JSD is descriptive and biased by sparse counts.
        observer_jsd = entropy(subject_ps.mean(axis=0)) - np.mean([entropy(p) for p in subject_ps])
        uncertainty.append(dict(image=s['name'], ensemble_disagreement_bits=float(max(0,disagreement)),
                                empirical_observer_jsd_bits=float(observer_jsd),
                                outcome_entropy_bits=entropy(s['counts']/s['counts'].sum()),
                                regions=len(s['area'])))
    summary = dict(seed=SEED, drop_first=args.drop_first, models=summaries, paired=paired,
                   selection=selection, split_sizes={k:len(v) for k,v in subsets.items()},
                   split_fixations={k:sum(int(s['counts'].sum()) for s in v) for k,v in subsets.items()},
                   mean_ensemble_disagreement_bits=float(np.mean([r['ensemble_disagreement_bits'] for r in uncertainty])),
                   environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, h5py=h5py.__version__),
                   elapsed_seconds=time.monotonic()-start)
    for name, value in [('summary',summary),('coefficients',coefficients),('uncertainty',uncertainty)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    with (out/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0])); writer.writeheader(); writer.writerows(csvrows)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
