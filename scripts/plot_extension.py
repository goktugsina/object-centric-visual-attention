"""Compute the representation contrast and plot extension results."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from attention.metrics import bootstrap_mean_ci

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results/figures').mkdir(parents=True, exist_ok=True)
read=lambda name:json.loads((ROOT/'results'/name).read_text())
rob=read('robustness/summary.json');vision=read('image_only/summary.json');oracle=read('image_only/oracle_pixel_reference.json')
assert rob['status']==vision['status']=='complete'
with (ROOT/'results/image_only/per_image_metrics.csv').open() as f:
    rows=[r for r in csv.DictReader(f) if r['seed']=='20261102' and r['model']=='detected_class']
auto={r['image']:float(r['ig_uniform_bits']) for r in rows}
human={r['image']:r['ig_uniform_bits'] for r in oracle['models']['semantics']['per_image']}
assert set(auto)==set(human)
gap=np.array([human[n]-auto[n] for n in sorted(auto)])
gap_result={'human_semantics_minus_detected_class_bits':float(gap.mean()),'paired_ci95':bootstrap_mean_ci(gap),
            'interpretation':'Joint representation difference; mask geometry, classes and semantic attributes differ.'}
(ROOT/'results/image_only/representation_gap.json').write_text(json.dumps(gap_result,indent=2)+'\n')
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(3.48,3.2),layout='constrained')
for ax,runs,keys in [(axes[0],rob['runs'],[('semantics','All attributes'),('without_face_text','No face/text'),('permuted_semantics','Permuted')]),
                    (axes[1],vision['runs'],[('class','Detected classes')])]:
    for offset,(key,label) in enumerate(keys):
        ds=[r['contrasts']['class_over_geometry_confidence' if key=='class' else key+'_over_geometry_event'] for r in runs]
        means=np.array([d['improvement_bits'] for d in ds]);cis=np.array([d['ci95'] for d in ds])
        ax.errorbar(np.arange(1,6)+(offset-(len(keys)-1)/2)*.12,means,yerr=np.array([means-cis[:,0],cis[:,1]-means]),fmt='o',markersize=3,capsize=2,label=label)
    ax.axhline(0,color='gray',ls='--',lw=.7);ax.set_xticks(range(1,6));ax.legend(fontsize=7)
    ax.set_ylabel('Log-score gain (bits)')
axes[0].set_xlabel('(a) Annotated-object split');axes[1].set_xlabel('(b) Detected-object split')
axes[0].set_ylim(-.025,.25);axes[0].legend(fontsize=7,loc='upper center',ncol=3)
fig.savefig(ROOT/'results/figures/extension_splits.png',dpi=360);fig.savefig(ROOT/'results/figures/extension_splits.pdf');plt.close(fig)
print('Saved representation contrast and figures under results/')
