"""Publication figures built from measured results, never illustrative scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'paper/figures'; out.mkdir(parents=True,exist_ok=True)
data=json.loads((ROOT/'results/primary/summary.json').read_text())
models=data['models']
labels={'uniform_regions':'Uniform regions','area':'Area prior','center':'Center prior',
        'geometry':'Geometry','semantics':'Semantics','semantics_calibrated':'Semantics + T',
        'ensemble':'Ensemble','ensemble_calibrated':'Ensemble + T'}
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.48,2.35),layout='constrained')
keys=list(labels)
means=np.array([models[k]['nll_bits'] for k in keys])
cis=np.array([models[k]['nll_ci95'] for k in keys])
ax.barh(range(len(keys)),means,color=['#9CA3AF']*3+['#64748B']+['#1F566A']*4)
ax.errorbar(means,range(len(keys)),xerr=np.maximum(0,np.array([means-cis[:,0],cis[:,1]-means])),fmt='none',ecolor='#111827',capsize=3)
ax.set_yticks(range(len(keys)),[labels[k] for k in keys]); ax.invert_yaxis()
ax.set_xlabel('Negative log likelihood (bits per fixation)')
ax.set_xlim(0,max(cis[:,1])*1.06)
fig.savefig(out/'heldout_scores.png',dpi=360); fig.savefig(out/'heldout_scores.pdf'); plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(3.48,3.65),layout='constrained')
for key,marker in [('semantics','o'),('semantics_calibrated','s'),('ensemble_calibrated','^')]:
    bins=models[key]['calibration']['bins']
    axes[0].plot([b['confidence'] for b in bins],[b['observed'] for b in bins],marker=marker,label=labels[key],markersize=4)
axes[0].plot([0,1],[0,1],'--',color='gray',linewidth=1)
axes[0].set(xlabel='(a) Predicted top-region probability',ylabel='Observed fraction',xlim=(0,1),ylim=(0,1))
axes[0].legend(fontsize=7,loc='upper left')
u=json.loads((ROOT/'results/primary/uncertainty.json').read_text())
axes[1].scatter([r['outcome_entropy_bits'] for r in u],[r['ensemble_disagreement_bits'] for r in u],s=15,alpha=.65,color='#1F566A')
axes[1].set(xlabel='(b) Empirical outcome entropy (bits)',ylabel='Bootstrap disagreement (bits)')
fig.savefig(out/'calibration_uncertainty.png',dpi=360); fig.savefig(out/'calibration_uncertainty.pdf'); plt.close(fig)
print('Wrote four figure artifacts')
