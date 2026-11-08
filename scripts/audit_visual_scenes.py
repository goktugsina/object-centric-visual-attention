"""Ten deterministic training-image audits, without selecting appealing outcomes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from attention.data import load_osie
from attention.image_only import fixation_records
from attention.visual import human_regions, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'output/visual_audit'
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT/'results/primary/splits.json').read_text())
    names = sorted(np.random.default_rng(20261103).choice(split['train'], 10, replace=False).tolist())
    scenes, _ = load_osie(ROOT/'data/raw')
    scenes = {s['name']: s for s in scenes}
    fixations = fixation_records(ROOT/'data/raw/fixations.mat')
    records = []
    for name, masks, labels, active in human_regions(ROOT/'data/raw/attrs.mat', set(names)):
        scene = scenes[name]
        assert np.array_equal(active, scene['active_ids'])
        subjects = fixations[name]
        counts = np.sum([np.bincount(labels[y, x], minlength=int(labels.max())+1)[active]
                         for x, y in subjects], axis=0)
        np.testing.assert_array_equal(counts, scene['counts'])
        with Image.open(ROOT/'data/raw/stimuli'/name) as image:
            photograph = np.asarray(image.convert('RGB'))
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
        fig.suptitle(f'{name} | training split | {len(active)-int(0 in active)} active objects | {int(counts.sum())} fixations')
        for ax in axes.flat[:5]:
            ax.imshow(photograph)
            ax.set_xlim(-.5, labels.shape[1]-.5)
            ax.set_ylim(labels.shape[0]-.5, -.5)
            ax.set_xticks([]); ax.set_yticks([])
        for x, y in subjects:
            axes[0, 0].scatter(x, y, s=12, facecolors='none', edgecolors='#ff2a40', linewidths=.6)
        axes[0, 0].set_title('All retained fixations (>=100 ms)')
        for mask in masks:
            if mask.any() and not mask.all():
                axes[0, 1].contour(mask.astype(float), levels=[.5], colors=['#00ffff'], linewidths=.5)
        axes[0, 1].set_title('Source object-mask boundaries')
        axes[0, 2].imshow(np.ma.masked_where(labels == 0, labels), cmap='turbo', alpha=.55,
                          vmin=0, vmax=max(1, int(labels.max())))
        axes[0, 2].set_title('Disjoint regions; background unpainted')
        for ax, sid in zip(axes[1, :2], (0, 1)):
            x, y = subjects[sid]
            ax.scatter(x, y, c=np.arange(len(x)), cmap='plasma', s=45, edgecolors='white')
            for j, (px, py) in enumerate(zip(x, y), 1):
                ax.text(px+5, py, str(j), color='white', fontsize=7,
                        bbox=dict(facecolor='black', alpha=.5, pad=.2))
            ax.set_title(f'Observer {sid+1}; eligible fixation order')
        x = np.arange(len(active))
        axes[1, 2].bar(x, counts/counts.sum(), label='All events', color='#3569a8')
        for sid, color in [(0, '#de7d22'), (1, '#398a53')]:
            c = scene['subject_counts'][sid]
            if c.sum():
                axes[1, 2].plot(x, c/c.sum(), '.-', color=color, label=f'Observer {sid+1}')
        axes[1, 2].set_xticks(x, ['BG' if i == 0 else str(i) for i in active], rotation=90)
        axes[1, 2].set_ylabel('Fixation share'); axes[1, 2].set_title('Target distributions')
        axes[1, 2].legend(fontsize=8)
        filename = Path(name).stem+'.png'
        fig.savefig(out/filename, dpi=120); plt.close(fig)
        records.append(dict(image=name, split='train', figure=filename, objects=int((active != 0).sum()),
                            fixations=int(counts.sum()), background_fraction=float(counts[active == 0].sum()/counts.sum()),
                            coordinates_match_loader=True, figure_sha256=sha256(out/filename)))
    thumbnails = []
    for r in sorted(records, key=lambda r: r['image']):
        with Image.open(out/r['figure']) as image:
            thumbnails.append(image.resize((750, 450)))
    contact = Image.new('RGB', (1500, 2250), 'white')
    for i, thumbnail in enumerate(thumbnails):
        contact.paste(thumbnail, ((i % 2)*750, (i//2)*450))
    contact.save(out/'contact_sheet.jpg')
    report = dict(selection_seed=20261103, selection='10 training images without replacement',
                  source_split_sha256=sha256(ROOT/'results/primary/splits.json'), scenes=records)
    (ROOT/'results/visual_audit').mkdir(parents=True, exist_ok=True)
    (ROOT/'results/visual_audit/manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Audited {len(records)} training scenes: {out}', flush=True)


if __name__ == '__main__':
    main()
