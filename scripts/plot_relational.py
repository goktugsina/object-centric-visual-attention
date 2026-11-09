"""Plot paired relational-model contrasts from saved numerical results."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'results/relational')
    args = parser.parse_args()
    s = json.loads((args.results/'summary.json').read_text())
    if s['status'] != 'complete':
        raise ValueError('Experiment is incomplete')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    contrasts = [('visual_geometry_over_visual', 'Add geometry'),
                 ('full_self_over_visual_geometry', 'Add attributes / confidence+class'),
                 ('full_gat_over_full_self', 'GAT vs self-only'),
                 ('full_gat_over_full_uniform', 'GAT vs uniform mixing')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained', sharey=True)
    for ax, (arm, a) in zip(axes, s['arms'].items()):
        for i, (key, label) in enumerate(contrasts):
            c = a['contrasts'][key]; mean = c['improvement_bits']; lo, hi = c['image_bootstrap_ci95']
            ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', color='#28648b', capsize=4)
        ax.axvline(0, linestyle='--', linewidth=1, color='#666666')
        ax.set_yticks(range(len(contrasts)), [label for _, label in contrasts])
        ax.set_title(arm.capitalize()+' regions')
        ax.set_xlabel('Improvement (bits / fixation; positive is better)')
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Frozen ResNet50: paired region-choice contrasts\nSeed-average differences; image-bootstrap 95% intervals')
    folder = args.results/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'contrasts.png', dpi=160)
    fig.savefig(folder/'contrasts.pdf')
    plt.close(fig)
    print(args.results/'figures')

if __name__ == '__main__':
    main()
