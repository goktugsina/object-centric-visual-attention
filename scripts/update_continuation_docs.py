"""Refresh current status only after completed experiment and verification records."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    emod = json.loads((ROOT/'results/emod_transfer/summary.json').read_text())
    verification = json.loads((ROOT/'results/continuation_verification.json').read_text())
    assert emod['status'] == verification['status'] == 'complete'
    readme = ROOT/'README.md'; text = readme.read_text(encoding='utf-8')
    text = text.replace('This is an exploratory OSIE case study and candidate empirical preprint, with an executable baseline.',
                        'This is an exploratory OSIE and public EMOd study with executable baselines.')
    text = text.replace('Sequential prediction, stronger saliency comparisons and a second dataset are optional follow-ups for broader claims;',
                        'Sequential prediction and personalization remain future directions;')
    text = text.replace('Author review and submission preparation remain outstanding.', 'Author review and external submission remain outstanding.')
    text = text.replace('These exploratory follow-up results are separate from the existing six-page paper.',
                        'These exploratory follow-up results are incorporated into the revised paper alongside the 5 November saliency, transfer and observer experiments.')
    text = text.replace('The six-page manuscript follows', 'The revised manuscript follows').replace('20 references in citation order', '24 references in citation order')
    readme.write_text(text, encoding='utf-8')
    status = ROOT/'research/shared_plan_status.md'; text = status.read_text(encoding='utf-8')
    text = text.replace('EMOd aktarımı 515 görüntülük önbellekten sürdürülüyor; son sonuç tablosu ve yeniden hesaplama denetimi henüz bitmedi. Makale şablonu genişletildi; Word/PDF henüz yeni sonuçlarla üretilmedi.',
                        'EMOd aktarımı 698 görüntünün tamamında bitti: 6.282 görüntü-model kaydı. Tüm aktarım önbelleklerinin hash denetimi, gözlemci ayrımı kontrolleri ve üç görüntüde tahminlerin yeniden hesaplanması geçti. Sonuçlar `saliency_transfer_results.md` içinde; makalenin yeni sayısal alanları bu sonuçlardan üretilir.')
    status.write_text(text, encoding='utf-8')
    path = ROOT/'research/next_experiments.md'; text = path.read_text(encoding='utf-8')
    text = text.replace('with zero-shot transfer in progress.', 'with zero-shot transfer completed without target fitting.')
    path.write_text(text, encoding='utf-8')
    path = ROOT/'research/relational_results.md'; text = path.read_text(encoding='utf-8')
    marker = '# Frozen visual features and object relations: measured results\n'
    note = '\nUpdate, 5 November 2026: the historical next-stage list below is superseded by the [saliency and transfer report](saliency_transfer_results.md). Strong static saliency comparison, public EMOd transfer and observer exclusion are now complete and incorporated into the revised manuscript. ResNet+U-Net, fine-tuning and personalization remain future work.\n'
    if note not in text: text = text.replace(marker, marker+note)
    path.write_text(text, encoding='utf-8')
    path = ROOT/'notebooks/object_attention_relations.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    note = '## Saliency and transfer continuation\nThe README now includes the frozen DeepGaze IIE comparison, public EMOd audit/transfer and excluded-observer workflow. Those scripts expect checkpoints under `output/relational_checkpoints/relational/`, while this notebook names its run `relational_colab`. Reproduce a run named `relational` in a fresh workspace before using the default continuation commands. The source bundle includes baseline coefficients but not learned neural weights. No Colab execution of the continuation is claimed.\n'
    if not any('## Saliency and transfer continuation' in ''.join(c['source']) for c in notebook['cells']):
        notebook['cells'].append(dict(cell_type='markdown', metadata={}, source=note.splitlines(keepends=True)))
    path.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
    print('Current status and notebook documentation refreshed')


if __name__ == '__main__': main()
