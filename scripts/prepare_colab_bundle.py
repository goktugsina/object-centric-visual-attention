"""Package reproducible source and reference scores, without raw data or credentials."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT/'output/colab/object_attention_source.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT/name for name in ('README.md', 'pyproject.toml', 'requirements.txt', 'requirements-vision.txt', 'requirements-saliency.txt')]
    for folder, pattern in [('src', '*.py'), ('scripts', '*.py'), ('tests', '*.py'),
                            ('notebooks', '*.ipynb')]:
        files.extend((ROOT/folder).rglob(pattern))
    for name in ('results/primary/splits.json', 'results/primary/per_image_metrics.csv',
                 'results/image_only/per_image_metrics.csv', 'results/image_only/model.json',
                 'results/image_only/oracle_pixel_reference.json', 'results/primary/coefficients.json'):
        files.append(ROOT/name)
    # Frozen baseline tables and coefficients support comparison and transfer.
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path.is_symlink():
                raise ValueError(f'Refusing symlink: {path}')
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'{destination} ({destination.stat().st_size:,} bytes; {len(set(files))} files)')


if __name__ == '__main__':
    main()
