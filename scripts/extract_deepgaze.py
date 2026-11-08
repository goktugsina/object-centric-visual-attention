"""Official frozen DeepGaze IIE density cache on the fixed OSIE test split."""
import argparse
import importlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import torch.nn.functional as F

from attention.visual import sha256
from attention.saliency import normalize_density

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'c7db17e2d1d7ea6468ffdee2cfaddf141095dcff'


def official_iie():
    # The current package __init__ eagerly imports MSDB and requires unused CLIP.
    # Load the unchanged IIE submodule through a package namespace instead.
    spec = importlib.util.find_spec('deepgaze_pytorch')
    if spec is None:
        raise ImportError('Install requirements-saliency.txt')
    package = types.ModuleType('deepgaze_pytorch')
    package.__path__ = list(spec.submodule_search_locations)
    sys.modules['deepgaze_pytorch'] = package
    return importlib.import_module('deepgaze_pytorch.deepgaze2e').DeepGazeIIE, Path(package.__path__[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, help='Smoke test only; never marked complete unless all 140 images exist')
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic(); raw = ROOT/'data/raw/deepgaze_cache'
    raw.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(raw))
    center_path = raw/'centerbias_mit1003.npy'
    if not center_path.exists():
        with urlopen('https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy', timeout=120) as response:
            center_path.write_bytes(response.read())
    model_class, source = official_iie()
    model = model_class(pretrained=True).to(args.device).eval().requires_grad_(False)
    config = dict(model='DeepGazeIIE', revision=REVISION, device=args.device, torch=str(torch.__version__),
                  input_size=[1024, 768], output_size=[800, 600], input_rgb_range=[0, 255],
                  center_bias_sha256=sha256(center_path),
                  weights_sha256={p.name: sha256(p) for p in sorted((raw/'checkpoints').iterdir()) if p.is_file()},
                  package_source_sha256={p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob('*.py'))},
                  import_adapter='Namespace-only import avoids unused MSDB/CLIP initialization; IIE source unchanged',
                  density_resampling='bilinear probability interpolation; normalize at original support',
                  split_sha256=sha256(ROOT/'results/primary/splits.json'))
    out = ROOT/'data/processed/deepgaze_osie'; out.mkdir(parents=True, exist_ok=True)
    provenance = out/'config.json'
    if provenance.exists() and json.loads(provenance.read_text()) != config:
        raise ValueError('DeepGaze cache configuration mismatch')
    provenance.write_text(json.dumps(config, indent=2)+'\n')
    names = json.loads((ROOT/'results/primary/splits.json').read_text())['test']
    template = np.load(center_path, allow_pickle=False)
    center = zoom(template, (768/template.shape[0], 1024/template.shape[1]), order=0, mode='nearest')
    center = torch.tensor(center-logsumexp(center), dtype=torch.float32, device=args.device)[None]
    for i, name in enumerate(names[:args.limit] if args.limit else names, 1):
        path = ROOT/'data/raw/stimuli'/name; target = out/(Path(name).stem+'.npz')
        image_hash = sha256(path)
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                if str(data['image_sha256']) != image_hash:
                    raise ValueError('Source image changed')
        else:
            before = time.monotonic()
            with Image.open(path) as photograph:
                original_size = photograph.size
                image = np.array(photograph.convert('RGB').resize((1024, 768), Image.Resampling.LANCZOS))
            tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32, device=args.device)[None]
            with torch.inference_mode():
                logp = model(tensor, center)
                p = logp.exp().reshape(1, 1, 768, 1024)
                p = F.interpolate(p, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            density = normalize_density(p).astype(np.float32)
            np.savez_compressed(target, density=density, image=name, image_sha256=image_hash,
                                elapsed_seconds=time.monotonic()-before)
        if i == 1 or i % 10 == 0:
            print(f'DeepGaze {i}/{len(names)}; elapsed {time.monotonic()-start:.1f}s', flush=True)
    index = {name: sha256(out/(Path(name).stem+'.npz')) for name in names if (out/(Path(name).stem+'.npz')).exists()}
    report = dict(status='complete' if len(index) == len(names) else 'partial', images=len(index),
                  elapsed_seconds=time.monotonic()-start, config=config, predictions_sha256=index)
    result = ROOT/'results/saliency_comparison'; result.mkdir(parents=True, exist_ok=True)
    (result/'deepgaze_provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
