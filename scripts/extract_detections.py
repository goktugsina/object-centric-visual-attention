"""Frozen image-only detections; no gaze or human annotation is loaded."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'data/raw/model_cache'))
    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights,progress=False).eval()
    transform=weights.transforms()
    out=ROOT/'data/processed/detections';out.mkdir(parents=True,exist_ok=True)
    config={'model':'fasterrcnn_mobilenet_v3_large_320_fpn','weights':'COCO_V1','url':weights.url,
            'threshold':.5,'max_detections':50,'device':'cpu','threads':4,
            'torch':torch.__version__,'torchvision':torchvision.__version__,'categories':weights.meta['categories']}
    checkpoint=ROOT/'data/raw/model_cache/checkpoints'/weights.url.rsplit('/',1)[1]
    config['weights_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_path=out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Detection cache configuration mismatch')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    images=sorted((ROOT/'data/raw/stimuli').glob('*.jpg'))
    if args.limit:images=images[:args.limit]
    started=time.monotonic();computed=0
    with torch.inference_mode():
        for i,path in enumerate(images,1):
            target=out/(path.stem+'.json')
            if not target.exists():
                image=Image.open(path).convert('RGB');before=time.monotonic()
                pred=model([transform(image)])[0]
                keep=torch.nonzero(pred['scores']>=.5).flatten()[:50]
                result={'image':path.name,'width':image.width,'height':image.height,
                        'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'boxes':pred['boxes'][keep].tolist(),'scores':pred['scores'][keep].tolist(),
                        'classes':pred['labels'][keep].tolist(),'elapsed_seconds':time.monotonic()-before}
                target.write_text(json.dumps(result)+'\n');computed+=1
            if i==1 or i%50==0:print(f'{i}/{len(images)} processed; {computed} new; elapsed {time.monotonic()-started:.1f}s',flush=True)
    print('Detections complete',flush=True)


if __name__=='__main__':main()
