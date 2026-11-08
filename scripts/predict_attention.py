"""Predict a normalized attention map from an image without human annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.special import softmax
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from attention.image_only import region_features,probability_map

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('--model',type=Path,default=ROOT/'results/image_only/model.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/predictions')
    args=parser.parse_args()
    trained=json.loads(args.model.read_text());config=trained['detector']
    torch.set_num_threads(4);torch.hub.set_dir(str(ROOT/'data/raw/model_cache'))
    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    detector=fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights,progress=False).eval()
    checkpoint=ROOT/'data/raw/model_cache/checkpoints'/weights.url.rsplit('/',1)[1]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=config['weights_sha256']:
        raise ValueError('Detector weights differ from the trained attention model')
    image=Image.open(args.image).convert('RGB')
    with torch.inference_mode():result=detector([weights.transforms()(image)])[0]
    keep=torch.nonzero(result['scores']>=config['threshold']).flatten()[:config['max_detections']]
    detection={'image':args.image.name,'width':image.width,'height':image.height,
               'boxes':result['boxes'][keep].tolist(),'scores':result['scores'][keep].tolist(),
               'classes':result['labels'][keep].tolist()}
    labels,active,area,x=region_features(detection,len(config['categories']))
    p=softmax(x@np.asarray(trained['coefficients']))
    density=probability_map(labels,active,p)
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/(args.image.stem+'_probability.npy'),density)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].imshow(image);axes[0].set_title('Input image')
    shown=axes[1].imshow(density,cmap='magma');axes[1].set_title('Predicted probability per pixel')
    for ax in axes:ax.axis('off')
    fig.colorbar(shown,ax=axes[1],fraction=.04,pad=.02)
    fig.savefig(args.output/(args.image.stem+'_attention.png'),dpi=150);plt.close(fig)
    summary={'image':str(args.image),'probability_sum':float(density.sum()),'regions':[]}
    for index,mass,fraction in zip(active,p,area):
        summary['regions'].append({'source_detection':int(index)-1,'class':'background' if index==0 else config['categories'][detection['classes'][index-1]],
                                   'probability':float(mass),'area_fraction':float(fraction)})
    (args.output/(args.image.stem+'_attention.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'probability_sum':float(density.sum()),'regions':len(active)}))


if __name__=='__main__':main()
