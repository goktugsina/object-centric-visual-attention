"""Download pinned OSIE stimuli and verify each Git blob; do not redistribute."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
import time
from download_osie import REVISION

ROOT=Path(__file__).resolve().parents[1]


def blob_sha(payload):
    return hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()


def main():
    endpoint=f'https://api.github.com/repos/NUS-VIP/predicting-human-gaze-beyond-pixels/git/trees/{REVISION}?recursive=1'
    with urlopen(Request(endpoint,headers={'User-Agent':'object-attention-research'}),timeout=60) as r:tree=json.load(r)
    if tree.get('truncated'):raise RuntimeError('Incomplete upstream tree')
    entries=[e for e in tree['tree'] if e['type']=='blob' and e['path'].startswith('data/stimuli/') and e['path'].endswith('.jpg')]
    assert len(entries)==700,len(entries)
    print(f'Pinned image tree: {len(entries)} images, {sum(e["size"] for e in entries)/1e6:.1f} MB',flush=True)
    folder=ROOT/'data/raw/stimuli';folder.mkdir(parents=True,exist_ok=True)
    def download(entry):
        name=Path(entry['path']).name;path=folder/name
        payload=path.read_bytes() if path.exists() else b''
        if blob_sha(payload)!=entry['sha']:
            url=f'https://raw.githubusercontent.com/NUS-VIP/predicting-human-gaze-beyond-pixels/{REVISION}/{entry["path"]}'
            for attempt in range(3):
                try:
                    with urlopen(url,timeout=60) as r:payload=r.read()
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(attempt+1)
            if blob_sha(payload)!=entry['sha']:raise ValueError(f'Git blob mismatch: {name}')
            path.write_bytes(payload)
        return {'name':name,'bytes':len(payload),'git_blob':entry['sha'],'sha256':hashlib.sha256(payload).hexdigest()}
    manifest=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,future in enumerate(as_completed([pool.submit(download,e) for e in entries]),1):
            manifest.append(future.result())
            if i%50==0:print(f'Verified {i}/700 images',flush=True)
    out=ROOT/'results/image_only';out.mkdir(parents=True,exist_ok=True)
    (out/'image_manifest.json').write_text(json.dumps({'revision':REVISION,'images':sorted(manifest,key=lambda r:r['name'])},indent=2)+'\n')


if __name__=='__main__':main()
