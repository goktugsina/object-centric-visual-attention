"""Pinned public EMOd subset and metadata; IAPS originals are not included."""
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'd0d1758ac4eef9e7db6ceef78d1bfdf3dfa14d02'
BASE = f'https://raw.githubusercontent.com/Fanshaojing/emotionalattention/{REVISION}'
FILES = {
    'fixations1019_renamed.mat': (BASE+'/fixations1019_renamed.mat', 'b881ac6149b849ce513f0d09bbef38c7c7efa2f1fb7689f3691738f0ddeca18c'),
    'allfindata1019_renamed.mat': (BASE+'/allfindata1019_renamed.mat', 'e38e3333a7866375b7b11c5afcf8f56e5d0df78eb680c0c7e719edbae7ff7bc6'),
    'AllLabelData1019.mat': (BASE+'/AllLabelData1019.mat', '85d5d55c6cd1737d6bb696cb61264b9e7a75dec5977b7b709feccf9d3db25706'),
    'EMOdImages1019.zip': ('https://www-users.cse.umn.edu/~qzhao/resources/emoattention/EMOdImages1019.zip', 'd7c479e32aea7d28d219cf6bd2ce71650c99543f7de380f22c47312956a19e94'),
    'FixationMap.zip': (BASE+'/FixationMap.zip', 'de49d44c6227927a85c4d5722bcd4edea5ea7b938db537cf796ba43c0c90b0b3'),
}


def main():
    root = ROOT/'data/raw/emod'; root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (url, expected) in FILES.items():
        path = root/name
        if not path.exists():
            partial = path.with_suffix(path.suffix+'.partial')
            try:
                with urlopen(url, timeout=180) as response:
                    partial.write_bytes(response.read())
            except URLError:
                # Native curl can use the OS certificate store; TLS is never disabled.
                subprocess.run(['curl', '-fsSL', '--max-time', '180', url, '-o', str(partial)], check=True)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Download checksum mismatch: {name}')
            partial.replace(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        records.append(dict(file=name, url=url, sha256=actual, bytes=path.stat().st_size))
    out = root/'stimuli'; out.mkdir(exist_ok=True)
    with ZipFile(root/'EMOdImages1019.zip') as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n]
        if len(names) != 698 or len({Path(n).name for n in names}) != 698:
            raise ValueError('Unexpected public image archive')
        for name in names:
            (out/Path(name).name).write_bytes(archive.read(name))
        readme = archive.read('EMOdImages1019/ReadMe.txt').decode()
    manifest = dict(repository_revision=REVISION, files=records, available_images=698,
                    unavailable_IAPS_images=321, archive_readme=readme)
    (root/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
