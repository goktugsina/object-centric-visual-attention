"""Download only the two public annotations needed for the pilot, verifying hashes."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "4a467890fcb40a2fec04cdded552760942bac7a4"
BASE = f"https://raw.githubusercontent.com/NUS-VIP/predicting-human-gaze-beyond-pixels/{REVISION}/data"
FILES = {
    "attrs.mat": ("attrs.mat", "51e9747fc6364102080cd5bac8921a279fc3f9e2a022119ae691c4bee4e8b74e"),
    "fixations.mat": ("eye/fixations.mat", "2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9"),
}


def main():
    root = Path(__file__).resolve().parents[1] / "data/raw"
    root.mkdir(parents=True, exist_ok=True)
    for name, (remote, expected) in FILES.items():
        path = root / name
        if not path.exists():
            with urlopen(f"{BASE}/{remote}", timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Unexpected upstream checksum: {name}")
            path.write_bytes(payload)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}; expected {expected}, got {actual}")
        print(f"Verified {name}: {actual}")


if __name__ == "__main__":
    main()
