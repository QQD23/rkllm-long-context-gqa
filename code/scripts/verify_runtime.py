"""Hash gate for the private ABI adapter. Does not load or execute libraries."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNTIME_LOCK = {'vendor_version': '1.3.0', 'internal_patch_version': '23', 'architecture': 'Linux aarch64 / RK3588', 'runtime_sha256': {'librkllmrt.so': '6a9e4fc5324c68921c3a900340361e107af7599fe34dc8fa7759b2c5ae22a6e6', 'librknnrt.so': '73993ed4b440460825f21611731564503cc1d5a0c123746477da6cd574f34885'}, 'historical_patch_binary_sha256': '96d37e2c515936224d164b305db239235e2663a20c6aae5618ec881d600b711b', 'note': 'Historical patch hash is evidence, not an expected hash for a new compiler build. Runtime pins are mandatory.'}


def verify(directory, lock=None):
    lock = lock or RUNTIME_LOCK
    checked = {}
    for name, expected in lock['runtime_sha256'].items():
        path = Path(directory) / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Unsupported runtime hash: ' + name)
        checked[name] = actual
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.runtime_dir), indent=2))
