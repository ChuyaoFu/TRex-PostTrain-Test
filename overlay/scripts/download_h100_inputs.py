"""Pinned public snapshots; HF local-dir downloads are resumable without cache copies."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_REPO = 'miniFranka/trex_gateway_tong_transfer_sf_norawtac_20260820'
DATA_REV = '9e20b13d042c708e1546138adda25c13ee6aa4e7'
WEIGHTS = [
    ('miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6',
     '62efb3bcb45a3df0e088c8909d759b582cfb98af', 'T-Rex_midtrain_epoch6',
     ['model.pt', 'config.json', 'training_args.json', 'stats_data.json', 'processor/*'],
     'model.pt', '5c86368a2c8ee0e3d06638d5b64b8e30494e836b5776eecd8f93b6c94a84582b'),
    ('Qwen/Qwen3-VL-2B-Instruct', '89644892e4d85e24eaac8bacfd4f463576704203',
     'Qwen3-VL-2B-Instruct', ['*.json', '*.safetensors', '*.txt'],
     'model.safetensors', '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'),
]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights-root', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--offline', action='store_true')
    a = p.parse_args()
    manifest = []
    for repo, rev, name, patterns, filename, expected in WEIGHTS:
        target = a.weights_root / name
        if not a.offline:
            snapshot_download(repo, revision=rev, local_dir=target,
                              allow_patterns=patterns, max_workers=4)
        actual = digest(target / filename)
        if actual != expected:
            raise RuntimeError(f'Weight checksum mismatch: {target / filename}')
        required = (['config.json', 'training_args.json', 'stats_data.json',
                     'processor/tokenizer_config.json'] if name == 'T-Rex_midtrain_epoch6'
                    else ['config.json', 'tokenizer_config.json'])
        for relative in required:
            if not (target / relative).is_file():
                raise RuntimeError(f'Missing model input: {target / relative}')
        manifest.append(dict(repo=repo, revision=rev, path=str(target.resolve()), sha256=actual))
    if digest(a.raw_root / 'meta/info.json') != 'e028f38c55350b9bdfd2b313b0f98863b269317fe6dad48b06c0202bd1b22bb9':
        raise RuntimeError('Raw dataset metadata does not match the pinned HF snapshot.')
    info = json.loads((a.raw_root / 'meta/info.json').read_text())
    if (info['total_episodes'], info['total_frames']) != (200, 208581):
        raise RuntimeError('Unexpected dataset snapshot; expected 200 episodes / 208581 frames.')
    for key in ['observation.state', 'action']:
        if info['features'][key]['shape'] != [58]:
            raise RuntimeError(f'Unexpected raw feature shape: {key}')
    manifest.append(dict(repo=DATA_REPO, revision=DATA_REV, path=str(a.raw_root.resolve()),
                         episodes=200, frames=208581))
    a.weights_root.mkdir(parents=True, exist_ok=True)
    (a.weights_root / 'h100_inputs_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
