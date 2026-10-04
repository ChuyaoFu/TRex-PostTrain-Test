"""Check JSON/PNG completion and paired release-processor tensors against video."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('trex_json_verification_train', ROOT / 'scripts/train.py')
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)
from qwen_vla.lerobot_dataset import TRexLeRobotDataset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lerobot_root', type=Path, required=True)
    p.add_argument('--image_root', type=Path, required=True)
    p.add_argument('--processor_path', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    root = args.image_root.resolve()
    assert not (root / 'PREPROCESSING_IN_PROGRESS').exists(), 'Image export is incomplete.'
    manifest = json.loads((root / 'preprocessing_manifest.json').read_text())
    info = json.loads((args.lerobot_root / 'meta/info.json').read_text())
    assert manifest['frames'] == info['total_frames']
    assert manifest['episodes'] == info['total_episodes']
    assert manifest['png_count'] == info['total_frames'] * 13
    assert (root / 'task_statistics.json').read_bytes() == (args.lerobot_root / 'meta/trex_norm_stats.json').read_bytes()
    cfg = SimpleNamespace(data_path=str(root / 'task.json'), lerobot_root=str(args.lerobot_root),
        lerobot_video_backend='pyav', image_size=[384,288], action_dim=62, action_chunk=16,
        use_flare=1, n_flare_steps=8, flare_frame_stride=4, use_tactile_vec=1,
        use_tactile_deform=1, use_tactile_vqvae=1, use_tactile_code=1,
        use_robot_state=0, vqvae_window=16, val_split_by_episode=1)
    processor = AutoProcessor.from_pretrained(str(args.processor_path))
    logger = SimpleNamespace(print=print)
    video = TRexLeRobotDataset(cfg, processor, logger)
    images = train.SftDataset(cfg, processor, logger)
    assert len(video) == len(images) == manifest['frames']
    episodes = [r for f in sorted((args.lerobot_root / 'meta/episodes').rglob('*.parquet')) for r in pq.read_table(f).to_pylist()]
    episodes.sort(key=lambda r:r['episode_index'])
    indices = {0,1,len(video)-1}
    for ep in (episodes[0], episodes[min(1,len(episodes)-1)], episodes[-1]):
        start = int(ep['dataset_from_index']); length = int(ep['length'])
        indices.update([start, start + length//2, start + length-1])
    indices = sorted(indices)
    def seed(): random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    seed(); a = video.collate_fn([video[i] for i in indices])
    seed(); b = images.collate_fn([images[i] for i in indices])
    tensors = {}
    assert a.keys() == b.keys()
    for key,x in a.items():
        y = b[key]
        if torch.is_tensor(x):
            assert torch.equal(x,y), f'Paired tensor mismatch: {key}'
            tensors[key] = {'shape':list(x.shape), 'exact':True}
        else: assert x == y, key
    # Both paths must retain the established episode split, not silently switch to frame split.
    video_val = video.create_val_split(0.05,42)
    image_val = images.create_val_split(0.05,42)
    assert len(video) == len(images) and len(video_val) == len(image_val)
    report = {'status':'PASS', 'indices':indices, 'tensors':tensors,
              'train_frames':len(images), 'val_frames':len(image_val),
              'val_split':'episode', 'identity':manifest['identity']}
    (root / 'paired_batch_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS', 'paired_tensors':len(tensors), 'train_frames':len(images), 'val_frames':len(image_val)}),flush=True)

if __name__ == '__main__': main()
