"""Real videos/joints/tactile -> trainer batch; processor is a small test stub.

Does not validate a released Qwen processor or released policy weights.
Run under the train environment (source scripts/env_ppu.sh on PPU).
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import torch
from qwen_vla.lerobot_dataset import TRexLeRobotDataset
from utils.lerobot_common import KEY_ACTION, KEY_HEAD, KEY_TACF6, DEFORM_KEYS

class Batch(dict):
    __getattr__ = dict.__getitem__

class ProcessorStub:
    tokenizer = SimpleNamespace(pad_token_id=0)
    def apply_chat_template(self, *args, **kwargs):
        return 'real-data-loader-smoke'
    def __call__(self, **kwargs):
        assert len(kwargs['images']) == 3
        assert all(image.size == (384, 288) for image in kwargs['images'])
        return Batch(input_ids=torch.tensor([[1, 2, 3, 4]]))
    def image_processor(self, images, **kwargs):
        assert all(image.size == (384, 288) for image in images)
        return Batch(pixel_values=torch.zeros(len(images), 3),
                     image_grid_thw=torch.tensor([[1, 1, 1]] * len(images)))


def load_real_batch(root):
    cfg = SimpleNamespace(lerobot_root=str(root), lerobot_video_backend='pyav',
        image_size=[384, 288], use_flare=1, n_flare_steps=8, flare_frame_stride=1,
        use_tactile_vec=1, use_tactile_deform=1, use_tactile_vqvae=1,
        use_robot_state=1, vqvae_window=16, action_dim=62)
    ds = TRexLeRobotDataset(cfg, ProcessorStub(), SimpleNamespace(print=print))
    ep = pq.read_table(next((Path(root) / 'meta/episodes').rglob('*.parquet'))).to_pylist()[0]
    # Check beginning and end: temporal history/future must clamp within episode.
    first = ds[0]
    last = ds[ep['dataset_to_index'] - 1]
    tail = ds[len(ds) - 1]
    assert tail['episode_index'].item() == ds._base_ds.meta.total_episodes - 1
    assert tail[f'{KEY_HEAD}_is_pad'][1:].all()
    assert tail[KEY_ACTION].shape == (16, 62) and torch.isfinite(tail[KEY_ACTION]).all()
    assert first[KEY_ACTION].shape == last[KEY_ACTION].shape == (16, 62)
    assert first[KEY_TACF6].shape == (16, 10, 6)
    assert first[KEY_HEAD].shape == (9, 3, 360, 640)
    assert first[f'{KEY_TACF6}_is_pad'][:-1].all()
    assert last[f'{KEY_HEAD}_is_pad'][1:].all()
    assert torch.equal(last[KEY_HEAD][0], last[KEY_HEAD][-1])
    for key in DEFORM_KEYS:
        assert first[key].shape == (3, 240, 240)
    old_size = ds.image_size
    ds.image_size = None
    cropped = ds._head_to_pil(first[KEY_HEAD][0])
    assert cropped.size == (400, 300)
    expected = (first[KEY_HEAD][0, :, :300, 140:540].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
    np.testing.assert_array_equal(np.asarray(cropped), expected)
    ds.image_size = old_size
    batch = ds.collate_fn([first])
    for key, shape in [('norm_actions', (1, 16, 62)), ('tactile_f6s', (1, 10, 6)),
                       ('tactile_deforms', (1, 10, 1, 240, 240)),
                       ('tactile_f6_history', (1, 16, 10, 6))]:
        value = batch[key]
        assert tuple(value.shape) == shape and torch.isfinite(value).all()
    if ds._base_ds.meta.total_episodes >= 2:
        full_length = len(ds)
        validation = ds.create_val_split(val_ratio=0.5, seed=42)
        assert len(ds) + len(validation) == full_length
        assert np.intersect1d(ds.ds.indices, validation.ds.indices).size == 0
        for subset in (ds, validation):
            item = subset[0]
            assert item['episode_index'].item() in subset.ds.episodes
            assert item[f'{KEY_TACF6}_is_pad'][:-1].all()
        print(json.dumps({'episode_train_val_split': 'PASS', 'train_frames': len(ds),
                          'val_frames': len(validation), 'frame_overlap': 0}))
    print(json.dumps({'real_video_decode_and_collate': 'PASS', 'frames': len(ds._base_ds),
                      'head_crop': list(cropped.size), 'history_and_future_padding': 'PASS',
                      'batch_shapes': {key: list(value.shape) for key, value in batch.items()
                                       if isinstance(value, torch.Tensor)}}))
    return batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    load_real_batch(args.root)
