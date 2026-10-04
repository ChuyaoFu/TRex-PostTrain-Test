"""Check converted data against recorded joints and independently composed SE(3) targets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.convert_joint_lerobot_to_trex import read_matrix, joint_poses, robot, schema

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
provenance = json.loads((a.root / 'meta/trex_conversion.json').read_text())
source = Path(provenance['source_root'])
eps = []
for path in sorted((a.root / 'meta/episodes').rglob('*.parquet')):
    eps.extend(pq.read_table(path).to_pylist())
import pyarrow as pa
out = pa.concat_tables([pq.read_table(path) for path in sorted((a.root / 'data').rglob('*.parquet'))])
raw = pa.concat_tables([pq.read_table(path) for path in sorted((source / 'data').rglob('*.parquet'))])
states = read_matrix(out, 'observation.state', 62)
absolute = read_matrix(out, 'action_abs', 62)
force = out['observation.tactile_f6'].combine_chunks().values.values.to_numpy().reshape(-1, 10, 6)
chunks = out['action'].combine_chunks().values.values.to_numpy().reshape(-1, 16, 62)
joints = read_matrix(raw, 'observation.state', 58)[:len(states)]
targets = read_matrix(raw, 'action', 58)[:len(states)]
assert chunks.shape == (len(states), 16, 62)
assert np.isfinite(chunks).all()
np.testing.assert_array_equal(states[:, 9:31], joints[:, schema.LEFT_HAND])
np.testing.assert_array_equal(states[:, 40:62], joints[:, schema.RIGHT_HAND])
np.testing.assert_array_equal(absolute[:, 9:31], targets[:, schema.LEFT_HAND])
np.testing.assert_array_equal(absolute[:, 40:62], targets[:, schema.RIGHT_HAND])
np.testing.assert_array_equal(force.reshape(-1, 60), read_matrix(raw, 'observation.tactile_force', 60)[:len(states)])
model, assemble, _ = robot.build_reduced_bimanual_robot(
    {'torso': np.asarray(provenance['torso']), 'head': np.asarray(provenance['head'])})
max_error = 0
checked = 0
for ep in eps:
    start, end = ep['dataset_from_index'], ep['dataset_to_index']
    for i in sorted(set([start, min(start + 10, end - 1), max(start, end - 2), end - 1])):
        current_left, current_right = joint_poses(joints[i:i + 1], model, assemble)
        for offset, pose in [(0, current_left[0]), (31, current_right[0])]:
            expected_state = np.concatenate([pose[:3, 3], pose[:3, 0], pose[:3, 1]])
            np.testing.assert_allclose(states[i, offset:offset + 9], expected_state, atol=2e-6)
        for k in [0, 1, 15]:
            future = min(i + k, end - 1)
            target_left, target_right = joint_poses(targets[future:future + 1], model, assemble)
            for offset, expected, hand in [(0, target_left[0], schema.LEFT_HAND),
                                            (31, target_right[0], schema.RIGHT_HAND)]:
                s = states[i, offset:offset + 9]
                d = chunks[i, k, offset:offset + 9]
                base_rot = np.column_stack([s[3:6], s[6:9], np.cross(s[3:6], s[6:9])])
                delta_rot = np.column_stack([d[3:6], d[6:9], np.cross(d[3:6], d[6:9])])
                reconstructed_xyz = s[:3] + base_rot @ d[:3]
                reconstructed_rot = base_rot @ delta_rot
                np.testing.assert_allclose(reconstructed_xyz, expected[:3, 3], atol=2e-6)
                np.testing.assert_allclose(reconstructed_rot, expected[:3, :3], atol=2e-6)
                np.testing.assert_allclose(base_rot.T @ base_rot, np.eye(3), atol=2e-6)
                np.testing.assert_array_equal(chunks[i, k, offset + 9:offset + 31], targets[future, hand])
                max_error = max(max_error, float(np.max(np.abs(reconstructed_xyz - expected[:3, 3]))),
                                float(np.max(np.abs(reconstructed_rot - expected[:3, :3]))))
            checked += 1
stats = json.loads((a.root / 'meta/trex_norm_stats.json').read_text())['rlbench']
for key, shape in [('action', (16, 62)), ('state', (62,)), ('tactile_f6', (60,))]:
    for stat in ['q01', 'q99', 'mean', 'std']:
        v = np.asarray(stats[key][stat]); assert v.shape == shape and np.isfinite(v).all()
assert len(stats['tracking_error']['mean']) == 56
assert stats['num_transitions'] == len(states) and stats['num_trajectories'] == len(eps)
print(json.dumps({'verified_frames': len(states), 'episodes': len(eps),
                  'sampled_chunk_steps': checked, 'max_se3_reconstruction_error': max_error,
                  'hand_targets_and_force_preserved': True, 'episode_boundaries': 'PASS',
                  'normalization_shapes': 'PASS'}))
