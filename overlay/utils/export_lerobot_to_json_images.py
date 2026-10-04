"""Export canonical T-Rex EEF62 LeRobot videos to the official JSON/PNG contract.

Sequential decode once per video, atomic files, resumable per-video completion,
and an exclusive output lock. No robot math, normalization, or VQ code changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

RGB = ['observation.images.head', 'observation.images.wrist_right', 'observation.images.wrist_left']
DEFORM = [f'observation.tactile_deform.{hand}{i}' for hand in ('l', 'r') for i in range(5)]
VIEWS = RGB + DEFORM
VERSION = 1

def emit(**fields): print(json.dumps(fields, ensure_ascii=False), flush=True)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)
def suffix(key): return key.removeprefix('observation.images.') if key in RGB else key.rsplit('.', 1)[-1]
def image_path(out, episode, frame, key): return out / 'images' / f'episode_{episode}' / f'image{frame}_{suffix(key)}.png'

def validate_image_inventory(out, episodes):
    """Reject an incomplete cache before a multi-hour training run starts."""
    for ep in episodes:
        folder = out / 'images' / f"episode_{ep['episode_index']}"
        expected = {f'image{i}_{suffix(key)}.png' for i in range(ep['length']) for key in VIEWS}
        if not folder.is_dir():
            raise RuntimeError(f'Missing image directory: {folder}')
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name in expected and entry.is_file() and entry.stat().st_size > 0:
                    expected.remove(entry.name)
        if expected:
            raise RuntimeError(f'Incomplete image cache: {folder / min(expected)}; '
                               'restore missing images or choose a new output directory.')

def remaining_free_gib(out, requested):
    # A retry already owns some of the PNG allocation. Do not demand the full
    # fresh-export allowance again; reserve at least 8 GiB for JSON and scratch.
    completed = sum(json.loads(p.read_text()).get('png_bytes', 0)
                    for p in (out / 'jobs').glob('*.json'))
    return max(min(8.0, requested), requested - completed / 2**30)

def export(source, out, workers, min_free_gib):
    source, out = source.resolve(), out.resolve()
    if (source / 'CONVERSION_IN_PROGRESS').exists(): raise RuntimeError('EEF conversion is incomplete.')
    info = json.loads((source / 'meta/info.json').read_text())
    assert info['features']['observation.state']['shape'] == [62], 'Run joint58 -> EEF62 conversion first.'
    assert info['features']['action']['shape'] == [16, 62]
    crop = info['trex_head_crop_box']
    eps = [row for p in sorted((source / 'meta/episodes').rglob('*.parquet')) for row in pq.read_table(p).to_pylist()]
    eps.sort(key=lambda r: r['episode_index'])
    assert sum(ep['length'] for ep in eps) == info['total_frames']
    parquet_files = sorted((source / 'data').rglob('*.parquet'))
    identity = {'version': VERSION, 'source': str(source), 'info_sha256': sha(source / 'meta/info.json'),
                'statistics_sha256': sha(source / 'meta/trex_norm_stats.json'),
                'parquet_sha256': {str(p.relative_to(source)): sha(p) for p in parquet_files},
                'video_metadata': {str(p.relative_to(source)): [p.stat().st_size, p.stat().st_mtime_ns]
                                   for p in sorted((source / 'videos').rglob('*.mp4'))},
                'crop': crop, 'png_compress_level': 1}
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.preprocess.lock').open('a') as lock:
        emit(stage='waiting_for_output_lock', output=str(out))
        fcntl.flock(lock, fcntl.LOCK_EX)
        progress = out / 'PREPROCESSING_IN_PROGRESS'
        manifest_path = out / 'preprocessing_manifest.json'
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old['identity'] != identity: raise RuntimeError('Output belongs to different input data; choose another output directory.')
            assert (out / 'task.json').stat().st_size == old['json_bytes']
            assert sha(out / 'task_statistics.json') == identity['statistics_sha256']
            validate_image_inventory(out, eps)
            progress.unlink(missing_ok=True)
            emit(stage='reuse_completed_export', **{k: old[k] for k in ('frames', 'episodes', 'png_count', 'png_bytes')})
            return old
        identity_path = out / 'source_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError('Partial output belongs to different inputs; choose a fresh directory.')
        atomic_json(identity_path, identity)
        free = shutil.disk_usage(out).free / 2**30
        required = remaining_free_gib(out, min_free_gib)
        if free < required: raise RuntimeError(f'Image output has {free:.1f} GiB free; require {required:.1f} GiB for remaining export.')
        progress.write_text(f'pid={os.getpid()}\nstarted={time.time()}\n')
        (out / 'jobs').mkdir(exist_ok=True)
        jobs = {}
        for ep in eps:
            (out / 'images' / f"episode_{ep['episode_index']}").mkdir(parents=True, exist_ok=True)
            for key in VIEWS:
                rel = info['video_path'].format(video_key=key, chunk_index=ep[f'videos/{key}/chunk_index'],
                                                file_index=ep[f'videos/{key}/file_index'])
                start = round(ep[f'videos/{key}/from_timestamp'] * info['fps'])
                jobs.setdefault((rel, key), []).append((start, ep['length'], ep['episode_index']))
        started = time.perf_counter()
        def extract(job):
            (rel, key), segments = job
            video = source / rel
            spec = {'video': rel, 'size': video.stat().st_size, 'mtime_ns': video.stat().st_mtime_ns,
                    'key': key, 'segments': segments}
            job_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            done = out / 'jobs' / f'{job_id}.json'
            if done.exists(): return json.loads(done.read_text())
            wanted, total_bytes = {}, 0
            for start, length, episode in segments:
                for i in range(length):
                    target = image_path(out, episode, i, key)
                    if target.exists() and target.stat().st_size > 0: total_bytes += target.stat().st_size
                    else:
                        frame_number = start + i
                        if frame_number in wanted: raise RuntimeError('Overlapping video segments.')
                        wanted[frame_number] = target
            written = 0
            if wanted:
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    for frame in container.decode(stream):
                        number = round(float(frame.pts * stream.time_base) * info['fps'])
                        target = wanted.pop(number, None)
                        if target is not None:
                            rgb = frame.to_ndarray(format='rgb24')
                            if key in DEFORM:
                                image = Image.fromarray(rgb[:, :, 0], mode='L')
                            else:
                                # Match LeRobot's float -> uint8 conversion exactly.
                                rgb = ((rgb.astype(np.float32) / 255.0) * 255.0).astype(np.uint8)
                                if key == RGB[0]:
                                    y0, y1, x0, x1 = crop
                                    rgb = rgb[y0:y1, x0:x1]
                                image = Image.fromarray(rgb, mode='RGB')
                            tmp = target.with_suffix('.png.tmp')
                            image.save(tmp, format='PNG', compress_level=1)
                            os.replace(tmp, target)
                            total_bytes += target.stat().st_size
                            written += 1
                        if not wanted: break
                if wanted: raise RuntimeError(f'{rel}: missing {len(wanted)} frames, first={min(wanted)}')
            result = {'video': rel, 'key': key, 'png_count': sum(n for _, n, _ in segments), 'png_bytes': total_bytes}
            atomic_json(done, result)
            emit(stage='video_complete', newly_written=written, **result)
            return result
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(extract, job) for job in jobs.items()]
            for future in as_completed(futures): results.append(future.result())
        extract_seconds = time.perf_counter() - started
        eps_by_id = {ep['episode_index']: ep for ep in eps}
        json_tmp = out / 'task.json.tmp'
        frames = 0
        columns = ['episode_index', 'frame_index', 'action', 'observation.state', 'observation.tactile_f6']
        with json_tmp.open('w') as f:
            for p in parquet_files:
                for batch in pq.ParquetFile(p).iter_batches(batch_size=256, columns=columns):
                    for row in batch.to_pylist():
                        ep, frame = int(row['episode_index']), int(row['frame_index'])
                        image = lambda key: str(image_path(out, ep, frame, key))
                        sample = {'action': row['action'], 'state_fast': row['observation.state'],
                                  'tactile_f6': row['observation.tactile_f6'], 'input_prompt': eps_by_id[ep]['tasks'][0],
                                  'input_image_slow': [image(RGB[0])],
                                  'input_image_fast': [image(RGB[1]), image(RGB[2])],
                                  'tactile_image_deform': [image(k) for k in DEFORM]}
                        f.write(json.dumps(sample, separators=(',', ':'), allow_nan=False) + '\n')
                        frames += 1
        assert frames == info['total_frames']
        assert sum(r['png_count'] for r in results) == frames * len(VIEWS)
        validate_image_inventory(out, eps)
        os.replace(json_tmp, out / 'task.json')
        shutil.copyfile(source / 'meta/trex_norm_stats.json', out / 'task_statistics.json.tmp')
        os.replace(out / 'task_statistics.json.tmp', out / 'task_statistics.json')
        report = {'identity': identity, 'frames': frames, 'episodes': len(eps), 'png_count': frames * len(VIEWS),
                  'png_bytes': sum(r['png_bytes'] for r in results), 'json_bytes': (out / 'task.json').stat().st_size,
                  'extract_seconds': extract_seconds, 'total_seconds': time.perf_counter() - started,
                  'workers': workers, 'created_at_unix': time.time(), 'views': results}
        atomic_json(manifest_path, report)
        progress.unlink()
        emit(stage='export_complete', **{k: report[k] for k in ('frames', 'episodes', 'png_count', 'png_bytes', 'json_bytes', 'total_seconds')})
        return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_root', type=Path, required=True)
    p.add_argument('--output_root', type=Path, required=True, help='Local SSD recommended; output paths are absolute and should not be moved.')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--min_free_gib', type=float, default=140, help='Free capacity required for a fresh export; completed exports are reused.')
    args = p.parse_args()
    if args.workers < 1: p.error('--workers must be positive')
    export(args.source_root, args.output_root, args.workers, args.min_free_gib)
