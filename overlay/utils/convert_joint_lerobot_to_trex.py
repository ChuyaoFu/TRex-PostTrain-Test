"""Public/task LeRobot v3 joint-58 -> canonical T-Rex EEF-62 dataset.

Reuse the official quickstart FK, chunk math and normalization accumulator.
Videos are linked or copied without re-encoding. A metadata crop is applied
by TRexLeRobotDataset to current and future head frames, matching deployment.
The source dataset is never modified; the output directory must not exist.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dataset_quickstart/src"))

from trex_dataset_quickstart import robot, schema
from utils.lerobot_common import (
    ACTION_CHUNK, ACTION_DIM, DEFORM_KEYS, KEY_ACTION, KEY_ACTION_ABS,
    KEY_HEAD, KEY_STATE, KEY_TACF6, KEY_WRIST_L, KEY_WRIST_R,
    NormStatsAccumulator, build_action_chunk, calculate_stats, pose_matrix_to_9d,
)

VIDEO_MAP = {
    "observation.images.head_left": KEY_HEAD,
    "observation.images.left_wrist": KEY_WRIST_L,
    "observation.images.right_wrist": KEY_WRIST_R,
    **{f"observation.images.tactile_{side}_deform_{finger}": DEFORM_KEYS[hand * 5 + i]
       for hand, side in enumerate(("left", "right"))
       for i, finger in enumerate(schema.FINGER_NAMES)},
}
KEY_MAP = {**VIDEO_MAP, "observation.tactile_force": KEY_TACF6}


def rename_meta_key(key):
    for prefix in ("videos/", "stats/"):
        if key.startswith(prefix):
            feature, suffix = key[len(prefix):].rsplit("/", 1)
            return prefix + KEY_MAP.get(feature, feature) + "/" + suffix
    return key


def array_column(values):
    """Build nested Arrow list columns without materializing Python floats."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    column = pa.array(values.reshape(-1), type=pa.float32())
    for width in reversed(values.shape[1:]):
        offsets = np.arange(0, len(column) + 1, width, dtype=np.int32)
        column = pa.ListArray.from_arrays(offsets, column)
    return column


def read_matrix(table, key, width):
    column = table[key].combine_chunks()
    if not pa.types.is_list(column.type) or column.null_count:
        raise ValueError(f"{key}: expected non-null list column")
    if not np.all(np.diff(column.offsets.to_numpy()) == width):
        raise ValueError(f"{key}: expected {width} values per row")
    result = column.values.to_numpy().reshape(-1, width)
    if not np.isfinite(result).all():
        raise ValueError(f"{key}: non-finite values")
    return result


def joint_poses(joints, model, assemble):
    left = np.empty((len(joints), 4, 4))
    right = np.empty_like(left)
    for i, joint in enumerate(joints):
        components = schema.split_state(joint)
        qpos = assemble({key: components[key] for key in ("left_arm", "right_arm")})
        poses = robot.forward_kinematics(model, qpos, ["L_ee", "R_ee"])
        left[i] = poses["L_ee"].homogeneous
        right[i] = poses["R_ee"].homogeneous
    return left, right


def native_stats(stats, count):
    return {key: value for key, value in stats.items() if key != "mask"} | {"count": [count]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_episodes", type=int, default=0, help="First N complete episodes; 0 = all")
    parser.add_argument("--torso", type=float, nargs=3, default=robot.DEFAULT_TORSO.tolist())
    parser.add_argument("--head", type=float, nargs=3, default=robot.DEFAULT_HEAD.tolist())
    parser.add_argument("--head_crop_box", type=int, nargs=4, default=[0, 300, 140, 540],
                        metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument("--no_head_crop", action="store_true")
    parser.add_argument("--video_mode", choices=["symlink", "copy"], default="symlink",
                        help="Use copy for a self-contained dataset to send to H100")
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Output must be a new directory outside the source dataset")
    if args.num_episodes < 0:
        raise ValueError("num_episodes must be >= 0")
    info = json.loads((source / "meta/info.json").read_text())
    for key, shape in ((KEY_STATE, [58]), (KEY_ACTION, [58]), ("observation.tactile_force", [60])):
        if info["features"].get(key, {}).get("shape") != shape:
            raise ValueError(f"Expected {key} shape {shape}")
    for key in VIDEO_MAP:
        if info["features"].get(key, {}).get("dtype") != "video":
            raise ValueError(f"Missing video feature {key}")
    episodes = []
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda ep: ep["episode_index"])
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Incomplete episode metadata")
    if args.num_episodes:
        episodes = episodes[:args.num_episodes]
    if not episodes or [ep["episode_index"] for ep in episodes] != list(range(len(episodes))):
        raise ValueError("Expected contiguous episodes beginning at 0")
    end = 0
    video_requirements = {}
    for ep in episodes:
        if ep["dataset_from_index"] != end or ep["dataset_to_index"] - end != ep["length"]:
            raise ValueError("Non-contiguous or inconsistent episode boundaries")
        end = ep["dataset_to_index"]
        for key in VIDEO_MAP:
            video = source / info["video_path"].format(
                video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"],
                file_index=ep[f"videos/{key}/file_index"])
            if not video.is_file() or not video.stat().st_size:
                raise FileNotFoundError(video)
            video_requirements[video] = max(video_requirements.get(video, 0), ep[f"videos/{key}/to_timestamp"])
    for video, needed in video_requirements.items():
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base) if stream.duration else container.duration / av.time_base
            if duration + 1 / info["fps"] < needed:
                raise ValueError(f"Incomplete video: {video}, duration {duration}, required {needed}")
    sources = []
    tables = []
    total = 0
    for path in sorted((source / "data").rglob("*.parquet")):
        if total >= end:
            break
        table = pq.read_table(path)
        table = table.slice(0, min(table.num_rows, end - total))
        sources.append((path.relative_to(source), total, table.num_rows))
        tables.append(table)
        total += table.num_rows
    if total != end:
        raise ValueError("Incomplete source data")
    table = pa.concat_tables(tables).replace_schema_metadata(None)
    if not np.array_equal(table["index"].to_numpy(), np.arange(end)):
        raise ValueError("Source frame index order is inconsistent")
    joints = read_matrix(table, KEY_STATE, 58)
    targets = read_matrix(table, KEY_ACTION, 58)
    force = read_matrix(table, "observation.tactile_force", 60)
    model, assemble, _ = robot.build_reduced_bimanual_robot(
        {"torso": np.asarray(args.torso), "head": np.asarray(args.head)})
    if any(not model.model.existFrame(name) for name in ("L_ee", "R_ee")):
        raise ValueError("URDF lacks required end-effector frames")
    print(f"Vega-1 FK: nq={model.nq}; {len(episodes)} episodes, {end} frames", flush=True)
    crop = None if args.no_head_crop else args.head_crop_box
    if crop:
        height = info["features"]["observation.images.head_left"]["shape"][0]
        width = info["features"]["observation.images.head_left"]["shape"][1]
        y0, y1, x0, x1 = crop
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("Crop outside source head image")
    output.mkdir(parents=True)
    (output / "meta").mkdir()
    (output / "CONVERSION_IN_PROGRESS").write_text("Do not train until conversion completes.\n")
    states = np.empty((end, ACTION_DIM), np.float32)
    absolute = np.empty_like(states)
    chunks = np.empty((end, ACTION_CHUNK, ACTION_DIM), np.float32)
    tactile = np.ascontiguousarray(force.reshape(end, 10, 6), dtype=np.float32)
    accumulator = NormStatsAccumulator()
    converted_episodes = []
    for ep in episodes:
        start, stop = ep["dataset_from_index"], ep["dataset_to_index"]
        n = stop - start
        if not np.all(table["episode_index"].slice(start, n).to_numpy() == ep["episode_index"]):
            raise ValueError("Episode boundary disagrees with frame episode_index")
        sl, sr = joint_poses(joints[start:stop], model, assemble)
        al, ar = joint_poses(targets[start:stop], model, assemble)
        states[start:stop] = np.concatenate([
            pose_matrix_to_9d(sl), joints[start:stop, schema.LEFT_HAND],
            pose_matrix_to_9d(sr), joints[start:stop, schema.RIGHT_HAND]], axis=1)
        absolute[start:stop] = np.concatenate([
            pose_matrix_to_9d(al), targets[start:stop, schema.LEFT_HAND],
            pose_matrix_to_9d(ar), targets[start:stop, schema.RIGHT_HAND]], axis=1)
        for i in range(n):
            chunk = build_action_chunk(sl, al, targets[start:stop, schema.LEFT_HAND],
                                       sr, ar, targets[start:stop, schema.RIGHT_HAND], i, n)
            chunks[start + i] = chunk
            accumulator.add_frame(chunks[start + i], states[start + i], tactile[start + i])
        accumulator.add_episode_tracking(states[start:stop], absolute[start:stop])
        new_ep = {rename_meta_key(key): value for key, value in ep.items()
                  if not key.startswith(("stats/observation.state/", "stats/action/",
                                         "stats/observation.tactile_force/"))}
        for key, values in ((KEY_STATE, states[start:stop]), (KEY_ACTION, chunks[start:stop]),
                            (KEY_ACTION_ABS, absolute[start:stop]), (KEY_TACF6, tactile[start:stop])):
            for stat, value in native_stats(calculate_stats(values), n).items():
                new_ep[f"stats/{key}/{stat}"] = value
        converted_episodes.append(new_ep)
        print(f"  episode {ep['episode_index']}: {n} frames", flush=True)
    replacements = {KEY_STATE: states, KEY_ACTION: chunks, KEY_ACTION_ABS: absolute, KEY_TACF6: tactile}
    for relative, start, count in sources:
        dst = output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        original = table.slice(start, count)
        columns = {key: original[key] for key in original.column_names
                   if key not in (KEY_STATE, KEY_ACTION, "observation.tactile_force")}
        columns.update({key: array_column(value[start:start + count]) for key, value in replacements.items()})
        pq.write_table(pa.table(columns), dst, compression="zstd")
    grouped = {}
    for ep in converted_episodes:
        key = (ep["meta/episodes/chunk_index"], ep["meta/episodes/file_index"])
        grouped.setdefault(key, []).append(ep)
    for (chunk, file), rows in grouped.items():
        dst = output / f"meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), dst, compression="zstd")
    shutil.copy2(source / "meta/tasks.parquet", output / "meta/tasks.parquet")
    for old_key, new_key in VIDEO_MAP.items():
        for ep in episodes:
            relative = info["video_path"].format(
                video_key=old_key, chunk_index=ep[f"videos/{old_key}/chunk_index"],
                file_index=ep[f"videos/{old_key}/file_index"])
            src = source / relative
            new_relative = info["video_path"].format(
                video_key=new_key, chunk_index=ep[f"videos/{old_key}/chunk_index"],
                file_index=ep[f"videos/{old_key}/file_index"])
            dst = output / new_relative
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.video_mode == "copy":
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
    print("Calculating exact per-step q01/q99 normalization...", flush=True)
    block = accumulator.assemble()
    (output / "meta/trex_norm_stats.json").write_text(json.dumps(block, indent=2, allow_nan=False))
    stats = {KEY_MAP.get(key, key): value for key, value in
             json.loads((source / "meta/stats.json").read_text()).items()
             if key not in (KEY_STATE, KEY_ACTION, "observation.tactile_force")}
    norm = block[next(iter(block))]
    for key, value in ((KEY_STATE, norm["state"]), (KEY_ACTION, norm["action"]),
                       (KEY_TACF6, calculate_stats(tactile)), (KEY_ACTION_ABS, calculate_stats(absolute))):
        stats[key] = native_stats(value, end)
    # Recalculate index/timestamp/task statistics for a prefix subset as well.
    for key in ("index", "frame_index", "episode_index", "task_index", "timestamp"):
        stats[key] = native_stats(calculate_stats(table[key].to_numpy().reshape(-1, 1)), end)
    (output / "meta/stats.json").write_text(json.dumps(stats, indent=2, allow_nan=False))
    features = {key: copy.deepcopy(value) for key, value in info["features"].items()
                if value["dtype"] != "video" and key not in (KEY_STATE, KEY_ACTION, "observation.tactile_force")}
    for key, value in replacements.items():
        features[key] = {"dtype": "float32", "shape": list(value.shape[1:]), "names": None}
    for old_key, new_key in VIDEO_MAP.items():
        features[new_key] = copy.deepcopy(info["features"][old_key])
    info.update(features=features, robot_type="trex_bimanual", total_episodes=len(episodes),
                total_frames=end, splits={"train": f"0:{len(episodes)}"}, trex_head_crop_box=crop)
    info["data_files_size_in_mb"] = 100  # retained file assignment; informational target size
    (output / "meta/info.json").write_text(json.dumps(info, indent=2, allow_nan=False))
    urdf = Path(robot.robots.humanoid.vega_1.vega_1.urdf)
    provenance = {"source_root": str(source), "source_info_sha256": hashlib.sha256(
        (source / "meta/info.json").read_bytes()).hexdigest(), "robot": "Dexmate Vega-1",
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "torso": args.torso, "head": args.head, "eef_frames": ["L_ee", "R_ee"],
        "target_pose_source": "FK of recorded target joints (not original pre-IK teleop pose)",
        "head_crop_box": crop, "video_mode": args.video_mode, "episodes": len(episodes),
        "frames": end, "action_chunk": ACTION_CHUNK}
    (output / "meta/trex_conversion.json").write_text(json.dumps(provenance, indent=2))
    (output / "CONVERSION_IN_PROGRESS").unlink()
    print(f"Ready: {output}", flush=True)


if __name__ == "__main__":
    main()
