#!/usr/bin/env python3
"""Create a GVHMR input video from an EgoBody EgoAllo input NPZ and run GVHMR.

This script is intentionally thin: it uses the frame list already selected by
convert_egobody_to_egoallo.py, writes a short mp4 for the requested interval,
and then calls GVHMR's official tools/demo/demo.py entry point.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import os
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]




def _default_gvhmr_python() -> Path:
    candidates = [
        Path("/public/home/wenxin/miniconda3/envs/gvhmr/bin/python"),
        Path("/public/home/wenxin/miniconda3/envs/GVHMR/bin/python"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _resolve_traj_root(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        return path
    raise FileNotFoundError(path)


def _load_image_paths(input_npz: Path) -> list[Path]:
    data = np.load(input_npz, allow_pickle=True)
    if "image_paths" not in data.files:
        raise KeyError(f"{input_npz} does not contain image_paths")
    base_path = Path("/public/home/wenxin/egobody")
    paths = [base_path / Path(str(x)) for x in data["image_paths"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image files, first: {missing[0]}")
    return paths


def _write_video(image_paths: list[Path], out_path: Path, fps: int) -> None:
    if not image_paths:
        raise ValueError("No image paths selected")
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read {image_paths[0]}")
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    try:
        for path in image_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read {path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-root", type=Path, required=True, help="Directory created by convert_egobody_to_egoallo.py")
    parser.add_argument("--input-name", default="egobody_egoallo_input.npz")
    parser.add_argument("--gvhmr-root", type=Path, default=Path("/public/home/wenxin/GVHMR"))
    parser.add_argument("--python", type=Path, default=None, help="Python executable used to run GVHMR demo.py; defaults to the gvhmr conda env if present")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--traj-length", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-name", default=None, help="Defaults to <recording>_<start>-<end>")
    parser.add_argument("--output-root", type=Path, default=None, help="GVHMR demo output root; defaults to <traj-root>/gvhmr_exo")
    parser.add_argument("--static-cam", action="store_true", help="Pass --static_cam to GVHMR demo")
    parser.add_argument("--use-dpvo", action="store_true", help="Pass --use_dpvo to GVHMR demo")
    parser.add_argument("--f-mm", type=int, default=None, help="Pass --f_mm to GVHMR demo. Ignored when real EgoBody K is enabled.")
    parser.add_argument("--no-real-k", action="store_true", help="Do not pass EgoBody PV K_fullimg to GVHMR demo; use GVHMR's estimated/f_mm intrinsics instead.")
    parser.add_argument("--force", action="store_true", help="Pass --force to GVHMR demo so hmr4d_results.pt and render videos are recomputed.")
    parser.add_argument("--no-run", action="store_true", help="Only create the mp4, camera K npy, and metadata")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    traj_root = _resolve_traj_root(args.traj_root)
    input_npz = traj_root / args.input_name
    if not input_npz.exists():
        raise FileNotFoundError(input_npz)
    image_paths = _load_image_paths(input_npz)
    start = int(args.start_index)
    end = len(image_paths) if args.traj_length is None else start + int(args.traj_length)
    if start < 0 or end > len(image_paths) or start >= end:
        raise ValueError(f"Invalid interval [{start}, {end}) for {len(image_paths)} frames")
    selected_paths = image_paths[start:end]

    data = np.load(input_npz, allow_pickle=True)
    recording = str(data["recording"].item() if np.asarray(data["recording"]).shape == () else data["recording"])
    video_name = args.video_name or f"{recording}_{start:06d}_{end:06d}"
    output_root = args.output_root.resolve() if args.output_root is not None else (traj_root / "gvhmr_exo").resolve()
    video_path = output_root / "input_videos" / f"{video_name}.mp4"
    _write_video(selected_paths, video_path, args.fps)

    k_fullimg_npy = None
    if not args.no_real_k:
        if "K_fullimg" not in data.files:
            raise KeyError(f"{input_npz} does not contain K_fullimg; rerun convert_egobody_to_egoallo.py or use --no-real-k")
        K_fullimg = np.asarray(data["K_fullimg"][start:end], dtype=np.float32)
        if K_fullimg.shape != (end - start, 3, 3):
            raise ValueError(f"Expected K_fullimg shape {(end - start, 3, 3)}, got {K_fullimg.shape}")
        k_dir = output_root / "camera_intrinsics"
        k_dir.mkdir(parents=True, exist_ok=True)
        k_fullimg_npy = k_dir / f"{video_name}_K_fullimg.npy"
        np.save(k_fullimg_npy, K_fullimg)

    meta = {
        "input_npz": str(input_npz.resolve()),
        "video_path": str(video_path.resolve()),
        "gvhmr_root": str(args.gvhmr_root.resolve()),
        "output_root": str(output_root),
        "video_name": video_name,
        "start_index": start,
        "end_index": end,
        "fps": args.fps,
        "image_paths": [str(p) for p in selected_paths],
        "K_fullimg_npy": str(k_fullimg_npy.resolve()) if k_fullimg_npy is not None else None,
        "uses_real_egobody_K": k_fullimg_npy is not None,
    }
    meta_path = output_root / f"{video_name}_egobody_frames.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved GVHMR input video: {video_path}")
    print(f"Saved metadata: {meta_path}")

    gvhmr_python = args.python.resolve() if args.python is not None else _default_gvhmr_python()
    if not gvhmr_python.exists():
        raise FileNotFoundError(gvhmr_python)
    cmd = [
        str(gvhmr_python),
        "tools/demo/demo.py",
        "--video",
        str(video_path),
        "--output_root",
        str(output_root),
    ]
    if args.static_cam:
        cmd.append("--static_cam")
    if args.use_dpvo:
        cmd.append("--use_dpvo")
    if k_fullimg_npy is not None:
        cmd += ["--K_fullimg_npy", str(k_fullimg_npy.resolve())]
    elif args.f_mm is not None:
        cmd += ["--f_mm", str(args.f_mm)]
    if args.force:
        cmd.append("--force")

    print("GVHMR command:")
    print(" ".join(cmd))
    if args.no_run:
        return
    env = dict(os.environ)
    gvhmr_bin = str(gvhmr_python.parent)
    env["PATH"] = gvhmr_bin + os.pathsep + env.get("PATH", "")
    if gvhmr_python.parent.name == "bin":
        env.setdefault("CONDA_PREFIX", str(gvhmr_python.parent.parent))
    subprocess.run(cmd, cwd=args.gvhmr_root, check=True, env=env)
    print(f"Expected GVHMR result: {output_root / video_name / 'hmr4d_results.pt'}")


if __name__ == "__main__":
    main()
