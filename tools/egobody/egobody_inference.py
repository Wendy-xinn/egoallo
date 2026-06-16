#!/usr/bin/env python3
"""Run EgoAllo sampling from an EgoBody-converted trajectory NPZ.

Run this inside the EgoAllo environment. The script intentionally avoids VRS,
MPS point clouds, HaMeR, and Aria hand tracking for the first smoke test.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _maybe_add_egoallo_src(path: Path) -> None:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-root", type=Path, required=True, help="Directory created by convert_egobody_to_egoallo.py")
    parser.add_argument("--input-name", default="egobody_egoallo_input.npz")
    parser.add_argument("--egoallo-src", type=Path, default=repo_root / "src")
    parser.add_argument("--checkpoint-dir", type=Path, default=repo_root / "egoallo_checkpoint_april13" / "checkpoints_3000000")
    parser.add_argument("--smplh-npz-path", type=Path, default=repo_root / "data" / "smplh" / "neutral" / "model.npz")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--traj-length", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--glasses-x-angle-offset", type=float, default=0.0)
    parser.add_argument("--floor-z", type=float, default=None)
    parser.add_argument("--save-traj", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _maybe_add_egoallo_src(args.egoallo_src)

    from egoallo import fncsmpl, fncsmpl_extensions
    from egoallo.inference_utils import load_denoiser
    from egoallo.sampling import run_sampling_with_stitching
    from egoallo.transforms import SE3, SO3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = args.traj_root / args.input_name
    data = np.load(input_path, allow_pickle=True)
    mats = data["Ts_world_cpf_mats"]
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"Expected Ts_world_cpf_mats with shape (T,4,4), got {mats.shape}")
    if args.start_index + args.traj_length + 1 > mats.shape[0]:
        raise ValueError(
            f"Need {args.start_index + args.traj_length + 1} poses for traj_length={args.traj_length}, "
            f"but input has {mats.shape[0]}"
        )

    Ts_world_cpf_all = SE3.from_matrix(torch.from_numpy(mats).to(torch.float32)).parameters()
    Ts_world_cpf = (
        SE3(Ts_world_cpf_all[args.start_index : args.start_index + args.traj_length + 1])
        @ SE3.from_rotation(
            SO3.from_x_radians(
                Ts_world_cpf_all.new_tensor(args.glasses_x_angle_offset)
            )
        )
    ).parameters().to(device)

    pose_timestamps_sec = data["pose_timestamps_sec"][args.start_index + 1 : args.start_index + args.traj_length + 1]
    frame_nums = data["frame_nums"][args.start_index : args.start_index + args.traj_length]
    floor_z = float(data["floor_z"]) if args.floor_z is None else args.floor_z

    pose_source = str(data["pose_source"]) if "pose_source" in data.files else "unknown"
    cpf_position_mode = str(data["cpf_position_mode"]) if "cpf_position_mode" in data.files else "unknown"
    scene_points = data["scene_points"].shape[0] if "scene_points" in data.files else 0

    print(f"Loaded {input_path}")
    print(
        f"Ts_world_cpf={tuple(Ts_world_cpf.shape)}, floor_z={floor_z:.4f}, "
        f"pose_source={pose_source}, cpf_position_mode={cpf_position_mode}, "
        f"scene_points={scene_points}, device={device}"
    )
    print("Loading model...")
    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    print("Sampling with guidance_mode='off' (no VRS/HaMeR/Aria hand inputs)...")
    traj = run_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode="no_hands",
        guidance_inner=False,
        guidance_post=False,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=None,
        aria_detections=None,
        num_samples=args.num_samples,
        device=device,
        floor_z=floor_z,
    )

    if args.save_traj:
        out_dir = args.traj_root / "egoallo_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_name = time.strftime("%Y%m%d-%H%M%S") + f"_{args.start_index}-{args.start_index + args.traj_length}"
        out_path = out_dir / f"{save_name}.npz"
        posed = traj.apply_to_body(body_model)
        Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            posed, Ts_world_cpf[..., 1:, :]
        )
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
            Ts_world_root=Ts_world_root.numpy(force=True),
            body_quats=posed.local_quats[..., :21, :].numpy(force=True),
            left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
            right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
            contacts=traj.contacts.numpy(force=True),
            betas=traj.betas.numpy(force=True),
            frame_nums=frame_nums,
            timestamps_ns=(np.asarray(pose_timestamps_sec) * 1e9).astype(np.int64),
            source_input=str(input_path),
        )
        serializable_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        (out_dir / f"{save_name}_args.yaml").write_text(yaml.safe_dump(serializable_args))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
