#!/usr/bin/env python3
"""Run and summarize the EgoBody ego/exo pipeline over multiple windows.

This script is the experiment-level wrapper around run_egobody_ego_exo_pipeline.py.
It avoids manually running raw/optimized/diagnose for one 128-frame window at a
time.

Typical use:
  python tools/egobody/run_egobody_window_sweep.py \
    --recording recording_20210907_S02_S01_01 \
    --start-index 0 --traj-length 128 --stride 128 --max-windows 5 \
    --run-name sweep_realK_smooth \
    --run-optimize --run-diagnose

It writes:
  <recording>/pipeline_runs/<run-name>_sweep_summary.json
  <recording>/pipeline_runs/<run-name>_sweep_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_ego_python() -> Path:
    candidate = Path('/public/home/wenxin/miniconda3/envs/ego/bin/python')
    return candidate if candidate.exists() else Path(sys.executable)


def _cmd_to_str(cmd: list[str | Path]) -> str:
    return ' '.join(shlex.quote(str(x)) for x in cmd)


def _scalar(metrics: dict[str, Any], path: list[str], default: float | None = None) -> float | None:
    obj: Any = metrics
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    if isinstance(obj, (float, int)):
        return float(obj)
    return default


def _read_metric_row(metrics_path: Path, start: int, end: int, video_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        'start_index': start,
        'end_index': end,
        'video_name': video_name,
        'metrics_path': str(metrics_path),
        'status': 'missing_metrics',
    }
    if not metrics_path.exists():
        return row
    metrics = json.loads(metrics_path.read_text())
    row['status'] = 'ok'
    row['root_l2_mean_m'] = _scalar(metrics, ['root_l2_error_m', 'mean'])
    row['root_z_mean_m'] = _scalar(metrics, ['root_z_error_m', 'mean'])
    row['mpjpe_mean_m'] = _scalar(metrics, ['exo_body_joint_metrics', 'mpjpe_m', 'mean'])
    row['root_aligned_mpjpe_mean_m'] = _scalar(metrics, ['exo_body_joint_metrics', 'root_aligned_mpjpe_m', 'mean'])
    row['pa_mpjpe_mean_m'] = _scalar(metrics, ['exo_body_joint_metrics', 'pa_mpjpe_m', 'mean'])
    row['reproj_vertex_in_image_mean'] = _scalar(metrics, ['overlay_vertex_in_image_fraction_mean'])
    row['reproj_joint_in_image_mean'] = _scalar(metrics, ['overlay_joint_in_image_fraction_mean'])
    row['baseline_root_l2_mean_m'] = _scalar(metrics, ['baseline_comparison', 'root_l2_error_m', 'mean'])
    row['baseline_mpjpe_mean_m'] = _scalar(metrics, ['baseline_comparison', 'exo_body_joint_metrics', 'mpjpe_m', 'mean'])
    row['baseline_pa_mpjpe_mean_m'] = _scalar(metrics, ['baseline_comparison', 'exo_body_joint_metrics', 'pa_mpjpe_m', 'mean'])
    row['delta_root_l2_mean_m'] = _scalar(metrics, ['baseline_comparison', 'delta_current_minus_baseline_root_l2_mean_m'])
    row['delta_mpjpe_mean_m'] = _scalar(metrics, ['baseline_comparison', 'delta_current_minus_baseline_mpjpe_mean_m'])
    row['delta_pa_mpjpe_mean_m'] = _scalar(metrics, ['baseline_comparison', 'delta_current_minus_baseline_pa_mpjpe_mean_m'])
    row['ego_sensor_head_l2_mean_m'] = _scalar(metrics, ['ego_sensor_head_track_error_m', 'l2_m', 'mean'])
    row['ego_sensor_left_wrist_l2_mean_m'] = _scalar(metrics, ['ego_sensor_left_wrist_track_error_m', 'l2_m', 'mean'])
    row['ego_sensor_right_wrist_l2_mean_m'] = _scalar(metrics, ['ego_sensor_right_wrist_track_error_m', 'l2_m', 'mean'])
    return row


def _mean_ok(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in rows if r.get('status') == 'ok' and isinstance(r.get(key), (float, int))]
    if not vals:
        return None
    return float(np.mean(vals))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--ego-python', type=Path, default=_default_ego_python())
    parser.add_argument('--egoallo-root', type=Path, default=_repo_root())
    parser.add_argument('--output-root', type=Path, default=_repo_root() / 'egoallo_egobody_trajectories')
    parser.add_argument('--input-name', default='egobody_egoallo_input.npz')
    parser.add_argument('--run-name', default='sweep_realK_smooth')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--traj-length', type=int, default=128)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--max-windows', type=int, default=5)
    parser.add_argument('--starts', type=int, nargs='*', default=None, help='Explicit start indices; overrides stride/max-windows')
    parser.add_argument('--run-optimize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-diagnose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-egoallo', action=argparse.BooleanOptionalAction, default=True, help='Run or reuse EgoAllo sampling for each window so joint ego-exo constraints can be active.')
    parser.add_argument('--force-egoallo', action='store_true', help='Re-run EgoAllo sampling for each window instead of reusing deterministic aliases')
    parser.add_argument('--force-gvhmr', action='store_true')
    parser.add_argument('--force-export', action='store_true')
    parser.add_argument('--force-optimize', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--extra-pipeline-args', nargs=argparse.REMAINDER, default=None, help='Additional args appended to run_egobody_ego_exo_pipeline.py')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    egoallo_root = args.egoallo_root.resolve()
    traj_root = (args.output_root / args.recording).resolve()
    input_npz = traj_root / args.input_name
    if not args.dry_run and not input_npz.exists():
        raise FileNotFoundError(f'Missing {input_npz}. Run convert or one full pipeline pass first.')
    if args.dry_run:
        total_len = args.start_index + args.traj_length + args.stride * max(args.max_windows - 1, 0) + 1
    else:
        data = np.load(input_npz, allow_pickle=True)
        if 'image_paths' in data.files:
            total_len = len(data['image_paths'])
        elif 'Ts_world_pv_mats' in data.files:
            total_len = len(data['Ts_world_pv_mats'])
        else:
            raise KeyError(f'{input_npz} has neither image_paths nor Ts_world_pv_mats')

    if args.starts is not None and len(args.starts) > 0:
        starts = [int(s) for s in args.starts]
    else:
        starts = []
        cur = int(args.start_index)
        while cur + args.traj_length <= total_len and len(starts) < args.max_windows:
            starts.append(cur)
            cur += args.stride
    if not starts:
        raise ValueError('No valid windows selected')

    pipeline_script = egoallo_root / 'tools' / 'egobody' / 'run_egobody_ego_exo_pipeline.py'
    rows: list[dict[str, Any]] = []
    for start in starts:
        end = start + args.traj_length
        video_name = f'{args.recording}_{start:06d}_{end:06d}_{args.run_name}'
        cmd: list[str | Path] = [
            args.ego_python,
            pipeline_script,
            '--recording', args.recording,
            '--output-root', args.output_root,
            '--input-name', args.input_name,
            '--start-index', str(start),
            '--traj-length', str(args.traj_length),
            '--run-name', args.run_name,
        ]
        if args.run_optimize:
            cmd.append('--run-optimize')
        else:
            cmd.append('--no-run-optimize')
        if args.run_diagnose:
            cmd.append('--run-diagnose')
        else:
            cmd.append('--no-run-diagnose')
        if not args.run_egoallo:
            cmd.append('--no-run-egoallo')
        if args.force_egoallo:
            cmd.append('--force-egoallo')
        if args.force_gvhmr:
            cmd.append('--force-gvhmr')
        if args.force_export:
            cmd.append('--force-export')
        if args.force_optimize:
            cmd.append('--force-optimize')
        if args.dry_run:
            cmd.append('--dry-run')
        if args.extra_pipeline_args:
            cmd.extend(args.extra_pipeline_args)

        print('\n[window]', start, end)
        print(_cmd_to_str(cmd))
        if not args.dry_run:
            subprocess.run([str(x) for x in cmd], cwd=str(egoallo_root), check=True)

        metrics_stem = f'{video_name}_exo_world_optimized' if args.run_optimize else f'{video_name}_exo_world'
        metrics_path = traj_root / 'gvhmr_exo' / f'{metrics_stem}_reprojection_debug' / 'metrics.json'
        rows.append(_read_metric_row(metrics_path, start, end, video_name))

    summary = {
        'recording': args.recording,
        'run_name': args.run_name,
        'traj_length': args.traj_length,
        'stride': args.stride,
        'starts': starts,
        'num_windows': len(rows),
        'num_ok': sum(1 for r in rows if r.get('status') == 'ok'),
        'means': {},
        'windows': rows,
    }
    metric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (float, int)) and k not in {'start_index', 'end_index'}})
    for key in metric_keys:
        summary['means'][key] = _mean_ok(rows, key)

    out_dir = traj_root / 'pipeline_runs'
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f'{args.run_name}_sweep_summary.json'
    csv_path = out_dir / f'{args.run_name}_sweep_summary.csv'
    json_path.write_text(json.dumps(summary, indent=2))
    fieldnames = ['start_index', 'end_index', 'video_name', 'status'] + [k for k in metric_keys if k not in {'start_index', 'end_index'}] + ['metrics_path']
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print('\nSweep summary:')
    print(json_path)
    print(csv_path)
    print(json.dumps(summary['means'], indent=2))


if __name__ == '__main__':
    main()
