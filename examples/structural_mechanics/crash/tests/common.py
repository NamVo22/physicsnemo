from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

CRASH_DIR = Path(__file__).resolve().parents[1]
if str(CRASH_DIR) not in sys.path:
    sys.path.insert(0, str(CRASH_DIR))

DEFAULT_RUN_DIR = CRASH_DIR / "runs" / "tiny_case"


def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Crash case directory containing raw_data, data_vtp, and output.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-time-steps", type=int, default=11)
    parser.add_argument("--dt", type=float, default=0.0001)
    return parser


def resolve_run_paths(run_dir: Path) -> dict[str, Path]:
    run_dir = run_dir.expanduser().resolve()
    return {
        "run_dir": run_dir,
        "raw_data": run_dir / "raw_data",
        "data_vtp": run_dir / "data_vtp",
        "output": run_dir / "output",
        "checkpoints": run_dir / "output" / "checkpoints",
    }


def make_point_cloud_dataset(run_dir: Path, num_samples: int, num_time_steps: int, dt: float):
    from datapipe import CrashPointCloudDataset
    from vtp_reader import Reader

    paths = resolve_run_paths(run_dir)
    return CrashPointCloudDataset(
        name=paths["run_dir"].name,
        reader=Reader(),
        data_dir=str(paths["data_vtp"]),
        split="train",
        num_samples=num_samples,
        num_steps=num_time_steps,
        features=["thickness"],
        dt=dt,
    )


def make_single_sample_loader(dataset):
    from datapipe import simsample_collate

    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=False,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
        sampler=sampler,
        collate_fn=simsample_collate,
    )
