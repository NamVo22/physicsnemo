from pathlib import Path
import argparse
import io
import json
import re
import zipfile

import torch


# ============================================================
# Paths
# ============================================================
CRASH_DIR = Path(__file__).resolve().parents[1]

# Default value nếu không truyền từ command line
DEFAULT_RUN_DIR = CRASH_DIR / "runs" / "tiny_case"
DEFAULT_CHECKPOINT_DIR = DEFAULT_RUN_DIR / "output" / "checkpoints"


def find_latest_checkpoint_pair(checkpoint_dir: Path):
    """
    Find the latest common epoch for:

        checkpoint.0.<epoch>.pt
        Transolver.0.<epoch>.mdlus
    """

    train_pattern = re.compile(r"checkpoint\.0\.(\d+)\.pt$")
    model_pattern = re.compile(r"Transolver\.0\.(\d+)\.mdlus$")

    train_files = {}
    model_files = {}

    for path in checkpoint_dir.glob("*"):
        m = train_pattern.match(path.name)
        if m:
            train_files[int(m.group(1))] = path

        m = model_pattern.match(path.name)
        if m:
            model_files[int(m.group(1))] = path

    common_epochs = sorted(set(train_files) & set(model_files))

    if not common_epochs:
        raise FileNotFoundError(
            f"No checkpoint pair found in:\n{checkpoint_dir}"
        )

    epoch = common_epochs[-1]

    return (
        epoch,
        train_files[epoch],
        model_files[epoch],
    )


# ============================================================
# Read training-state checkpoint
# ============================================================
def inspect_training_checkpoint(path: Path):

    print("\n" + "=" * 80)
    print("TRAINING CHECKPOINT")
    print("=" * 80)

    print("file :", path)
    print("size :", path.stat().st_size, "bytes")

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print("\nTop-level keys:")
    for key in ckpt.keys():
        print("  ", key)

    # --------------------------------------------------------
    # Epoch
    # --------------------------------------------------------
    if "epoch" in ckpt:
        print("\nepoch:")
        print("  ", ckpt["epoch"])

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    if "optimizer_state_dict" in ckpt:
        opt = ckpt["optimizer_state_dict"]

        print("\noptimizer_state_dict:")
        print("  keys:", list(opt.keys()))

        state = opt.get("state", {})
        param_groups = opt.get("param_groups", [])

        print("  number of parameter states:", len(state))
        print("  number of param groups     :", len(param_groups))

        for i, group in enumerate(param_groups):
            print(f"\n  param_group[{i}]:")

            for key, value in group.items():
                if key == "params":
                    print(f"    params: {len(value)} parameters")
                else:
                    print(f"    {key}: {value}")

        # Inspect one Adam state as example
        if len(state) > 0:
            first_key = next(iter(state))
            first_state = state[first_key]

            print("\n  Example optimizer state:")
            print("    parameter id:", first_key)

            for key, value in first_state.items():
                if torch.is_tensor(value):
                    print(
                        f"    {key:15s} "
                        f"shape={tuple(value.shape)}, "
                        f"dtype={value.dtype}"
                    )
                else:
                    print(f"    {key}: {value}")

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    if "scheduler_state_dict" in ckpt:
        print("\nscheduler_state_dict:")

        scheduler = ckpt["scheduler_state_dict"]

        for key, value in scheduler.items():
            print(f"  {key}: {value}")

    # --------------------------------------------------------
    # AMP GradScaler
    # --------------------------------------------------------
    if "scaler_state_dict" in ckpt:
        print("\nscaler_state_dict:")

        scaler = ckpt["scaler_state_dict"]

        if len(scaler) == 0:
            print("  empty")
        else:
            for key, value in scaler.items():
                print(f"  {key}: {value}")


# ============================================================
# Read PhysicsNeMo model checkpoint
# ============================================================
def inspect_model_checkpoint(path: Path):

    print("\n" + "=" * 80)
    print("MODEL CHECKPOINT (.mdlus)")
    print("=" * 80)

    print("file :", path)
    print("size :", path.stat().st_size, "bytes")

    with zipfile.ZipFile(path, "r") as archive:

        print("\nFiles inside .mdlus:")

        for name in archive.namelist():
            info = archive.getinfo(name)
            print(f"  {name:20s} {info.file_size:12d} bytes")

        # ----------------------------------------------------
        # args.json
        # ----------------------------------------------------
        args = json.loads(archive.read("args.json"))

        print("\nargs.json:")
        print(json.dumps(args, indent=2))

        # ----------------------------------------------------
        # metadata.json
        # ----------------------------------------------------
        metadata = json.loads(archive.read("metadata.json"))

        print("\nmetadata.json:")
        print(json.dumps(metadata, indent=2))

        # ----------------------------------------------------
        # model.pt
        # ----------------------------------------------------
        model_bytes = archive.read("model.pt")

    state_dict = torch.load(
        io.BytesIO(model_bytes),
        map_location="cpu",
        weights_only=False,
    )

    print("\n" + "-" * 80)
    print("MODEL STATE DICT")
    print("-" * 80)

    print("number of tensors:", len(state_dict))

    total_numel = 0

    for name, tensor in state_dict.items():

        numel = tensor.numel()
        total_numel += numel

        print(
            f"{name:70s} "
            f"shape={str(tuple(tensor.shape)):20s} "
            f"dtype={str(tensor.dtype):15s} "
            f"numel={numel}"
        )

    print("\nTotal number of stored parameters/buffers:")
    print(f"  {total_numel:,}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Crash case directory containing output/checkpoints.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing PhysicsNeMo checkpoint files",
    )

    args = parser.parse_args()

    CHECKPOINT_DIR = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else args.run_dir / "output" / "checkpoints"
    ).resolve()

    print("=" * 80)
    print("Crash checkpoint inspection")
    print("=" * 80)

    print("\ncheckpoint directory:")
    print(CHECKPOINT_DIR)

    epoch, training_ckpt, model_ckpt = find_latest_checkpoint_pair(
        CHECKPOINT_DIR
    )

    print("\nLatest common checkpoint:")
    print("  epoch :", epoch)
    print("  train :", training_ckpt.name)
    print("  model :", model_ckpt.name)

    inspect_training_checkpoint(training_ckpt)
    inspect_model_checkpoint(model_ckpt)
