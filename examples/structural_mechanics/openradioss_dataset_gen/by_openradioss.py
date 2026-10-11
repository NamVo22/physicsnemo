from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

sys.path.insert(0, str(HERE))

from bumper_beam.generate_dataset import generate_dataset  # noqa: E402
from bumper_beam.restructure_global_features import convert_global_features  # noqa: E402
from common.radioss_runner import RunnerConfig, run_batch  # noqa: E402
from common.rename_d3plot import rename_all  # noqa: E402


def parse_float_spec(spec: str) -> list[float]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty value specification.")

    if ":" in spec:
        start_s, step_s, end_s = [p.strip() for p in spec.split(":")]
        start = Decimal(start_s)
        step = Decimal(step_s)
        end = Decimal(end_s)
        if step == 0:
            raise ValueError("Step cannot be zero.")
        if start < end and step < 0:
            raise ValueError("Step must be positive when start < end.")
        if start > end and step > 0:
            raise ValueError("Step must be negative when start > end.")

        values = []
        x = start
        if step > 0:
            while x <= end:
                values.append(float(x))
                x += step
        else:
            while x >= end:
                values.append(float(x))
                x += step
        return values

    return [float(p) for p in re.split(r"[,\s]+", spec.strip("[]")) if p]


def parse_optional_float_spec(spec: str) -> list[float | None]:
    spec = spec.strip()
    if spec.lower() in {"", "none", "default"}:
        return [None]
    if ";" in spec:
        values = []
        for item in spec.split(";"):
            item = item.strip()
            if item.lower() in {"", "none", "default"}:
                values.append(None)
            else:
                values.append(float(item))
        return values
    return parse_float_spec(spec)


def parse_vector_spec(spec: str) -> list[tuple[float, float, float]]:
    vectors = []
    for item in spec.split(";"):
        item = item.strip().strip("[]()")
        if not item:
            continue
        values = [float(p) for p in re.split(r"[,\s]+", item) if p]
        if len(values) != 3:
            raise ValueError(f"Expected 3 values per vector, got {len(values)}: {item}")
        vectors.append((values[0], values[1], values[2]))
    if not vectors:
        raise ValueError("Expected at least one vector.")
    return vectors


def parse_optional_vector_spec(spec: str) -> list[tuple[float, float, float] | None]:
    spec = spec.strip()
    if spec.lower() in {"", "none", "default"}:
        return [None]

    vectors = []
    for item in spec.split(";"):
        item = item.strip()
        if item.lower() in {"", "none", "default"}:
            vectors.append(None)
            continue
        values = parse_vector_spec(item)
        if len(values) != 1:
            raise ValueError(f"Expected one vector per item, got: {item}")
        vectors.append(values[0])
    return vectors


def deck_names(model_name: str) -> tuple[str, str]:
    return f"{model_name}_0000.rad", f"{model_name}_0001.rad"


def clean_dataset_dir(dataset_dir: Path):
    if not dataset_dir.exists():
        return
    for child in dataset_dir.iterdir():
        if child.is_dir() and child.name.lower().startswith("run"):
            shutil.rmtree(child)
        elif child.name in {"summary.json", "summary.csv"}:
            child.unlink()


def modify_engine_runtime(
    input_file: Path,
    stop_time: float | None,
    anim_dt: float | None,
):
    if stop_time is None and anim_dt is None:
        return

    lines = input_file.read_text(encoding="utf-8", errors="replace").splitlines(True)
    out = []
    expect_run_time = False
    expect_anim_dt = False

    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("/RUN/"):
            expect_run_time = True
            out.append(line)
            continue

        if stripped.upper().startswith("/ANIM/DT"):
            expect_anim_dt = True
            out.append(line)
            continue

        if expect_run_time and stripped and not stripped.startswith("#"):
            if stop_time is not None:
                line = f"{stop_time:20.6f}\n"
            expect_run_time = False

        if expect_anim_dt and stripped and not stripped.startswith("#"):
            if anim_dt is not None:
                parts = stripped.split()
                tstart = float(parts[0]) if parts else 0.0
                line = f"{tstart:10.6f}{anim_dt:10.6f}\n"
            expect_anim_dt = False

        out.append(line)

    input_file.write_text("".join(out), encoding="utf-8")


def apply_engine_smoke_overrides(
    dataset_dir: Path,
    model_name: str,
    stop_time: float | None,
    anim_dt: float | None,
):
    if stop_time is None and anim_dt is None:
        return
    _, engine_name = deck_names(model_name)
    for run_dir in sorted(dataset_dir.glob("run*")):
        engine_file = run_dir / engine_name
        if engine_file.exists():
            modify_engine_runtime(engine_file, stop_time=stop_time, anim_dt=anim_dt)


def run_curator(
    run_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    curator_root: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(REPO_ROOT / ".venv/bin/physicsnemo-curator-etl"),
        "--config-dir=examples/structural_mechanics/crash/config",
        "--config-name=crash_etl",
        "serialization_format=vtp",
        f"etl.source.input_dir={dataset_dir}",
        f"serialization_format.sink.output_dir={output_dir}",
        "etl.transformations.crash_transform.wall_threshold=-1.0",
        "etl.processing.num_processes=1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{curator_root}:{curator_root / 'examples'}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    print("\nRunning Curator:")
    print(" ".join(command))
    subprocess.run(command, cwd=curator_root, env=env, check=True)

    vtps = sorted(output_dir.glob("*.vtp"))
    if not vtps:
        raise RuntimeError(f"Curator completed but no VTP files were found in {output_dir}")
    print(f"Curator wrote {len(vtps)} VTP file(s) to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bumper data with OpenRadioss and prepare VTP files."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional campaign root. Used only as a default for raw-data-dir, "
            "vtp-output-dir, global-features-path, and template-dir."
        ),
    )
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    parser.add_argument("--vtp-output-dir", type=Path, default=None)
    parser.add_argument("--global-features-path", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="Bumper_Beam_AP_meshed")
    parser.add_argument("--geometry-scales", default="1.0,1.0,1.0")
    parser.add_argument("--velocities", default="none")
    parser.add_argument("--thickness-scales", required=True)
    parser.add_argument("--rwall-diameters", default="none")
    parser.add_argument("--rwall-origins", default="none")
    parser.add_argument("--openradioss-root", default=os.environ.get("OPENRADIOSS_ROOT"))
    parser.add_argument("--max-parallel-jobs", type=int, default=1)
    parser.add_argument("--num-threads", default=os.environ.get("OMP_NUM_THREADS", "2"))
    parser.add_argument("--engine-stop-time", type=float, default=None)
    parser.add_argument("--anim-dt", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-curator", action="store_true")
    parser.add_argument(
        "--stream-solver-output",
        action="store_true",
        help="Print Starter and Engine output to the terminal while also writing logs.",
    )
    parser.add_argument(
        "--curator-root",
        type=Path,
        default=REPO_ROOT.parent / "physicsnemo-curator",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir is not None else None
    if args.raw_data_dir is not None:
        dataset_dir = args.raw_data_dir.resolve()
    elif run_dir is not None:
        dataset_dir = run_dir / "raw_data"
    else:
        raise ValueError("Pass --raw-data-dir, or pass --run-dir as a fallback root.")

    template_dir = (
        args.template_dir.resolve()
        if args.template_dir is not None
        else (dataset_dir / "template" if run_dir is None else run_dir / "raw_data" / "template")
    )
    output_vtp_dir = (
        args.vtp_output_dir.resolve()
        if args.vtp_output_dir is not None
        else (run_dir / "data_vtp" if run_dir is not None else dataset_dir.parent / "data_vtp")
    )
    global_features_path = (
        args.global_features_path.resolve()
        if args.global_features_path is not None
        else (
            run_dir / "global_features.json"
            if run_dir is not None
            else dataset_dir.parent / "global_features.json"
        )
    )

    starter_name, engine_name = deck_names(args.model_name)
    starter_template = template_dir / starter_name
    engine_template = template_dir / engine_name

    if not starter_template.exists():
        raise FileNotFoundError(f"Missing starter template: {starter_template}")
    if not engine_template.exists():
        raise FileNotFoundError(f"Missing engine template: {engine_template}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clean_dataset_dir(dataset_dir)

    geometry_scales = parse_vector_spec(args.geometry_scales)
    velocities = parse_optional_vector_spec(args.velocities)
    thickness_scales = parse_float_spec(args.thickness_scales)
    rwall_diameters = parse_optional_float_spec(args.rwall_diameters)
    rwall_origins = parse_optional_vector_spec(args.rwall_origins)
    variations = {
        "geometry_scales": geometry_scales,
        "velocities": velocities,
        "thickness_scales": thickness_scales,
        "rwall_diameters": rwall_diameters,
        "rwall_origins": rwall_origins,
    }

    print("=" * 80)
    print("OPENRADIOSS BUMPER DATASET GENERATION")
    print("=" * 80)
    print(f"run_dir          : {run_dir}")
    print(f"raw_data_dir     : {dataset_dir}")
    print(f"vtp_output_dir   : {output_vtp_dir}")
    print(f"global_features  : {global_features_path}")
    print(f"template_dir     : {template_dir}")
    print(f"model_name       : {args.model_name}")
    print(f"geometry_scales  : {geometry_scales}")
    print(f"velocities       : {velocities}")
    print(f"thickness_scales : {thickness_scales}")
    print(f"rwall_diameters  : {rwall_diameters}")
    print(f"rwall_origins    : {rwall_origins}")
    print(f"openradioss_root : {args.openradioss_root}")
    print("=" * 80)

    generate_dataset(
        str(starter_template),
        str(engine_template),
        str(dataset_dir),
        variations,
    )
    apply_engine_smoke_overrides(
        dataset_dir,
        args.model_name,
        stop_time=args.engine_stop_time,
        anim_dt=args.anim_dt,
    )

    global_features_path.parent.mkdir(parents=True, exist_ok=True)
    convert_global_features(dataset_dir / "summary.json", global_features_path)

    if args.skip_run:
        print("Skipping OpenRadioss run by request.")
        return

    if not args.openradioss_root:
        raise RuntimeError(
            "OPENRADIOSS_ROOT is not set. Install/download OpenRadioss and pass "
            "--openradioss-root or set the OPENRADIOSS_ROOT environment variable."
        )

    cfg = RunnerConfig(
        openradioss_root=str(Path(args.openradioss_root).resolve()),
        dataset_dir=str(dataset_dir),
        input_base_name=args.model_name,
        max_parallel_jobs=args.max_parallel_jobs,
        omp_num_threads=str(args.num_threads),
        debug_mode=False,
        stream_solver_output=args.stream_solver_output,
    )
    run_batch(cfg)

    rename_all(dataset_dir, args.model_name)

    if not args.skip_curator:
        run_curator(
            run_dir=run_dir or dataset_dir.parent,
            dataset_dir=dataset_dir,
            output_dir=output_vtp_dir,
            curator_root=args.curator_root.resolve(),
        )


if __name__ == "__main__":
    main()
