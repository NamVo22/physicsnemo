import argparse
import json
import os
import re
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

# Same properties modified by the original Bumper Beam dataset generator.
TARGET_PROP_IDS = {2, 7}

# Licensed Altair Radioss installation on this Windows workstation.
DEFAULT_RADIOSS_STARTER = (
    r"C:\Program Files\Altair\2022.3\hwsolvers\radioss\bin\win64"
    r"\s_2022.3_win64.exe"
)
DEFAULT_RADIOSS_ENGINE = (
    r"C:\Program Files\Altair\2022.3\hwsolvers\radioss\bin\win64"
    r"\e_2022.3_win64.exe"
)
DEFAULT_ALTAIR_HOME = r"C:\Program Files\Altair\2022.3"
DEFAULT_RAD_CFG_PATH = (
    r"C:\Program Files\Altair\2022.3\hwsolvers\radioss\cfg"
)
DEFAULT_RADIOSS_PATH_PREFIX = (
    r"C:\Program Files\Altair\2022.3\hwsolvers\common\bin\win64;"
    r"C:\Program Files\Altair\2022.3\hwsolvers\radioss\bin\win64"
)

# Solver resources for ONE Radioss simulation.
DEFAULT_NP = 1
DEFAULT_NT = 8


def deck_names(model_name: str) -> tuple[str, str]:
    return f"{model_name}_0000.rad", f"{model_name}_0001.rad"


# =============================================================================
# Thickness specification
# =============================================================================

def parse_thickness_spec(spec: str) -> list[float]:
    """
    Supported formats:

        Explicit values:
            1.2,1.4,1.6,1.8,2.0
            [1.2, 1.4, 1.6]

        Range:
            start:step:end

        Example:
            1.2:0.2:2.0
        gives:
            [1.2, 1.4, 1.6, 1.8, 2.0]
    """

    spec = spec.strip()

    if not spec:
        raise ValueError("Thickness specification is empty.")

    # -------------------------------------------------------------------------
    # Range: start:step:end
    # -------------------------------------------------------------------------
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]

        if len(parts) != 3:
            raise ValueError(
                "Range format must be: start:step:end, "
                "for example 1.2:0.2:2.0"
            )

        start = Decimal(parts[0])
        step = Decimal(parts[1])
        end = Decimal(parts[2])

        if step == 0:
            raise ValueError("Thickness step cannot be zero.")

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

    # -------------------------------------------------------------------------
    # Explicit list
    # -------------------------------------------------------------------------
    spec = spec.strip("[]")

    parts = [
        p
        for p in re.split(r"[,\s]+", spec)
        if p.strip()
    ]

    values = [float(p) for p in parts]

    if not values:
        raise ValueError("No thickness values found.")

    return values


# =============================================================================
# Radioss deck modification
# =============================================================================

def modify_shell_thickness(
    input_file: Path,
    output_file: Path,
    thickness: float,
):
    """
    Modify shell thickness for /PROP/SHELL/2 and /PROP/SHELL/7.

    For the Altair Bumper Beam deck, the relevant block looks like:

        /PROP/SHELL/2
        ...
        #        N   Istrain               Thick ...
                 5         0                 1.8 ...

    Thickness occupies columns 31-40 in this specific deck format.
    """

    with input_file.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_prop_id = None
    thickness_line_next = False
    changed = {}

    output_lines = []

    for line in lines:

        stripped = line.strip()

        # ---------------------------------------------------------------------
        # Detect a new Radioss block
        # ---------------------------------------------------------------------
        if stripped.startswith("/"):
            current_prop_id = None
            thickness_line_next = False

            match = re.match(
                r"/PROP/SHELL/(\d+)",
                stripped,
                flags=re.IGNORECASE,
            )

            if match:
                current_prop_id = int(match.group(1))

            output_lines.append(line)
            continue

        # ---------------------------------------------------------------------
        # Find the comment describing the thickness data line
        # ---------------------------------------------------------------------
        if current_prop_id in TARGET_PROP_IDS:
            if (
                stripped.startswith("#")
                and "Istrain" in line
                and "Thick" in line
            ):
                thickness_line_next = True
                output_lines.append(line)
                continue

            # -----------------------------------------------------------------
            # Modify the next actual data line
            # -----------------------------------------------------------------
            if (
                thickness_line_next
                and stripped
                and not stripped.startswith("#")
            ):
                if len(line.rstrip("\n")) < 40:
                    raise RuntimeError(
                        f"Unexpected /PROP/SHELL/{current_prop_id} format."
                    )

                old_field = line[30:40]

                try:
                    old_value = float(old_field.strip())
                except ValueError as exc:
                    raise RuntimeError(
                        f"Cannot parse thickness in "
                        f"/PROP/SHELL/{current_prop_id}: {old_field!r}"
                    ) from exc

                new_field = f"{thickness:10.4f}"

                line = (
                    line[:30]
                    + new_field
                    + line[40:]
                )

                changed[current_prop_id] = {
                    "old": old_value,
                    "new": thickness,
                }

                thickness_line_next = False

        output_lines.append(line)

    # -------------------------------------------------------------------------
    # Safety check
    # -------------------------------------------------------------------------
    missing = TARGET_PROP_IDS - set(changed)

    if missing:
        raise RuntimeError(
            f"Could not modify PROP/SHELL IDs: {sorted(missing)}"
        )

    with output_file.open("w", encoding="utf-8") as f:
        f.writelines(output_lines)

    return changed


def modify_engine_runtime(
    input_file: Path,
    output_file: Path,
    stop_time: float | None,
    anim_dt: float | None,
):
    """Optionally shorten Radioss engine runtime/output frequency for smoke runs."""

    with input_file.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    output_lines = []
    expect_run_time = False
    expect_anim_dt = False
    changed = {}

    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("/RUN/"):
            expect_run_time = True
            output_lines.append(line)
            continue

        if stripped.upper().startswith("/ANIM/DT"):
            expect_anim_dt = True
            output_lines.append(line)
            continue

        if expect_run_time and stripped and not stripped.startswith("#"):
            if stop_time is not None:
                old_value = stripped.split()[0]
                line = f"{stop_time:20.6f}\n"
                changed["stop_time"] = {"old": old_value, "new": stop_time}
            expect_run_time = False

        if expect_anim_dt and stripped and not stripped.startswith("#"):
            if anim_dt is not None:
                parts = stripped.split()
                old_values = parts[:2]
                tstart = float(parts[0]) if parts else 0.0
                line = f"{tstart:10.6f}{anim_dt:10.6f}\n"
                changed["anim_dt"] = {"old": old_values, "new": [tstart, anim_dt]}
            expect_anim_dt = False

        output_lines.append(line)

    with output_file.open("w", encoding="utf-8") as f:
        f.writelines(output_lines)

    return changed


def windows_path(path: Path) -> str:
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_powershell_command(
    case_dir: Path,
    executable: str,
    input_name: str,
    np_domains: int,
    nt_threads: int,
    log_file: Path,
    use_np: bool,
):
    case_dir_win = windows_path(case_dir)
    np_arg = f" -np {np_domains}" if use_np else ""
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            f"$ErrorActionPreference='Stop'; "
            f"$env:ALTAIR_HOME='{DEFAULT_ALTAIR_HOME}'; "
            f"$env:RAD_CFG_PATH='{DEFAULT_RAD_CFG_PATH}'; "
            f"$env:PATH='{DEFAULT_RADIOSS_PATH_PREFIX};' + $env:PATH; "
            f"Set-Location -LiteralPath '{case_dir_win}'; "
            f"& '{executable}' -i '{input_name}'{np_arg} -nt {nt_threads}"
        ),
    ]

    print("\nRunning:")
    print(" ".join(command))
    print(f"cwd = {case_dir}")

    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(
            f"Radioss failed in {case_dir}\n"
            f"return code = {returncode}\n"
            f"See: {log_file}"
        )


def require_log_contains(path: Path, pattern: str, description: str):
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(pattern, text, flags=re.IGNORECASE) is None:
        raise RuntimeError(f"{description} not found in {path}")


def require_log_not_contains(path: Path, pattern: str, description: str):
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(pattern, text, flags=re.IGNORECASE) is not None:
        raise RuntimeError(f"{description} found in {path}")


# =============================================================================
# Run one Radioss case
# =============================================================================

def run_radioss(
    case_dir: Path,
    model_name: str,
    starter_exe: str,
    engine_exe: str,
    np_domains: int,
    nt_threads: int,
):
    """
    Run licensed Radioss.

    Run Starter and Engine explicitly. This is more predictable than the GUI
    wrapper for parameter sweeps.
    """

    starter_name, engine_name = deck_names(model_name)

    run_powershell_command(
        case_dir=case_dir,
        executable=starter_exe,
        input_name=starter_name,
        np_domains=np_domains,
        nt_threads=nt_threads,
        log_file=case_dir / "radioss_starter.log",
        use_np=True,
    )

    starter_out = case_dir / f"{model_name}_0000.out"
    if not starter_out.exists():
        raise RuntimeError(f"Starter output was not found: {starter_out}")
    require_log_contains(starter_out, r"\b0\s+ERROR\(S\)", "Starter 0 ERROR(S)")

    run_powershell_command(
        case_dir=case_dir,
        executable=engine_exe,
        input_name=engine_name,
        np_domains=np_domains,
        nt_threads=nt_threads,
        log_file=case_dir / "radioss_engine.log",
        use_np=False,
    )

    # -------------------------------------------------------------------------
    # Basic result check
    # -------------------------------------------------------------------------
    animation_files = sorted(
        case_dir.glob(f"{model_name}A*")
    )

    engine_out = case_dir / f"{model_name}_0001.out"

    if not engine_out.exists():
        raise RuntimeError(
            f"Radioss returned normally but Engine output was not found:\n"
            f"{engine_out}"
        )

    require_log_contains(engine_out, r"NORMAL\s+TERMINATION", "Engine normal termination")
    require_log_not_contains(engine_out, r"ERROR\s+TERMINATION", "Engine error termination")

    if not animation_files:
        raise RuntimeError(
            f"Radioss returned normally but no animation files were found "
            f"in {case_dir}"
        )

    return animation_files


# =============================================================================
# Generate dataset
# =============================================================================

def generate_dataset(
    run_dir: Path,
    model_name: str,
    thickness_values: list[float],
    starter_exe: str,
    engine_exe: str,
    np_domains: int,
    nt_threads: int,
    start_index: int,
    overwrite: bool,
    engine_stop_time: float | None,
    anim_dt: float | None,
):
    """
    Expected input structure:

        run_dir/
        └── raw_data/
            └── template/
                ├── <model_name>_0000.rad
                └── <model_name>_0001.rad

    Generated:

        run_dir/
        └── raw_data/
            ├── template/
            ├── Run001/
            ├── Run002/
            └── ...
    """

    run_dir = run_dir.resolve()

    raw_data_dir = run_dir / "raw_data"
    template_dir = raw_data_dir / "template"

    starter_name, engine_name = deck_names(model_name)
    starter_template = template_dir / starter_name
    engine_template = template_dir / engine_name

    # -------------------------------------------------------------------------
    # Validate directory structure
    # -------------------------------------------------------------------------
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Crash run directory does not exist:\n{run_dir}"
        )

    if not raw_data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing directory:\n{raw_data_dir}"
        )

    if not template_dir.is_dir():
        raise FileNotFoundError(
            f"Missing template directory:\n{template_dir}"
        )

    if not starter_template.is_file():
        raise FileNotFoundError(
            f"Missing Starter template:\n{starter_template}"
        )

    if not engine_template.is_file():
        raise FileNotFoundError(
            f"Missing Engine template:\n{engine_template}"
        )

    print("=" * 80)
    print("BUMPER BEAM DATASET GENERATION")
    print("=" * 80)
    print(f"Run directory : {run_dir}")
    print(f"Model name    : {model_name}")
    print(f"Template      : {template_dir}")
    print(f"Starter exe   : {starter_exe}")
    print(f"Engine exe    : {engine_exe}")
    print(f"NP            : {np_domains}")
    print(f"NT            : {nt_threads}")
    print(f"Engine stop   : {engine_stop_time}")
    print(f"Animation dt  : {anim_dt}")
    print(f"Thicknesses   : {thickness_values}")
    print(f"Number cases  : {len(thickness_values)}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Sequential simulations
    # -------------------------------------------------------------------------
    for local_idx, thickness in enumerate(thickness_values):

        run_number = start_index + local_idx
        run_name = f"Run{run_number:03d}"

        case_dir = raw_data_dir / run_name

        print()
        print("=" * 80)
        print(
            f"[{local_idx + 1}/{len(thickness_values)}] "
            f"{run_name}: thickness = {thickness}"
        )
        print("=" * 80)

        if case_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{case_dir} already exists.\n"
                    f"Delete it first or use --overwrite."
                )

            shutil.rmtree(case_dir)

        case_dir.mkdir(parents=True)

        # ---------------------------------------------------------------------
        # Create Starter deck with modified thickness
        # ---------------------------------------------------------------------
        starter_output = case_dir / starter_name

        changes = modify_shell_thickness(
            starter_template,
            starter_output,
            thickness,
        )

        # ---------------------------------------------------------------------
        # Engine deck is unchanged
        # ---------------------------------------------------------------------
        engine_changes = modify_engine_runtime(
            engine_template,
            case_dir / engine_name,
            stop_time=engine_stop_time,
            anim_dt=anim_dt,
        )

        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------
        metadata = {
            "run": run_name,
            "thickness": thickness,
            "modified_prop_shell_ids": sorted(TARGET_PROP_IDS),
            "changes": changes,
            "engine_changes": engine_changes,
        }

        with (case_dir / "case.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=4)

        print("Created case:")
        print(f"  {starter_output}")
        print(
            "  thickness:",
            ", ".join(
                f"PROP {k}: {v['old']} -> {v['new']}"
                for k, v in changes.items()
            ),
        )

        # ---------------------------------------------------------------------
        # Run licensed Radioss
        # ---------------------------------------------------------------------
        animation_files = run_radioss(
            case_dir=case_dir,
            model_name=model_name,
            starter_exe=starter_exe,
            engine_exe=engine_exe,
            np_domains=np_domains,
            nt_threads=nt_threads,
        )

        print(
            f"{run_name} completed successfully: "
            f"{len(animation_files)} animation file(s)"
        )

    print()
    print("=" * 80)
    print("ALL RADIOSs RUNS COMPLETED")
    print("=" * 80)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Bumper Beam crash data using licensed Radioss."
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Experiment directory, for example "
            ".../crash/runs/bumper"
        ),
    )

    parser.add_argument(
        "--h",
        required=True,
        help=(
            "Thickness values. "
            "Examples: '1.2,1.6,2.0' or '1.2:0.2:2.0'"
        ),
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Radioss model root name, e.g. Bumper_Beam_AP_meshed.",
    )

    parser.add_argument(
        "--radioss-starter",
        default=os.environ.get("RADIOSS_STARTER", DEFAULT_RADIOSS_STARTER),
        help="Path to Radioss Starter executable.",
    )

    parser.add_argument(
        "--radioss-engine",
        default=os.environ.get("RADIOSS_ENGINE", DEFAULT_RADIOSS_ENGINE),
        help="Path to Radioss Engine executable.",
    )

    parser.add_argument(
        "--np",
        type=int,
        default=DEFAULT_NP,
        help="Number of Radioss SPMD domains.",
    )

    parser.add_argument(
        "--nt",
        type=int,
        default=DEFAULT_NT,
        help="Number of threads per SPMD domain.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First Run index. Default: 1",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing RunXXX directories.",
    )

    parser.add_argument(
        "--engine-stop-time",
        type=float,
        default=None,
        help="Override Engine /RUN stop time. Useful for smoke tests.",
    )

    parser.add_argument(
        "--anim-dt",
        type=float,
        default=None,
        help="Override /ANIM/DT output interval.",
    )

    args = parser.parse_args()

    thickness_values = parse_thickness_spec(args.h)

    generate_dataset(
        run_dir=Path(args.run_dir),
        model_name=args.model_name,
        thickness_values=thickness_values,
        starter_exe=args.radioss_starter,
        engine_exe=args.radioss_engine,
        np_domains=args.np,
        nt_threads=args.nt,
        start_index=args.start_index,
        overwrite=args.overwrite,
        engine_stop_time=args.engine_stop_time,
        anim_dt=args.anim_dt,
    )


if __name__ == "__main__":
    main()
