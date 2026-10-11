import argparse
import os
import signal
import time
from pathlib import Path


SOLVER_NAMES = {
    "starter_linux64_gf",
    "starter_linux64_gf_sp",
    "engine_linux64_gf",
    "engine_linux64_gf_sp",
    "engine_linux64_gf_ompi",
    "engine_linux64_gf_ompi_sp",
}

CONVERTER_NAMES = {
    "anim_to_vtk_linux64_gf",
    "th_to_csv_linux64_gf",
}

DRIVER_MARKERS = {
    "by_openradioss.py",
    "openradioss_dataset_gen",
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def iter_processes(include_converters: bool, include_drivers: bool):
    current_pid = os.getpid()
    parent_pid = os.getppid()
    names = set(SOLVER_NAMES)
    if include_converters:
        names.update(CONVERTER_NAMES)

    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid in {current_pid, parent_pid}:
            continue

        comm = read_text(proc_dir / "comm").strip()
        cmdline = read_text(proc_dir / "cmdline").replace("\x00", " ").strip()
        if not cmdline:
            cmdline = comm

        exe_name = Path(cmdline.split()[0]).name if cmdline.split() else comm
        is_solver = comm in names or exe_name in names
        is_driver = include_drivers and any(marker in cmdline for marker in DRIVER_MARKERS)

        if is_solver or is_driver:
            yield {
                "pid": pid,
                "name": comm or exe_name,
                "cmdline": cmdline,
            }


def stop_process(pid: int, timeout: float) -> str:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already exited"
    except PermissionError as exc:
        return f"permission denied: {exc}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return "terminated"
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    except PermissionError as exc:
        return f"permission denied: {exc}"
    return "killed"


def main():
    parser = argparse.ArgumentParser(
        description="Stop running OpenCourant/OpenRadioss Linux processes in WSL."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching processes without stopping them.",
    )
    parser.add_argument(
        "--include-converters",
        action="store_true",
        help="Also stop anim_to_vtk and th_to_csv converter processes.",
    )
    parser.add_argument(
        "--include-drivers",
        action="store_true",
        help="Also stop Python dataset generation drivers.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Seconds to wait after SIGTERM before SIGKILL.",
    )
    args = parser.parse_args()

    processes = list(
        iter_processes(
            include_converters=args.include_converters,
            include_drivers=args.include_drivers,
        )
    )
    if not processes:
        print("No running OpenCourant processes found.")
        return

    for proc in processes:
        if args.dry_run:
            status = "Would stop"
        else:
            status = stop_process(proc["pid"], timeout=args.timeout)
        print(f"{status}: pid={proc['pid']} name={proc['name']}")
        print(f"  {proc['cmdline']}")


if __name__ == "__main__":
    main()
