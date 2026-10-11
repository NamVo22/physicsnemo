import argparse
import json
import os
import subprocess
import sys


POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'

$includeLaunchers = [System.Convert]::ToBoolean($env:INCLUDE_RADIOSS_LAUNCHERS)

$solverPattern = '^(s|e)_20\d{2}\.\d+_win64.*|^radflex_20\d{2}.*'
$launcherPattern = 'hwsolver\.tcl.*-solver\s+RD'

$items = Get-CimInstance Win32_Process |
    Where-Object {
        ($_.Name -match $solverPattern) -or
        ($includeLaunchers -and ($_.CommandLine -match $launcherPattern))
    } |
    Select-Object ProcessId, Name, CommandLine

$result = @()

foreach ($item in $items) {
    $entry = [ordered]@{
        pid = $item.ProcessId
        name = $item.Name
        commandLine = $item.CommandLine
        stopped = $false
        error = $null
    }

    if ($env:DRY_RUN -ne '1') {
        try {
            Stop-Process -Id $item.ProcessId -Force -ErrorAction Stop
            $entry.stopped = $true
        }
        catch {
            $entry.error = $_.Exception.Message
        }
    }

    $result += [pscustomobject]$entry
}

$result | ConvertTo-Json -Depth 4
"""


def run_powershell(dry_run: bool, include_launchers: bool):
    env = {
        "DRY_RUN": "1" if dry_run else "0",
        "INCLUDE_RADIOSS_LAUNCHERS": "true" if include_launchers else "false",
    }

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        POWERSHELL_SCRIPT,
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )

    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)

    output = completed.stdout.strip()
    if not output:
        return []

    data = json.loads(output)
    if isinstance(data, dict):
        return [data]
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Stop running Altair Radioss solver processes on Windows."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching processes without stopping them.",
    )
    parser.add_argument(
        "--include-launchers",
        action="store_true",
        help="Also stop Altair hwsolver.tcl launcher processes for Radioss jobs.",
    )
    args = parser.parse_args()

    processes = run_powershell(
        dry_run=args.dry_run,
        include_launchers=args.include_launchers,
    )

    if not processes:
        print("No running Radioss solver processes found.")
        return

    action = "Would stop" if args.dry_run else "Stopped"
    for proc in processes:
        status = action if proc.get("stopped") or args.dry_run else "Failed"
        print(f"{status}: pid={proc['pid']} name={proc['name']}")
        if proc.get("error"):
            print(f"  error: {proc['error']}")


if __name__ == "__main__":
    main()
