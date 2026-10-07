"""Time the serial and MPI matchers and plot the comparison for the report.

Usage:
    python benchmark.py [pattern] [max-processes]

Runs `serial-proteins.py` once and `mpi-proteins.py` on every process count
from 1 up to the number of cores, keeping the fastest of `REPEATS` runs of
each, then writes `benchmark-times.png`, `benchmark-speedup.png` and
`benchmark.csv`.

This is a development tool for producing the figures of the report; it is not
part of the lab delivery. The two programs it drives are untouched by it -- it
only feeds them a pattern on stdin and reads the time they print. The programs
run with `MPLBACKEND=Agg` so that their `plt.show()` does not block.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SERIAL = "serial-proteins.py"
PARALLEL = "mpi-proteins.py"
TIMES_CHART = Path("benchmark-times.png")
SPEEDUP_CHART = Path("benchmark-speedup.png")
TABLE = Path("benchmark.csv")

# Runs per configuration. The fastest is kept: a slower run only ever means
# the machine was busy with something else, which is noise, not signal.
REPEATS = 3

# Not anchored to the start of a line: the `input()` prompt is flushed without
# a newline, so the elapsed time is printed right after it on the same line.
ELAPSED = re.compile(r"Elapsed time: ([0-9.]+) s")


def run(command: list[str], pattern: str) -> float:
    """Run `command`, feed it `pattern` on stdin and return its elapsed time.

    Args:
        command: Program to run, as an `argv` list.
        pattern: Pattern typed into the program's prompt.

    Returns:
        The number of seconds the program reported searching for.

    Raises:
        RuntimeError: If the program failed or printed no elapsed time.
    """
    done = subprocess.run(  # noqa: S603
        command,
        input=f"{pattern}\n",
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"MPLBACKEND": "Agg"},
    )
    if done.returncode != 0:
        message = f"{' '.join(command)} failed:\n{done.stderr}"
        raise RuntimeError(message)

    found = ELAPSED.search(done.stdout)
    if found is None:
        message = f"{' '.join(command)} printed no elapsed time:\n{done.stdout}"
        raise RuntimeError(message)
    return float(found.group(1))


def fastest(command: list[str], pattern: str, label: str) -> float:
    """Return the fastest of `REPEATS` runs of `command`.

    Args:
        command: Program to run, as an `argv` list.
        pattern: Pattern typed into the program's prompt.
        label: Name of the configuration, for the progress line.

    Returns:
        The lowest elapsed time of the runs, in seconds.
    """
    times = [run(command, pattern) for _ in range(REPEATS)]
    best = min(times)
    print(f"{label:>14}: {best:7.3f} s  (runs: {', '.join(f'{t:.3f}' for t in times)})")
    return best


def usable(mpiexec: str, limit: int) -> int:
    """Return the largest process count up to `limit` that `mpiexec` accepts.

    An MPI launcher refuses by default to put more processes on a node than it
    counts slots, and a slot is a physical core, not a hardware thread. Asking
    it is more reliable than guessing the topology.

    Args:
        mpiexec: Path to the MPI launcher.
        limit: Largest number of processes to try.

    Returns:
        The largest count that launched successfully, at least 1.
    """
    while limit > 1:
        probe = subprocess.run(  # noqa: S603
            [mpiexec, "-n", str(limit), sys.executable, "-c", ""],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            break
        limit -= 1
    return limit


def times_chart(processes: list[int], parallel: list[float], serial: float) -> None:
    """Plot elapsed time against process count and save it.

    Args:
        processes: Process counts measured.
        parallel: Elapsed time of the MPI version at each process count.
        serial: Elapsed time of the serial version.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(processes, parallel, "o-", color="indianred", label="MPI")
    plt.axhline(serial, color="steelblue", linestyle="--", label="serial")
    plt.xticks(processes)
    plt.ylim(bottom=0)
    plt.title("Search time against number of MPI processes")
    plt.xlabel("processes")
    plt.ylabel("elapsed time (s)")
    plt.legend()
    plt.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TIMES_CHART, dpi=150)


def speedup_chart(processes: list[int], speedups: list[float]) -> None:
    """Plot measured speedup against the linear ideal and save it.

    Args:
        processes: Process counts measured.
        speedups: Serial time divided by parallel time at each count.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(processes, speedups, "o-", color="indianred", label="measured")
    plt.plot(processes, processes, "--", color="grey", label="ideal (linear)")
    plt.xticks(processes)
    plt.title("Speedup of the MPI version over the serial version")
    plt.xlabel("processes")
    plt.ylabel("speedup")
    plt.legend()
    plt.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SPEEDUP_CHART, dpi=150)


def main() -> None:
    """Benchmark both programs and write the charts and table."""
    args = sys.argv[1:]
    pattern = args[0] if args else "ABCD"
    limit = int(args[1]) if len(args) > 1 else os.cpu_count() or 1

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        sys.exit("mpiexec not found: install an MPI implementation first.")

    limit = usable(mpiexec, limit)
    print(f'Pattern "{pattern}", up to {limit} processes, best of {REPEATS} runs:')
    serial = fastest([sys.executable, SERIAL], pattern, "serial")

    processes = list(range(1, limit + 1))
    parallel = [
        fastest(
            [mpiexec, "-n", str(n), sys.executable, PARALLEL], pattern, f"MPI n={n}"
        )
        for n in processes
    ]
    speedups = [serial / t for t in parallel]

    with TABLE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("processes", "seconds", "speedup", "efficiency"))
        for n, t, s in zip(processes, parallel, speedups, strict=True):
            writer.writerow((n, f"{t:.3f}", f"{s:.2f}", f"{s / n:.2f}"))

    times_chart(processes, parallel, serial)
    speedup_chart(processes, speedups)
    print(f"\nWrote {TABLE}, {TIMES_CHART} and {SPEEDUP_CHART}")


if __name__ == "__main__":
    main()
