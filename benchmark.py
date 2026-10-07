"""Time the serial and MPI matchers and print the comparison.

Usage:
    python benchmark.py [pattern] [max-processes]

Runs `serial-proteins.py` once and `mpi-proteins.py` on every process count
from 1 up to the number of cores, keeping the fastest of `REPEATS` runs of
each, and prints a table of elapsed times, speedups and parallel efficiencies.

The two programs are untouched by this: it only feeds them a pattern on stdin
and reads back the time they print. They run with `MPLBACKEND=Agg` so that
their `plt.show()` returns instead of blocking on a window.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

SERIAL = "serial-proteins.py"
PARALLEL = "mpi-proteins.py"

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


def fastest(command: list[str], pattern: str) -> float:
    """Return the fastest of `REPEATS` runs of `command`.

    Args:
        command: Program to run, as an `argv` list.
        pattern: Pattern typed into the program's prompt.

    Returns:
        The lowest elapsed time of the runs, in seconds.
    """
    return min(run(command, pattern) for _ in range(REPEATS))


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


def main() -> None:
    """Benchmark both programs and print the comparison."""
    args = sys.argv[1:]
    pattern = args[0] if args else "ABCD"
    limit = int(args[1]) if len(args) > 1 else os.cpu_count() or 1

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        sys.exit("mpiexec not found: install an MPI implementation first.")

    limit = usable(mpiexec, limit)
    print(f'Pattern "{pattern}", best of {REPEATS} runs per configuration.\n')

    serial = fastest([sys.executable, SERIAL], pattern)
    print(f"{'serial':>12}  {serial:7.3f} s")
    print(f"\n{'processes':>12}  {'time':>9}  {'speedup':>8}  {'efficiency':>10}")

    for size in range(1, limit + 1):
        elapsed = fastest([mpiexec, "-n", str(size), sys.executable, PARALLEL], pattern)
        gain = serial / elapsed
        print(f"{size:>12}  {elapsed:7.3f} s  {gain:7.2f}x  {gain / size:9.0%}")


if __name__ == "__main__":
    main()
