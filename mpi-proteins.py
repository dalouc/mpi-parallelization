"""Distributed protein matcher over the `proteins.csv` dataset, using MPI.

Usage:
    mpiexec -n <processes> python mpi-proteins.py

The parallel counterpart of `serial-proteins.py`. Rank 0 reads the pattern from
the keyboard and broadcasts it; every rank then searches one contiguous byte
range of the dataset and keeps only its ten best proteins, which are gathered
on rank 0 and merged into the global ranking.

The dataset itself never travels through MPI: every rank memory-maps the file
and scans its own range of it, so the only messages exchanged are the pattern
and ten triples per rank. The program reports its own elapsed time; comparing
it against the serial version is what `benchmark.py` is for.
"""

from __future__ import annotations

import heapq
import mmap
import sys
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
from mpi4py import MPI

DATASET = Path("proteins.csv")

# Proteins shown in the barchart and named in the ranking.
TOP_N = 10

NEWLINE = b"\n"
COMMA = b","

# A matching protein, ordered so that plain tuple comparison ranks it: more
# occurrences first, then higher hydrophobicity, then the lower id. Storing the
# id negated is what makes that last tie-break fall out of the comparison too.
Match = tuple[int, int, int]


def keep(best: list[Match], match: Match) -> None:
    """Add `match` to `best`, a bounded min-heap of the `TOP_N` best proteins.

    Keeping only `TOP_N` entries is what makes the search run in constant
    memory, and it is also what keeps the gather cheap: a rank sends ten
    triples however many proteins its range matched.

    Args:
        best: Min-heap of the best proteins seen so far, modified in place.
        match: Candidate `(occurrences, hydrofob, -protid)` triple.
    """
    if len(best) < TOP_N:
        heapq.heappush(best, match)
    elif match > best[0]:
        heapq.heapreplace(best, match)


def scan(data: mmap.mmap, pattern: bytes, start: int, stop: int) -> list[Match]:
    """Return the best proteins among the rows of `data` starting in the range.

    `data.find` jumps straight from one candidate row to the next, at the speed
    of a raw memory scan, so the rows that do not match -- the vast majority --
    are never split into fields. Only a candidate is parsed, and its
    occurrences are then counted in the `sequence` field alone, which is what
    keeps a hit inside `protid` or `hydrofob` from being counted.

    A row belongs to the range when it *starts* in it. The scan therefore runs
    from the first row boundary at or after `start` to the end of the row
    holding the byte before `stop`; searching from `start - 1` is what makes
    `start` count as a boundary when it already is one. That rule is the whole
    of the orchestration: the row straddling a boundary is read by the rank
    below it and skipped by the rank above it, so the ranges are disjoint and
    complete without a single message being exchanged.

    Args:
        data: The whole dataset, memory-mapped read-only.
        pattern: Uppercase pattern to look for, as bytes.
        start: First byte of the range; must be 1 or greater.
        stop: One past the last byte of the range.

    Returns:
        At most `TOP_N` `(occurrences, hydrofob, -protid)` triples, as an
        unordered heap.
    """
    begin = data.find(NEWLINE, start - 1)
    begin = len(data) if begin < 0 else begin + 1
    limit = data.find(NEWLINE, stop - 1)
    limit = len(data) if limit < 0 else limit + 1

    best: list[Match] = []
    pos = data.find(pattern, begin, limit)
    while pos >= 0:
        row = data.rfind(NEWLINE, 0, pos) + 1
        end = data.find(NEWLINE, pos)
        if end < 0:
            end = len(data)
        protid, _enzyme, hydrofob, sequence = data[row:end].split(COMMA)
        occurrences = sequence.count(pattern)
        if occurrences:
            keep(best, (occurrences, int(hydrofob), -int(protid)))
        # Nothing left to find in this row: resume after its last byte.
        pos = data.find(pattern, end, limit)
    return best


def search(pattern: bytes, rank: int, size: int) -> list[Match]:
    """Return the best proteins matching `pattern` in this rank's byte range.

    The dataset is split into `size` ranges of equal byte length rather than of
    equal row count. Rows vary in length only within a bounded range, so equal
    byte shares balance the work, and a rank can compute its own share from the
    file size alone -- there is nothing to scatter.

    Memory-mapping rather than reading is what makes that split pay off: the
    ranks scan one shared copy of the page cache in place, instead of each
    copying its share of the 670 MB of a 5,000,000-row file into a buffer of
    its own. Copying is pure memory traffic, and memory bandwidth is the one
    resource the cores cannot have more of by being more numerous.

    Args:
        pattern: Uppercase pattern to look for, as bytes.
        rank: Id of this process, selecting which range it searches.
        size: Total number of processes the dataset is split between.

    Returns:
        At most `TOP_N` `(occurrences, hydrofob, -protid)` triples, as an
        unordered heap.
    """
    with (
        DATASET.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        # The header row is not a protein: the ranges start just after it.
        first = data.find(NEWLINE) + 1
        span = len(data) - first
        return scan(
            data,
            pattern,
            first + span * rank // size,
            first + span * (rank + 1) // size,
        )


def barchart(ranked: list[Match], pattern: str) -> None:
    """Plot the proteins in `ranked` as a barchart.

    Args:
        ranked: Matching proteins, best first, as merged on rank 0.
        pattern: The pattern that was searched for, shown in the title.
    """
    ids = [str(-protid) for _, _, protid in ranked]
    occurrences = [count for count, _, _ in ranked]

    plt.figure(figsize=(10, 5))
    plt.bar(ids, occurrences, color="indianred")
    plt.title(f'Top {len(ranked)} proteins matching "{pattern}" (MPI)')
    plt.xlabel("protein id")
    plt.ylabel("occurrences")
    plt.tight_layout()


def report(ranked: list[Match], pattern: str) -> None:
    """Print the ranking and the best protein, and build the barchart.

    Args:
        ranked: Matching proteins, best first, as merged on rank 0.
        pattern: The pattern that was searched for.
    """
    if not ranked:
        print(f'No protein sequence contains "{pattern}".')
        return

    print(f'\nTop {len(ranked)} proteins matching "{pattern}":')
    print(f"{'protid':>10} {'occurrences':>12} {'hydrofob':>9}")
    for count, hydrofob, protid in ranked:
        print(f"{-protid:>10} {count:>12} {hydrofob:>9}")

    barchart(ranked, pattern)

    count, hydrofob, protid = ranked[0]
    print(
        f"\nProtein with most occurrences: id {-protid} "
        f"with {count} occurrences (hydrofob {hydrofob})"
    )


def main() -> None:
    """Search the dataset in parallel for a pattern read from the keyboard."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 has a keyboard, and every rank needs the pattern: broadcast
    # it. `None` travels just as well as a string, so an empty pattern is a
    # value every rank can agree to stop on -- no rank is left waiting.
    pattern = input("Pattern to search: ").strip().upper() if rank == 0 else None
    pattern = comm.bcast(pattern, root=0)
    if not pattern:
        if rank == 0:
            sys.exit("The pattern cannot be empty.")
        return

    # No rank may start its clock while another is still waiting on `input`.
    comm.Barrier()
    begin = MPI.Wtime()

    best = search(pattern.encode(), rank, size)

    # Ten triples per rank, so the gather itself costs nothing measurable. It
    # also synchronises the ranks, which is why the elapsed times reduced just
    # after it all cover the whole search, and their maximum is the wall time.
    gathered = comm.gather(best, root=0)
    elapsed = comm.reduce(MPI.Wtime() - begin, op=MPI.MAX, root=0)

    if rank != 0 or gathered is None or elapsed is None:
        return

    # Only a rank's own ten best can reach the global ten best, so merging the
    # gathered heaps yields exactly the ranking the serial version computes.
    ranked = heapq.nlargest(TOP_N, chain.from_iterable(gathered))

    print(f"Elapsed time: {elapsed:.3f} s on {size} processes")
    report(ranked, pattern)
    plt.show()


if __name__ == "__main__":
    main()
