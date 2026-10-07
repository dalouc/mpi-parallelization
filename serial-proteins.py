"""Serial protein matcher over the `proteins.csv` dataset.

Usage:
    python serial-proteins.py

Reads a pattern from the keyboard, counts how many times it occurs in the
`sequence` field of every protein, reports the elapsed search time, plots the
ten proteins with the most matches and prints the single best one.

The companion `mpi-proteins.py` splits exactly this search across MPI
processes: it reuses `scan` unchanged and only hands it a different byte range
per process. The elapsed time written to `serial-time.txt` is the baseline it
reports its speedup against.
"""

from __future__ import annotations

import heapq
import mmap
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

DATASET = Path("proteins.csv")
CHART = Path("serial-matches.png")
# Baseline the MPI version reads back to report its speedup.
TIMING = Path("serial-time.txt")

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
    memory: a short pattern matches millions of proteins, and all but ten of
    them can be dropped the moment they are read.

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
    `start` count as a boundary when it already is one. That rule is what lets
    `mpi-proteins.py` hand each process a byte range of its own and be sure
    every row is read by exactly one of them, with no communication at all.

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


def search(pattern: bytes) -> list[Match]:
    """Return the `TOP_N` proteins matching `pattern`, best first.

    The dataset is memory-mapped rather than read into a buffer: the pattern
    search then runs straight over the page cache, with none of the copying
    that `read` would do for every one of the 670 MB of a 5,000,000-row file.

    Args:
        pattern: Uppercase pattern to look for, as bytes.

    Returns:
        At most `TOP_N` `(occurrences, hydrofob, -protid)` triples, ordered by
        descending occurrences, then descending hydrophobicity, then ascending
        protein id.
    """
    with (
        DATASET.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        # The header row is not a protein: the search starts just after it.
        best = scan(data, pattern, data.find(NEWLINE) + 1, len(data))
    return sorted(best, reverse=True)


def barchart(ranked: list[Match], pattern: str, path: Path) -> None:
    """Plot the proteins in `ranked` as a barchart and save it to `path`.

    Args:
        ranked: Matching proteins, best first, as returned by `search`.
        pattern: The pattern that was searched for, shown in the title.
        path: File the figure is written to, for inclusion in the report.
    """
    ids = [str(-protid) for _, _, protid in ranked]
    occurrences = [count for count, _, _ in ranked]

    plt.figure(figsize=(10, 5))
    plt.bar(ids, occurrences, color="steelblue")
    plt.title(f'Top {len(ranked)} proteins matching "{pattern}"')
    plt.xlabel("protein id")
    plt.ylabel("occurrences")
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def report(ranked: list[Match], pattern: str, chart: Path) -> None:
    """Print the ranking and the best protein, and build the barchart.

    Args:
        ranked: Matching proteins, best first, as returned by `search`.
        pattern: The pattern that was searched for.
        chart: File the barchart is written to.
    """
    if not ranked:
        print(f'No protein sequence contains "{pattern}".')
        return

    print(f'\nTop {len(ranked)} proteins matching "{pattern}":')
    print(f"{'protid':>10} {'occurrences':>12} {'hydrofob':>9}")
    for count, hydrofob, protid in ranked:
        print(f"{-protid:>10} {count:>12} {hydrofob:>9}")

    barchart(ranked, pattern, chart)
    print(f"Barchart saved to {chart}")

    count, hydrofob, protid = ranked[0]
    print(
        f"\nProtein with most occurrences: id {-protid} "
        f"with {count} occurrences (hydrofob {hydrofob})"
    )


def main() -> None:
    """Search the dataset for a pattern read from the keyboard."""
    pattern = input("Pattern to search: ").strip().upper()
    if not pattern:
        sys.exit("The pattern cannot be empty.")

    begin = time.perf_counter()
    ranked = search(pattern.encode())
    elapsed = time.perf_counter() - begin

    print(f"Elapsed time: {elapsed:.3f} s")
    TIMING.write_text(f"{elapsed:.6f}\n")

    report(ranked, pattern, CHART)
    plt.show()


if __name__ == "__main__":
    main()
