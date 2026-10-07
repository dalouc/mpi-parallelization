# Distributed protein matching with MPI

Lab 2 of *Technological Fundamentals in the Big Data World*. A pattern is typed
at the keyboard and matched against the `sequence` field of every protein in
`proteins.csv`; the program reports how long the search took, charts the ten
proteins with the most occurrences, and names the best one.

| File | Role |
| ---- | ---- |
| `proteins-generator.py` | Dataset generator supplied with the lab. **Do not modify.** |
| `serial-proteins.py` | Serial version (parts 1–9 of the assignment). |
| `mpi-proteins.py` | Distributed version using `mpi4py` (parts 10–12). |
| `benchmark.py` | Drives both over every process count and plots the comparison. Development tool, not delivered. |
| `authors.txt` | One line per author. |

## Setup

`mpi4py` needs an MPI implementation on the system — Open MPI or MPICH — for
`mpiexec` to exist:

```bash
sudo pacman -S openmpi          # or: sudo apt install libopenmpi-dev openmpi-bin

python -m venv .venv
source .venv/bin/activate
python -m pip install --group dev --group lab
pre-commit install
```

## Running

Generate the dataset first. Seed 42 is what every number in the report was
measured with; 50,000 rows is plenty while developing.

```bash
python proteins-generator.py 5000000 42      # 670 MB, for the real timings
python proteins-generator.py 50000 42        # for development
```

Then run either version. Both prompt for the pattern, upper-case it, and print
the elapsed time before drawing the chart.

```bash
python serial-proteins.py                    # writes serial-time.txt
mpiexec -n 12 python mpi-proteins.py         # reads it back for the speedup
```

`serial-proteins.py` records its elapsed time in `serial-time.txt`, which
`mpi-proteins.py` reads so it can print the speedup itself. Run the serial
version once first, or the MPI version will say the baseline is missing.

Open MPI counts *physical cores* as slots and refuses to launch more processes
than that; `nproc` reports hardware threads, which can be higher.

## Benchmarking

```bash
python benchmark.py ABCD                     # -> benchmark.csv + two charts
```

`benchmark.py` runs the serial version and then the MPI version on every
process count up to the core limit, keeping the fastest of three runs each. It
writes `benchmark.csv` along with the time and speedup charts that the report
is built from. The report itself is kept outside this repository.

## Results

On an Intel Core i5-13500H (4 P-cores + 8 E-cores), 5,000,000 proteins
(670 MB), pattern `ABCD`:

| | 1 process | 4 | 12 |
| --- | --- | --- | --- |
| Serial | 1.126 s | | |
| MPI | 1.109 s | 0.302 s | **0.173 s** |
| Speedup | 1.02× | 3.73× | **6.51×** |

Scaling is near-ideal (93–99% efficiency) up to four processes and then settles
on a plateau, because the fifth rank is the first to land on a slower E-core
and the wall time of a parallel run is the time of its slowest process. The
report works this through with the per-rank measurements.

## How it works

Both programs share one `scan` function. It memory-maps the dataset and calls
`find` on the whole mapped region, jumping from one candidate row to the next
at the speed of a raw memory scan; only a row that actually contains the
pattern is split into fields and counted. Matches are kept in a bounded
ten-entry heap, so memory is constant even for a pattern that matches every
protein.

The MPI version hands each rank one contiguous byte range of the file and the
same `scan`. A rank owns the rows that *start* in its range, so the row
straddling a boundary is read by the rank below it and skipped by the rank
above — disjoint and complete with no communication. The dataset never travels
through MPI: the only messages are one `bcast` of the pattern, one `gather` of
ten triples per rank, and one `reduce` for the wall time.

## Delivery

The delivery is a single zip holding the report, the authors file and the two
programs — the report PDF is not tracked here, so point the command at wherever
you keep it:

```bash
zip <your-nia>_lab2_2026.zip \
    authors.txt serial-proteins.py mpi-proteins.py
zip <your-nia>_lab2_2026.zip -j path/to/lab2-report.pdf
```

## Development commands

| Task | Command |
| ---- | ------- |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Type-check | `pyright` |
| Every hook | `pre-commit run --all-files` |
