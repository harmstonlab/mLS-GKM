#!/usr/bin/env python3
"""Build bigBed counts/list TSV files for downstream SLURM arrays.

Outputs:
1) bigbed_counts.tsv with columns: Filename, N_records
2) bigbed_list.tsv with columns: SLURM_ARRAY_TASK_ID, FILE, EXP_NAME
   where files are included only if N_records >= --min-records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pyBigWig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count bigBed entries and generate filtered TSV lists."
    )
    parser.add_argument(
        "--bigbed-dir",
        default="./bigbed",
        help="Directory containing .bb files (default: ./bigbed)",
    )
    parser.add_argument(
        "--counts-out",
        default="bigbed_counts.tsv",
        help="Output path for counts table (default: bigbed_counts.tsv)",
    )
    parser.add_argument(
        "--list-out",
        default="bigbed_list.tsv",
        help="Output path for filtered list table (default: bigbed_list.tsv)",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=5000,
        help="Minimum N_records needed to include a file in bigbed_list.tsv (default: 5000)",
    )
    return parser.parse_args()


def count_records_in_bigbed(bb_path: Path) -> Tuple[int, int]:
    bw = pyBigWig.open(str(bb_path))
    if bw is None:
        raise RuntimeError(f"Could not open bigBed file: {bb_path}")

    total = 0
    long_records = 0
    try:
        for chrom, size in bw.chroms().items():
            entries = bw.entries(chrom, 0, size)
            if entries:
                total += len(entries)
                for start, end, *_ in entries:
                    if (end - start) > 1000:
                        long_records += 1
    finally:
        bw.close()

    return total, long_records


def derive_exp_name(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) >= 3:
        return parts[2].split("_")[0]
    return Path(filename).stem


def write_counts_tsv(rows: List[Tuple[str, int]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("Filename\tN_records\n")
        for filename, n_records in rows:
            handle.write(f"{filename}\t{n_records}\n")


def write_list_tsv(rows: List[Tuple[str, int]], out_path: Path, min_records: int) -> int:
    kept = [(fn, n) for fn, n in rows if n >= min_records]

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("SLURM_ARRAY_TASK_ID\tFILE\tEXP_NAME\n")
        for idx, (filename, _) in enumerate(kept, start=1):
            exp_name = derive_exp_name(filename)
            handle.write(f"{idx}\t{filename}\t{exp_name}\n")

    return len(kept)


def main() -> None:
    args = parse_args()

    bigbed_dir = Path(args.bigbed_dir)
    if not bigbed_dir.exists() or not bigbed_dir.is_dir():
        raise FileNotFoundError(f"bigBed directory not found: {bigbed_dir}")

    bb_files = sorted(bigbed_dir.glob("*.bb"))
    if not bb_files:
        raise FileNotFoundError(f"No .bb files found in {bigbed_dir}")

    rows: List[Tuple[str, int]] = []
    files_with_long_records = 0
    total_records = 0
    total_long_records = 0
    for bb_file in bb_files:
        n_records, n_long_records = count_records_in_bigbed(bb_file)
        rows.append((bb_file.name, n_records))
        total_records += n_records
        if n_long_records > 0:
            files_with_long_records += 1
        total_long_records += n_long_records

    # Keep outputs deterministic: smallest count first, then filename.
    rows.sort(key=lambda item: (item[1], item[0]))

    counts_out = Path(args.counts_out)
    list_out = Path(args.list_out)
    write_counts_tsv(rows, counts_out)
    n_kept = write_list_tsv(rows, list_out, args.min_records)

    print(f"Processed {len(rows)} files from {bigbed_dir}")
    print(f"Wrote counts: {counts_out}")
    print(f"Wrote filtered list: {list_out} (kept {n_kept} files with N_records >= {args.min_records})")
    print(f"Files with records >1000bp: {files_with_long_records}")
    pct_long = (100.0 * total_long_records / total_records) if total_records > 0 else 0.0
    print(
        "Total records >1000bp across all files: "
        f"{total_long_records} / {total_records} ({pct_long:.2f}%)"
    )


if __name__ == "__main__":
    main()
