#!/usr/bin/env python3
"""Generate positive/negative FASTA pairs from one ENCODE bigBed dataset.

For an input ``*.bb`` file, this script writes outputs directly into a
user-provided output directory:

- ``positive.fa`` and ``negative.fa`` for all retained regions
- ``train_positive.fa`` and ``train_negative.fa`` for training regions
- ``test_positive.fa`` and ``test_negative.fa`` for test regions
- ``test_combined.fa`` for all test regions

Test regions are defined by chromosome (default: chr1 and chr2), and
training excludes those chromosomes.

Repeat-fraction is estimated from soft-masked genome FASTA input as the fraction
of lowercase bases in each sequence.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyBigWig
from tqdm import tqdm


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For one .bb dataset, create positive and GC/repeat-matched "
            "negative FASTA files, with train/test splits (memory-only FASTA)."
        )
    )
    parser.add_argument(
        "--input-bb",
        required=True,
        help="Path to one input *.bb file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder where FASTA files are written",
    )
    parser.add_argument(
        "--genome-fasta",
        required=True,
        help="Path to soft-masked reference genome FASTA (.fa/.fasta)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gc-tol",
        type=float,
        default=0.05,
        help=(
            "GC-content tolerance as absolute fraction difference "
            "(default: 0.05; e.g., pos GC=0.20 allows 0.15-0.25)"
        ),
    )
    parser.add_argument(
        "--repeat-tol",
        type=float,
        default=0.05,
        help=(
            "Repeat-fraction tolerance as absolute fraction difference "
            "(default: 0.05; e.g., pos rep=0.20 allows 0.15-0.25)"
        ),
    )
    parser.add_argument(
        "--max-tries-per-region",
        type=int,
        default=100000,
        help=(
            "Maximum random samples to try for each negative region "
            "(default: 100,000; use -1 to retry until a match is found)"
        ),
    )
    parser.add_argument(
        "--chrom-regex",
        default=r"^chr([0-9]+|X|Y)$",
        help=(
            "Regex for allowed chromosome names when sampling negatives "
            r"(default: '^chr([0-9]+|X|Y)$')"
        ),
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1000,
        help="Ignore positive regions longer than this length (default: 1000)",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=5000,
        help="Minimum number of retained regions required to proceed (default: 5000)",
    )
    parser.add_argument(
        "--test-chroms",
        default="chr1,chr2",
        help="Comma-separated chromosomes to use as test set (default: chr1,chr2)",
    )
    return parser.parse_args()


def bb_to_regions(bb_path: Path, max_len: int) -> List[Region]:
    regions: List[Region] = []
    bw = pyBigWig.open(str(bb_path))
    if bw is None:
        raise RuntimeError(f"Could not open bigBed file: {bb_path}")
    try:
        chrom_sizes = bw.chroms()
        for chrom, size in chrom_sizes.items():
            intervals = bw.entries(chrom, 0, size)
            if not intervals:
                continue
            for start, end, *_ in intervals:
                if end <= start:
                    continue
                if (end - start) > max_len:
                    continue
                regions.append(Region(chrom=chrom, start=start, end=end))
    finally:
        bw.close()
    return regions


def seq_metrics(seq: str) -> Tuple[float, float]:
    if not seq:
        return 0.0, 0.0

    valid = [b for b in seq if b.upper() in {"A", "C", "G", "T"}]
    if not valid:
        return 0.0, 0.0

    gc = sum(1 for b in valid if b.upper() in {"G", "C"}) / len(valid)
    repeat = sum(1 for b in valid if b.islower()) / len(valid)
    return gc, repeat


def load_fasta_into_memory(fasta_path: Path) -> Dict[str, str]:
    genome: Dict[str, str] = {}
    current_chrom: Optional[str] = None
    current_chunks: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_chrom is not None:
                    genome[current_chrom] = "".join(current_chunks)
                    current_chunks.clear()

                current_chrom = line[1:].split()[0]
                continue

            if current_chrom is None:
                raise ValueError(
                    "FASTA appears malformed: sequence data before header."
                )
            current_chunks.append(line)

    if current_chrom is not None:
        genome[current_chrom] = "".join(current_chunks)

    if not genome:
        raise ValueError(f"No FASTA records found in {fasta_path}")

    return genome


def build_weighted_chroms(
    genome: Dict[str, str],
    chrom_pattern: re.Pattern[str],
) -> List[Tuple[str, int]]:
    weighted: List[Tuple[str, int]] = []
    for chrom, seq in genome.items():
        if chrom_pattern.match(chrom):
            weighted.append((chrom, len(seq)))
    if not weighted:
        raise ValueError("No chromosomes matched --chrom-regex in the FASTA.")
    return weighted


def random_region_same_length(
    rng: random.Random,
    weighted_chroms: Sequence[Tuple[str, int]],
    length: int,
) -> Optional[Region]:
    valid = [
        (chrom, chrom_len) for chrom, chrom_len in weighted_chroms if chrom_len > length
    ]
    if not valid:
        return None

    total = sum(chrom_len for _, chrom_len in valid)
    pick = rng.randint(1, total)
    running = 0
    chosen_chrom = valid[0][0]
    chosen_len = valid[0][1]
    for chrom, chrom_len in valid:
        running += chrom_len
        if pick <= running:
            chosen_chrom = chrom
            chosen_len = chrom_len
            break

    start = rng.randint(0, chosen_len - length)
    end = start + length
    return Region(chrom=chosen_chrom, start=start, end=end)


def format_header(
    prefix: str, idx: int, region: Region, gc: float, repeat: float
) -> str:
    return (
        f">{prefix}_{idx}"
        f"|{region.chrom}:{region.start}-{region.end}"
        f"|len={region.length}|gc={gc:.4f}|rep={repeat:.4f}"
    )


def write_fasta(path: Path, records: Iterable[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(header + "\n")
            handle.write(seq + "\n")


def quiet_progress_points(total: int) -> set[int]:
    """Return milestone indices for quiet-mode progress logging.

    Milestones are 10% increments plus the first and last item.
    """
    if total <= 0:
        return set()

    points = {1, total}
    for pct in range(1, 100, 1):
        idx = max(1, int(total * pct / 100))
        points.add(idx)
    return points


def get_region_seq(genome: Dict[str, str], region: Region) -> str:
    seq = genome.get(region.chrom)
    if seq is None:
        return ""
    if region.start < 0 or region.end > len(seq) or region.end <= region.start:
        return ""
    return seq[region.start : region.end]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_bb = Path(args.input_bb)
    if not input_bb.exists():
        raise FileNotFoundError(f"Input .bb file not found: {input_bb}")

    genome_fasta = Path(args.genome_fasta)
    if not genome_fasta.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {genome_fasta}")

    chrom_pattern = re.compile(args.chrom_regex)
    test_chroms = {
        chrom.strip() for chrom in args.test_chroms.split(",") if chrom.strip()
    }
    if not test_chroms:
        raise ValueError("--test-chroms cannot be empty.")

    regions = bb_to_regions(input_bb, args.max_len)
    n_regions = len(regions)

    if n_regions < args.min_records:
        print(
            f"Stopping early: retained region count is below --min-records.\nn_regions={n_regions}, min_records={args.min_records}."
        )
        sys.exit(0)

    print("Loading FASTA into memory...")
    genome = load_fasta_into_memory(genome_fasta)
    weighted_chroms = build_weighted_chroms(genome, chrom_pattern)

    progress_points = quiet_progress_points(n_regions)

    positives_all: List[Tuple[str, str]] = []
    negatives_all: List[Tuple[str, str]] = []
    positives_train: List[Tuple[str, str]] = []
    negatives_train: List[Tuple[str, str]] = []
    positives_test: List[Tuple[str, str]] = []
    negatives_test: List[Tuple[str, str]] = []

    for i, region in enumerate(
        tqdm(regions, desc="Processing regions", disable=args.quiet), start=1
    ):
        pos_seq = get_region_seq(genome, region)
        if len(pos_seq) != region.length:
            raise RuntimeError(
                "Failed to extract positive sequence with expected length for "
                f"{region.chrom}:{region.start}-{region.end}."
            )

        pos_gc, pos_rep = seq_metrics(pos_seq)
        pos_record = (
            format_header("pos", i, region, pos_gc, pos_rep),
            pos_seq.upper(),
        )

        neg_region: Optional[Region] = None
        neg_seq = ""
        neg_gc = 0.0
        neg_rep = 0.0

        tries = 0
        while args.max_tries_per_region == -1 or tries < args.max_tries_per_region:
            tries += 1
            candidate = random_region_same_length(rng, weighted_chroms, region.length)
            if candidate is None:
                break

            seq = get_region_seq(genome, candidate)
            if len(seq) != region.length or "N" in seq.upper():
                continue

            gc, rep = seq_metrics(seq)
            if (
                abs(gc - pos_gc) <= args.gc_tol
                and abs(rep - pos_rep) <= args.repeat_tol
            ):
                neg_region = candidate
                neg_seq = seq.upper()
                neg_gc = gc
                neg_rep = rep
                break

        if neg_region is None:
            raise RuntimeError(
                "Failed to find matched negative region for "
                f"{input_bb.name} at positive index {i}. "
                "Consider increasing --max-tries-per-region or relaxing tolerances."
            )

        neg_record = (
            format_header("neg", i, neg_region, neg_gc, neg_rep),
            neg_seq,
        )

        positives_all.append(pos_record)
        negatives_all.append(neg_record)

        if region.chrom in test_chroms:
            positives_test.append(pos_record)
            negatives_test.append(neg_record)
        else:
            positives_train.append(pos_record)
            negatives_train.append(neg_record)

        if args.quiet and i in progress_points:
            pct_done = (100.0 * i) / n_regions
            print(f"Progress: {i}/{n_regions} regions ({pct_done:.1f}%)", flush=True)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_fasta(output_root / "positive.fa", positives_all)
    write_fasta(output_root / "negative.fa", negatives_all)
    write_fasta(output_root / "train_positive.fa", positives_train)
    write_fasta(output_root / "train_negative.fa", negatives_train)
    write_fasta(output_root / "test_positive.fa", positives_test)
    write_fasta(output_root / "test_negative.fa", negatives_test)
    write_fasta(output_root / "test_combined.fa", positives_test + negatives_test)

    print(
        f"Processed {input_bb.name}: "
        f"{n_regions} regions (<= {args.max_len} bp) -> {output_root}"
    )


if __name__ == "__main__":
    main()
