#!/usr/bin/python
"""
    make_dnshuff_fasta.py: generate dinuc-shuffled FASTA files

    Copyright (C) 2026 Kieran Howard

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from Bio import SeqIO
from deeplift.dinuc_shuffle import dinuc_shuffle
from tqdm import tqdm


def _read_fasta_sequences(path: Path):
    return [record for record in SeqIO.parse(path, "fasta")]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate dinuc-shuffled FASTA (uses deeplift dinuc_shuffle).")
    p.add_argument("--in-fasta", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("."), type=Path)
    p.add_argument("--suffix", default=".shuffled.fasta", type=str)
    p.add_argument(
        "--mult",
        type=int,
        default=5,
        help="Number of dinuc-shuffled sequences to generate per input sequence in each class.",
    )
    p.add_argument("--classes", type=str, required=True, help="Comma-separated list of class labels to process.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    fasta_seqs = _read_fasta_sequences(args.in_fasta)
    if not fasta_seqs:
        raise SystemExit(f"No sequences found in {args.in_fasta}")
    
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    for class_label in classes:
        print(f"Class: {class_label}")
        class_records = [rec for rec in fasta_seqs if class_label in rec.description]
        num_sequences = len(class_records)
        if num_sequences == 0:
            print(f"No sequences found for class {class_label}, skipping...")
            continue
        if args.mult <= 0:
            raise SystemExit("--mult must be a positive integer")

        np.random.seed(args.seed)
        random.seed(args.seed)

        if not args.out_dir.exists():
            args.out_dir.mkdir(parents=True, exist_ok=True)
        outfile = (args.out_dir.joinpath(f"{class_label}{args.suffix}"))

        total = args.mult * num_sequences
        print(f"Input sequences for {class_label}: {num_sequences}; generating {total} shuffles ({args.mult}x each)")
        with outfile.open("w", encoding="utf-8", newline="\n") as out:
            out_i = 0
            for rec in tqdm(class_records, total=num_sequences):
                seq = str(rec.seq)
                for j in range(args.mult):
                    out.write(f">seq{out_i}|source={class_label}|orig={rec.id}|rep={j}\n")
                    out.write(dinuc_shuffle(seq))
                    out.write("\n")
                    out_i += 1
        print(f"Wrote {total} dinuc-shuffled sequences to {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
