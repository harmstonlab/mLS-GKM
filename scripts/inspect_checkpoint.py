
#!/usr/bin/python
"""
    inspect_checkpoint.py: inspect LS-GKM checkpoint files

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
import ctypes
import os
import struct
import sys
from dataclasses import dataclass
from typing import BinaryIO


TRAIN_MAGIC = b"LSGKM_TRAIN_CKPT\x00"
CV_MAGIC = b"LSGKM_CV_CKPT\x00"
TRAIN_MAGIC_TEXT = TRAIN_MAGIC.rstrip(b"\x00")
CV_MAGIC_TEXT = CV_MAGIC.rstrip(b"\x00")


class TrainHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_char * len(TRAIN_MAGIC)),
        ("version", ctypes.c_int32),
        ("l", ctypes.c_int32),
        ("nr_class", ctypes.c_int32),
        ("svm_type", ctypes.c_int32),
        ("kernel_type", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("k", ctypes.c_int32),
        ("d", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("norc", ctypes.c_int32),
        ("probability", ctypes.c_int32),
        ("shrinking", ctypes.c_int32),
        ("nr_weight", ctypes.c_int32),
        ("H", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("C", ctypes.c_double),
        ("eps", ctypes.c_double),
        ("perm_hash", ctypes.c_uint64),
    ]


class CvHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_char * len(CV_MAGIC)),
        ("version", ctypes.c_int32),
        ("l", ctypes.c_int32),
        ("nr_fold", ctypes.c_int32),
        ("nr_class", ctypes.c_int32),
        ("num_dec_values", ctypes.c_int32),
        ("icv", ctypes.c_int32),
        ("probability", ctypes.c_int32),
        ("svm_type", ctypes.c_int32),
        ("kernel_type", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("k", ctypes.c_int32),
        ("d", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("norc", ctypes.c_int32),
        ("shrinking", ctypes.c_int32),
        ("H", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("C", ctypes.c_double),
        ("eps", ctypes.c_double),
        ("perm_hash", ctypes.c_uint64),
        ("fold_hash", ctypes.c_uint64),
    ]


@dataclass
class PairRecord:
    i: int
    j: int
    alpha_len: int
    rho: float
    probA: float
    probB: float
    alpha: list[float] | None = None


@dataclass
class FoldItem:
    data_idx: int
    target: float
    dec_values: list[float]


@dataclass
class FoldRecord:
    fold_index: int
    fold_count: int
    items: list[FoldItem] | None = None


def read_exact(fp: BinaryIO, n: int) -> bytes:
    data = fp.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


def cstring(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def read_int32_array(fp: BinaryIO, n: int) -> list[int]:
    if n <= 0:
        return []
    data = read_exact(fp, 4 * n)
    return list(struct.unpack(f"={n}i", data))


def read_double_array(fp: BinaryIO, n: int) -> list[float]:
    if n <= 0:
        return []
    data = read_exact(fp, 8 * n)
    return list(struct.unpack(f"={n}d", data))


def detect_kind(path: str) -> str:
    with open(path, "rb") as fp:
        head = fp.read(max(len(TRAIN_MAGIC), len(CV_MAGIC)))
    if head.startswith(TRAIN_MAGIC_TEXT):
        return "train"
    if head.startswith(CV_MAGIC_TEXT):
        return "cv"
    return "unknown"


def parse_train(path: str, show_records: bool, max_records: int, show_all: bool) -> int:
    with open(path, "rb") as fp:
        header_raw = read_exact(fp, ctypes.sizeof(TrainHeader))
        header = TrainHeader.from_buffer_copy(header_raw)

        if cstring(bytes(header.magic)) != TRAIN_MAGIC_TEXT.decode("ascii"):
            print("Not a train checkpoint (magic mismatch)", file=sys.stderr)
            return 2

        labels = read_int32_array(fp, header.nr_class)
        counts = read_int32_array(fp, header.nr_class)

        weight_labels: list[int] = []
        weights: list[float] = []
        if header.nr_weight > 0:
            weight_labels = read_int32_array(fp, header.nr_weight)
            weights = read_double_array(fp, header.nr_weight)

        records: list[PairRecord] = []
        while True:
            prefix = fp.read(3 * 4 + 3 * 8)
            if not prefix:
                break
            if len(prefix) != (3 * 4 + 3 * 8):
                print("Truncated pair record header", file=sys.stderr)
                return 2

            i, j, alpha_len, rho, probA, probB = struct.unpack("=iiiddd", prefix)
            alpha = None
            if show_all:
                alpha = read_double_array(fp, alpha_len)
            else:
                _ = read_exact(fp, alpha_len * 8)
            records.append(PairRecord(i=i, j=j, alpha_len=alpha_len, rho=rho, probA=probA, probB=probB, alpha=alpha))

    print(f"File: {path}")
    print("Kind: train checkpoint")
    print(f"Magic: {cstring(bytes(header.magic))}")
    print(f"Version: {header.version}")
    print(f"l={header.l}, nr_class={header.nr_class}, nr_weight={header.nr_weight}")
    print(
        f"svm_type={header.svm_type}, kernel_type={header.kernel_type}, "
        f"L={header.L}, k={header.k}, d={header.d}, M={header.M}, norc={header.norc}"
    )
    print(
        f"probability={header.probability}, shrinking={header.shrinking}, "
        f"H={header.H}, gamma={header.gamma}, C={header.C}, eps={header.eps}"
    )
    print(f"perm_hash={header.perm_hash}")
    print(f"labels={labels}")
    print(f"counts={counts}")
    if weight_labels:
        print(f"weight_label={weight_labels}")
        print(f"weight={weights}")

    total_pairs = header.nr_class * (header.nr_class - 1) // 2
    print(f"Pair records: {len(records)} / {total_pairs}")

    if show_all and records:
        print("All pair records:")
        for rec in records:
            print(
                f"  ({rec.i},{rec.j}) len={rec.alpha_len} "
                f"rho={rec.rho:.6g} probA={rec.probA:.6g} probB={rec.probB:.6g}"
            )
            print(f"    alpha={rec.alpha}")
    elif show_records and records:
        n = min(max_records, len(records))
        print(f"Showing first {n} pair records:")
        for rec in records[:n]:
            print(
                f"  ({rec.i},{rec.j}) len={rec.alpha_len} "
                f"rho={rec.rho:.6g} probA={rec.probA:.6g} probB={rec.probB:.6g}"
            )

    return 0


def parse_cv(path: str, show_records: bool, max_records: int, show_all: bool) -> int:
    with open(path, "rb") as fp:
        header_raw = read_exact(fp, ctypes.sizeof(CvHeader))
        header = CvHeader.from_buffer_copy(header_raw)

        if cstring(bytes(header.magic)) != CV_MAGIC_TEXT.decode("ascii"):
            print("Not a CV checkpoint (magic mismatch)", file=sys.stderr)
            return 2

        records: list[FoldRecord] = []
        while True:
            prefix = fp.read(8)
            if not prefix:
                break
            if len(prefix) != 8:
                print("Truncated fold record header", file=sys.stderr)
                return 2
            fold_index, fold_count = struct.unpack("=ii", prefix)

            items = None
            if show_all:
                items = []
                for _ in range(fold_count):
                    data_idx = struct.unpack("=i", read_exact(fp, 4))[0]
                    target = struct.unpack("=d", read_exact(fp, 8))[0]
                    dec_values = read_double_array(fp, header.num_dec_values)
                    items.append(FoldItem(data_idx=data_idx, target=target, dec_values=dec_values))
            else:
                per_item_bytes = 4 + 8 + (8 * header.num_dec_values)
                _ = read_exact(fp, fold_count * per_item_bytes)
            records.append(FoldRecord(fold_index=fold_index, fold_count=fold_count, items=items))

    print(f"File: {path}")
    print("Kind: cv checkpoint")
    print(f"Magic: {cstring(bytes(header.magic))}")
    print(f"Version: {header.version}")
    print(
        f"l={header.l}, nr_fold={header.nr_fold}, nr_class={header.nr_class}, "
        f"num_dec_values={header.num_dec_values}, icv={header.icv}"
    )
    print(
        f"probability={header.probability}, svm_type={header.svm_type}, kernel_type={header.kernel_type}, "
        f"L={header.L}, k={header.k}, d={header.d}, M={header.M}, norc={header.norc}, shrinking={header.shrinking}"
    )
    print(f"H={header.H}, gamma={header.gamma}, C={header.C}, eps={header.eps}")
    print(f"perm_hash={header.perm_hash}, fold_hash={header.fold_hash}")

    done = sorted({rec.fold_index for rec in records})
    print(f"Fold records: {len(records)} (unique done folds: {done})")

    if show_all and records:
        print("All fold records:")
        for rec in records:
            print(f"  fold={rec.fold_index} count={rec.fold_count}")
            if rec.items:
                for item in rec.items:
                    print(f"    data_idx={item.data_idx} target={item.target:.6g} dec_values={item.dec_values}")
    elif show_records and records:
        n = min(max_records, len(records))
        print(f"Showing first {n} fold records:")
        for rec in records[:n]:
            print(f"  fold={rec.fold_index} count={rec.fold_count}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LS-GKM binary checkpoint files")
    parser.add_argument("checkpoint", help="Path to .train.ckpt or .cv.ckpt file")
    parser.add_argument("--records", action="store_true", help="Print record-level details")
    parser.add_argument("--all", action="store_true", help="Print all records and full payload data")
    parser.add_argument("--max-records", type=int, default=20, help="Max records to print with --records")
    args = parser.parse_args()

    show_records = args.records or args.all

    if not os.path.exists(args.checkpoint):
        print(f"File not found: {args.checkpoint}", file=sys.stderr)
        return 2

    kind = detect_kind(args.checkpoint)
    if kind == "train":
        return parse_train(args.checkpoint, show_records, args.max_records, args.all)
    if kind == "cv":
        return parse_cv(args.checkpoint, show_records, args.max_records, args.all)

    print("Unrecognized checkpoint magic. Not a supported LS-GKM checkpoint file.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
