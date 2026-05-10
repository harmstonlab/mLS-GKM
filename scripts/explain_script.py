
#!/usr/bin/python
"""
    explain_script.py: explain model predictions with TF-MoDISco and summarise results.

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

import contextlib
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore")


def _silence_showwarning(*_args, **_kwargs):
    return None


warnings.showwarning = _silence_showwarning


@contextlib.contextmanager
def silence_stderr():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stderr(devnull):
            yield


@contextlib.contextmanager
def chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
bg_freqs = np.array([0.29, 0.21, 0.21, 0.29])


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str
    label: str


@dataclass(frozen=True)
class ExplainRecord:
    header: str
    class_scores: np.ndarray
    contrib: np.ndarray


def label_from_header(header: str) -> str:
    match = re.search(r"(?:^|\|)source=([^\|]+)", header)
    return match.group(1) if match else "NA"


def read_fasta(path: Path) -> List[FastaRecord]:
    records: List[FastaRecord] = []
    header: Optional[str] = None
    seq_chunks: List[str] = []

    with path.open("r", newline="") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_chunks)
                    records.append(FastaRecord(header=header, sequence=seq, label=label_from_header(header)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if header is not None:
        seq = "".join(seq_chunks)
        records.append(FastaRecord(header=header, sequence=seq, label=label_from_header(header)))

    return records


def seq_to_onehot(sequence: str) -> np.ndarray:
    arr = np.zeros((len(sequence), 4), dtype=np.int8)
    for i, ch in enumerate(sequence):
        base = ch.upper()
        if base == "N":
            continue
        arr[i, BASE_TO_INDEX[base]] = 1
    return arr


def _parse_contrib_field(field: str) -> np.ndarray:
    pos_strs = field.strip().split(";")
    out = np.empty((len(pos_strs), 4), dtype=np.float32)
    for i, pos in enumerate(pos_strs):
        vals = pos.split(",")
        if len(vals) != 4:
            raise ValueError(f"Expected 4 comma-separated values at position {i}, got {len(vals)}")
        out[i, 0] = float(vals[0])
        out[i, 1] = float(vals[1])
        out[i, 2] = float(vals[2])
        out[i, 3] = float(vals[3])
    return out


def pairwise_indices(class_count: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(class_count):
        for j in range(i + 1, class_count):
            pairs.append((i, j))
    return pairs


def pairwise_to_margins(scores_pairwise: np.ndarray, class_count: int) -> np.ndarray:
    margins = np.zeros(class_count, dtype=np.float32)
    for (i, j), score in zip(pairwise_indices(class_count), scores_pairwise.tolist()):
        margins[i] += score
        margins[j] -= score
    return margins


def pairwise_predict_index(scores_pairwise: np.ndarray, class_count: int) -> int:
    votes = np.zeros(class_count, dtype=np.int32)
    margins = np.zeros(class_count, dtype=np.float32)
    for (i, j), score in zip(pairwise_indices(class_count), scores_pairwise.tolist()):
        if score >= 0:
            votes[i] += 1
        else:
            votes[j] += 1
        margins[i] += score
        margins[j] -= score
    top_votes = np.max(votes)
    top = np.where(votes == top_votes)[0]
    if len(top) == 1:
        return int(top[0])
    return int(top[np.argmax(margins[top])])


def read_explain_tsv(
    path: Path,
    *,
    score_count: int,
    base_block_count: int,
    base_block_index: int,
) -> List[ExplainRecord]:
    records: List[ExplainRecord] = []
    with path.open("r", newline="") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"{path}: line {line_no}: expected >=3 tab-separated fields")
            header = parts[0]

            if len(parts) < 1 + score_count + base_block_count:
                raise ValueError(
                    f"{path}: line {line_no}: expected {score_count} score fields and {base_block_count} base-score fields"
                )

            score_fields = parts[1 : 1 + score_count]
            base_fields = parts[1 + score_count : 1 + score_count + base_block_count]

            try:
                scores = np.array([float(x) for x in score_fields], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_no}: non-numeric score field") from exc

            if base_block_index < 0 or base_block_index >= len(base_fields):
                raise ValueError(
                    f"{path}: line {line_no}: base_block_index={base_block_index} out of range for {len(base_fields)} base blocks"
                )

            contrib = _parse_contrib_field(base_fields[base_block_index])
            records.append(ExplainRecord(header=header, class_scores=scores, contrib=contrib))

    return records


def normalise_scores(impscores, hyp_impscores, onehot_data):
    normed_hyp = []
    normed_imp = []
    for imp, hyp, oh in zip(impscores, hyp_impscores, onehot_data):
        perpos_imp = np.sum(imp, axis=-1)
        perpos_sign = np.sign(perpos_imp)
        same_sign = (np.sign(hyp) * perpos_sign[:, None]) > 0
        denom = np.sum(hyp * same_sign, axis=-1)
        scale = np.zeros_like(denom, dtype=np.float32)
        ok = denom != 0
        scale[ok] = (perpos_imp[ok] / denom[ok]).astype(np.float32)
        hyp_n = hyp * scale[:, None]
        normed_hyp.append(hyp_n)
        normed_imp.append(hyp_n * oh)
    return normed_imp, normed_hyp


def null_perpos_from_explain_tsv(
    path: Path,
    *,
    score_count: int,
    base_block_count: int,
    base_block_index: int,
):
    recs = read_explain_tsv(
        path,
        score_count=score_count,
        base_block_count=base_block_count,
        base_block_index=base_block_index,
    )
    return [np.sum(r.contrib, axis=-1) for r in recs]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse LS-GKM gkmexplain multiclass output, run TF-MoDISco to identify patterns, and summarise results."
        )
    )
    parser.add_argument("--fasta", type=Path, required=True, help="FASTA file to explain")
    parser.add_argument("--explain", type=Path, required=True, help="gkmexplain output (pairwise score format)")
    parser.add_argument("--hyp-explain", type=Path, required=True, help="gkmexplain -m 1 output")
    parser.add_argument("--null-explain", type=Path, required=True, help="null/background gkmexplain output")
    parser.add_argument("--class-names", type=parse_csv_list, required=True, help="Comma-separated list of class names (in order of class indices)")
    parser.add_argument("--analyse-class-index", type=int, default=0)
    parser.add_argument(
        "--analyse-boundary-index",
        type=int,
        default=None,
        help="Score/boundary index (0-based, order i<j) to select base-score block for explanations.",
    )
    parser.add_argument(
        "--analyse-pair-index",
        type=int,
        default=None,
        help=(
            "Optional pairwise score index (0-based, order i<j). When set, report correlations against "
            "that pairwise score in addition to class margins."
        ),
    )
    parser.add_argument(
        "--pair-order",
        choices=["forward", "reverse"],
        default="forward",
        help="Direction for the selected boundary (forward=i_vs_j, reverse=j_vs_i).",
    )
    parser.add_argument(
        "--score-filter",
        choices=["none", "positive", "negative"],
        default="none",
        help="Optional filter on the selected boundary score before TF-MoDISco.",
    )
    parser.add_argument(
        "--score-quantile",
        type=float,
        default=0.1,
        help="Quantile fraction for score filtering (e.g. 0.1 keeps top/bottom 10%%).",
    )
    parser.add_argument("--window-size", type=int, default=20, help="window size for trimming TF-MoDISco patterns")
    parser.add_argument("--tfmodisco-target-seqlet-fdr", type=float, default=0.15, help="TF-MoDISco target seqlet FDR")
    parser.add_argument("--tfmodisco-kmer-len", type=int, default=6, help="TF-MoDISco k-mer length")
    parser.add_argument("--tfmodisco-num-gaps", type=int, default=1, help="TF-MoDISco number of gaps")
    parser.add_argument("--tfmodisco-num-mismatches", type=int, default=0, help="TF-MoDISco number of mismatches")
    parser.add_argument("--run-tag", type=str, default=None, help="tag to disambiguate TF-MoDISco figures")
    parser.add_argument("--drop-n", action="store_true", help="drop sequences containing N from FASTA")
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="output directory for saved plots")
    parser.add_argument("--threads", type=int, default=4, help="number of threads for TF-MoDISco (default=4)")
    return parser


def load_label_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def main() -> None:
    repo_root = Path.cwd()
    if not repo_root.joinpath("bin").exists() and repo_root.parent.joinpath("bin").exists():
        repo_root = repo_root.parent

    parser = build_parser(repo_root)
    args = parser.parse_args()

    class_names = args.class_names
    class_count = len(class_names)
    analyse_class_index = args.analyse_class_index
    analyse_boundary_index = args.analyse_boundary_index
    analyse_pair_index = args.analyse_pair_index
    pair_order = args.pair_order
    score_filter = args.score_filter
    score_quantile = args.score_quantile
    analyse_label = class_names[analyse_class_index] if analyse_class_index < len(class_names) else f"class{analyse_class_index}"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_records = read_fasta(args.fasta)
    if args.drop_n:
        fasta_records = [r for r in fasta_records if "N" not in r.sequence.upper()]
    print("FASTA records:", len(fasta_records))

    n_dec = class_count * (class_count - 1) // 2
    if analyse_boundary_index is None and analyse_pair_index is not None:
        analyse_boundary_index = analyse_pair_index
    if analyse_boundary_index is None:
        raise ValueError("--analyse-boundary-index is required")

    explain_records = read_explain_tsv(
        args.explain,
        score_count=n_dec,
        base_block_count=n_dec,
        base_block_index=analyse_boundary_index,
    )
    print("Explain records:", len(explain_records))

    fasta_by_header = {r.header: r for r in fasta_records}
    explain_by_header = {r.header: r for r in explain_records}

    missing_in_explain = sorted(set(fasta_by_header) - set(explain_by_header))
    missing_in_fasta = sorted(set(explain_by_header) - set(fasta_by_header))
    print("Missing in explain:", len(missing_in_explain))
    print("Missing in fasta:", len(missing_in_fasta))
    if missing_in_explain[:3]:
        print("  examples:", missing_in_explain[:3])
    if missing_in_fasta[:3]:
        print("  examples:", missing_in_fasta[:3])

    if missing_in_explain or missing_in_fasta:
        raise SystemExit("FASTA/explain headers are not aligned")

    headers = []
    seqs = []
    labels = []
    pairwise_scores_list = []
    contrib_raw = []
    onehot = []

    for fr in fasta_records:
        er = explain_by_header[fr.header]
        if er.contrib.shape[0] != len(fr.sequence):
            raise ValueError(
                f"Length mismatch for {fr.header}: fasta={len(fr.sequence)} explain={er.contrib.shape[0]}"
            )
        headers.append(fr.header)
        seqs.append(fr.sequence)
        labels.append(fr.label)
        pairwise_scores_list.append(er.class_scores)
        contrib_raw.append(er.contrib.astype(np.float32, copy=False))
        onehot.append(seq_to_onehot(fr.sequence))

    scores_pairwise = np.stack(pairwise_scores_list, axis=0)
    scores = np.stack(
        [pairwise_to_margins(row, class_count=class_count) for row in scores_pairwise],
        axis=0,
    )
    pair_labels = [
        f"{class_names[i] if i < len(class_names) else f'class{i}'}_vs_{class_names[j] if j < len(class_names) else f'class{j}'}"
        for i, j in pairwise_indices(class_count)
    ]
    print("Pairwise scores shape:", scores_pairwise.shape)
    print("Margins shape:", scores.shape)
    print("Analysing class:", analyse_label, "(index", analyse_class_index, ")")
    print("Analysing boundary:", pair_labels[analyse_boundary_index], "(index", analyse_boundary_index, ")")
    if analyse_pair_index is not None:
        if analyse_pair_index < 0 or analyse_pair_index >= len(pair_labels):
            raise ValueError(
                f"analyse_pair_index={analyse_pair_index} out of range for {len(pair_labels)} pairwise scores"
            )
        print("Analysing pairwise score:", pair_labels[analyse_pair_index], "(index", analyse_pair_index, ")")

    

    raw_is_hyp = any(
        np.any(np.sum(m != 0.0, axis=1) > 1)
        for m in contrib_raw[: min(10, len(contrib_raw))]
    )
    print("Contrib looks like:", "hypothetical" if raw_is_hyp else "importance")

    if raw_is_hyp:
        raise SystemExit("Expected raw contrib to be importance scores, but it looks like hypothetical scores. Please check your input files.")
    else:
        impscores = contrib_raw

    hyp_records = read_explain_tsv(
        args.hyp_explain,
        score_count=n_dec,
        base_block_count=n_dec,
        base_block_index=analyse_boundary_index,
    )
    hyp_by_header = {r.header: r for r in hyp_records}
    hyp_scores = []
    for h, seq in zip(headers, seqs):
        mat = hyp_by_header[h].contrib
        if mat.shape[0] != len(seq):
            raise ValueError(f"Length mismatch in HYP_EXPLAIN for {h}: fasta={len(seq)} explain={mat.shape[0]}")
        hyp_scores.append(mat.astype(np.float32, copy=False))
    print("Loaded hyp_scores:", len(hyp_scores), "for class", analyse_label)

    if hyp_scores is None:
        raise SystemExit("Hypothetical scores are required for normalisation and TF-MoDISco")
    if pair_order == "reverse":
        scores_pairwise[:, analyse_boundary_index] *= -1.0
        impscores = [(-1.0 * m) for m in impscores]
        hyp_scores = [(-1.0 * m) for m in hyp_scores]
    impscores, hyp_scores = normalise_scores(impscores, hyp_scores, onehot)
    print("Normalised", len(impscores), "sequences")

    total_imp = np.array([float(np.sum(m)) for m in impscores], dtype=np.float32)
    print("Total importance stats:", float(np.min(total_imp)), float(np.median(total_imp)), float(np.max(total_imp)))

    chosen = scores[:, analyse_class_index].astype(np.float32)
    if np.std(chosen) == 0 or np.std(total_imp) == 0:
        corr_chosen = float("nan")
    else:
        corr_chosen = float(np.corrcoef(total_imp, chosen)[0, 1])
    print(f"corr(total_imp, margin[{analyse_label}]): {corr_chosen:.4f}")

    if analyse_pair_index is not None:
        pair_vals = scores_pairwise[:, analyse_pair_index].astype(np.float32)
        if np.std(pair_vals) == 0 or np.std(total_imp) == 0:
            corr_pair = float("nan")
        else:
            corr_pair = float(np.corrcoef(total_imp, pair_vals)[0, 1])
        print(f"corr(total_imp, pairwise[{pair_labels[analyse_pair_index]}]): {corr_pair:.4f}")

    print("(Optional) correlations to other class margins:")
    for j in range(scores.shape[1]):
        x = scores[:, j].astype(np.float32)
        if np.std(x) == 0 or np.std(total_imp) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(total_imp, x)[0, 1])
        name = class_names[j] if j < len(class_names) else f"class{j}"
        print(f"  corr(total_imp, margin[{name}]): {corr:.4f}")

    if args.null_explain is None or not args.null_explain.exists():
        raise SystemExit("TF-MoDISco requires --null-explain")

    dnshuff_perposimp = null_perpos_from_explain_tsv(
        args.null_explain,
        score_count=n_dec,
        base_block_count=n_dec,
        base_block_index=analyse_boundary_index,
    )
    print("Loaded null sequences:", len(dnshuff_perposimp), "for label", analyse_label)

    def subset_by_label(target_label: str):
        idx = [i for i, lab in enumerate(labels) if lab == target_label]
        if not idx:
            raise ValueError(f"No sequences with label={target_label!r} in FASTA")
        if score_filter != "none":
            if score_quantile <= 0 or score_quantile > 1:
                raise ValueError("--score-quantile must be in (0, 1]")
            boundary_scores = scores_pairwise[:, analyse_boundary_index]
            score_subset = boundary_scores[idx]
            if score_filter == "positive":
                cutoff = float(np.quantile(score_subset, 1.0 - score_quantile))
                idx = [i for i in idx if boundary_scores[i] >= cutoff]
            elif score_filter == "negative":
                cutoff = float(np.quantile(score_subset, score_quantile))
                idx = [i for i in idx if boundary_scores[i] <= cutoff]
            if not idx:
                raise ValueError("No sequences left after score filtering")
        sub_imps = [impscores[i] for i in idx]
        sub_hyp = [hyp_scores[i] for i in idx]
        sub_oh = [onehot[i] for i in idx]
        return idx, sub_imps, sub_hyp, sub_oh

    idx, sub_imps, sub_hyp, sub_oh = subset_by_label(analyse_label)
    print(
        f"Running TF-MoDISco for label={analyse_label} (n={len(idx)}, n_null={len(dnshuff_perposimp)})"
        f" using class_index={analyse_class_index}", flush=True
    )
    with silence_stderr(), chdir(out_dir):
        import tensorflow as tf
        import modisco
        from modisco.aggregator import TrimToBestWindow
        from modisco.visualization import viz_sequence

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
            verbose=False,
            target_seqlet_fdr=args.tfmodisco_target_seqlet_fdr,
            seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                n_cores=args.threads,
                nn_n_jobs=args.threads,
                kmer_len=args.tfmodisco_kmer_len,
                num_gaps=args.tfmodisco_num_gaps,
                num_mismatches=args.tfmodisco_num_mismatches,
            ),
        )(
            task_names=["task0"],
            contrib_scores={"task0": sub_imps},
            hypothetical_contribs={"task0": sub_hyp},
            one_hot=sub_oh,
            null_per_pos_scores={"task0": dnshuff_perposimp},
        )

        sub = tfmodisco_results.metacluster_idx_to_submetacluster_results[0]

        def write_patterns_to_meme(pattern_list, meme_path, label_prefix="motif"):
            """
            Write a list of patterns to a MEME format file.
            """
            trimmer = TrimToBestWindow(window_size=args.window_size, track_names=["task0_contrib_scores"])
            with open(meme_path, "w") as f:
                # Write MEME file header
                f.write("MEME version 4\n\n")
                f.write("ALPHABET= ACGT\n\n")
                f.write("strands: + -\n\n")
                f.write("Background letter frequencies:\n")
                f.write(f"A {bg_freqs[0]:.4f} C {bg_freqs[1]:.4f} G {bg_freqs[2]:.4f} T {bg_freqs[3]:.4f}\n\n")
                for i, pattern in enumerate(trimmer(pattern_list)):
                    pwm = pattern["sequence"].fwd
                    motif_len = pwm.shape[0]
                    num_seqlets = len(pattern.seqlets)
                    # Convert to probabilities (normalise rows)
                    pwm = np.clip(pwm, 1e-6, None)
                    row_sums = pwm.sum(axis=1, keepdims=True)
                    pwm = pwm / row_sums
                    f.write(f"MOTIF {label_prefix}_{i}\n")
                    f.write(f"letter-probability matrix: alength= 4 w= {motif_len} nsites= {num_seqlets}\n")
                    for row in pwm:
                        f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
                    f.write("\n")

        def render_patterns(pattern_list, *, label: str, target_dir: Path):
            target_dir.mkdir(parents=True, exist_ok=True)
            if len(pattern_list) == 0:
                (target_dir / "NONE_IDENTIFIED").touch()
                print("No patterns identified; created marker in:", target_dir)
                return

            trimmer = TrimToBestWindow(window_size=args.window_size, track_names=["task0_contrib_scores"])
            # Try to get total number of input sequences for enrichment
            try:
                total_input_seqs = len(sub_oh)
            except Exception:
                total_input_seqs = None
            for i, pattern in enumerate(trimmer(pattern_list)):
                print("pattern", i, "num seqlets", len(pattern.seqlets))

                plt.figure(figsize=(8, 4))
                viz_sequence.plot_weights(viz_sequence.ic_scale(pattern["sequence"].fwd, background=bg_freqs))
                fwd_name = target_dir / f"{label}_{i}_FWD.png"
                plt.savefig(fwd_name, dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close()

                plt.figure(figsize=(8, 4))
                viz_sequence.plot_weights(viz_sequence.ic_scale(pattern["sequence"].rev, background=bg_freqs))
                rev_name = target_dir / f"{label}_{i}_REV.png"
                plt.savefig(rev_name, dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close()

                fwd_img = Image.open(fwd_name).convert("RGB")
                rev_img = Image.open(rev_name).convert("RGB")
                width = max(fwd_img.width, rev_img.width)
                title_h = 48
                gap = 8
                height = fwd_img.height + rev_img.height + (title_h * 2) + gap
                combined = Image.new("RGB", (width, height), color="white")
                draw = ImageDraw.Draw(combined)
                font = load_label_font(size=32)

                y = 0
                num_seqlets = len(pattern.seqlets)
                contrib_fwd = pattern["task0_contrib_scores"].fwd
                mean_contrib = contrib_fwd.mean()
                seqlet_indices = set(s.coor.example_idx for s in pattern.seqlets)
                num_unique_seqs_fwd = len(seqlet_indices)
                # Seqlet enrichment: fraction of unique input sequences with seqlets
                if total_input_seqs:
                    seqlet_enrichment = num_unique_seqs_fwd / total_input_seqs
                else:
                    seqlet_enrichment = float('nan')
                additional_text = (
                    f"Seqlets: {num_seqlets} "
                    f"Mean contrib: {mean_contrib:.3f} "
                    f"Unique seqs: {num_unique_seqs_fwd} "
                    f"Enrichment: {seqlet_enrichment:.2%}"
                )
                draw.text((8, y + 6), f"FWD", fill="black", font=font)
                draw.text((180, y + 6), additional_text, fill="black", font=font)
                y += title_h
                combined.paste(fwd_img, (0, y))
                y += fwd_img.height + gap
                draw.text((8, y + 6), "REV", fill="black", font=font)
                y += title_h
                combined.paste(rev_img, (0, y))

                out_name = target_dir / f"{label}_{i}.png"
                combined.save(out_name)
                print("Saved combined image:", out_name)
                os.unlink(fwd_name)
                os.unlink(rev_name)

        patterns = sub.seqlets_to_patterns_result.patterns
        merged_patterns = sub.seqlets_to_patterns_result.merged_patterns
        # Render images
        try:
            render_patterns(
                patterns,
                label=analyse_label,
                target_dir=out_dir / "patterns",
            )
        except:
            print("Error rendering patterns:")
            import traceback
            traceback.print_exc()
        try:
            render_patterns(
                merged_patterns,
                label=analyse_label,
                target_dir=out_dir / "merged_patterns",
            )
        except:
            print("Error rendering merged patterns:")
            import traceback
            traceback.print_exc()
        # Write MEME files
        meme_patterns_path = out_dir / f"{os.path.basename(out_dir)}_patterns.meme"
        meme_merged_path = out_dir / f"{os.path.basename(out_dir)}_merged_patterns.meme"
        try:
            write_patterns_to_meme(patterns, meme_patterns_path, label_prefix=f"{os.path.basename(out_dir)}_pattern")
        except:
            print("Error writing MEME for patterns:")
            import traceback
            traceback.print_exc()
        try:
            write_patterns_to_meme(merged_patterns, meme_merged_path, label_prefix=f"{os.path.basename(out_dir)}_merged")
        except:
            print("Error writing MEME for merged patterns:")
            import traceback
            traceback.print_exc()
        print(f"Wrote MEME file for patterns: {meme_patterns_path}")
        print(f"Wrote MEME file for merged_patterns: {meme_merged_path}")
    if out_dir.joinpath("figures").exists():
        for f in out_dir.joinpath("figures").iterdir():
            if f.is_file():
                os.rename(f, out_dir / f.name)
        out_dir.joinpath("figures").rmdir()


if __name__ == "__main__":
    main()
