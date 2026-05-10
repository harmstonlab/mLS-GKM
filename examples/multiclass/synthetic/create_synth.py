
import os
import random

N_SEQS = 15000
BP_LEN = 500

CLASSES = ["A", "B", "C"]

TRAIN_FRAC = 0.7

MIN_MOTIFS = 1
MAX_MOTIFS = 3

OVERLAPPING_MOTIFS = False

SEED = 42

BG_FREQs = {"A":0.3, "C":0.2, "G":0.2, "T":0.3}

MOTIF_A = "GTAAACA"  # Simplified FOXK2, MA1103.2 - https://jaspar.elixir.no/matrix/MA1103.2/
MOTIF_B = "GGGGAGGGG"  # Simplified SP1, MA0079.5 - https://jaspar.elixir.no/matrix/MA0079.5/
MOTIF_C = "CTTATCT" # Simplified GATA2, MA0036.2 - https://jaspar.elixir.no/matrix/MA0036.2/




def _random_background(length, freqs):
    nucleotides = list(freqs.keys())
    weights = list(freqs.values())
    return "".join(random.choices(nucleotides, weights=weights, k=length))


def _intervals_overlap(start, end, intervals):
    for used_start, used_end in intervals:
        if start < used_end and end > used_start:
            return True
    return False


def _place_motifs(sequence, motifs, allow_overlap):
    seq_list = list(sequence)
    used_intervals = []
    for motif in motifs:
        motif_len = len(motif)
        if motif_len > len(sequence):
            raise ValueError("Motif length exceeds sequence length.")
        for _ in range(2000):
            start = random.randint(0, len(sequence) - motif_len)
            end = start + motif_len
            if allow_overlap or not _intervals_overlap(start, end, used_intervals):
                seq_list[start:end] = list(motif)
                used_intervals.append((start, end))
                break
        else:
            raise RuntimeError("Failed to place motif without overlap.")
    return "".join(seq_list)


def _motifs_for_class(class_name):
    if class_name == "A":
        count = random.randint(MIN_MOTIFS, MAX_MOTIFS)
        motifs = [MOTIF_A] * count
    elif class_name == "B":
        count = random.randint(MIN_MOTIFS, MAX_MOTIFS)
        motifs = [MOTIF_B] * count
    elif class_name == "C":
        count = random.randint(MIN_MOTIFS, MAX_MOTIFS)
        motifs = [MOTIF_C] * count
    else:
        raise ValueError(f"Unknown class: {class_name}")
    random.shuffle(motifs)
    return motifs


def _generate_records(class_name, start_index):
    records = []
    for idx in range(start_index, start_index + N_SEQS):
        background = _random_background(BP_LEN, BG_FREQs)
        motifs = _motifs_for_class(class_name)
        if not OVERLAPPING_MOTIFS:
            total_motif_len = sum(len(m) for m in motifs)
            if total_motif_len > BP_LEN:
                raise ValueError("Motifs do not fit without overlap.")
        seq = _place_motifs(background, motifs, OVERLAPPING_MOTIFS)
        header = f"seq{idx}|source={class_name}"
        records.append((header, seq))
    return records


def _write_fasta(records, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(f">{header}\n{seq}\n")


def _split_records(records):
    shuffled = records[:]
    random.shuffle(shuffled)
    train_count = int(N_SEQS * TRAIN_FRAC)
    return shuffled[:train_count], shuffled[train_count:]


def main():
    random.seed(SEED)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FASTAs")
    os.makedirs(output_dir, exist_ok=True)
    combined_test_records = []
    next_index = 1
    for class_name in CLASSES:
        records = _generate_records(class_name, next_index)
        next_index += N_SEQS
        train_records, test_records = _split_records(records)
        train_path = os.path.join(output_dir, f"{class_name}_train.fa")
        _write_fasta(train_records, train_path)
        combined_test_records.extend(test_records)

    combined_test_path = os.path.join(output_dir, "combined_test.fa")
    _write_fasta(combined_test_records, combined_test_path)


if __name__ == "__main__":
    main()