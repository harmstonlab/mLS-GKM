#! /usr/bin/env Rscript
library(Biostrings)
library(GenomicRanges)
library(rtracklayer)
library(BSgenome.Hsapiens.UCSC.hg38)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#### LOAD DATA ####
gff_file <- "./Homo_sapiens.GRCh38.regulatory_features.v115.gff3"
output_dir <- "./FASTAs"
dir.create(output_dir)

# Load GFF3 file
gff <- rtracklayer::import(gff_file)

# Standardize chromosome names to have "chr" prefix
seqlevels(gff) <- paste0("chr", gsub("^chr", "", seqlevels(gff)))
strand(gff) <- "*"
# Get the reference genome
genome <- BSgenome.Hsapiens.UCSC.hg38

#### PARSE FEATURES BY TYPE ####
feature_types <- c("promoter", "enhancer", "CTCF_binding_site")

# Create lists to store sequences for each feature type
feature_list <- list()

for (ftype in feature_types) {
    # Subset GFF to this feature type
    subset_gff <- gff[gff$type == ftype]
    
    if (length(subset_gff) > 0) {
        # Extract sequences from genome
        seqs <- getSeq(genome, subset_gff)
        
        # Create informative names with chromosome and coordinates
        names(seqs) <- paste0(
            as.character(seqnames(subset_gff)), ":",
            start(subset_gff), "-",
            end(subset_gff), "|",
            subset_gff$ID, "|",
            "source=", ftype
        )
        
        # Split into test (chr1, chr2) and train (others)
        chrs <- as.character(seqnames(subset_gff))
        test_idx <- chrs %in% c("chr1", "chr2")
        train_idx <- !test_idx
        
        feature_list[[ftype]] <- list(
            full = seqs,
            train = seqs[train_idx],
            test = seqs[test_idx],
            train_idx = train_idx,
            test_idx = test_idx
        )
    }
}

#### WRITE FULL FASTA FILES ####
for (ftype in feature_types) {
    if (!is.null(feature_list[[ftype]])) {
        full_path <- file.path(output_dir, paste0(ftype, "_FULL.fa"))
        writeXStringSet(feature_list[[ftype]]$full, filepath = full_path, format = "fasta")
        cat("Written:", full_path, "\n")
    }
}

#### WRITE TRAIN FASTA FILES ####
for (ftype in feature_types) {
    if (!is.null(feature_list[[ftype]]) && length(feature_list[[ftype]]$train) > 0) {
        train_path <- file.path(output_dir, paste0(ftype, "_train.fa"))
        writeXStringSet(feature_list[[ftype]]$train, filepath = train_path, format = "fasta")
        cat("Written:", train_path, "\n")
    }
}

#### WRITE TEST FASTA FILES ####
for (ftype in feature_types) {
    if (!is.null(feature_list[[ftype]]) && length(feature_list[[ftype]]$test) > 0) {
        test_path <- file.path(output_dir, paste0(ftype, "_test.fa"))
        writeXStringSet(feature_list[[ftype]]$test, filepath = test_path, format = "fasta")
        cat("Written:", test_path, "\n")
    }
}

#### WRITE COMBINED TEST FASTA ####
all_test_seqs <- do.call(c, lapply(feature_types, function(ftype) {
    if (!is.null(feature_list[[ftype]])) feature_list[[ftype]]$test else NULL
}))
if (length(all_test_seqs) > 0) {
    combined_test_path <- file.path(output_dir, "combined_test.fa")
    writeXStringSet(all_test_seqs, filepath = combined_test_path, format = "fasta")
    cat("Written:", combined_test_path, "\n")
}

#### SUMMARY ####
for (ftype in feature_types) {
    if (!is.null(feature_list[[ftype]])) {
        cat("\n", ftype, ":\n", sep = "")
        cat("  Total:", length(feature_list[[ftype]]$full), "\n")
        cat("  Train:", length(feature_list[[ftype]]$train), "\n")
        cat("  Test:", length(feature_list[[ftype]]$test), "\n")
    }
}

