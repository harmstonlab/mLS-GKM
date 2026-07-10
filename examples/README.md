# Reproducing the mLS-GKM Application Note analyses

This directory contains the scripts and configuration needed to reproduce the benchmarking and multi-class analyses reported in the mLS-GKM Application Note. Each subsection below corresponds to one figure panel or supplementary result; everything assumes you are running from the `examples/` directory of the repository unless stated otherwise.

## Prerequisites

- A built and installed `mLS-GKM` (`gkmtrain`, `gkmpredict`,
  `gkmexplain` in `../bin/`).
- The original [LS-GKM + gkmexplain](https://github.com/kundajelab/lsgkm) build, for the head-to-head equivalence and speed comparisons. 
- SLURM (for the job scripts).
    - Alternatively these scripts can be adapted to run on other schedulers or locally; the SLURM job scripts are provided as-is and may require adjustments to fit your cluster configuration (e.g. partition names, memory limits, job array syntax). 
    - If using SLURM the `job` files will need to be edited to set up the correct partition name for your cluster. Alternatively, the `sbatch` commands can be run directly on the command line with the appropriate arguments for your cluster, e.g. `sbatch --partition=your_partition_name <JOB_FILE>`.
 

A reference Conda environment is provided as `environment.yml` in the scripts directory. To create and activate:

```sh
conda env create -f ../scripts/environment.yml
conda activate mLS-GKM
```

## Reference genomes

The analyses use:

- `hg19` human reference ([UCSC](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz)).
    - Used for generating the FASTA records from the ENCODE bigBed files, as well as the repeat/GC matched negatives
- `GRCh38 regulatory features` from Ensembl Regulatory Build release v115 ([Ensembl FTP](https://ftp.ensembl.org/pub/release-115/regulation/homo_sapiens/GRCh38/annotation/)), specifically `Homo_sapiens.GRCh38.regulatory_features.v115.gff3.gz`

## Directory layout
```
examples/
├── figure1_panel.R                         Recreate Fig. 1
├── README.md                               (this file)
├── supplemental_figures.R                  Recreate all Supplemental Figures
├── SPEEDUP/
│   └── ENCODE_DATA/                        Raw and processed ENCODE datasets
│        ├── evalute_metrics.R              Compute ROC/PR/MCC metrics + plots
│        ├── filter_bigbed.py               Count records and build bigbed_list.tsv
│        │── plot_resource_comparison.R     Plot memory and run time comparisons
│        ├── Predict_LS-GKM.job             Predict on test FASTA (LS-GKM)
│        ├── Predict_mLS-GKM.job            Predict on test FASTA (mLS-GKM)
│        ├── process_bigbed.py              BigBed -> FASTA + matched negatives
│        ├── Process_ENCODE_bigbed.job      SLURM array job for process_bigbed.py
│        ├── Speed_EVAL_Predict_LS-GKM.job  Speed/memory for gkmpredict (LS-GKM)
│        ├── Speed_EVAL_Predict_mLS-GKM.job Speed/memory for gkmpredict (mLS-GKM)
│        ├── Speed_EVAL_Explain_LS-GKM.job  Speed/memory for gkmexplain (LS-GKM)
│        ├── Speed_EVAL_Explain_mLS-GKM.job Speed/memory for gkmexplain (mLS-GKM)
│        ├── Train_LS-GKM.job               Train with original LS-GKM
│        └── Train_mLS-GKM.job              Train with mLS-GKM
└── multiclass/
     ├── regulatory_features/
     │   ├── evaluate_performance.R         Evaluate performance metrics/plots
     │   ├── interpret_explain.job          Create PWM/ICM from gkmexplain output
     │   ├── make_FASTAs.R                  Build FASTA files
     │   ├── make_explain.job               Run gkmexplain jobs
     │   ├── plot_meme.R                    Plot MEME motifs from svmw_emalign_k11 output
     │   ├── predict.job                    Run gkmpredict
     │   ├── svmw_emalign_k11.job           Recover motifs with svwm_emalign (k=11)
     │   └── train.job                      Train multi-class model
     └── synthetic/
         ├── create_synth.py                Generate synthetic sequences
         ├── evaluate_performance.R         Evaluate performance metrics/plots
         ├── interpret_explain.job          Create PWM/ICM from gkmexplain output
         ├── make_explain.job               Run gkmexplain jobs
         ├── plot_meme.R                    Plot MEME motifs
         ├── predict.job                    Run gkmpredict
         ├── svmw_emalign_k11.job           Recover motifs with svwm_emalign (k=11)
         └── train.job                      Train multi-class model

```



## 1. ENCODE ChIP-seq benchmark (Fig. 1A, B, C, D)

This reproduces the 333-dataset equivalence comparison and the threading/memory benchmarks on `H1hescCtcf`.

### 1a. Download and prepare the reference genome
Download the `hg19` reference genome from UCSC:

```sh
cd SPEEDUP/ENCODE_DATA
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
```

### 1b. Download and process the ENCODE peak files

The original LS-GKM paper used ENCODE ChIP-seq peak files with at least 5,000 peaks. We use the same source:

```
https://ftp.ebi.ac.uk/pub/databases/ensembl/encode/integration_data_jan2011/byDataType/peaks/jan2011/spp/optimal/hub/
```

To download and filter:

```sh
rsync -Lav rsync://ftp.ebi.ac.uk/pub/databases/ensembl/encode/integration_data_jan2011/byDataType/peaks/jan2011/spp/optimal/hub/ ./bigbed/
python filter_bigbed.py
```
This yields `bigbed_counts.tsv` listing the 333 retained datasets.

To convert each bigBed to a positives FASTA + matched negatives FASTA (GC-content and repeat-fraction matched, regions > 1 kb removed for direct comparability with Lee 2016):

```sh
sbatch Process_ENCODE_bigbed.job
```

The output is one positives and one matched-negatives FASTA per dataset under `FASTAs/<DATASET_NAME>`. Chromosomes 1 and 2 are held out as the test set; all other chromosomes are used for training.


### 1c. Train with original LS-GKM and with mLS-GKM

```sh
sbatch Train_LS-GKM.job
sbatch Train_mLS-GKM.job
```

Both job scripts use the gkmrbf kernel (`-t 3`) with the LS-GKM-recommended hyperparameters:

```
-t 3 -c 10 -g 2 -m 4000 -T 4
```

producing one model per dataset under `Models/LS-GKM` and `Models/mLS-GKM` respectively.

### 1d. Predict on the test set and evaluate metrics

```sh
sbatch Predict_LS-GKM.job
sbatch Predict_mLS-GKM.job
```

This produces three predictions TSV per dataset under `Predictions/LS-GKM_Models/<DATASET_NAME>` and `Predictions/mLS-GKM_Models/<DATASET_NAME>` respectively:
- `positive_predictions.tsv` (chromosomes 1–2 positives)
- `negative_predictions.tsv` (chromosomes 1–2 negatives)
- `all_predictions.tsv` (chromosomes 1–2 positives and negatives combined)

Then compute ROC AUC, PR AUC, and MCC for each dataset and plot the comparisons:

Either run the R script:

```sh
Rscript evalute_metrics.R
```
Or run the R script interactively within RStudio. The required packages are:

```R
dplyr
readr
tibble
pROC
ggplot2
```

### 1e. Speed and memory benchmarks (Fig. 1B, C, D)

It is best to run these benchmarks in a controlled environment with minimal background load. We used a computed node with 2x AMD EPYC 7543 and a job reservation to ensure exclusive access. 

Use the `run_benchmarks.sh` wrapper to submit all four scripts the desired number of times. Multiple repeats give a more robust estimate of the median and interquartile range. The outputs are subdivided into `SPEED_EVAL/Predict_LS-GKM/THREADS_<N>/RPT_<M>/` etc...

```sh
# from examples/SPEEDUP/ENCODE_DATA/
./run_benchmarks.sh -n 5                 # 5 repeats, exclusive nodes (publication runs)
./run_benchmarks.sh -n 1 --no-exclusive  # quick test runs; share nodes to run many at once
```

The wrapper loops `RPT` from 1 to `-n` and submits every script for each repeat. For the multithreaded scripts it submits one job per thread count, with `--cpus-per-task` set equal to that thread count. The thread sweeps (which used to be SLURM `--array` lists) are defined near the top of `run_benchmarks.sh`.

`--exclusive` is the default and reserves whole nodes, which is best for optaining reliable benchmarking results. Pass `--no-exclusive` while testing will mean that each job only reserves as many CPUs as it uses, so many low-thread jobs can share a node and run concurrently (timings in this mode are not publication-representative). Defaults for `-n` and exclusivity can also be edited at the top of the wrapper.

For the most consistent timings, pin every job to the same machine and/or a dedicated reservation via the `EXTRA_ARGS` variable near the top of `run_benchmarks.sh` (empty by default). It is passed to every `sbatch` call, e.g.:

```sh
EXTRA_ARGS="--nodelist=node01 --reservation=my_benchmark_res"
```

- The Predict and Explain jobs for mLS-GKM use 1,2,4,8,16,32,64,128 threads used for per sequence parallelism
- The Predict job for LS-GKM uses 1,4,16 threads, used for kernel computations
- The Explain job for LS-GKM uses 1 thread only, as the original implementation does not support multi-threading.

In all cases wall-clock time and peak resident memory are recorded via `psrecord`. Creating a `resource_usage.png` plot for each run, as well as a `resource_usage.csv` that will be used for the final summary plots.

To plot:

```sh
Rscript plot_resource_comparison.R
```

Or run the R script interactively within RStudio. The required packages are:

```R
dplyr
ggplot2
readr
stringr
purrr
tidyr
forcats
```
This will create `SPEED_EVAL/plots/` with the resource comparison plots for both prediction and explanation.

## 2. Multi-class regulatory-feature classification (Fig. 1E)

This reproduces the 3-class Ensembl Regulatory Build experiment (Enhancer / Promoter / CTCF Binding Site).

### 2a. Extract Ensembl regulatory features

Download and process the Ensembl Regulatory Build GFF3 file to produce FASTA files for each class:

```sh
cd multiclass/regulatory_features
wget https://ftp.ensembl.org/pub/release-115/regulation/homo_sapiens/GRCh38/annotation/Homo_sapiens.GRCh38.regulatory_features.v115.gff3.gz
gunzip Homo_sapiens.GRCh38.regulatory_features.v115.gff3.gz
Rscript make_FASTAs.R
```
Alternatively the R Script can be run interactively in RStudio

For each class (enhancer, promoter, CTCF_binding_site) the produces:
- <CLASS>_FULL.fa - The full FASTA file from that class
- <CLASS>_test.fa - The test FASTA file from that class (held out chr1 and chr2)
- <CLASS>_train.fa - The training FASTA file from that class (`FULL` - `TEST`)

Aditionally there is a `combined_test.fa` which is a concatenation of all `_test.fa` files

### 2b. Train the 3-class probability-calibrated model

NOTE - this job will take a significant amount of time to complete, as such a pre-trained model is provided at `multiclass/regulatory_features/enhancer_vs_promoter_vs_CTCF.t3.model.txt`, and so the training step can be skipped if desired.

```sh
sbatch train.job
```

The job runs:

```sh
../../../bin/gkmtrain \
    -P \
    -T 16 \
    -m 248000 \
    -t 3 -c 10 -g 2 \
    -C ./enhancer_vs_promoter_vs_CTCF.t3 \
    ./FASTAs/enhancer_train.fa \
    ./FASTAs/promoter_train.fa \
    ./FASTAs/CTCF_binding_site_train.fa \
    ./enhancer_vs_promoter_vs_CTCF.t3
```

The job submits 10 copies of itself, with an array limit of 1 to ensure that only one runs at a time. This is so that is a job gets killed/times out it will automatically resume from the last completed checkpoint. After training has successfully completed all pending jobs are cancelled.

### 2c. Generate non-redundant 11mers

```sb
python ../../../scripts/nrkmers.py 11 FASTAs/kmers_11.fa
```

### 2d. Score the test set + kmers and produce Fig. 1E

```sh
sbatch predict.job
```

The `evaluate_performance.R` script and then be used to compute the ROC AUC, PR AUC, and MCC metrics for each class.

```sh
Rscript evaluate_performance.R
```

Alternatively the R Script can be run interactively in RStudio. The required packages are:

```R
dplyr
readr
stringr
tidyr
ggplot2
glue
pROC
tibble
scales
patchwork
```


## 3. Motif recovery (Fig. 1F)

This reproduces the FOS::JUN motif recovery on the Ensembl 3-class dataset.


### 3a. Prepare shuffled background sequences

The TF-MoDISco workflow requires dinucleotide-preserved shuffled background sequences. This can be done with the `make_dnshuff_fasta.py` script:

```sh
cd multiclass/regulatory_features
conda activate mLS-GKM
python ../../../scripts/make_dnshuff_fasta.py \
    --in-fasta FASTAs/combined_test.fa \
    --out-dir FASTAs/ \
    --classes enhancer,promoter,CTCF_binding_site
```
### 3c. Run gkmexplain

```sh
sbatch make_explain.job
```
This will run `gkmexplain` on the shuffled datasets, as well as the original combined test set, with the `-m 1` option to get the hypothetical importance scores needed for TF-MoDISco. 

### 3b. Run TF-MoDISco

The TF-MoDISco workflow follows the original gkmexplain notebooks ([kundajelab/gkmexplain](https://github.com/kundajelab/gkmexplain)) with adaptations to handle the multi-class output format of mLS-GKM gkmexplain.


To run TF-MoDISco with the default parameters:

```sh
sbatch interpret_explain.job
```

The FOS::JUN motif will be in `multiclass/regulatory_features/gkmexplain/BASE_PARAMS/enhancer_vs_promoter_D/patterns/enhancer_0.png`

### 3c. Recover motifs with svmw_emalign

```sh
sbatch svmw_emalign_k11.job
```

This takes the top 1%, 5% and 10% of kmers by importance score for each class, and aligns the kmers with an expectation-maximization algorithm to produce a MEME file. The resulting meme files can then be plotted with the `plot_meme.R` script:

```sh
Rscript plot_meme.R
```

Alternatively the R Script can be run interactively in RStudio. The required packages are:

```R
TFBSTools
universalmotif
glue
ggbio
ggpattern
ggplot2
GenomicRanges
regioneR
```


## 4. Synthetic motif-recovery benchmark (Supplementary Fig. S1)

A controlled benchmark with three implanted motifs.

### 4a. Generate the synthetic dataset

```sh
cd multiclass/synthetic
conda activate mLS-GKM
python create_synth.py
```

This generates three sets of sequences, each with 15,000 sequences containing 1–3 non-overlapping copies of:

- Motif A: `GTAAACA`   (simplified FOXK2; JASPAR MA1103.2)
- Motif B: `GGGGAGGGG` (simplified SP1;   JASPAR MA0079.5)
- Motif C: `CTTATCG`   (simplified GATA2; JASPAR MA0036.2)

Each FASTA is then split 70/30 into training and test sets.

Resulting in:

- A_train.fa, A_test.fa
- B_train.fa, B_test.fa
- C_train.fa, C_test.fa
- combined_test.fa (A_test + B_test + C_test)

### 4b. Generate non-redundant 11mers

```sh
python ../../../scripts/nrkmers.py 11 FASTAs/kmers_11.fa
```

### 4c. Train

```sh
sbatch train.job
```
This runs `gkmtrain -t 3 -c 10 -g 2 -P` on the three training files.

### 4d. Run predictions

```sh
sbatch predict.job
```

### 4e. Recover motifs with svmw_emalign

```sh
sbatch svmw_emalign_k11.job
```

This takes the top 1%, 5% and 10% of kmers by importance score for each class, and aligns the kmers with an expectation-maximization algorithm to produce a MEME file. The resulting meme files can then be plotted with the `plot_meme.R` script:

```sh
Rscript plot_meme.R
```

Alternatively the R Script can be run interactively in RStudio. The required packages are:

```R
TFBSTools
universalmotif
glue
ggbio
ggpattern
ggplot2
GenomicRanges
regioneR
```
### 4f. gkmexplain and TF-MoDISco

Motifs can also be recovered with the gkmexplain + TF-MoDISco workflow as described in the main text for the Ensembl regulatory feature dataset. 

First the shuffled background FASTA files need to be generated:

```sh
python ../../../scripts/make_dnshuff_fasta.py \
    --in-fasta FASTAs/combined_test.fa \
    --out-dir FASTAs/ \
    --classes A,B,C
```

Next the gkmexplain steps need to be run:

```sh
sbatch make_explain.job
```

Then the TF-MoDISco workflow can be run with the default parameters:

```sh
sbatch interpret_explain.job
```

## 5. Generating Figures Used in the Application Note

The R scripts `figure1_panel.R` and `supplemental_figures.R` can be used to generate the panels for the main figure and the supplemental figures respectively. These scripts assume that all the necessary data (predictions, resource usage, performance metrics) have already been generated by running the previous steps.


## Contact

If you have problems reproducing any of the analyses, please open an issue on the GitHub repository, or email  (howardkj1 AT cardiff DOT ac DOT uk)
