## mLS-GKM: a multi-class, multi-threaded, memory-efficient gkm-SVM

`mLS-GKM` builds upon [LS-GKM](https://github.com/Dongwon-Lee/lsgkm), ending gkm-SVM training, prediction and per-base interpretation in four ways:

1. **Multi-class classification.** `gkmtrain` accepts any number of class FASTA files and trains a one-vs-one model with `K(K−1)/2` binary sub-classifiers stored in a single model file.
2. **Calibrated probability outputs.** With `-P`, training fits per-pair Platt sigmoids by internal 5-fold cross-validation, and prediction produces a calibrated probability vector over all classes via Wu-Lin-Weng pairwise coupling.
3. **Per-sequence multi-threading.** `gkmpredict` and `gkmexplain` now parallelise over input sequences with the new `-t` flag, in addition to the original `-T` kernel-internal threading.
4. **Streaming `gkmexplain`.** Per-base contribution scores are accumulated directly into the per-position output rather than materialising the full per-position-per-base tensor, reducing peak memory by >65% on representative ENCODE workloads.

A separate on-disk checkpointing system (`-C` flag on `gkmtrain`) makes long-running probability-calibrated training jobs robust to wall-clock limits and node failures.

`mLS-GKM` is backwards-compatible with LS-GKM in the binary case: training a two-class problem produces a model whose support vectors and α coefficients are bit-identical to LS-GKM, except for a change in internal class-label encoding (1/2 rather than 1/−1).

### Citation

Please cite the original LS-GKM and gkm-SVM papers if you use this software in your research:

- Ghandi, M.†, Lee, D.†, Mohammad-Noori, M. & Beer, M. A. *Enhanced Regulatory Sequence Prediction Using Gapped k-mer Features.* PLoS Comput Biol 10, e1003711 (2014). doi:10.1371/journal.pcbi.1003711 *† Co-first authors*

- Lee, D. *LS-GKM: A new gkm-SVM for large-scale Datasets.* Bioinformatics btw142 (2016). doi:10.1093/bioinformatics/btw142

If you use the `gkmexplain` interpretation tool, please additionally cite:

- Shrikumar, A., Prakash, E. & Kundaje, A. *GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs.* Bioinformatics 35, i173–i182 (2019). doi:10.1093/bioinformatics/btz322

Please cite the `mLS-GKM` Application Note if you use the multi-class, probability, or streaming `gkmexplain` extensions:

- XXXXXXXXXX (citation to be added on publication).


### Installation

After downloading and extracting the source code, type:

```sh
cd src
make
```

If successful, you should be able to find the following executables in the current (`src`) directory:

```
gkmtrain
gkmpredict
gkmexplain
svmw_emalign
```

`make install` will copy these executables to the `../bin` directory.

### Tutorial

Introduces users to the basic workflow of `mLS-GKM`. Please refer to the help messages for more detailed information on each program. You can access them by running each program without any arguments or with `-h/--help`.


#### Training

Train a binary or multi-class SVM classifier using `gkmtrain`. It takes *at least* three arguments: one or more class sequence files (FASTA), followed by an output prefix.

For binary classification (drop-in compatible with LS-GKM):

```
Usage: gkmtrain [options] <posfile> <negfile> <outprefix>
```
For multi-class classification with `K` ≥ 2 classes:

```
Usage: gkmtrain [options] <class1.fa> <class2.fa> [class3.fa ...] <outprefix>
```

The output is `<outprefix>.model.txt`. With `-x N` cross-validation, the output is `<outprefix>.cvpred.txt`.

Available options (run `gkmtrain` with no arguments for the full list):

```
-t <0 ~ 5>   set kernel function (default: 2 gkm)
        NOTE: RBF kernels (3 and 5) work best with -c 10 -g 2
        0 -- gapped-kmer
        1 -- estimated l-mer with full filter
        2 -- estimated l-mer with truncated filter (gkm)
        3 -- gkm + RBF (gkmrbf)
        4 -- gkm + center weighted (wgkm)
                [weight = max(M, floor(M*exp(-ln(2)*D/H)+1))]
        5 -- gkm + center weighted + RBF (wgkmrbf)
 -l <int>     set word length, 3<=l<=12 (default: 11)
 -k <int>     set number of informative column, k<=l (default: 7)
 -d <int>     set maximum number of mismatches to consider, d<=4 (default: 3)
 -g <float>   set gamma for RBF kernel. -t 3 or 5 only (default: 1.0)
 -M <int>     set the initial value (M) of the exponential decay function
              for wgkm-kernels. max=255, -t 4 or 5 only (default: 50)
 -H <float>   set the half-life parameter (H) that is the distance (D) required
              to fall to half of its initial value in the exponential decay
              function for wgkm-kernels. -t 4 or 5 only (default: 50)
 -R           if set, reverse-complement is not considered as the same feature
 -c <float>   set the regularization parameter SVM-C (default: 1.0)
 -e <float>   set the precision parameter epsilon (default: 0.001)
 -w <float>   set the parameter SVM-C to w*C for the positive set (default: 1.0)
 -m <float>   set cache memory size in MB (default: 100.0)
              NOTE: Large cache signifcantly reduces runtime. >4Gb is recommended
 -s           if set, use the shrinking heuristics
 -P           enable probability estimates
 -C <prefix> set checkpoint prefix (writes <prefix>.train.ckpt and <prefix>.cv.ckpt)
 -x <int>     set N-fold cross validation mode (default: no cross validation)
 -i <int>     run i-th cross validation only 1<=i<=ncv (default: all)
 -r <int>     set random seed for shuffling (CV and probability calibration) (default: 1)
 -v <0 ~ 4>   set the level of verbosity (default: 2)
                0 -- error msgs only (ERROR)
                1 -- warning msgs (WARN)
                2 -- progress msgs at coarse-grained level (INFO)
                3 -- progress msgs at fine-grained level (DEBUG)
                4 -- progress msgs at finer-grained level (TRACE)
-T <1|4|16>   set the number of threads for parallel calculation, 1, 4, or 16
                 (default: 1)
```



To train a 3-class probability-calibrated model with checkpointing:

```
mkdir -p ckpt
../bin/gkmtrain -t 3 -c 10 -g 2 -m 4000 -T 4 -P -C ckpt/run1 enhancer.fa promoter.fa ctcf.fa run1
```

If the job is killed before it finishes, simply re-run the same command to resume from the last completed pair or fold.


#### Prediction

Use `gkmpredict` to score sequences against a trained model.

```
Usage: gkmpredict [options] <test_seqfile> <model_file> <output_file>
```

Output modes (mutually exclusive):
```
-D             output SVM decision values per sequence.
               Binary: one value. Multiclass: k*(k-1)/2 values (i<j order).
-S             output aggregated per-class scores (margin-based).
               Score_i = (1/(k-1)) * sum_{j!=i} sign(i,j)*margin(i,j).
-P             output per-class probability estimates (requires probA/probB in model).
```

Threading:
```
-t <int>        worker threads across sequences (default: 1)
-T <1|4|16>     threads used *inside* the kernel calculation (default: 1)
                Notes:
                - For many sequences: set -t to ~#CPU cores and keep -T 1.
                  This maximizes throughput and avoids nested threading.
                - For very few sequences (or a single long sequence): keep -t 1
                  and try -T 4 or -T 16 to speed up each prediction.
                - If you set both -t>1 and -T>1, the program may create roughly
                  -t * -T runnable threads during the busy part of computation.
```

Logging and progress:
```
-v <0|1|2|3|4>  set the level of verbosity (default: 2)
                0 -- error msgs only (ERROR)
                1 -- warning msgs (WARN)
                2 -- progress msgs at coarse-grained level (INFO)
                3 -- progress msgs at fine-grained level (DEBUG)
                4 -- progress msgs at finer-grained level (TRACE)
-p <int>       print progress every N sequences (default: 100; 0 disables)
```

For most workloads (many sequences) the recommended setting is `-t <#CPU cores> -T 1`, which maximises throughput and avoids nested thread creation. For workloads with very few but long sequences, keep `-t 1` and use `-T 4` or `-T 16`.


#### Per-base interpretation (gkmexplain)

`gkmexplain` decomposes a model's prediction for each test sequence into per-base contribution scores.

```
Usage: gkmexplain [options] <test_seqfile> <model_file> <output_file>
```

Explanation modes (`-m`):
```
-m 0    importance scores (default)
-m 1    hypothetical importance scores (lmers with d mismatches)
-m 2    hypothetical importance scores (d+1 mismatches)
-m 3    perturbation effect estimation (lmers with d mismatches)
-m 4    perturbation effect estimation (d+1 mismatches)
-m 5    score perturbations for only the central position
```
Output modes (mutually exclusive, multi-class):
```
-D                  decision values + one base-score block per pair
                    (in i<j pair order).
-S [-C <label>]     aggregated per-class scores + one base-score
                    block per class. With -C <label>, only the
                    named class is explained.
-P                  per-class probabilities + probability-scaled
                    base-score blocks (one per class).
                    Requires probA/probB in the model.
```

Threading flags `-t` and `-T` work as in `gkmpredict`.

`gkmexplain` in `mLS-GKM` accumulates per-base contributions into the output as it processes each support vector, never materialising the full per-position-per-base contribution tensor. Output is bit-equivalent to the original `gkmexplain` but peak memory is reduced from O(L × SV) to O(L + SV).


#### Recovering motifs

There are two main approaches to motif discovery with `mLS-GKM`:

1. **TF-MoDISco.** Run `gkmexplain` to generate contribution scores for the test sequences, with both `-m 0` and `-m 1` (hypothetical importance scores). Then generate shuffled control sequences with [make_dnshuff_fasta.py](scripts/make_dnshuff_fasta.py) and run `gkmexplain` on these as well. Finally, run [explain_script.py](scripts/explain_script.py) to run TF-MoDISco and summarise results.

2, **svmw_emalign.** Generate all non-redundant *k*-mers using `scripts/nrkmers.py` with `K` being the word size of the the model and score them with `gkmpredict`. Then run `svmw_emalign` to cluster and align them into PWMs. See the help message of `svmw_emalign` for details on the expected input format and available options.


#### Inspecting checkpoint files

Checkpoint files are binary. To inspect their contents (header, completed pairs/folds, optionally the full payload), use:

```sh
scripts/inspect_checkpoint.py <prefix>.train.ckpt
scripts/inspect_checkpoint.py <prefix>.cv.ckpt --records
scripts/inspect_checkpoint.py <prefix>.cv.ckpt --all
```

### Example workflow

A worked example reproducing the analyses in the mLS-GKM Application Note is provided in the `examples/` directory. See [`examples/README.md`](examples/README.md) for step-by-step instructions covering data download, processing, training, prediction, and the benchmarking sweeps reported in the paper.


### Contact

For questions about `mLS-GKM`, please open an issue on the GitHub repository or email Kieran Howard (howardkj1 AT cardiff DOT ac DOT uk)

For questions about the original LS-GKM, please contact Dongwon Lee (dongwon.lee AT childrens DOT harvard DOT edu).
