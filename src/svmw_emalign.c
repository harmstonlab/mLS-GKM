// 
// 
/* svmw_emalign.c: C reimplementation of ./scripts/svmw_emalign.py
 * Build de novo PWMs from SVM scores (multi-class)
 * Copyright (C) 2014 Dongwon Lee (original Python)
 * Copyright (C) 2026 Kieran Howard (C reimplementation and multi-class support)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <getopt.h>

#define CLOG_MAIN
#include "clog.h"

#define SMALLP 1e-3

// Background letter frequencies (A, C, G, T)
static const double BG_FREQ[4] = {0.29, 0.21, 0.21, 0.29};

typedef struct {
    double alpha;
    int iterations;
    int nmaxpwms;
    int nminkmers;
    int top_frac;
    double cutoff;
    char *classes;
    int threads;
} Options;

typedef struct {
    char *key;
    double score;
    size_t order;
    int used;
} HashEntry;

typedef struct {
    HashEntry *entries;
    size_t size;
    size_t count;
    size_t next_order;
} HashTable;

typedef struct {
    char *kmer;
    double score;
    size_t order;
} KmerScore;

typedef struct {
    double *pwm;     // len x 4
    double *pwm_rc;  // len x 4
    int len;
} KmerPwm;

typedef struct {
    int skmerind;
    double los;
    int mpos;
    int rc;
} Alignment;

static char *xstrdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrtok_r(char *s, const char *delim, char **saveptr) {
    char *p = s ? s : *saveptr;
    if (!p) return NULL;

    // skip leading delimiters
    p += strspn(p, delim);
    if (*p == '\0') {
        *saveptr = NULL;
        return NULL;
    }

    char *token = p;
    p = strpbrk(token, delim);
    if (p) {
        *p = '\0';
        *saveptr = p + 1;
    } else {
        *saveptr = NULL;
    }
    return token;
}

static double log2_safe(double x) {
    return log(x) / log(2.0);
}

static void print_info(const char *msg) {
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char buf[64];
    strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S", tm_info);
    fprintf(stderr, "INFO  @ %s: %s\n", buf, msg);
}

static char complement_base(char c) {
    switch (c) {
        case 'A': return 'T';
        case 'C': return 'G';
        case 'G': return 'C';
        case 'T': return 'A';
        default: return 'N';
    }
}

static char *revcomp(const char *seq) {
    size_t n = strlen(seq);
    char *rc = (char *)malloc(n + 1);
    if (!rc) return NULL;
    for (size_t i = 0; i < n; i++) {
        rc[i] = complement_base(seq[n - 1 - i]);
    }
    rc[n] = '\0';
    return rc;
}

static int nt2idx(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return 0;
    }
}

static double *seq2pwm(const char *seq, int pwmlen) {
    double columns[4][4] = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };
    int seqlen = (int)strlen(seq);
    double *pwm = (double *)calloc((size_t)pwmlen * 4, sizeof(double));
    if (!pwm) return NULL;

    if (pwmlen < seqlen) {
        int offset = (seqlen - pwmlen) / 2;
        for (int i = 0; i < pwmlen; i++) {
            int idx = nt2idx(seq[offset + i]);
            memcpy(&pwm[i * 4], columns[idx], 4 * sizeof(double));
        }
    } else if (pwmlen == seqlen) {
        for (int i = 0; i < pwmlen; i++) {
            int idx = nt2idx(seq[i]);
            memcpy(&pwm[i * 4], columns[idx], 4 * sizeof(double));
        }
    } else {
        int paddinglen = (pwmlen - seqlen) / 2;
        int pos = 0;
        for (int i = 0; i < paddinglen; i++) {
            memcpy(&pwm[pos * 4], BG_FREQ, 4 * sizeof(double));
            pos++;
        }
        for (int i = 0; i < seqlen; i++) {
            int idx = nt2idx(seq[i]);
            memcpy(&pwm[pos * 4], columns[idx], 4 * sizeof(double));
            pos++;
        }
        while (pos < pwmlen) {
            memcpy(&pwm[pos * 4], BG_FREQ, 4 * sizeof(double));
            pos++;
        }
    }
    return pwm;
}

static unsigned long hash_djb2(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + (unsigned long)c;
    }
    return hash;
}

static void hash_init(HashTable *ht, size_t initial_size) {
    ht->size = initial_size;
    ht->count = 0;
    ht->next_order = 0;
    ht->entries = (HashEntry *)calloc(ht->size, sizeof(HashEntry));
}

static void hash_free(HashTable *ht) {
    if (!ht || !ht->entries) return;
    for (size_t i = 0; i < ht->size; i++) {
        if (ht->entries[i].used) {
            free(ht->entries[i].key);
        }
    }
    free(ht->entries);
    ht->entries = NULL;
    ht->size = 0;
    ht->count = 0;
}

static void hash_resize(HashTable *ht, size_t new_size) {
    HashEntry *old = ht->entries;
    size_t old_size = ht->size;
    ht->entries = (HashEntry *)calloc(new_size, sizeof(HashEntry));
    ht->size = new_size;
    ht->count = 0;

    for (size_t i = 0; i < old_size; i++) {
        if (old[i].used) {
            unsigned long h = hash_djb2(old[i].key);
            size_t idx = h % ht->size;
            while (ht->entries[idx].used) {
                idx = (idx + 1) % ht->size;
            }
            ht->entries[idx] = old[i];
            ht->count++;
        }
    }
    free(old);
}

static void hash_put(HashTable *ht, const char *key, double score) {
    if (ht->count * 10 >= ht->size * 7) {
        hash_resize(ht, ht->size * 2 + 1);
    }
    unsigned long h = hash_djb2(key);
    size_t idx = h % ht->size;
    while (ht->entries[idx].used) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            ht->entries[idx].score = score;
            return;  // keep order
        }
        idx = (idx + 1) % ht->size;
    }
    ht->entries[idx].used = 1;
    ht->entries[idx].key = xstrdup(key);
    ht->entries[idx].score = score;
    ht->entries[idx].order = ht->next_order++;
    ht->count++;
}

static int hash_get(const HashTable *ht, const char *key, double *out_score) {
    unsigned long h = hash_djb2(key);
    size_t idx = h % ht->size;
    size_t start = idx;
    while (ht->entries[idx].used) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            if (out_score) *out_score = ht->entries[idx].score;
            return 1;
        }
        idx = (idx + 1) % ht->size;
        if (idx == start) break;
    }
    return 0;
}

static int compare_kmer_score_desc(const void *a, const void *b) {
    const KmerScore *ka = (const KmerScore *)a;
    const KmerScore *kb = (const KmerScore *)b;
    if (ka->score < kb->score) return 1;
    if (ka->score > kb->score) return -1;
    if (ka->order < kb->order) return -1;
    if (ka->order > kb->order) return 1;
    return 0;
}

static int ensure_dir(const char *path) {
    if (!path || path[0] == '\0') return 0;
    char *tmp = xstrdup(path);
    if (!tmp) return -1;

    size_t len = strlen(tmp);
    if (len == 0) { free(tmp); return 0; }
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) && errno != EEXIST) {
                free(tmp);
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0775) && errno != EEXIST) {
        free(tmp);
        return -1;
    }
    free(tmp);
    return 0;
}

static void write_memefile(double **models, int *model_len,
                           int *nsites, double *sum_los,
                           int nmodels, const char *class_name,
                           const char *outfile) {
    FILE *f = fopen(outfile, "w");
    if (!f) {
        fprintf(stderr, "I/O error: %s\n", strerror(errno));
        return;
    }

    fprintf(f, "MEME version 4\n\n");
    fprintf(f, "ALPHABET= ACGT\n\n");
    fprintf(f, "strands: + -\n\n");
    fprintf(f, "Background letter frequencies (from entire human genome)\n");
        fprintf(f, "A %.2f C %.2f G %.2f T %.2f\n\n",
            BG_FREQ[0], BG_FREQ[1], BG_FREQ[2], BG_FREQ[3]);

    for (int centerid = 0; centerid < nmodels; centerid++) {
        if (!models[centerid]) continue;
        int pwmlen = model_len[centerid];
        int ns = nsites[centerid];
        fprintf(f, "MOTIF %s.%d\n", class_name, centerid + 1);
        if (sum_los) {
            // sum_los is the total aligned log-odds over all used k-mers; higher is better.
            // mean_los is sum_los divided by nsites to normalize across motif support size.
            fprintf(f, "# sum_los=%.6f mean_los=%.6f\n",
                sum_los[centerid],
                (ns > 0) ? (sum_los[centerid] / ns) : 0.0);
        }
        fprintf(f, "letter-probability matrix: alength= 4 w= %d nsites= %d E= 0\n", pwmlen, ns);
        for (int j = 0; j < pwmlen; j++) {
            int freqs[4];
            int sum = 0;
            for (int k = 0; k < 4; k++) {
                double p = models[centerid][j * 4 + k];
                freqs[k] = (int)(p * ns);
                sum += freqs[k];
            }
            if (sum != ns) {
                int diff = ns - sum;
                int diff_inc = (diff > 0) ? 1 : -1;
                // sort indices by freqs desc; tie-break on lower index (Python stable sort) 
                int idxs[4] = {0, 1, 2, 3};
                for (int a = 0; a < 4; a++) {
                    for (int b = a + 1; b < 4; b++) {
                        if (freqs[idxs[b]] > freqs[idxs[a]] ||
                            (freqs[idxs[b]] == freqs[idxs[a]] && idxs[b] < idxs[a])) {
                            int t = idxs[a]; idxs[a] = idxs[b]; idxs[b] = t;
                        }
                    }
                }
                for (int i = 0; i < abs(diff); i++) {
                    freqs[idxs[i % 4]] += diff_inc;
                }
            }
            fprintf(f, "%.3f %.3f %.3f %.3f\n",
                    freqs[0] / (double)ns,
                    freqs[1] / (double)ns,
                    freqs[2] / (double)ns,
                    freqs[3] / (double)ns);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void emalign(const char *skmer, int skmerind,
                    char **seedkmers, int nseed,
                    HashTable *svmw, int pwmlen, const Options *options,
                    double **out_model, Alignment *out_alignments,
                    int *out_used, double *out_sum_los) {
    int window = (int)strlen(skmer);
    double bglogp[4];
    for (int i = 0; i < 4; i++) {
        bglogp[i] = log2_safe((BG_FREQ[i] + SMALLP) / (1.0 + 4.0 * SMALLP));
    }

    KmerPwm *kmerpwms = (KmerPwm *)calloc(nseed, sizeof(KmerPwm));
    for (int i = 0; i < nseed; i++) {
        if (!seedkmers[i]) continue;
        char *rc = revcomp(seedkmers[i]);
        kmerpwms[i].len = (int)strlen(seedkmers[i]);
        kmerpwms[i].pwm = seq2pwm(seedkmers[i], kmerpwms[i].len);
        kmerpwms[i].pwm_rc = seq2pwm(rc, kmerpwms[i].len);
        free(rc);
    }

    double *model = seq2pwm(skmer, pwmlen);
    double *prev_model = NULL;

    double los_max_total_prev = 0.0;
    double los_max_total_curr = 0.0;

    for (int iround = 0; iround < options->iterations; iround++) {
        // clear used and alignments 
        for (int i = 0; i < nseed; i++) {
            out_used[i] = 0;
        }

        double *lomodel = (double *)calloc((size_t)pwmlen * 4, sizeof(double));
        for (int j = 0; j < pwmlen; j++) {
            for (int k = 0; k < 4; k++) {
                double val = model[j * 4 + k];
                lomodel[j * 4 + k] = log2_safe((val + SMALLP) / (1.0 + 4.0 * SMALLP)) - bglogp[k];
            }
        }

        // find best alignments 
        for (int ind = 0; ind < nseed; ind++) {
            if (!seedkmers[ind]) continue;
            double los_max = -9999.0;
            int mpos_max = -1;
            int rc_max = 0;
            const char *kmer = seedkmers[ind];
            char *kmer_rc = revcomp(kmer);

            for (int rc = 0; rc < 2; rc++) {
                const char *curr = (rc == 0) ? kmer : kmer_rc;
                for (int mpos = 0; mpos <= pwmlen - window; mpos++) {
                    double los = 0.0;
                    for (int i = 0; i < window; i++) {
                        int idx = nt2idx(curr[i]);
                        los += lomodel[(mpos + i) * 4 + idx];
                    }
                    if (los > los_max) {
                        los_max = los;
                        mpos_max = mpos;
                        rc_max = rc;
                    }
                }
            }

            out_alignments[ind].skmerind = skmerind;
            out_alignments[ind].los = los_max;
            out_alignments[ind].mpos = mpos_max;
            out_alignments[ind].rc = rc_max;
            free(kmer_rc);
        }

        // update model
        los_max_total_curr = 0.0;
        prev_model = model;
        model = (double *)calloc((size_t)pwmlen * 4, sizeof(double));

        for (int ind = 0; ind < nseed; ind++) {
            if (!seedkmers[ind]) continue;
            Alignment *aln = &out_alignments[ind];
            if (aln->los < options->cutoff) {
                continue;
            }
            // sum of per-kmer best log-odds alignments (higher implies stronger motif support)
            los_max_total_curr += aln->los;
            out_used[ind] = 1;

            const char *kmer = seedkmers[ind];
            double score = 0.0;
            if (!hash_get(svmw, kmer, &score)) score = 0.0;

            double *kmerpwm = (aln->rc == 0) ? kmerpwms[ind].pwm : kmerpwms[ind].pwm_rc;
            int klen = kmerpwms[ind].len;
            int offset = aln->mpos;

            // build kmerinstance 
            double *kmerinstance = (double *)calloc((size_t)pwmlen * 4, sizeof(double));
            for (int i = 0; i < klen; i++) {
                int pos = offset + i;
                if (pos >= 0 && pos < pwmlen) {
                    memcpy(&kmerinstance[pos * 4], &kmerpwm[i * 4], 4 * sizeof(double));
                }
            }

            double mult = options->alpha * score;
            for (int i = 0; i < pwmlen * 4; i++) {
                model[i] += exp(mult * kmerinstance[i]);
            }
            free(kmerinstance);
        }

        // normalize model 
        int used_any = 0;
        for (int i = 0; i < nseed; i++) if (out_used[i]) { used_any = 1; break; }
        if (!used_any) {
            free(model);
            model = prev_model;
            free(lomodel);
            break;
        }

        for (int j = 0; j < pwmlen; j++) {
            double rowsum = 0.0;
            for (int k = 0; k < 4; k++) rowsum += model[j * 4 + k];
            if (rowsum == 0.0) {
                for (int k = 0; k < 4; k++) model[j * 4 + k] = BG_FREQ[k];
                rowsum = 1.0;
            }
            for (int k = 0; k < 4; k++) model[j * 4 + k] /= rowsum;
        }

        free(lomodel);
        if (fabs(los_max_total_curr - los_max_total_prev) < 1e-9) {
            break;
        }
        los_max_total_prev = los_max_total_curr;
    }

    *out_model = model;
    if (out_sum_los) {
        *out_sum_los = los_max_total_curr;
    }

    // cleanup
    for (int i = 0; i < nseed; i++) {
        free(kmerpwms[i].pwm);
        free(kmerpwms[i].pwm_rc);
    }
    free(kmerpwms);
    if (prev_model && prev_model != model) {
        free(prev_model);
    }
}

static void build_motifs_for_class(const char *class_label,
                                   char **kmers, double *scores,
                                   int nkmers, int pwmlen,
                                   const Options *options,
                                   const char *outdir,
                                   int score_mode,
                                   int quiet) {
    char msg[4096 + 16];
    snprintf(msg, sizeof(msg), "=== Building motifs for class: %s ===", class_label);
    print_info(msg);

    HashTable svmw;
    hash_init(&svmw, (size_t)(nkmers * 2 + 7));

    for (int i = 0; i < nkmers; i++) {
        double raw = scores[i];
        if ((score_mode > 0 && raw <= 0.0) || (score_mode < 0 && raw >= 0.0)) {
            continue;
        }
        double score = (score_mode < 0) ? -raw : raw;
        hash_put(&svmw, kmers[i], score);
    }
    for (int i = 0; i < nkmers; i++) {
        double raw = scores[i];
        if ((score_mode > 0 && raw <= 0.0) || (score_mode < 0 && raw >= 0.0)) {
            continue;
        }
        double score = (score_mode < 0) ? -raw : raw;
        char *rc = revcomp(kmers[i]);
        hash_put(&svmw, rc, score);
        free(rc);
    }

    // collect keys for sorting
    KmerScore *all = (KmerScore *)malloc(svmw.count * sizeof(KmerScore));
    size_t idx = 0;
    for (size_t i = 0; i < svmw.size; i++) {
        if (svmw.entries[i].used) {
            all[idx].kmer = svmw.entries[i].key;
            all[idx].score = svmw.entries[i].score;
            all[idx].order = svmw.entries[i].order;
            idx++;
        }
    }

    qsort(all, svmw.count, sizeof(KmerScore), compare_kmer_score_desc);

    int topN = (int)((svmw.count * (size_t)options->top_frac) / 100);
    if (topN < 0) topN = 0;
    char **seedkmers = NULL;
    if (topN > 0) {
        seedkmers = (char **)calloc((size_t)topN, sizeof(char *));
        for (int i = 0; i < topN; i++) {
            seedkmers[i] = all[i].kmer;
        }
    }

    int *visited = (int *)calloc((size_t)topN, sizeof(int));
    int remainings = topN;
    int npwms = 0;
    int failed = 0;

    double **models = (double **)calloc((size_t)topN, sizeof(double *));
    int *model_len = (int *)calloc((size_t)topN, sizeof(int));
    int *nsites = (int *)calloc((size_t)topN, sizeof(int));
    double *sum_los = (double *)calloc((size_t)topN, sizeof(double));

    while (remainings > 0 && npwms < options->nmaxpwms && failed < 10) {
        const char *skmer = NULL;
        int skmerind = -1;

        for (int i = 0; i < topN; i++) {
            if (!visited[i]) {
                if (skmerind == -1) {
                    skmer = seedkmers[i];
                    skmerind = i;
                }
            }
        }

        if (!skmer) break;

        Alignment *alignments = (Alignment *)calloc((size_t)topN, sizeof(Alignment));
        int *used = (int *)calloc((size_t)topN, sizeof(int));
        char **curr_kmers = (char **)calloc((size_t)topN, sizeof(char *));
        for (int i = 0; i < topN; i++) {
            if (!visited[i]) {
                curr_kmers[i] = seedkmers[i];
            }
        }

        double *model = NULL;
        double model_sum_los = 0.0;
        emalign(skmer, skmerind, curr_kmers, topN, &svmw, pwmlen, options,
            &model, alignments, used, &model_sum_los);

        int nkmers_used = 0;
        for (int i = 0; i < topN; i++) {
            if (used[i]) {
                nkmers_used++;
                visited[i] = 1;
            }
        }

        if (nkmers_used < options->nminkmers) {
            if (!quiet) {
                snprintf(msg, sizeof(msg), "skip %s (number of kmers aligned is %d < %d)",
                         skmer, nkmers_used, options->nminkmers);
                print_info(msg);
            }
            if (skmerind >= 0) visited[skmerind] = 1;
            failed++;
            free(model);
            free(alignments);
            free(used);
            continue;
        }

        models[skmerind] = model;
        model_len[skmerind] = pwmlen;
        nsites[skmerind] = nkmers_used;
        sum_los[skmerind] = model_sum_los;

        remainings = 0;
        for (int i = 0; i < topN; i++) {
            if (!visited[i]) remainings++;
        }
        npwms++;

        free(curr_kmers);
        free(alignments);
        free(used);
    }

    if (outdir && outdir[0]) {
        ensure_dir(outdir);
    }

    char outpath[4096];
    if (outdir && outdir[0]) {
        snprintf(outpath, sizeof(outpath), "%s/%s.meme", outdir, class_label);
    } else {
        snprintf(outpath, sizeof(outpath), "%s.meme", class_label);
    }
    write_memefile(models, model_len, nsites, sum_los, topN, class_label, outpath);

    if (!quiet) {
        snprintf(msg, sizeof(msg), "Wrote %s", outpath);
        print_info(msg);
    }

    for (int i = 0; i < topN; i++) {
        free(models[i]);
    }
    free(models);
    free(model_len);
    free(nsites);
    free(sum_los);
    free(visited);
    free(seedkmers);
    free(all);
    hash_free(&svmw);
}

static int parse_scores_file(const char *filename, char ***out_kmers,
                             double ***out_scores_by_class, int *out_kmer_count,
                             int *out_class_count) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "I/O error: %s\n", strerror(errno));
        return -1;
    }

    char line[8192];
    char **kmers = NULL;
    double **scores = NULL; // [class][i]
    int kmer_cap = 0;
    int kmer_count = 0;
    int class_count = 0;

    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        // tokenize
        char *tokens[1024];
        int ntok = 0;
        char *saveptr = NULL;
        char *tok = xstrtok_r(p, " \t\r\n", &saveptr);
        while (tok && ntok < 1024) {
            tokens[ntok++] = tok;
            tok = xstrtok_r(NULL, " \t\r\n", &saveptr);
        }
        if (ntok < 2) continue;

        // parse scores
        int scores_ok = 1;
        double vals[1024];
        for (int i = 1; i < ntok; i++) {
            char *endptr = NULL;
            double v = strtod(tokens[i], &endptr);
            if (endptr == tokens[i] || *endptr != '\0') {
                scores_ok = 0;
                break;
            }
            vals[i - 1] = v;
        }
        if (!scores_ok) continue;

        if (class_count == 0) {
            class_count = ntok - 1;
            scores = (double **)calloc((size_t)class_count, sizeof(double *));
            for (int c = 0; c < class_count; c++) {
                scores[c] = NULL;
            }
        } else if (ntok - 1 != class_count) {
            fprintf(stderr, "Inconsistent number of score columns: expected %d, saw %d\n",
                    class_count, ntok - 1);
            fclose(f);
            return -1;
        }

        if (kmer_count >= kmer_cap) {
            kmer_cap = (kmer_cap == 0) ? 1024 : (kmer_cap * 2);
            kmers = (char **)realloc(kmers, (size_t)kmer_cap * sizeof(char *));
            for (int c = 0; c < class_count; c++) {
                scores[c] = (double *)realloc(scores[c], (size_t)kmer_cap * sizeof(double));
            }
        }

        kmers[kmer_count] = xstrdup(tokens[0]);
        for (int c = 0; c < class_count; c++) {
            scores[c][kmer_count] = vals[c];
        }
        kmer_count++;
    }
    fclose(f);

    if (class_count == 0) {
        fprintf(stderr, "No k-mer score rows found in %s\n", filename);
        return -1;
    }

    *out_kmers = kmers;
    *out_scores_by_class = scores;
    *out_kmer_count = kmer_count;
    *out_class_count = class_count;
    return 0;
}

typedef struct {
    char **kmers;
    double **scores_by_class;
    int nkmers;
    int pwmlen;
    Options *options;
    char **class_names;
    int nclasses;
    const char *outdir;
    int quiet;
    int next_job;
    pthread_mutex_t lock;
} ThreadPoolCtx;

static void *worker_thread_pos(void *arg) {
    ThreadPoolCtx *ctx = (ThreadPoolCtx *)arg;
    while (1) {
        pthread_mutex_lock(&ctx->lock);
        int job = ctx->next_job;
        ctx->next_job++;
        pthread_mutex_unlock(&ctx->lock);
        if (job >= ctx->nclasses * 2) break;

        int idx = job % ctx->nclasses;
        int score_mode = (job < ctx->nclasses) ? 1 : -1;
        char label[512];
        snprintf(label, sizeof(label), "%s_%s", ctx->class_names[idx],
                 (score_mode > 0) ? "pos" : "neg");

        build_motifs_for_class(label, ctx->kmers,
                               ctx->scores_by_class[idx], ctx->nkmers,
                               ctx->pwmlen, ctx->options, ctx->outdir,
                               score_mode, ctx->quiet);
    }
    return NULL;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options] KMER_SCORES PWM_LENGTH [OUT_DIR]\n", prog);
    fprintf(stderr, "\nOptions (defaults in parentheses):\n");
    fprintf(stderr, "  -a, --alpha FLOAT       alpha multiplier (3.0)\n");
    fprintf(stderr, "  -i, --iterations INT    max EM iterations (100)\n");
    fprintf(stderr, "  -n, --nmaxpwms INT       max PWMs to build (5)\n");
    fprintf(stderr, "  -m, --nminkmers INT      min kmers per PWM (100)\n");
    fprintf(stderr, "  -f, --top_frac INT       top %% k-mers as seeds (1)\n");
    fprintf(stderr, "  -c, --cutoff FLOAT       log-odds cutoff (5)\n");
    fprintf(stderr, "  --classes LIST  comma-separated class names (CNE,Enhancer,Random)\n");
    fprintf(stderr, "  -t, --threads N         max classes in parallel (1)\n");
    fprintf(stderr, "  -h, --help      show this help message\n");
}

int main(int argc, char **argv) {
    Options options;
    options.alpha = 3.0;
    options.iterations = 100;
    options.nmaxpwms = 5;
    options.nminkmers = 100;
    options.top_frac = 1;
    options.cutoff = 5.0;
    options.classes = xstrdup("CNE,Enhancer,Random");
    options.threads = 1;

    static struct option long_options[] = {
        {"alpha", required_argument, 0, 'a'},
        {"iterations", required_argument, 0, 'i'},
        {"nmaxpwms", required_argument, 0, 'n'},
        {"nminkmers", required_argument, 0, 'm'},
        {"top_frac", required_argument, 0, 'f'},
        {"cutoff", required_argument, 0, 'c'},
        {"classes", required_argument, 0, 1000},
        {"threads", required_argument, 0, 't'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "a:i:n:m:f:c:t:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'a': options.alpha = atof(optarg); break;
            case 'i': options.iterations = atoi(optarg); break;
            case 'n': options.nmaxpwms = atoi(optarg); break;
            case 'm': options.nminkmers = atoi(optarg); break;
            case 'f': options.top_frac = atoi(optarg); break;
            case 'c': options.cutoff = atof(optarg); break;
            case 't': options.threads = atoi(optarg); break;
            case 'h':
                usage(argv[0]);
                return 0;
            case 1000:
                free(options.classes);
                options.classes = xstrdup(optarg);
                break;
            default:
                usage(argv[0]);
                return 0;
        }
    }

    int remaining = argc - optind;
    if (remaining == 0) {
        usage(argv[0]);
        return 0;
    }
    if (remaining < 2 || remaining > 3) {
        fprintf(stderr, "incorrect number of arguments\n");
        return 0;
    }

    const char *svmscorefile = argv[optind];
    int pwmlen = atoi(argv[optind + 1]);
    const char *outdir = (remaining == 3) ? argv[optind + 2] : ".";

    // parse class names
    char *classes_copy = xstrdup(options.classes);
    char *class_names[256];
    int class_count = 0;
    char *saveptr = NULL;
    char *tok = xstrtok_r(classes_copy, ",", &saveptr);
    while (tok && class_count < 256) {
        while (isspace((unsigned char)*tok)) tok++;
        char *end = tok + strlen(tok) - 1;
        while (end > tok && isspace((unsigned char)*end)) { *end = '\0'; end--; }
        if (*tok) {
            class_names[class_count++] = xstrdup(tok);
        }
        tok = xstrtok_r(NULL, ",", &saveptr);
    }
    free(classes_copy);

    if (class_count == 0) {
        fprintf(stderr, "--classes must contain at least one class name\n");
        return 1;
    }

    char **kmers = NULL;
    double **scores_by_class = NULL;
    int nkmers = 0;
    int score_class_count = 0;
    if (parse_scores_file(svmscorefile, &kmers, &scores_by_class, &nkmers, &score_class_count) != 0) {
        return 1;
    }

    if (score_class_count != class_count) {
        fprintf(stderr,
                "Number of score columns (%d) does not match number of class names (%d). "
                "Pass --classes with exactly %d names.\n",
                score_class_count, class_count, score_class_count);
        return 1;
    }

    if (options.threads < 1) {
        fprintf(stderr, "-t/--threads must be >= 1\n");
        return 1;
    }
    if (options.top_frac < 0 || options.top_frac > 100) {
        fprintf(stderr, "-f/--top_frac must be between 0 and 100\n");
        return 1;
    }
    if (options.threads > class_count * 2) {
        options.threads = class_count * 2;
    }

    if (options.threads == 1 || class_count == 1) {
        for (int i = 0; i < class_count; i++) {
            char label_pos[512];
            char label_neg[512];
            snprintf(label_pos, sizeof(label_pos), "%s_pos", class_names[i]);
            snprintf(label_neg, sizeof(label_neg), "%s_neg", class_names[i]);

            build_motifs_for_class(label_pos, kmers, scores_by_class[i],
                                   nkmers, pwmlen, &options, outdir, 1, 0);
            build_motifs_for_class(label_neg, kmers, scores_by_class[i],
                                   nkmers, pwmlen, &options, outdir, -1, 0);
        }
    } else {
        int total_jobs = class_count * 2;
        int procs = (options.threads < total_jobs) ? options.threads : total_jobs;
        char msg[128];
        snprintf(msg, sizeof(msg), "Running %d jobs in parallel (threads=%d)", procs, options.threads);
        print_info(msg);

        ThreadPoolCtx ctx;
        ctx.kmers = kmers;
        ctx.scores_by_class = scores_by_class;
        ctx.nkmers = nkmers;
        ctx.pwmlen = pwmlen;
        ctx.options = &options;
        ctx.class_names = class_names;
        ctx.nclasses = class_count;
        ctx.outdir = outdir;
        ctx.quiet = 1;
        ctx.next_job = 0;
        pthread_mutex_init(&ctx.lock, NULL);

        pthread_t *threads = (pthread_t *)calloc((size_t)procs, sizeof(pthread_t));
        for (int i = 0; i < procs; i++) {
            pthread_create(&threads[i], NULL, worker_thread_pos, &ctx);
        }
        for (int i = 0; i < procs; i++) {
            pthread_join(threads[i], NULL);
        }
        pthread_mutex_destroy(&ctx.lock);
        free(threads);
    }

    for (int i = 0; i < nkmers; i++) {
        free(kmers[i]);
    }
    free(kmers);
    for (int c = 0; c < score_class_count; c++) {
        free(scores_by_class[c]);
    }
    free(scores_by_class);

    for (int i = 0; i < class_count; i++) {
        free(class_names[i]);
    }
    free(options.classes);

    return 0;
}
