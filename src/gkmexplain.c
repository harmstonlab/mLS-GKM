/* gkmpredict.c
 *
 * Copyright (C) 2015 Dongwon Lee
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
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <getopt.h>
#include <pthread.h>
#include <stdarg.h>
#include <math.h>

#include "libsvm_gkm.h"

#define CLOG_MAIN
#include "clog.h"

void print_usage_and_exit()
{
    printf(
            "\n"
            "Program: gkmexplain (lsgkm program for explaining predictions using a trained model)\n"
            "Version: "
            LSGKM_VERSION
            "\n\n"
            "Usage: gkmexplain [options] <test_seqfile> <model_file> <output_file>\n"
            " explain prediction on test sequences using trained gkm-SVM\n"
            "\n"
            "Arguments:\n"
            " test_seqfile: sequence file for test (fasta format)\n"
            " model_file: output of gkmtrain\n"
            " output_file: name of output file\n"
            "\n"
            "Options:\n"
            " -v <0|1|2|3|4>  set the level of verbosity (default: 2)\n"
            "                   0 -- error msgs only (ERROR)\n"
            "                   1 -- warning msgs (WARN)\n"
            "                   2 -- progress msgs at coarse-grained level (INFO)\n"
            "                   3 -- progress msgs at fine-grained level (DEBUG)\n"
            "                   4 -- progress msgs at finer-grained level (TRACE)\n"
            " -m <0..5>  set the explanation mode (default: 0)\n"
            "                   0 -- importance scores\n"
            "                   1 -- hypothetical importance scores (considering lmers with d mismatches)\n"
            "                   2 -- hypothetical importance scores (considering d+1 mismatches)\n"
            "                   3 -- perturbation effect estimation (considering lmers with d mismatches)\n"
            "                   4 -- perturbation effect estimation (considering d+1 mismatches)\n"
            "                   5 -- score perturbations for only the central position in the region\n"
            " -D              output SVM decision values per sequence.\n"
            "                 Binary: one value + one base-score block.\n"
            "                 Multiclass: k*(k-1)/2 values (i<j order) + one base-score block per pair (same order).\n"
            " -S              output aggregated per-class scores (margin-based).\n"
            "                 Score_i = (1/(k-1)) * sum_{j!=i} sign(i,j)*margin(i,j).\n"
            "                 Output: scores then per-class base-score blocks (all classes unless -C is used).\n"
            " -P              output per-class probabilities and probability-scaled base scores.\n"
            "                 Output: probabilities then per-class base-score blocks (all classes).\n"
            "                 Requires probA/probB in the model.\n"
            "                 Note: -D, -S, and -P are mutually exclusive.\n"
            " -C <label>      (multiclass + -S) explain the score for the given class label.\n"
            "                 Without -C, and when nr_class>2, gkmexplain outputs scores + base-scores for *all* classes.\n"
            " -t <int>         worker threads across sequences (default: 1)\n"
            " -T <1|4|16>      threads used *inside* the kernel calculation (default: 1)\n"
            "                 Notes:\n"
            "                 - For many sequences: set -t to ~#CPU cores and keep -T 1.\n"
            "                 - For very few sequences (or a single long sequence): keep -t 1\n"
            "                   and try -T 4 or -T 16.\n"
            "                 - If you set both -t>1 and -T>1, you can create roughly -t * -T\n"
            "                   runnable threads during the busy part of computation.\n"
            " -p <int>         print progress every N sequences (default: 100; 0 disables)\n"
            "\n");
    exit(0);
}

static struct svm_model* model;

static char *line = NULL;
static int max_line_len;
static int output_decision_values = 0; // -D decision values
static int output_scores = 0; // -S per-class aggregated scores
static int output_probabilities = 0; // -P per-class probabilities (appended)
static int gkmexplain_num_threads = 1;
static int gkmexplain_progress_every = 100;
static int explain_score_label = -1; // -C <label> (only for multiclass -S)

// this function was copied from libsvm & slightly modified 
static char* readline(FILE *input)
{
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line, (size_t) max_line_len);
        int len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    
    line[strcspn(line, "\r\n")] = '\0';

    return line;
}


typedef struct {
    char *data;
    size_t len;
    size_t cap;
} strbuf_t;

static void sb_init(strbuf_t *sb)
{
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static void sb_reserve(strbuf_t *sb, size_t extra)
{
    size_t need = sb->len + extra + 1;
    if (need <= sb->cap) return;
    size_t new_cap = sb->cap ? sb->cap : 256;
    while (new_cap < need) new_cap *= 2;
    sb->data = (char *)realloc(sb->data, new_cap);
    if (!sb->data) {
        clog_error(CLOG(LOGGER_ID), "realloc failed for output buffer");
        exit(1);
    }
    sb->cap = new_cap;
}

static void sb_appendf(strbuf_t *sb, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (needed < 0) {
        va_end(ap2);
        clog_error(CLOG(LOGGER_ID), "vsnprintf sizing failed");
        exit(1);
    }
    sb_reserve(sb, (size_t)needed);
    int written = vsnprintf(sb->data + sb->len, sb->cap - sb->len, fmt, ap2);
    va_end(ap2);
    if (written < 0 || written != needed) {
        clog_error(CLOG(LOGGER_ID), "vsnprintf write failed");
        exit(1);
    }
    sb->len += (size_t)written;
}

static void sb_append_base_scores(strbuf_t *sb, double **explanation, int seqlen, int mode)
{
    int positions_to_print = (mode == 5) ? 1 : seqlen;
    for (int i = 0; i < positions_to_print; i++) {
        if (i > 0) sb_appendf(sb, ";");
        for (int j = 0; j < MAX_ALPHABET_SIZE; j++) {
            if (j > 0) sb_appendf(sb, ",");
            sb_appendf(sb, "%g", explanation[i][j]);
        }
    }
}

static double sigmoid_predict_local(double decision_value, double A, double B)
{
    double fApB = decision_value * A + B;
    if (fApB >= 0) {
        return exp(-fApB) / (1.0 + exp(-fApB));
    }
    return 1.0 / (1.0 + exp(fApB));
}

static void multiclass_probability_local(int k, double **r, double *p)
{
    int t, j;
    int iter = 0, max_iter = (k > 100) ? k : 100;
    double **Q = (double **)malloc(sizeof(double *) * (size_t)k);
    double *Qp = (double *)malloc(sizeof(double) * (size_t)k);
    if (!Q || !Qp) {
        clog_error(CLOG(LOGGER_ID), "malloc failed for multiclass_probability");
        exit(1);
    }

    for (t = 0; t < k; t++) {
        p[t] = 1.0 / k;
        Q[t] = (double *)malloc(sizeof(double) * (size_t)k);
        if (!Q[t]) {
            clog_error(CLOG(LOGGER_ID), "malloc failed for multiclass_probability row");
            exit(1);
        }
        Q[t][t] = 0;
        for (j = 0; j < t; j++) {
            Q[t][t] += r[j][t] * r[j][t];
            Q[t][j] = Q[j][t];
        }
        for (j = t + 1; j < k; j++) {
            Q[t][t] += r[j][t] * r[j][t];
            Q[t][j] = -r[j][t] * r[t][j];
        }
    }

    for (iter = 0; iter < max_iter; iter++) {
        double pQp = 0;
        for (t = 0; t < k; t++) {
            Qp[t] = 0;
            for (j = 0; j < k; j++) Qp[t] += Q[t][j] * p[j];
            pQp += p[t] * Qp[t];
        }
        double max_error = 0;
        for (t = 0; t < k; t++) {
            double error = fabs(Qp[t] - pQp);
            if (error > max_error) max_error = error;
        }
        if (max_error < 0.005 / k) break;

        for (t = 0; t < k; t++) {
            double diff = (-Qp[t] + pQp) / Q[t][t];
            p[t] += diff;
            pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
            for (j = 0; j < k; j++) {
                Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
                p[j] /= (1 + diff);
            }
        }
    }

    for (t = 0; t < k; t++) free(Q[t]);
    free(Q);
    free(Qp);
}

typedef struct {
    int idx;
    char *sid;
    char *seq;
    int seqlen;
} explain_job_t;

typedef struct {
    explain_job_t **buf;
    int cap;
    int head;
    int tail;
    int count;
    int done;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} job_queue_t;

typedef struct {
    job_queue_t q;
    int mode;

    // ordered output
    char **out_lines;
    int out_cap;
    int total_jobs;
    int next_to_write;
    pthread_mutex_t out_mutex;
    pthread_cond_t out_cond;
} explain_mt_ctx_t;

static void job_queue_init(job_queue_t *q, int cap)
{
    q->buf = (explain_job_t **)calloc((size_t)cap, sizeof(explain_job_t *));
    if (!q->buf) {
        clog_error(CLOG(LOGGER_ID), "calloc failed for job queue");
        exit(1);
    }
    q->cap = cap;
    q->head = 0;
    q->tail = 0;
    q->count = 0;
    q->done = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void job_queue_destroy(job_queue_t *q)
{
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
    pthread_mutex_destroy(&q->mutex);
    free(q->buf);
}

static void job_queue_push(job_queue_t *q, explain_job_t *job)
{
    pthread_mutex_lock(&q->mutex);
    while (q->count == q->cap) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }
    q->buf[q->tail] = job;
    q->tail = (q->tail + 1) % q->cap;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

static explain_job_t *job_queue_pop(job_queue_t *q)
{
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0 && !q->done) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }
    if (q->count == 0 && q->done) {
        pthread_mutex_unlock(&q->mutex);
        return NULL;
    }
    explain_job_t *job = q->buf[q->head];
    q->buf[q->head] = NULL;
    q->head = (q->head + 1) % q->cap;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return job;
}

static void job_queue_mark_done(job_queue_t *q)
{
    pthread_mutex_lock(&q->mutex);
    q->done = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

static char *build_output_line_for_record(const char *sid, const char *seq, int seqlen, int mode)
{
    // Allocate explanation buffers sized to seqlen.
    // mode==5 uses single-base explanation (length 4) and prints one position.
    double **explanation = NULL;
    double *singlebase_expl = NULL;

    if (mode == 5) {
        singlebase_expl = (double *)malloc(sizeof(double) * ((size_t)MAX_ALPHABET_SIZE));
        if (!singlebase_expl) {
            clog_error(CLOG(LOGGER_ID), "malloc failed for singlebase_expl");
            exit(1);
        }
        for (int j = 0; j < MAX_ALPHABET_SIZE; j++) singlebase_expl[j] = 0.0;
        // Reuse existing calculation function by creating a 2D wrapper with one row.
        explanation = (double **)malloc(sizeof(double *));
        if (!explanation) {
            clog_error(CLOG(LOGGER_ID), "malloc failed for explanation");
            exit(1);
        }
        explanation[0] = singlebase_expl;
    } else {
        explanation = (double **)malloc(sizeof(double *) * (size_t)seqlen);
        if (!explanation) {
            clog_error(CLOG(LOGGER_ID), "malloc failed for explanation");
            exit(1);
        }
        for (int i = 0; i < seqlen; i++) {
            explanation[i] = (double *)malloc(sizeof(double) * ((size_t)MAX_ALPHABET_SIZE));
            if (!explanation[i]) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for explanation[%d]", i);
                exit(1);
            }
            for (int j = 0; j < MAX_ALPHABET_SIZE; j++) explanation[i][j] = 0.0;
        }
    }

    const int nr_class = model ? model->nr_class : 2;
    const int n_dec = (nr_class <= 2) ? 1 : (nr_class * (nr_class - 1) / 2);
    double *dec_values = (double *)malloc(sizeof(double) * (size_t)n_dec);
    if (!dec_values) {
        clog_error(CLOG(LOGGER_ID), "malloc failed for dec_values");
        exit(1);
    }

    strbuf_t sb;
    sb_init(&sb);

    // Multiclass + -S default: output all class scores and all class explanations.
    // -C <label> selects a single class explanation (and prints only that class score).
    int explain_one_class = -1;
    if (output_scores && nr_class > 2 && explain_score_label != -1) {
        for (int i = 0; i < nr_class; i++) {
            if (model->label && model->label[i] == explain_score_label) {
                explain_one_class = i;
                break;
            }
        }
        if (explain_one_class < 0) {
            clog_error(CLOG(LOGGER_ID), "label %d not found in model labels", explain_score_label);
            exit(1);
        }
    }

    sb_appendf(&sb, "%s", sid);

    if (output_decision_values) {
        // Decision values are not per-class scores.
        // For multiclass, emit per-pair base scores (1vs2, 1vs3, ...).
        union svm_data x;
        x.d = gkmkernel_new_object((char *)seq, NULL, 0);
        svm_predict_values(model, x, dec_values);

        for (int di = 0; di < n_dec; di++) sb_appendf(&sb, "\t%g", dec_values[di]);

        if (nr_class <= 2) {
            if (mode == 5) {
                svm_predict_and_singlebaseexplain_values(model, x, dec_values, explanation[0]);
            } else {
                svm_predict_and_explain_values(model, x, dec_values, explanation, mode);
            }
            sb_appendf(&sb, "\t");
            sb_append_base_scores(&sb, explanation, seqlen, mode);
            gkmkernel_delete_object(x.d);
        } else {
            const int l = model->l;
            int *start = Malloc(int, nr_class);
            start[0] = 0;
            for (int i = 1; i < nr_class; i++) start[i] = start[i - 1] + model->nSV[i - 1];

            double *kvalue = Malloc(double, l);
            double *sv_weight = Malloc(double, l);

            double **singlebasepersv_explanation = NULL;
            if (mode == 5) {
                singlebasepersv_explanation = (double **)malloc(sizeof(double *) * ((size_t)MAX_ALPHABET_SIZE));
                if (!singlebasepersv_explanation) {
                    clog_error(CLOG(LOGGER_ID), "malloc failed for singlebasepersv_explanation");
                    exit(1);
                }
                for (int g = 0; g < MAX_ALPHABET_SIZE; g++) {
                    singlebasepersv_explanation[g] = (double *)malloc(sizeof(double) * ((size_t)l));
                    if (!singlebasepersv_explanation[g]) {
                        clog_error(CLOG(LOGGER_ID), "malloc failed for singlebasepersv_explanation[%d]", g);
                        exit(1);
                    }
                    for (int k = 0; k < l; k++) singlebasepersv_explanation[g][k] = 0.0;
                }
                gkmexplainsinglebasekernel_kernelfunc_batch_sv(x.d, kvalue, singlebasepersv_explanation);
            }

            int p = 0;
            for (int i = 0; i < nr_class; i++) {
                for (int j = i + 1; j < nr_class; j++) {
                    for (int k = 0; k < l; k++) sv_weight[k] = 0.0;

                    int si = start[i];
                    int sj = start[j];
                    int ci = model->nSV[i];
                    int cj = model->nSV[j];
                    double *coef1 = model->sv_coef[j - 1];
                    double *coef2 = model->sv_coef[i];

                    for (int k = 0; k < ci; k++) sv_weight[si + k] += coef1[si + k];
                    for (int k = 0; k < cj; k++) sv_weight[sj + k] += coef2[sj + k];

                    if (mode == 5) {
                        for (int g = 0; g < MAX_ALPHABET_SIZE; g++) singlebase_expl[g] = 0.0;
                        for (int g = 0; g < MAX_ALPHABET_SIZE; g++) {
                            for (int k = 0; k < l; k++) {
                                singlebase_expl[g] += singlebasepersv_explanation[g][k] * sv_weight[k];
                            }
                        }
                    } else {
                        for (int r = 0; r < seqlen; r++)
                            for (int g = 0; g < MAX_ALPHABET_SIZE; g++)
                                explanation[r][g] = 0.0;

                        gkmexplainkernel_kernelfunc_batch_sv_weighted(x.d, kvalue, explanation, mode, sv_weight);
                    }

                    sb_appendf(&sb, "\t");
                    sb_append_base_scores(&sb, explanation, seqlen, mode);
                    p++;
                }
            }

            if (singlebasepersv_explanation) {
                for (int g = 0; g < MAX_ALPHABET_SIZE; g++) free(singlebasepersv_explanation[g]);
                free(singlebasepersv_explanation);
            }
            free(sv_weight);
            free(kvalue);
            free(start);
            gkmkernel_delete_object(x.d);
        }
    } else if (output_probabilities) {
        if (!model->probA || !model->probB) {
            clog_error(CLOG(LOGGER_ID), "model does not contain probability estimates (probA/probB)");
            exit(1);
        }

        union svm_data x;
        x.d = gkmkernel_new_object((char *)seq, NULL, 0);
        svm_predict_values(model, x, dec_values);

        double **pairwise_prob = (double **)malloc(sizeof(double *) * (size_t)nr_class);
        double **pairwise_prob_work = (double **)malloc(sizeof(double *) * (size_t)nr_class);
        double *prob_estimates = (double *)malloc(sizeof(double) * (size_t)nr_class);
        double *prob_eps = (double *)malloc(sizeof(double) * (size_t)nr_class);
        double *pair_slope = (double *)malloc(sizeof(double) * (size_t)n_dec);
        if (!pairwise_prob || !pairwise_prob_work || !prob_estimates || !prob_eps || !pair_slope) {
            clog_error(CLOG(LOGGER_ID), "malloc failed for probability buffers");
            exit(1);
        }
        for (int i = 0; i < nr_class; i++) {
            pairwise_prob[i] = (double *)malloc(sizeof(double) * (size_t)nr_class);
            pairwise_prob_work[i] = (double *)malloc(sizeof(double) * (size_t)nr_class);
            if (!pairwise_prob[i] || !pairwise_prob_work[i]) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for pairwise probability matrix");
                exit(1);
            }
        }

        const double min_prob = 1e-7;
        int p = 0;
        for (int i = 0; i < nr_class; i++) {
            for (int j = i + 1; j < nr_class; j++) {
                double pr = sigmoid_predict_local(dec_values[p], model->probA[p], model->probB[p]);
                if (pr < min_prob) pr = min_prob;
                if (pr > 1.0 - min_prob) pr = 1.0 - min_prob;
                pairwise_prob[i][j] = pr;
                pairwise_prob[j][i] = 1.0 - pr;
                pair_slope[p] = pr * (1.0 - pr) * model->probA[p];
                p++;
            }
        }
        multiclass_probability_local(nr_class, pairwise_prob, prob_estimates);

        for (int i = 0; i < nr_class; i++) sb_appendf(&sb, "\t%g", prob_estimates[i]);

        double **dP_dR = (double **)malloc(sizeof(double *) * (size_t)nr_class);
        if (!dP_dR) {
            clog_error(CLOG(LOGGER_ID), "malloc failed for dP_dR");
            exit(1);
        }
        for (int c = 0; c < nr_class; c++) {
            dP_dR[c] = (double *)malloc(sizeof(double) * (size_t)n_dec);
            if (!dP_dR[c]) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for dP_dR[%d]", c);
                exit(1);
            }
            for (int k = 0; k < n_dec; k++) dP_dR[c][k] = 0.0;
        }

        const double eps = 1e-6;
        for (int i = 0; i < nr_class; i++) {
            for (int j = 0; j < nr_class; j++) {
                pairwise_prob_work[i][j] = pairwise_prob[i][j];
            }
        }

        p = 0;
        for (int i = 0; i < nr_class; i++) {
            for (int j = i + 1; j < nr_class; j++) {
                double orig = pairwise_prob_work[i][j];
                double bumped = orig + eps;
                if (bumped > 1.0 - min_prob) bumped = 1.0 - min_prob;
                if (bumped < min_prob) bumped = min_prob;
                pairwise_prob_work[i][j] = bumped;
                pairwise_prob_work[j][i] = 1.0 - bumped;
                multiclass_probability_local(nr_class, pairwise_prob_work, prob_eps);
                double denom = bumped - orig;
                if (denom == 0.0) denom = eps;
                for (int c = 0; c < nr_class; c++) {
                    dP_dR[c][p] = (prob_eps[c] - prob_estimates[c]) / denom;
                }
                pairwise_prob_work[i][j] = orig;
                pairwise_prob_work[j][i] = 1.0 - orig;
                p++;
            }
        }

        if (mode == 5) {
            double **class_singlebase = (double **)malloc(sizeof(double *) * (size_t)nr_class);
            if (!class_singlebase) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for class_singlebase");
                exit(1);
            }
            for (int c = 0; c < nr_class; c++) {
                class_singlebase[c] = (double *)calloc((size_t)MAX_ALPHABET_SIZE, sizeof(double));
                if (!class_singlebase[c]) {
                    clog_error(CLOG(LOGGER_ID), "calloc failed for class_singlebase[%d]", c);
                    exit(1);
                }
            }

            const int l = model->l;
            int *start = Malloc(int, nr_class);
            start[0] = 0;
            for (int i = 1; i < nr_class; i++) start[i] = start[i - 1] + model->nSV[i - 1];

            double *kvalue = Malloc(double, l);
            double *sv_weight = Malloc(double, l);
            double **singlebasepersv_explanation = (double **)malloc(sizeof(double *) * ((size_t)MAX_ALPHABET_SIZE));
            if (!singlebasepersv_explanation) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for singlebasepersv_explanation");
                exit(1);
            }
            for (int g = 0; g < MAX_ALPHABET_SIZE; g++) {
                singlebasepersv_explanation[g] = (double *)malloc(sizeof(double) * ((size_t)l));
                if (!singlebasepersv_explanation[g]) {
                    clog_error(CLOG(LOGGER_ID), "malloc failed for singlebasepersv_explanation[%d]", g);
                    exit(1);
                }
                for (int k = 0; k < l; k++) singlebasepersv_explanation[g][k] = 0.0;
            }
            gkmexplainsinglebasekernel_kernelfunc_batch_sv(x.d, kvalue, singlebasepersv_explanation);

            p = 0;
            for (int i = 0; i < nr_class; i++) {
                for (int j = i + 1; j < nr_class; j++) {
                    for (int k = 0; k < l; k++) sv_weight[k] = 0.0;

                    int si = start[i];
                    int sj = start[j];
                    int ci = model->nSV[i];
                    int cj = model->nSV[j];
                    double *coef1 = model->sv_coef[j - 1];
                    double *coef2 = model->sv_coef[i];

                    for (int k = 0; k < ci; k++) sv_weight[si + k] += coef1[si + k];
                    for (int k = 0; k < cj; k++) sv_weight[sj + k] += coef2[sj + k];

                    for (int g = 0; g < MAX_ALPHABET_SIZE; g++) singlebase_expl[g] = 0.0;
                    for (int g = 0; g < MAX_ALPHABET_SIZE; g++) {
                        for (int k = 0; k < l; k++) {
                            singlebase_expl[g] += singlebasepersv_explanation[g][k] * sv_weight[k];
                        }
                    }

                    for (int c = 0; c < nr_class; c++) {
                        double w = dP_dR[c][p] * pair_slope[p];
                        if (w == 0.0) continue;
                        for (int g = 0; g < MAX_ALPHABET_SIZE; g++) {
                            class_singlebase[c][g] += w * singlebase_expl[g];
                        }
                    }
                    p++;
                }
            }

            for (int c = 0; c < nr_class; c++) {
                sb_appendf(&sb, "\t");
                for (int g = 0; g < MAX_ALPHABET_SIZE; g++) {
                    if (g > 0) sb_appendf(&sb, ",");
                    sb_appendf(&sb, "%g", class_singlebase[c][g]);
                }
            }

            for (int g = 0; g < MAX_ALPHABET_SIZE; g++) free(singlebasepersv_explanation[g]);
            free(singlebasepersv_explanation);
            free(kvalue);
            free(sv_weight);
            free(start);
            for (int c = 0; c < nr_class; c++) free(class_singlebase[c]);
            free(class_singlebase);
        } else {
            double ***class_expl = (double ***)malloc(sizeof(double **) * (size_t)nr_class);
            if (!class_expl) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for class_expl");
                exit(1);
            }
            for (int c = 0; c < nr_class; c++) {
                class_expl[c] = (double **)malloc(sizeof(double *) * (size_t)seqlen);
                if (!class_expl[c]) {
                    clog_error(CLOG(LOGGER_ID), "malloc failed for class_expl[%d]", c);
                    exit(1);
                }
                for (int r = 0; r < seqlen; r++) {
                    class_expl[c][r] = (double *)calloc((size_t)MAX_ALPHABET_SIZE, sizeof(double));
                    if (!class_expl[c][r]) {
                        clog_error(CLOG(LOGGER_ID), "calloc failed for class_expl[%d][%d]", c, r);
                        exit(1);
                    }
                }
            }

            const int l = model->l;
            int *start = Malloc(int, nr_class);
            start[0] = 0;
            for (int i = 1; i < nr_class; i++) start[i] = start[i - 1] + model->nSV[i - 1];

            double *kvalue = Malloc(double, l);
            double *sv_weight = Malloc(double, l);

            p = 0;
            for (int i = 0; i < nr_class; i++) {
                for (int j = i + 1; j < nr_class; j++) {
                    for (int k = 0; k < l; k++) sv_weight[k] = 0.0;

                    int si = start[i];
                    int sj = start[j];
                    int ci = model->nSV[i];
                    int cj = model->nSV[j];
                    double *coef1 = model->sv_coef[j - 1];
                    double *coef2 = model->sv_coef[i];

                    for (int k = 0; k < ci; k++) sv_weight[si + k] += coef1[si + k];
                    for (int k = 0; k < cj; k++) sv_weight[sj + k] += coef2[sj + k];

                    for (int r = 0; r < seqlen; r++)
                        for (int g = 0; g < MAX_ALPHABET_SIZE; g++)
                            explanation[r][g] = 0.0;

                    gkmexplainkernel_kernelfunc_batch_sv_weighted(x.d, kvalue, explanation, mode, sv_weight);

                    for (int c = 0; c < nr_class; c++) {
                        double w = dP_dR[c][p] * pair_slope[p];
                        if (w == 0.0) continue;
                        for (int r = 0; r < seqlen; r++) {
                            for (int g = 0; g < MAX_ALPHABET_SIZE; g++) {
                                class_expl[c][r][g] += w * explanation[r][g];
                            }
                        }
                    }
                    p++;
                }
            }

            for (int c = 0; c < nr_class; c++) {
                sb_appendf(&sb, "\t");
                sb_append_base_scores(&sb, class_expl[c], seqlen, mode);
            }

            free(kvalue);
            free(sv_weight);
            free(start);
            for (int c = 0; c < nr_class; c++) {
                for (int r = 0; r < seqlen; r++) free(class_expl[c][r]);
                free(class_expl[c]);
            }
            free(class_expl);
        }

        for (int i = 0; i < nr_class; i++) {
            free(pairwise_prob[i]);
            free(pairwise_prob_work[i]);
        }
        free(pairwise_prob);
        free(pairwise_prob_work);
        for (int c = 0; c < nr_class; c++) free(dP_dR[c]);
        free(dP_dR);
        free(prob_estimates);
        free(prob_eps);
        free(pair_slope);
        gkmkernel_delete_object(x.d);
    } else if (output_scores && nr_class > 2) {
        // Multiclass per-class scores.
        double *scores = (double *)calloc((size_t)nr_class, sizeof(double));
        if (!scores) {
            clog_error(CLOG(LOGGER_ID), "calloc failed for scores");
            exit(1);
        }

        union svm_data x;
        x.d = gkmkernel_new_object((char *)seq, NULL, 0);

        if (explain_one_class >= 0) {
            // Only one class requested.
            if (mode == 5) {
                svm_predict_and_singlebaseexplain_values_for_score_class(model, x, dec_values, explanation[0], explain_one_class);
            } else {
                svm_predict_and_explain_values_for_score_class(model, x, dec_values, explanation, mode, explain_one_class);
            }

            int idx = 0;
            for (int i = 0; i < nr_class; i++) {
                for (int j = i + 1; j < nr_class; j++) {
                    double v = dec_values[idx++];
                    scores[i] += v;
                    scores[j] -= v;
                }
            }
            for (int i = 0; i < nr_class; i++) scores[i] /= (nr_class - 1);

            sb_appendf(&sb, "\t%g\t", scores[explain_one_class]);
            sb_append_base_scores(&sb, explanation, seqlen, mode);
        } else {
            // Default: output all scores first.
            // Compute dec_values once (using class 0 explanation), then compute all scores.
            if (mode == 5) {
                svm_predict_and_singlebaseexplain_values_for_score_class(model, x, dec_values, explanation[0], 0);
            } else {
                svm_predict_and_explain_values_for_score_class(model, x, dec_values, explanation, mode, 0);
            }

            int idx = 0;
            for (int i = 0; i < nr_class; i++) {
                for (int j = i + 1; j < nr_class; j++) {
                    double v = dec_values[idx++];
                    scores[i] += v;
                    scores[j] -= v;
                }
            }
            for (int i = 0; i < nr_class; i++) scores[i] /= (nr_class - 1);

            for (int i = 0; i < nr_class; i++) sb_appendf(&sb, "\t%g", scores[i]);

            // Append base scores for class 0 (already computed).
            sb_appendf(&sb, "\t");
            sb_append_base_scores(&sb, explanation, seqlen, mode);

            // Append base scores for remaining classes.
            for (int c = 1; c < nr_class; c++) {
                // Explanation buffers are overwritten by the call.
                if (mode == 5) {
                    svm_predict_and_singlebaseexplain_values_for_score_class(model, x, dec_values, explanation[0], c);
                } else {
                    svm_predict_and_explain_values_for_score_class(model, x, dec_values, explanation, mode, c);
                }
                sb_appendf(&sb, "\t");
                sb_append_base_scores(&sb, explanation, seqlen, mode);
            }
        }

        gkmkernel_delete_object(x.d);
        free(scores);
        
    } else if (output_scores) {
        // Binary (or degenerate) per-class score output: keep historical format (single score).
        union svm_data x;
        x.d = gkmkernel_new_object((char *)seq, NULL, 0);
        if (mode == 5) {
            svm_predict_and_singlebaseexplain_values(model, x, dec_values, explanation[0]);
        } else {
            svm_predict_and_explain_values(model, x, dec_values, explanation, mode);
        }

        sb_appendf(&sb, "\t%g\t", dec_values[0]);
        sb_append_base_scores(&sb, explanation, seqlen, mode);
        gkmkernel_delete_object(x.d);
    } else {
        clog_error(CLOG(LOGGER_ID), "no score output mode selected");
        exit(1);
    }

    sb_appendf(&sb, "\n");

    char *linebuf = sb.data;

    free(dec_values);
    if (mode == 5) {
        free(explanation);
        free(singlebase_expl);
    } else {
        for (int i = 0; i < seqlen; i++) free(explanation[i]);
        free(explanation);
    }

    return linebuf;
}

static void *explain_worker_thread(void *ptr)
{
    explain_mt_ctx_t *ctx = (explain_mt_ctx_t *)ptr;

    while (1) {
        explain_job_t *job = job_queue_pop(&ctx->q);
        if (!job) break;

        char *out_line = build_output_line_for_record(job->sid, job->seq, job->seqlen, ctx->mode);

        pthread_mutex_lock(&ctx->out_mutex);
        ctx->out_lines[job->idx] = out_line;
        pthread_cond_broadcast(&ctx->out_cond);
        pthread_mutex_unlock(&ctx->out_mutex);

        free(job->sid);
        free(job->seq);
        free(job);
    }

    return NULL;
}



void predict_and_explain(FILE *input, FILE *output, int mode)
{
    int nthreads = gkmexplain_num_threads > 0 ? gkmexplain_num_threads : 1;
    if (nthreads < 1) nthreads = 1;

    explain_mt_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.mode = mode;
    pthread_mutex_init(&ctx.out_mutex, NULL);
    pthread_cond_init(&ctx.out_cond, NULL);

    // Bounded queue to cap memory usage.
    int qcap = nthreads * 2;
    if (qcap < 2) qcap = 2;
    job_queue_init(&ctx.q, qcap);

    ctx.out_cap = 0;
    ctx.out_lines = NULL;
    ctx.total_jobs = 0;
    ctx.next_to_write = 0;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)nthreads);
    if (!threads) {
        clog_error(CLOG(LOGGER_ID), "malloc failed for threads");
        exit(1);
    }
    for (int t = 0; t < nthreads; t++) {
        if (pthread_create(&threads[t], NULL, explain_worker_thread, &ctx) != 0) {
            clog_error(CLOG(LOGGER_ID), "pthread_create failed");
            exit(1);
        }
    }

    // FASTA streaming parser: build one record at a time and submit jobs.
    char *curr_sid = NULL;
    char *curr_seq = NULL;
    int curr_seqlen = 0;
    int curr_seq_cap = 0;
    int scored = 0;

    while (readline(input)) {
        if (line[0] == '>') {
            // flush previous
            if (curr_sid != NULL) {
                explain_job_t *job = (explain_job_t *)malloc(sizeof(explain_job_t));
                if (!job) {
                    clog_error(CLOG(LOGGER_ID), "malloc failed for job");
                    exit(1);
                }

                pthread_mutex_lock(&ctx.out_mutex);
                int idx = ctx.total_jobs;
                ctx.total_jobs++;
                if (ctx.total_jobs > ctx.out_cap) {
                    int new_cap = ctx.out_cap == 0 ? 256 : ctx.out_cap * 2;
                    while (new_cap < ctx.total_jobs) new_cap *= 2;
                    ctx.out_lines = (char **)realloc(ctx.out_lines, sizeof(char *) * (size_t)new_cap);
                    if (!ctx.out_lines) {
                        clog_error(CLOG(LOGGER_ID), "realloc failed for out_lines");
                        exit(1);
                    }
                    for (int i = ctx.out_cap; i < new_cap; i++) ctx.out_lines[i] = NULL;
                    ctx.out_cap = new_cap;
                }
                pthread_mutex_unlock(&ctx.out_mutex);

                job->idx = idx;
                job->sid = curr_sid;
                job->seq = curr_seq;
                job->seqlen = curr_seqlen;
                job_queue_push(&ctx.q, job);

                curr_sid = NULL;
                curr_seq = NULL;
                curr_seqlen = 0;
                curr_seq_cap = 0;
            }

            char *ptr = strtok(line, " \t\r\n");
            if (!ptr || strlen(ptr) < 2) {
                clog_error(CLOG(LOGGER_ID), "invalid FASTA header line");
                exit(1);
            }
            size_t sid_len = strlen(ptr + 1);
            if (sid_len >= MAX_SEQ_LENGTH) {
                clog_error(CLOG(LOGGER_ID), "maximum sequence id length is %d.\n", MAX_SEQ_LENGTH - 1);
                exit(1);
            }
            curr_sid = (char *)malloc(sid_len + 1);
            if (!curr_sid) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for sid");
                exit(1);
            }
            memcpy(curr_sid, ptr + 1, sid_len + 1);

            curr_seq_cap = 1024;
            curr_seq = (char *)malloc((size_t)curr_seq_cap);
            if (!curr_seq) {
                clog_error(CLOG(LOGGER_ID), "malloc failed for seq");
                exit(1);
            }
            curr_seq[0] = '\0';
            curr_seqlen = 0;
        } else {
            if (!curr_seq) continue;
            // append line (bounded by MAX_SEQ_LENGTH)
            int line_len = (int)strlen(line);
            if (curr_seqlen + line_len >= MAX_SEQ_LENGTH) {
                int remaining = (MAX_SEQ_LENGTH - 1) - curr_seqlen;
                if (remaining <= 0) continue;
                clog_warn(CLOG(LOGGER_ID),
                          "maximum sequence length allowed is %d. The first %d nucleotides of %s will only be used (Note: You can increase the MAX_SEQ_LENGTH parameter in libsvm_gkm.h and recompile).",
                          MAX_SEQ_LENGTH - 1, MAX_SEQ_LENGTH - 1, curr_sid ? curr_sid : "(unknown)");
                line[remaining] = '\0';
                line_len = remaining;
            }
            if (curr_seqlen + line_len + 1 > curr_seq_cap) {
                int new_cap = curr_seq_cap;
                while (new_cap < curr_seqlen + line_len + 1) new_cap *= 2;
                curr_seq = (char *)realloc(curr_seq, (size_t)new_cap);
                if (!curr_seq) {
                    clog_error(CLOG(LOGGER_ID), "realloc failed for seq");
                    exit(1);
                }
                curr_seq_cap = new_cap;
            }
            memcpy(curr_seq + curr_seqlen, line, (size_t)line_len);
            curr_seqlen += line_len;
            curr_seq[curr_seqlen] = '\0';
        }

        // flush completed outputs
        while (1) {
            pthread_mutex_lock(&ctx.out_mutex);
            if (ctx.next_to_write < ctx.total_jobs && ctx.out_lines[ctx.next_to_write] != NULL) {
                char *line_to_write = ctx.out_lines[ctx.next_to_write];
                ctx.out_lines[ctx.next_to_write] = NULL;
                ctx.next_to_write++;
                pthread_mutex_unlock(&ctx.out_mutex);
                fputs(line_to_write, output);
                free(line_to_write);
                scored++;
                if (gkmexplain_progress_every > 0 && (scored % gkmexplain_progress_every) == 0) {
                    clog_info(CLOG(LOGGER_ID), "%d scored", scored);
                    fflush(output);
                }
                continue;
            }
            pthread_mutex_unlock(&ctx.out_mutex);
            break;
        }
    }

    // flush last record if present
    if (curr_sid != NULL) {
        explain_job_t *job = (explain_job_t *)malloc(sizeof(explain_job_t));
        if (!job) {
            clog_error(CLOG(LOGGER_ID), "malloc failed for job");
            exit(1);
        }

        pthread_mutex_lock(&ctx.out_mutex);
        int idx = ctx.total_jobs;
        ctx.total_jobs++;
        if (ctx.total_jobs > ctx.out_cap) {
            int new_cap = ctx.out_cap == 0 ? 256 : ctx.out_cap * 2;
            while (new_cap < ctx.total_jobs) new_cap *= 2;
            ctx.out_lines = (char **)realloc(ctx.out_lines, sizeof(char *) * (size_t)new_cap);
            if (!ctx.out_lines) {
                clog_error(CLOG(LOGGER_ID), "realloc failed for out_lines");
                exit(1);
            }
            for (int i = ctx.out_cap; i < new_cap; i++) ctx.out_lines[i] = NULL;
            ctx.out_cap = new_cap;
        }
        pthread_mutex_unlock(&ctx.out_mutex);

        job->idx = idx;
        job->sid = curr_sid;
        job->seq = curr_seq;
        job->seqlen = curr_seqlen;
        job_queue_push(&ctx.q, job);
    }

    // stop workers once queue empty
    job_queue_mark_done(&ctx.q);

    // final ordered writer loop
    while (1) {
        pthread_mutex_lock(&ctx.out_mutex);
        while (ctx.next_to_write < ctx.total_jobs && ctx.out_lines[ctx.next_to_write] == NULL) {
            pthread_cond_wait(&ctx.out_cond, &ctx.out_mutex);
        }
        if (ctx.next_to_write >= ctx.total_jobs) {
            pthread_mutex_unlock(&ctx.out_mutex);
            break;
        }
        char *line_to_write = ctx.out_lines[ctx.next_to_write];
        ctx.out_lines[ctx.next_to_write] = NULL;
        ctx.next_to_write++;
        pthread_mutex_unlock(&ctx.out_mutex);
        fputs(line_to_write, output);
        free(line_to_write);
        scored++;
        if (gkmexplain_progress_every > 0 && (scored % gkmexplain_progress_every) == 0) {
            clog_info(CLOG(LOGGER_ID), "%d scored", scored);
            fflush(output);
        }
    }
    fflush(output);

    for (int t = 0; t < nthreads; t++) pthread_join(threads[t], NULL);
    free(threads);

    job_queue_destroy(&ctx.q);
    pthread_cond_destroy(&ctx.out_cond);
    pthread_mutex_destroy(&ctx.out_mutex);
    free(ctx.out_lines);

    clog_info(CLOG(LOGGER_ID), "%d scored", scored);

}


int main(int argc, char **argv)
{
    FILE *input, *output;
    int verbosity = 2;
    int nthreads = 1;
    int kthreads = 1;
    int progress_every = 100;
    int mode = 0;

    /* Initialize the logger */
    if (clog_init_fd(LOGGER_ID, 1) != 0) {
        fprintf(stderr, "Logger initialization failed.\n");
        return 1;
    }

    clog_set_fmt(LOGGER_ID, LOGGER_FORMAT);
    clog_set_level(LOGGER_ID, CLOG_INFO);

	if(argc == 1) { print_usage_and_exit(); }

	int c;
        while ((c = getopt (argc, argv, "v:m:t:T:p:DSC:P")) != -1) {
		switch (c) {
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 'm':
                mode = atoi(optarg);
                break;
            case 't':
                nthreads = atoi(optarg);
                if (nthreads < 1) {
                    fprintf(stderr, "Invalid thread count: %s\n", optarg);
                    print_usage_and_exit();
                }
                break;
            case 'T':
                kthreads = atoi(optarg);
                if (kthreads < 1) {
                    fprintf(stderr, "Invalid kernel thread count: %s\n", optarg);
                    print_usage_and_exit();
                }
                break;
            case 'p':
                progress_every = atoi(optarg);
                if (progress_every < 0) {
                    fprintf(stderr, "Invalid progress interval: %s\n", optarg);
                    print_usage_and_exit();
                }
                break;
            case 'D':
                output_decision_values = 1;
                break;
            case 'S':
                output_scores = 1;
                break;
            case 'C':
                explain_score_label = atoi(optarg);
                break;
            case 'P':
                output_probabilities = 1;
                break;
			default:
                fprintf(stderr,"Unknown option: -%c\n", c);
                print_usage_and_exit();
		}
	}

    if ((output_decision_values + output_scores + output_probabilities) > 1) {
        fprintf(stderr, "Options -D, -S, and -P are mutually exclusive.\n");
        print_usage_and_exit();
    }

    if (!output_decision_values && !output_scores && !output_probabilities) {
        fprintf(stderr, "One of -D, -S, or -P must be specified.\n");
        print_usage_and_exit();
    }

    if (explain_score_label != -1 && !output_scores) {
        fprintf(stderr, "Option -C requires -S (per-class scores).\n");
        print_usage_and_exit();
    }

    if (explain_score_label != -1 && output_probabilities) {
        fprintf(stderr, "Option -C cannot be used with -P.\n");
        print_usage_and_exit();
    }

    if (argc - optind != 3) {
        fprintf(stderr,"Wrong number of arguments [%d].\n", argc - optind);
        print_usage_and_exit();
    }

	int index = optind;
	char *testfile = argv[index++];
	char *modelfile = argv[index++];
	char *outfile = argv[index];

    switch(verbosity) 
    {
        case 0:
            clog_set_level(LOGGER_ID, CLOG_ERROR);
            break;
        case 1:
            clog_set_level(LOGGER_ID, CLOG_WARN);
            break;
        case 2:
            clog_set_level(LOGGER_ID, CLOG_INFO);
            break;
        case 3:
            clog_set_level(LOGGER_ID, CLOG_DEBUG);
            break;
        case 4:
            clog_set_level(LOGGER_ID, CLOG_TRACE);
            break;
        default:
            fprintf(stderr, "Unknown verbosity: %d\n", verbosity);
            print_usage_and_exit();
    }

    if (mode < 0 || mode > 5) {
            fprintf(stderr, "Unknown interpretation mode: %d\n", mode);
            print_usage_and_exit();
    }

    gkmexplain_num_threads = nthreads;
    gkmkernel_set_num_threads(kthreads);
    gkmexplain_progress_every = progress_every;

    input = fopen(testfile,"r");
    if(input == NULL) {
        clog_error(CLOG(LOGGER_ID),"can't open input file %s", testfile);
        exit(1);
    }

    output = fopen(outfile,"w");
    if(output == NULL) {
        clog_error(CLOG(LOGGER_ID),"can't open output file %s", outfile);
        exit(1);
    }

    clog_info(CLOG(LOGGER_ID), "load model %s", modelfile);
    if((model=svm_load_model(modelfile))==0) {
        clog_error(CLOG(LOGGER_ID),"can't open model file %s", modelfile);
        exit(1);
    }
    if (output_probabilities && !svm_check_probability_model(model)) {
        clog_error(CLOG(LOGGER_ID), "model does not contain probability estimates (probA/probB)");
        exit(1);
    }
    clog_info(CLOG(LOGGER_ID), "model loaded");
    max_line_len = 4096;
    line = (char *)malloc(((size_t) max_line_len) * sizeof(char));

    clog_info(CLOG(LOGGER_ID), "write prediction result to %s", outfile);
    predict_and_explain(input, output, mode);
    svm_free_and_destroy_model(&model);
    free(line);
    fclose(input);
    fclose(output);
    return 0;
}