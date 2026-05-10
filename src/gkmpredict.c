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

#include "libsvm_gkm.h"

#define CLOG_MAIN
#include "clog.h"

void print_usage_and_exit()
{
    printf(
            "\n"
            "Program: gkmpredict (lsgkm program for scoring sequences using a trained model)\n"
            "Version: "
            LSGKM_VERSION
            "\n\n"
            "Usage: gkmpredict [options] <test_seqfile> <model_file> <output_file>\n"
            "\n"
            " score test sequences using trained gkm-SVM.\n"
            " For binary models, outputs decision score per sequence.\n"
            " For multi-class models, outputs predicted class label per sequence.\n"
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
            "-t <int>         worker threads across sequences (default: 1)\n"
            "-T <1|4|16>      threads used *inside* the kernel calculation (default: 1)\n"
            "                 Notes:\n"
            "                 - For many sequences: set -t to ~#CPU cores and keep -T 1.\n"
            "                   This maximizes throughput and avoids nested threading.\n"
            "                 - For very few sequences (or a single long sequence): keep -t 1\n"
            "                   and try -T 4 or -T 16 to speed up each prediction.\n"
            "                 - If you set both -t>1 and -T>1, the program may create roughly\n"
            "                   -t * -T runnable threads during the busy part of computation.\n"
            " -D              output SVM decision values per sequence.\n"
            "                 Binary: one value. Multiclass: k*(k-1)/2 values (i<j order).\n"
            " -S              output aggregated per-class scores (margin-based).\n"
            "                 Score_i = (1/(k-1)) * sum_{j!=i} sign(i,j)*margin(i,j). Mutually exclusive with -D.\n"
            " -P              output per-class probability estimates (requires probA/probB in model).\n"
            " -p <int>         print progress every N sequences (default: 100; 0 disables)\n"
            "\n");
    exit(0);
}

static struct svm_model* model;
static int output_decision_values = 0; // -D
static int output_scores = 0; // -S per-class aggregated scores
static int output_probabilities = 0; // -P per-class probabilities

static char *line = NULL;
static int max_line_len;
static int gkmpredict_num_threads = 1;
static int gkmpredict_progress_every = 100;

// Simple bounded producer/consumer queue to avoid buffering the entire FASTA.
// This keeps memory stable even for very large input files.
typedef struct {
    int idx;
    char *sid;
    char *seq;
    int seqlen;
} predict_job_t;

typedef struct {
    predict_job_t **buf;
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
    int nr_class;
    char **out_lines;
    int out_cap;
    int total_jobs;
    int next_to_write;
    pthread_mutex_t out_mutex;
    pthread_cond_t out_cond;
} mt_ctx_t;

static void job_queue_init(job_queue_t *q, int cap)
{
    q->buf = (predict_job_t **)calloc((size_t)cap, sizeof(predict_job_t *));
    if (!q->buf) { clog_error(CLOG(LOGGER_ID), "calloc failed for job queue"); exit(1); }
    q->cap = cap; q->head = 0; q->tail = 0; q->count = 0; q->done = 0;
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

static void job_queue_push(job_queue_t *q, predict_job_t *job)
{
    pthread_mutex_lock(&q->mutex);
    while (q->count == q->cap) pthread_cond_wait(&q->not_full, &q->mutex);
    q->buf[q->tail] = job;
    q->tail = (q->tail + 1) % q->cap;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

static predict_job_t *job_queue_pop(job_queue_t *q)
{
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0 && !q->done) pthread_cond_wait(&q->not_empty, &q->mutex);
    if (q->count == 0 && q->done) { pthread_mutex_unlock(&q->mutex); return NULL; }
    predict_job_t *job = q->buf[q->head];
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

static char *predict_one_to_line(const char *sid, const char *seq, int nr_class)
{
    // Build output line as a string (thread-safe; workers never write to FILE*).
    union svm_data x; x.d = gkmkernel_new_object((char *)seq, NULL, 0);

    char *linebuf = NULL;
    if (output_probabilities) {
        double *prob_estimates = (double *)malloc(sizeof(double) * (size_t)nr_class);
        if (!prob_estimates) { clog_error(CLOG(LOGGER_ID), "malloc failed for prob_estimates"); exit(1); }
        svm_predict_probability(model, x, prob_estimates);
        size_t cap = strlen(sid) + 1;
        for (int i = 0; i < nr_class; i++) {
            int needed = snprintf(NULL, 0, "\t%g", prob_estimates[i]);
            cap += (size_t)needed;
        }
        cap += 2;
        linebuf = (char *)malloc(cap);
        if (!linebuf) { clog_error(CLOG(LOGGER_ID), "malloc failed for output line"); exit(1); }
        size_t off = 0;
        off += (size_t)snprintf(linebuf + off, cap - off, "%s", sid);
        for (int i = 0; i < nr_class; i++) off += (size_t)snprintf(linebuf + off, cap - off, "\t%g", prob_estimates[i]);
        linebuf[off++] = '\n'; linebuf[off] = '\0';
        free(prob_estimates);
    } else if (output_decision_values) {
        int ndec = (nr_class <= 2) ? 1 : (nr_class * (nr_class - 1) / 2);
        double *dec = (double *)malloc(sizeof(double) * (size_t)ndec);
        if (!dec) { clog_error(CLOG(LOGGER_ID), "malloc failed for dec"); exit(1); }
        svm_predict_values(model, x, dec);
        size_t cap = strlen(sid) + 1;
        for (int di = 0; di < ndec; di++) {
            int needed = snprintf(NULL, 0, "\t%g", dec[di]);
            cap += (size_t)needed;
        }
        cap += 2;
        linebuf = (char *)malloc(cap);
        if (!linebuf) { clog_error(CLOG(LOGGER_ID), "malloc failed for output line"); exit(1); }
        size_t off = 0;
        off += (size_t)snprintf(linebuf + off, cap - off, "%s", sid);
        for (int di = 0; di < ndec; di++) off += (size_t)snprintf(linebuf + off, cap - off, "\t%g", dec[di]);
        linebuf[off++] = '\n'; linebuf[off] = '\0';
        free(dec);
    } else if (output_scores) {
        int ndec = (nr_class <= 2) ? 1 : (nr_class * (nr_class - 1) / 2);
        double *dec = (double *)malloc(sizeof(double) * (size_t)ndec);
        if (!dec) { clog_error(CLOG(LOGGER_ID), "malloc failed for dec"); exit(1); }
        svm_predict_values(model, x, dec);
        double *scores = (double *)calloc((size_t)nr_class, sizeof(double));
        if (!scores) { clog_error(CLOG(LOGGER_ID), "calloc failed for scores"); exit(1); }
        int idxp = 0;
        for (int i = 0; i < nr_class; i++) {
            for (int j = i + 1; j < nr_class; j++) {
                double v = dec[idxp++];
                scores[i] += v;
                scores[j] -= v;
            }
        }
        for (int i = 0; i < nr_class; i++) scores[i] /= (nr_class - 1);
        size_t cap = strlen(sid) + 1;
        for (int i = 0; i < nr_class; i++) {
            int needed = snprintf(NULL, 0, "\t%g", scores[i]);
            cap += (size_t)needed;
        }
        cap += 2;
        linebuf = (char *)malloc(cap);
        if (!linebuf) { clog_error(CLOG(LOGGER_ID), "malloc failed for output line"); exit(1); }
        size_t off = 0;
        off += (size_t)snprintf(linebuf + off, cap - off, "%s", sid);
        for (int i = 0; i < nr_class; i++) off += (size_t)snprintf(linebuf + off, cap - off, "\t%g", scores[i]);
        linebuf[off++] = '\n'; linebuf[off] = '\0';
        free(scores);
        free(dec);
    } else if (nr_class <= 2) {
        double score;
        svm_predict_values(model, x, &score);
        int needed = snprintf(NULL, 0, "%s\t%g\n", sid, score);
        linebuf = (char *)malloc((size_t)needed + 1);
        if (!linebuf) { clog_error(CLOG(LOGGER_ID), "malloc failed for output line"); exit(1); }
        snprintf(linebuf, (size_t)needed + 1, "%s\t%g\n", sid, score);
    } else {
        double label = svm_predict(model, x);
        int needed = snprintf(NULL, 0, "%s\t%d\n", sid, (int)label);
        linebuf = (char *)malloc((size_t)needed + 1);
        if (!linebuf) { clog_error(CLOG(LOGGER_ID), "malloc failed for output line"); exit(1); }
        snprintf(linebuf, (size_t)needed + 1, "%s\t%d\n", sid, (int)label);
    }

    gkmkernel_delete_object(x.d);
    return linebuf;
}

static void *predict_worker(void *p)
{
    mt_ctx_t *ctx = (mt_ctx_t *)p;
    while (1) {
        predict_job_t *job = job_queue_pop(&ctx->q);
        if (!job) break;
        char *out_line = predict_one_to_line(job->sid, job->seq, ctx->nr_class);
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
    
    //remove CR ('\r') or LF ('\n'), whichever comes first
    line[strcspn(line, "\r\n")] = '\0';

    return line;
}

double calculate_score(char *seq)
{
    union svm_data x;
    double score;

    x.d = gkmkernel_new_object(seq, NULL, 0);

    svm_predict_values(model, x, &score);

    gkmkernel_delete_object(x.d);

    return score;
}

// static int predict_label(char *seq)
// {
//     union svm_data x;
//     x.d = gkmkernel_new_object(seq, NULL, 0);
//     double label = svm_predict(model, x);
//     gkmkernel_delete_object(x.d);
//     return (int)label;
// }

void predict(FILE *input, FILE *output)
{
    // Parallel prediction across sequences.
    // - We only keep a small bounded number of sequences in-flight.
    // - Output is written in the original input order.

    int nr_class = svm_get_nr_class(model);
    int nthreads = gkmpredict_num_threads > 0 ? gkmpredict_num_threads : 1;
    if (nthreads < 1) nthreads = 1;

    mt_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.nr_class = nr_class;
    pthread_mutex_init(&ctx.out_mutex, NULL);
    pthread_cond_init(&ctx.out_cond, NULL);

    int qcap = nthreads * 2;
    if (qcap < 2) qcap = 2;
    job_queue_init(&ctx.q, qcap);

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)nthreads);
    if (!threads) { clog_error(CLOG(LOGGER_ID), "malloc failed for threads"); exit(1); }
    for (int t = 0; t < nthreads; t++) {
        if (pthread_create(&threads[t], NULL, predict_worker, &ctx) != 0) {
            clog_error(CLOG(LOGGER_ID), "pthread_create failed");
            exit(1);
        }
    }

    // Stream FASTA and submit jobs
    char *curr_sid = NULL;
    char *curr_seq = NULL;
    int curr_seqlen = 0;
    int curr_seq_cap = 0;
    int scored = 0;

    while (readline(input)) {
        if (line[0] == '>') {
            if (curr_sid != NULL) {
                predict_job_t *job = (predict_job_t *)malloc(sizeof(predict_job_t));
                if (!job) { clog_error(CLOG(LOGGER_ID), "malloc failed for job"); exit(1); }

                pthread_mutex_lock(&ctx.out_mutex);
                int idx = ctx.total_jobs;
                ctx.total_jobs++;
                if (ctx.total_jobs > ctx.out_cap) {
                    int new_cap = ctx.out_cap == 0 ? 256 : ctx.out_cap * 2;
                    while (new_cap < ctx.total_jobs) new_cap *= 2;
                    ctx.out_lines = (char **)realloc(ctx.out_lines, sizeof(char *) * (size_t)new_cap);
                    if (!ctx.out_lines) { clog_error(CLOG(LOGGER_ID), "realloc failed for out_lines"); exit(1); }
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
            if (!ptr || strlen(ptr) < 2) { clog_error(CLOG(LOGGER_ID), "invalid FASTA header line"); exit(1); }
            size_t sid_len = strlen(ptr + 1);
            if (sid_len >= MAX_SEQ_LENGTH) {
                clog_error(CLOG(LOGGER_ID), "maximum sequence id length is %d.\n", MAX_SEQ_LENGTH - 1);
                exit(1);
            }
            curr_sid = (char *)malloc(sid_len + 1);
            if (!curr_sid) { clog_error(CLOG(LOGGER_ID), "malloc failed for sid"); exit(1); }
            memcpy(curr_sid, ptr + 1, sid_len + 1);

            curr_seq_cap = 1024;
            curr_seq = (char *)malloc((size_t)curr_seq_cap);
            if (!curr_seq) { clog_error(CLOG(LOGGER_ID), "malloc failed for seq"); exit(1); }
            curr_seq[0] = '\0';
            curr_seqlen = 0;
        } else {
            if (!curr_seq) continue;
            int line_len = (int)strlen(line);
            if (curr_seqlen + line_len >= MAX_SEQ_LENGTH) {
                int remaining = (MAX_SEQ_LENGTH - 1) - curr_seqlen;
                if (remaining <= 0) continue;
                line[remaining] = '\0';
                line_len = remaining;
            }
            if (curr_seqlen + line_len + 1 > curr_seq_cap) {
                int new_cap = curr_seq_cap;
                while (new_cap < curr_seqlen + line_len + 1) new_cap *= 2;
                curr_seq = (char *)realloc(curr_seq, (size_t)new_cap);
                if (!curr_seq) { clog_error(CLOG(LOGGER_ID), "realloc failed for seq"); exit(1); }
                curr_seq_cap = new_cap;
            }
            memcpy(curr_seq + curr_seqlen, line, (size_t)line_len);
            curr_seqlen += line_len;
            curr_seq[curr_seqlen] = '\0';
        }

        // flush ready outputs
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
                if (gkmpredict_progress_every > 0 &&
                    (scored % gkmpredict_progress_every) == 0) {
                    clog_info(CLOG(LOGGER_ID), "%d scored", scored);
                    fflush(output);
                }
                continue;
            }
            pthread_mutex_unlock(&ctx.out_mutex);
            break;
        }
    }

    if (curr_sid != NULL) {
        predict_job_t *job = (predict_job_t *)malloc(sizeof(predict_job_t));
        if (!job) { clog_error(CLOG(LOGGER_ID), "malloc failed for job"); exit(1); }

        pthread_mutex_lock(&ctx.out_mutex);
        int idx = ctx.total_jobs;
        ctx.total_jobs++;
        if (ctx.total_jobs > ctx.out_cap) {
            int new_cap = ctx.out_cap == 0 ? 256 : ctx.out_cap * 2;
            while (new_cap < ctx.total_jobs) new_cap *= 2;
            ctx.out_lines = (char **)realloc(ctx.out_lines, sizeof(char *) * (size_t)new_cap);
            if (!ctx.out_lines) { clog_error(CLOG(LOGGER_ID), "realloc failed for out_lines"); exit(1); }
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

    job_queue_mark_done(&ctx.q);

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
        if (gkmpredict_progress_every > 0 &&
            (scored % gkmpredict_progress_every) == 0) {
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
    int worker_threads = 1;
    int kernel_threads = 1;

    /* Initialize the logger */
    if (clog_init_fd(LOGGER_ID, 1) != 0) {
        fprintf(stderr, "Logger initialization failed.\n");
        return 1;
    }

    clog_set_fmt(LOGGER_ID, LOGGER_FORMAT);
    clog_set_level(LOGGER_ID, CLOG_INFO);

	if(argc == 1) { print_usage_and_exit(); }

	int c;
        while ((c = getopt (argc, argv, "v:T:t:p:DSP")) != -1) {
		switch (c) {
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 'T':
                kernel_threads = atoi(optarg);
                if (kernel_threads < 1) {
                    fprintf(stderr, "Invalid kernel thread count: %s\n", optarg);
                    print_usage_and_exit();
                }
                break;
            case 't':
                worker_threads = atoi(optarg);
                if (worker_threads < 1) {
                    fprintf(stderr, "Invalid worker thread count: %s\n", optarg);
                    print_usage_and_exit();
                }
                break;
            case 'p':
                gkmpredict_progress_every = atoi(optarg);
                if (gkmpredict_progress_every < 0) {
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
            case 'P':
                output_probabilities = 1;
                break;
			default:
                fprintf(stderr,"Unknown option: -%c\n", c);
                print_usage_and_exit();
		}
	}

    if (output_decision_values && output_scores) {
        fprintf(stderr, "Options -D and -S cannot be used together.\n");
        print_usage_and_exit();
    }

    if ((output_decision_values || output_scores) && output_probabilities) {
        fprintf(stderr, "Option -P cannot be combined with -D or -S.\n");
        print_usage_and_exit();
    }

    if (!output_decision_values && !output_scores && !output_probabilities) {
        fprintf(stderr, "At least one of -D, -S, or -P must be specified.\n");
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

    // Separate controls:
    // - worker_threads: parallelize across sequences
    // - kernel_threads: parallelize within a single kernel evaluation
    gkmpredict_num_threads = worker_threads;
    gkmkernel_set_num_threads(kernel_threads);

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

    max_line_len = 4096;
    line = (char *)malloc(((size_t) max_line_len) * sizeof(char));

    clog_info(CLOG(LOGGER_ID), "write prediction result to %s", outfile);
    predict(input, output);
    svm_free_and_destroy_model(&model);
    free(line);
    fclose(input);
    fclose(output);
    return 0;
}
