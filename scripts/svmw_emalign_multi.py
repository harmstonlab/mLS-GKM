#!/usr/bin/env python
"""
    svmw_emalign.py: build de novo PWMs from SVM scores

    Copyright (C) 2014 Dongwon Lee

    Modified 2026 by Kieran Howard to support multiple classes and parallel execution.

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

import sys
import numpy
import random
import optparse
import logging
import os
import multiprocessing
import time
from math import log, exp, sqrt

# Background letter frequencies (A, C, G, T)
BG_FREQ = [0.29, 0.21, 0.21, 0.29]

logging.basicConfig(level=20,
                    format='%(levelname)-5s @ %(asctime)s: %(message)s ',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    stream=sys.stderr,
                    filemode="w")


def _print_info(msg):
    # Match the basicConfig format closely: "INFO  @ Fri, 16 Jan 2026 12:12:05: ..."
    # Keep this separate so we can suppress noisy worker logs while still emitting
    # a few high-level progress lines.
    ts = time.strftime('%a, %d %b %Y %H:%M:%S')
    sys.stderr.write('INFO  @ %s: %s\n' % (ts, msg))

def revcomp(seq):
    """get reverse complement DNA sequence

    Arguments:
    seq -- string, DNA sequence

    Return:
    the reverse complement sequence of the given sequence
    """
    rc = {'A':'T', 'G':'C', 'C':'G', 'T':'A'}
    return ''.join([rc[seq[i]] for i in range(len(seq)-1, -1, -1)])


def read_fastafile(filename, subs=True):
    """Read a file in FASTA format

    Arguments:
    filename -- string, the name of the sequence file in FASTA format

    Return: 
    list of sequences, list of sequence ids
    """
    id = '' 
    ids = []
    seqs = []

    try:
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()

    except IOError as e:
        print("I/O error: %s" % e)
        sys.exit(0)

    seq = [] 
    for line in lines:
        if line[0] == '>':
            seqid = '_'.join(line[1:].rstrip('\n').split())
            ids.append(seqid)
            if seq != []: seqs.append("".join(seq))
            seq = []
        else:
            if subs:
                seq.append(line.rstrip('\n').replace('N', 'A').upper())
            else:
                seq.append(line.rstrip('\n').upper())

    if seq != []:
        seqs.append("".join(seq))

    return seqs, ids


def seq2pwm(seq, pwmlen):
    nt2idx = {'A':0, 'C':1, 'G':2, 'T':3}
    bgfreq = BG_FREQ
    columns = [[1.0, 0.0, 0.0, 0.0],\
            [0.0, 1.0, 0.0, 0.0],\
            [0.0, 0.0, 1.0, 0.0],\
            [0.0, 0.0, 0.0, 1.0]]
    seqlen = len(seq)

    if pwmlen < seqlen:
        offset = int((seqlen-pwmlen)/2) # floor
        pwm = [columns[nt2idx[nt]] for nt in seq[offset:offset+pwmlen]]
    elif pwmlen == seqlen:
        pwm = [columns[nt2idx[nt]] for nt in seq]
    else:
        pwm = []
        paddinglen = int((pwmlen-seqlen)/2)
        for i in range(paddinglen):
            pwm.append(list(bgfreq))

        for nt in seq:
            pwm.append(list(columns[nt2idx[nt]]))

        for i in range(pwmlen-paddinglen-seqlen):
            pwm.append(list(bgfreq))
    
    return pwm

def pearson(pwm1, pwm2):
    n = len(pwm1)
    avg1sq = (pwm1**2).sum()/n
    avg2sq = (pwm2**2).sum()/n
    avg12 = (pwm1*pwm2).sum()/n
    return (avg12 - (0.25**2))/sqrt((avg1sq-(0.25**2))*(avg2sq-(0.25**2)))


def pearsonc(pwm1, pwm2, avg1sq):
    n = len(pwm1)
    avg2sq = (pwm2**2).sum()/n
    avg12 = (pwm1*pwm2).sum()/n
    return (avg12 - (0.25**2))/sqrt((avg1sq-(0.25**2))*(avg2sq-(0.25**2)))


def euclidean(pwm1, pwm2):
    return numpy.sum(numpy.sqrt(numpy.sum((pwm1-pwm2)**2, axis=1)))


def write_memefile(models, modelnames, nsites, nameprefix, outfile):
    f = open(outfile, 'w')

    f.write("MEME version 4\n\n")
    f.write("ALPHABET= ACGT\n\n")
    f.write("strands: + -\n\n")
    f.write("Background letter frequencies (from entire human genome)\n")
    f.write("A %.2f C %.2f G %.2f T %.2f\n\n" % (BG_FREQ[0], BG_FREQ[1], BG_FREQ[2], BG_FREQ[3]))

    n=0
    for centerid in sorted(models.keys()):
        n += 1
        pwmlen = len(models[centerid])
        ns = nsites[centerid]
        #f.write("MOTIF %s.%d.%s\n" % (nameprefix, centerid+1, modelnames[centerid])) #0-base index of centerid to 1-base index
        f.write("MOTIF %s.%d\n" % (nameprefix, centerid+1)) #0-base index of centerid to 1-base index
        f.write("letter-probability matrix: alength= 4 w= %d nsites= %d E= 0\n" % (pwmlen, ns))
        for j in range(pwmlen):
            freqs = [ int(p * ns) for p in models[centerid][j] ]
            #ensure that the sum of row equal 1
            if sum(freqs) != ns:
                diff = ns - sum(freqs)
                if diff > 0:
                    diff_inc = 1
                else:
                    diff_inc= -1

                sorted_freqs = sorted(zip(range(4), freqs), key=lambda x: x[1], reverse=True)

                for i in range(abs(diff)):
                    freqs[sorted_freqs[i][0]] += diff_inc

            f.write(' '.join(["%.3f" % (x/float(ns)) for x in freqs ]) + '\n')
        f.write('\n')
    f.close()


def write_resultfile(seqids, seqs, aligninfo, outfile):
    f = open(outfile, 'w')

    for i in range(len(seqids)):
        coord = seqids[i].split('_')
        (chrom, position) = coord[0].split(':')
        (start, end) = position.split('-')

        (skmerind, los, mpos, rc) = aligninfo[i]

        f.write('\t'.join([seqids[i], seqs[i], str(skmerind+1), str(mpos), str(rc), str(los)]) + '\n')
    f.close()


def emalign(skmer, skmerind, kmers, svmw, pwmlen, options):
    bestmodel = None
    bestshft = -1
    bestobj = -9999
    bestalignments = None
    smallp = 1e-3
    bgfreq = numpy.array(BG_FREQ)
    bglogp = numpy.log2((bgfreq + smallp) / (1 + 4 * smallp))
    nt2idx = {'A':0, 'C':1, 'G':2, 'T':3}
    window = len(skmer)

    logging.info("seed kmer: %s" % (skmer))

    kmerpwms = dict()
    for ind in kmers.keys():
        kmer = kmers[ind]
        kmerrc = revcomp(kmer)
        kmerpwms[ind] = (seq2pwm(kmer, len(kmer)), seq2pwm(kmerrc, len(kmer)))

    model = numpy.array(seq2pwm(skmer, pwmlen))
    alignments = None

    los_max_total_prev = 0
    for iround in range(options.iterations):
        used_kmers = dict()
        alignments = dict() 
        lomodel = numpy.log2((model + smallp)/(1+4*smallp))-bglogp

        logging.info("round - %d/%d" % (iround, options.iterations))

        #1. find the best alignment to the current model
        for ind in kmers.keys():
            los_max = -9999
            mpos_max = -1
            rc_max = None
            rc = False
            for kmer in (kmers[ind], revcomp(kmers[ind])):
                for mpos in range(pwmlen-window+1):
                    los = 0
                    for i in range(window):
                        los += lomodel[mpos+i][nt2idx[kmer[i]]]

                    if los > los_max:
                        los_max = los
                        mpos_max = mpos
                        rc_max = rc
                rc = not rc

            alignments[ind] = (skmerind, los_max, mpos_max, rc_max)

        los_max_total_curr = 0
        #2. update the current model using the best alignments
        prev_model = model
        #reset the current model
        model = numpy.zeros((len(model), 4))
        for ind in kmerpwms.keys():
            (skmer_ind, los, mpos, rc) = alignments[ind]

            if los < options.cutoff:
                continue

            los_max_total_curr += los
            used_kmers[ind] = True
            kmerpwm = list(kmerpwms[ind][int(rc)])

            kmerinstance = kmerpwm
            offset = mpos
            while offset > 0:
                kmerinstance.insert(0, [0.0, 0.0, 0.0, 0.0])
                offset -= 1
            for i in range(pwmlen - len(kmerinstance)):
                kmerinstance.append([0.0, 0.0, 0.0, 0.0])

            #print ind, skmer_ind, los, mpos, rc, svmw[kmers[ind]]
            #print model
            #print numpy.array(kmerinstance)

            model += numpy.exp(options.alpha*svmw[kmers[ind]]*numpy.array(kmerinstance))

        row_sums = model.sum(axis=1)
        if len(used_kmers) == 0:
            logging.info("no kmers passed cutoff; stopping EM for this seed")
            model = prev_model
            break

        zero_rows = (row_sums == 0)
        if numpy.any(zero_rows):
            model[zero_rows, :] = bgfreq
            row_sums[zero_rows] = 1.0

        model = model / row_sums[:, numpy.newaxis]

        logging.info("obj: %.6f" % los_max_total_curr)
        #logging.info(model)

        #no changes. reached local optimum
        if abs(los_max_total_curr - los_max_total_prev) < 1e-9:
            break

        los_max_total_prev = los_max_total_curr

    if los_max_total_curr > bestobj:
        bestobj = los_max_total_curr
        bestmodel = model
        bestalignments = alignments

    return bestmodel, bestalignments, used_kmers


def read_svmscorefile(filename):
    """Read a k-mer score file.

    Expected format (whitespace-delimited):
        kmer  score_class1  score_class2  ...

    Lines starting with '#' are ignored.

    Return:
        (kmers, scores_by_class)
        - kmers: list of k-mers in file order
        - scores_by_class: list of dicts, one per class, mapping kmer->score
    """

    kmers = []
    scores_by_class = None

    try:
        f = open(filename, 'r')
        for raw in f:
            line = raw.strip()
            if line == "" or line[0] == '#':
                continue
            s = line.split()
            if len(s) < 2:
                continue

            kmer = s[0]
            try:
                scores = [float(x) for x in s[1:]]
            except ValueError:
                # allow skipping a header row if present
                continue

            if scores_by_class is None:
                scores_by_class = [dict() for _ in range(len(scores))]
            elif len(scores) != len(scores_by_class):
                raise RuntimeError(
                    "Inconsistent number of score columns: expected %d, saw %d" % (
                        len(scores_by_class), len(scores)
                    )
                )

            kmers.append(kmer)
            for i in range(len(scores)):
                scores_by_class[i][kmer] = scores[i]
        f.close()

    except IOError as e:
        print("I/O error: %s" % e)
        sys.exit(0)

    if scores_by_class is None:
        raise RuntimeError("No k-mer score rows found in %s" % filename)

    return kmers, scores_by_class


def _build_motifs_for_class(class_name, svmw_base, pwmlen, options, outdir):
    """Run the original EM motif discovery for one class and write CLASS.meme."""
    if getattr(options, '_quiet', False):
        _print_info("=== Building motifs for class: %s ===" % (class_name,))
    else:
        logging.info("=== Building motifs for class: %s ===" % (class_name,))

    # Build a local score dict and add reverse-complement scores.
    # NOTE: This doubles the k-mer dictionary size for the class.
    svmw = dict(svmw_base)
    svmw_rc = dict()
    for kmer in svmw_base.keys():
        svmw_rc[revcomp(kmer)] = svmw_base[kmer]
    svmw.update(svmw_rc)

    models = dict()
    modelnames = dict()
    nsites = dict()
    aligninfo = dict()
    visited = dict()

    # select k-mers of interest
    kmers_sorted = sorted(svmw.keys(), key=lambda s: svmw[s], reverse=True)
    topN = int(len(kmers_sorted) * options.top_frac / 100.0)
    seedkmers = kmers_sorted[:topN]

    remainings = len(seedkmers)
    npwms = 0
    failed = 0
    while (remainings > 0) and (npwms < options.nmaxpwms) and (failed < 10):
        skmer = ""
        skmerind = -1
        kmers_to_align = dict()
        for i in range(len(seedkmers)):
            if not i in visited:
                if skmerind == -1:
                    skmer = seedkmers[i]
                    skmerind = i

                kmers_to_align[i] = seedkmers[i]

        model, alignments, used_kmers = emalign(skmer, skmerind, kmers_to_align, svmw, pwmlen, options)
        nkmers = 0
        for ind in used_kmers.keys():
            nkmers += 1
            visited[ind] = True

        # skip weak seeds
        if nkmers < options.nminkmers:
            logging.info("skip %s (number of kmers aligned is %d < %d)" % (skmer, nkmers, options.nminkmers))
            visited[skmerind] = 1  # remove the seed
            failed += 1
            continue

        models[skmerind] = model
        modelnames[skmerind] = skmer
        nsites[skmerind] = nkmers
        aligninfo.update(alignments)

        remainings = 0
        for i in range(len(seedkmers)):
            if not i in visited:
                remainings += 1

        npwms += 1

    # ensure output directory exists
    try:
        if outdir and not os.path.isdir(outdir):
            os.makedirs(outdir)
    except Exception:
        pass

    outfile = outdir.rstrip("/") + "/" + class_name + ".meme" if outdir else class_name + ".meme"
    write_memefile(models, modelnames, nsites, class_name, outfile)
    if not getattr(options, '_quiet', False):
        logging.info("Wrote %s" % outfile)
    return outfile


# Globals for multiprocessing workers (avoids passing large dicts per task on fork-based systems)
_G_SCORES_BY_CLASS = None
_G_CLASS_NAMES = None
_G_PWMLEN = None
_G_OPTIONS = None
_G_OUTDIR = None
_G_QUIET = False


def _init_pool(scores_by_class, class_names, pwmlen, options, outdir, quiet):
    global _G_SCORES_BY_CLASS, _G_CLASS_NAMES, _G_PWMLEN, _G_OPTIONS, _G_OUTDIR, _G_QUIET
    _G_SCORES_BY_CLASS = scores_by_class
    _G_CLASS_NAMES = class_names
    _G_PWMLEN = pwmlen
    _G_OPTIONS = options
    _G_OUTDIR = outdir
    _G_QUIET = quiet

    # In parallel mode, suppress all the per-iteration logging coming from emalign().
    # We'll still emit a few high-level INFO lines via _print_info().
    if quiet:
        try:
            logging.getLogger().setLevel(logging.CRITICAL)
        except Exception:
            pass


def _worker_run_class(class_i):
    class_name = _G_CLASS_NAMES[class_i]
    svmw_base = _G_SCORES_BY_CLASS[class_i]
    if _G_QUIET:
        setattr(_G_OPTIONS, '_quiet', True)
    return _build_motifs_for_class(class_name, svmw_base, _G_PWMLEN, _G_OPTIONS, _G_OUTDIR)

def main(argv = sys.argv):
    usage = "Usage: %prog [options] KMER_SCORES PWM_LENGTH [OUT_DIR]"
    desc  = "1. take a k-mer score file (multiclass ok) 2. perform EM to align k-mers per class"

    parser = optparse.OptionParser(usage=usage, description=desc)                                                                              
    parser.add_option("-a", dest="alpha", type="float", default=3.0, \
            help="set the alpha (multiplier) for svm weights (default=3.0)")

    parser.add_option("-i", dest="iterations", type="int", default=100, \
            help="set the maximum number of iterations for EM (default=100)")

    parser.add_option("-n", dest="nmaxpwms", type="int", default=5, \
            help="set the maximum number of pwms to build (default=5)")

    parser.add_option("-m", dest="nminkmers", type="int", default=100, \
            help="set the minium number of kmers for each pwm (default=100)")

    parser.add_option("-f", dest="top_frac", type="int", default=1, \
            help="set the percentage of the top k-mers to be evaluated as seed k-mers (default=1(%))")

    parser.add_option("-c", dest="cutoff", type="float", default=5, \
            help="set the cut-off of minimum log oddratio sum from the model when model rebuilding (default=5)")

    parser.add_option("-p", dest="nameprefix", default="GKM", \
            help="set the prefix of pwm names (e.g. ${PREFIX}001. default='GKM')")

    parser.add_option("--classes", dest="classes", default="CNE,Enhancer,Random", \
            help="comma-separated class names; used for output filenames (default='CNE,Enhancer,Random')")

    parser.add_option("--max_parallel", dest="max_parallel", type="int", default=1, \
            help="Max number of classes to process in parallel (default=1; set to >1 to speed up)")

    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    if len(args) < 2 or len(args) > 3:
        parser.error("incorrect number of arguments")
        sys.exit(0)

    svmscorefile = args[0]
    pwmlen = int(args[1])
    outdir = "." if len(args) < 3 else args[2]

    class_names = [c.strip() for c in options.classes.split(',') if c.strip()]
    if len(class_names) == 0:
        raise RuntimeError("--classes must contain at least one class name")

    _, scores_by_class = read_svmscorefile(svmscorefile)
    if len(scores_by_class) != len(class_names):
        raise RuntimeError(
            "Number of score columns (%d) does not match number of class names (%d). "
            "Pass --classes with exactly %d names." % (
                len(scores_by_class), len(class_names), len(scores_by_class)
            )
        )

    # run motif discovery independently for each class score column
    max_p = int(options.max_parallel)
    if max_p < 1:
        raise RuntimeError("--max_parallel must be >= 1")

    if max_p == 1 or len(class_names) == 1:
        for class_i in range(len(class_names)):
            _build_motifs_for_class(class_names[class_i], scores_by_class[class_i], pwmlen, options, outdir)
    else:
        procs = min(max_p, len(class_names))
        _print_info("Running %d classes in parallel (max_parallel=%d)" % (procs, max_p))
        pool = multiprocessing.Pool(
            processes=procs,
            initializer=_init_pool,
            initargs=(scores_by_class, class_names, pwmlen, options, outdir, True),
        )
        try:
            pool.map(_worker_run_class, list(range(len(class_names))))
        finally:
            pool.close()
            pool.join()


if __name__=='__main__': main()
