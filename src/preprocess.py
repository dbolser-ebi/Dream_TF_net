from multiprocessing import Process
import argparse
from subprocess import Popen, PIPE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import os
import numpy as np
import gzip
import time
import re
import pdb
from datagen import DataGenerator

# Low mem overhead splitting
def split_iter(string):
    return (x.group(0) for x in re.finditer(r"[ \t\f\v0-9A-Za-z,.=']+", string))

def get_DNAse_fold_track(celltype, chromosome, left, right):
    fpath = os.path.join('../data/', 'dnase_fold_coverage/DNASE.%s.fc.signal.bigwig' % celltype)
    process = Popen(["wiggletools", "seek", chromosome, str(left), str(right), fpath],
                    stdout=PIPE)
    (wiggle_output, _) = process.communicate()
    track = np.zeros((right - left + 1,), dtype=np.float32)
    position = 0
    for line in split_iter(wiggle_output):
        tokens = line.split()
        if line.startswith('fixedStep'):
            continue
        elif line.startswith('chr'):
            start = int(tokens[1]) + 1
            end = int(tokens[2])
            length = end-start+1
            value = float(tokens[3])
            track[position:position+length] = value
            position += length
        else:
            value = float(tokens[0])
            track[position] = value
            position += 1
    return track


def parralelDNAseSignalProcessor(lines, fout_path, celltype, bin_size):
    bin_correction = max(0, (bin_size - 200) / 2)

    with gzip.open(fout_path, 'w') as fout:
        for line in split_iter(lines):
            tokens = line.split()
            chromosome = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            track = get_DNAse_fold_track(celltype, chromosome, start-bin_correction, end+bin_correction)

            for i in range(start, end - 200 + 1, 50):
                sbin = track[i-start:i-start+bin_size]
                num_bins = bin_size/10
                bins = np.split(sbin, num_bins)
                print>> fout, np.max(sbin), np.percentile(sbin, 90), np.mean(sbin),
                for j in bins:
                    print >>fout, np.mean(j),
                print>>fout


def preprocess_dnase(num_jobs, bin_size):
    datagen = DataGenerator()
    processes = []

    celltypes = datagen.get_celltypes()

    for part in ['train', 'ladder', 'test']:

        with open('../data/annotations/%s_regions.blacklistfiltered.merged.bed' % part) as fin:
            lines = fin.read()

        for celltype in celltypes:

            if not os.path.exists('../data/preprocess/DNASE_FEATURES/%s_%s_%d.txt' % (celltype, part, bin_size)):
                fout_path = '../data/preprocess/DNASE_FEATURES/%s_%s_%d.gz' % (celltype, part, bin_size)
                processes.append(
                    Process(
                        target=parralelDNAseSignalProcessor,
                        args=(lines, fout_path, celltype, bin_size)))

    num_processes = num_jobs
    for i in range(0, len(processes), num_processes):
        map(lambda x: x.start(), processes[i:i + num_processes])
        map(lambda x: x.join(), processes[i:i + num_processes])


def parallel_normalize_chipseq(fpath):
    a = np.loadtxt(fpath, dtype=np.float32)
    a = MinMaxScaler().fit_transform(a.reshape(-1, 1))
    with gzip.open(fpath, 'w') as fout:
        for i in np.nditer(a):
            print >>fout, i


def normalize_chipseq(num_jobs):
    dir_path = '../data/preprocess/CHIPSEQ_FEATURES/'
    processes = []
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        processes.append(Process(
            target=parallel_normalize_chipseq,
            args=(fpath,)
        ))
    for i in range(0, len(processes), num_jobs):
        map(lambda x: x.start(), processes[i:i + num_jobs])
        map(lambda x: x.join(), processes[i:i + num_jobs])



def get_ChIPSeq_fold_track(celltype, transcription_factor,  chromosome, left, right):
    fpath = os.path.join('../data/', 'chipseq_fold_change_signal/ChIPseq.%s.%s.fc.signal.train.bw' % (celltype, transcription_factor))
    process = Popen(["wiggletools", "seek", chromosome, str(left), str(right), fpath],
                    stdout=PIPE)
    (wiggle_output, _) = process.communicate()
    track = np.zeros((right - left + 1,), dtype=np.float32)
    position = 0
    for line in split_iter(wiggle_output):
        tokens = line.split()
        if line.startswith('fixedStep'):
            continue
        elif line.startswith('chr'):
            start = int(tokens[1]) + 1
            end = int(tokens[2])
            length = end-start+1
            value = float(tokens[3])
            track[position:position+length] = value
            position += length
        else:
            value = float(tokens[0])
            track[position] = value
            position += 1
    return track


def parralelChIPSeqSignalProcessor(lines, fout_path, celltype, transcription_factor, bin_size):
    bin_correction = max(0, (bin_size - 200) / 2)
    with gzip.open(fout_path, 'w') as fout:
        for line in split_iter(lines):
            tokens = line.split()
            chromosome = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            track = get_ChIPSeq_fold_track(celltype, transcription_factor, chromosome,
                                           start-bin_correction, end+bin_correction)
            for i in range(start, end - 200 + 1, 50):
                sbin = track[i-start:i - start + bin_size]
                assert(len(sbin) == bin_size)
                print>> fout, np.mean(sbin)


def preprocess_chipseq(num_jobs, bin_size):
    datagen = DataGenerator()
    processes = []

    celltypes = datagen.get_celltypes()
    transcription_factors = datagen.get_trans_fs()

    for part in ['train']:
        with open('../data/annotations/%s_regions.blacklistfiltered.merged.bed' % part) as fin:
            lines = fin.read()

        for celltype in celltypes:
            for transcription_factor in transcription_factors:
                if not os.path.exists('../data/chipseq_fold_change_signal/ChIPseq.%s.%s.fc.signal.train.bw'
                                      % (celltype, transcription_factor)):
                    continue
                fout_path = '../data/preprocess/CHIPSEQ_FEATURES/%s_%s_%d.gz' % (
                                    celltype, transcription_factor, bin_size)
                if not os.path.exists(fout_path):
                    processes.append(
                        Process(target=parralelChIPSeqSignalProcessor,
                                args=(lines, fout_path, celltype, transcription_factor, bin_size)))

    for i in range(0, len(processes), num_jobs):
        map(lambda x: x.start(), processes[i:i + num_jobs])
        map(lambda x: x.join(), processes[i:i + num_jobs])


def parallel_normalize_dnaseq(ifpath, ofpath):
    print ifpath
    dnase_features = pd.read_csv(ifpath, delimiter=" ", dtype=np.float32, header=None)

    dnase_norm = StandardScaler().fit_transform(dnase_features)
    dnase_norm[np.isnan(dnase_norm)] = 0
    dnase_norm[np.isinf(dnase_norm)] = 0

    np.save(ofpath+'.npy', dnase_norm)


def normalize_dnaseq(num_jobs, bin_size):
    in_dir_path = '../data/preprocess/DNASE_FEATURES/'
    out_dir_path = '../data/preprocess/DNASE_FEATURES_NORM/'
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    processes = []
    for fname in os.listdir(in_dir_path):
        if str(bin_size) in fname:
            ifpath = os.path.join(in_dir_path, fname)
            ofpath = os.path.join(out_dir_path, fname)

            processes.append(Process(
                target=parallel_normalize_dnaseq,
                args=(ifpath, ofpath,)
            ))

    for i in range(0, len(processes), num_jobs):
        map(lambda x: x.start(), processes[i:i + num_jobs])
        map(lambda x: x.join(), processes[i:i + num_jobs])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dnase', action='store_true', required=False)
    parser.add_argument('--chipseq', action='store_true', required=False)
    parser.add_argument('--norm_dnase', action='store_true', required=False)
    parser.add_argument('--num_jobs', '-nj', help="number of cores to use", required=True)
    parser.add_argument('--bin_size', '-bs', help="bin size", required=False)
    args = parser.parse_args()
    bin_size = 200 if args.bin_size is None else int(args.bin_size)
    bin_size = max(200, bin_size)
    bin_size -= bin_size % 2
    num_jobs = int(args.num_jobs)

    if args.dnase:
        preprocess_dnase(num_jobs, bin_size)
    if args.chipseq:
        preprocess_chipseq(num_jobs, bin_size)
    if args.norm_dnase:
        normalize_dnaseq(num_jobs, bin_size)

