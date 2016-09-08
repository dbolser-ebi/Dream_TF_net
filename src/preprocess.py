from datareader import *
from wiggleReader import *
from multiprocessing import Process
import argparse
import time


def print_num_instances_for_each_chromosome():
    chr_size = {}
    datareader = DataReader('../data/')
    for idx, instance in enumerate(datareader.generate_cross_celltype('train', 'CTCF', ['MCF-7'])):
        (chromosome, start), features, labels = instance
        if chromosome not in chr_size:
            chr_size[chromosome] = 0
        chr_size[chromosome] += 1

    return chr_size


def get_num_bound_lines(amb_as_bound=False):
    reader = DataReader('../data/')
    bound = {}
    for transcription_factor in reader.get_tfs():
        with gzip.open(reader.datapath + reader.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
            fin.readline()
            bound_lines = []
            # only the train set
            for line_idx, line in enumerate(fin):
                if 'B' in line or (amb_as_bound and 'A' in line):
                    bound_lines.append(line_idx)
            print transcription_factor, "num bound lines", len(bound_lines)
            bound[transcription_factor] = len(bound_lines)
    print bound


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
                assert(len(sbin) == bin_size)
                num_bins = bin_size/10
                bins = np.split(sbin, num_bins)
                print>> fout, np.max(sbin), np.percentile(sbin, 90), np.mean(sbin),
                for j in bins:
                    print >>fout, np.mean(j),
                print>>fout


def preprocess_dnase(num_jobs, bin_size):
    reader = DataReader('../data/')
    processes = []

    celltypes = reader.get_celltypes()

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


def get_ChIPSeq_fold_track(celltype, transcription_factor,  chromosome, left, right):
    fpath = os.path.join('../data/', 'ChIPseq.%s.%s.fc.signal.train.bw ' % (celltype, transcription_factor))
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


def parralelChIPSeqSignalProcessor(lines, fout_path, celltype, transcription_factor):
    with gzip.open(fout_path, 'w') as fout:
        for line in split_iter(lines):
            tokens = line.split()
            chromosome = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            track = get_ChIPSeq_fold_track(celltype, transcription_factor, chromosome, start, end)
            for i in range(0, track.size - 200, 50):
                sbin = track[i:i + 200]
                print>> fout, np.mean(sbin)


def preprocess_chipseq(num_jobs, bin_size):
    reader = DataReader('../data/')
    processes = []

    celltypes = reader.get_celltypes()
    transcription_factors = reader.get_tfs()

    for part in ['train']:
        with open('../data/annotations/%s_regions.blacklistfiltered.merged.bed' % part) as fin:
            lines = fin.read()

        for celltype in celltypes:
            for transcription_factor in transcription_factors:
                if not os.path.exists('../data/preprocess/CHIPSEQ_FEATURES/%s_%s_%s.txt' %
                                     (celltype, transcription_factor, part)):
                    fout_path = '../data/preprocess/CHIPSEQ_FEATURES/%s_%s_%s.gz' % (celltype, transcription_factor, part)
                    processes.append(
                        Process(target=parralelChIPSeqSignalProcessor,
                                args=(lines, fout_path, celltype, transcription_factor)))

    num_processes = num_jobs
    for i in range(0, len(processes), num_processes):
        map(lambda x: x.start(), processes[i:i + num_processes])
        map(lambda x: x.join(), processes[i:i + num_processes])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnase', action='store_true', required=False)
    parser.add_argument('--chipseq', action='store_true', required=False)
    parser.add_argument('--num_jobs', '-nj', help="number of cores to use", required=True)
    parser.add_argument('--bin_size', '-bs', help="bin size", required=False)
    args = parser.parse_args()
    bin_size = 200 if args.bin_size is None else int(args.bin_size)
    bin_size = max(200, bin_size)
    bin_size -= bin_size % 2

    if args.dnase:
        preprocess_dnase(int(args.num_jobs), bin_size)
    if args.chipseq:
        preprocess_chipseq(int(args.num_jobs), bin_size)

