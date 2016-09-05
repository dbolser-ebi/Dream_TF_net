from datareader import DataReader
from wiggleReader import *
import os
import pywt

def get_DNAse_fold_track(celltype, chromosome, left, right):
    fpath = os.path.join('../data/', 'dnase_fold_coverage/DNASE.%s.fc.signal.bigwig' % celltype)
    with open(fpath) as fin:
        fin.read()
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

def preprocess_dnase():
    reader = DataReader('../data/')
    num_train_instances = 51676736

    for celltype in ['A549', 'H1-hESC', 'HeLa-S3', 'HepG2', 'IMR-90', 'K562', 'MCF-7', 'GM12878']:
        print "CELLTYPE", celltype

        def process_dnase(fin):
            for line in fin:
                print line
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                track = get_DNAse_fold_track(celltype, chromosome, start, end)
                for i in range(0, track.size-200, 50):
                    sbin = track[i:i+200]
                    if sbin[sbin.nonzero()].size > 0:
                        print chromosome, i, i+200
                        print pywt.dwt(sbin, 'db1')
                        raw_input()

        with open('../data/annotations/train_regions.blacklistfiltered.merged.bed') as fin:
            process_dnase(fin)

preprocess_dnase()
