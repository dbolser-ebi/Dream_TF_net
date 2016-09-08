from datareader import DataReader
from wiggleReader import *
import os
import pywt
import matplotlib.pyplot as plt
import time
import gzip

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

    for celltype in ['MCF-7']:
        try:
            print "CELLTYPE", celltype

            c_binding = {}
            a_binding = {}

            tf = 'ATF2'

            for fname in os.listdir('../data/chipseq_labels'):
                if tf not in fname:
                    continue
                with gzip.open(os.path.join('../data/chipseq_labels', fname)) as fin:
                    header = fin.readline()
                    if celltype in header:
                        position = 0
                        tokens = header.split()
                        for idx, token in enumerate(tokens):
                            if celltype in token:
                                position = idx
                        for line in fin:
                            tokens = line.split()
                            chromosome = tokens[0]
                            start = int(tokens[1])
                            state = tokens[position]
                            if state == 'B':
                                if (chromosome, start) not in c_binding:
                                    c_binding[(chromosome, start)] = []
                                c_binding[(chromosome, start)].append(fname)
                            if state == 'A':
                                if (chromosome, start) not in a_binding:
                                    a_binding[(chromosome, start)] = []
                                a_binding[(chromosome, start)].append(fname)

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
                        if sbin[sbin.nonzero()].size > 0 or ((chromosome, start+i) in a_binding or (chromosome, start+i) in c_binding):
                            plt.plot(sbin)
                            print chromosome, start+i, start+i+200
                            if (chromosome, start+i) in a_binding: print 'ambg', a_binding[(chromosome, start+i)]
                            if (chromosome, start+i) in c_binding: print 'cons', c_binding[(chromosome, start+i)]
                            #print pywt.dwt(sbin, 'db1')
                            plt.show()

            with open('../data/annotations/train_regions.blacklistfiltered.merged.bed') as fin:
                process_dnase(fin)
        except KeyboardInterrupt:
            pass

preprocess_dnase()
