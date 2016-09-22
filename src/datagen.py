import numpy as np
from pyfasta import Fasta
import os
import warnings
import argparse
import gzip
import pdb
import pandas as pd


class DataGenerator:
    def __init__(self):
        self.datapath = '../data/'
        self.dna_peak_c_path = 'dnase_peaks_conservative/'
        self.label_path = os.path.join(self.datapath, 'chipseq_labels/')
        self.hg19 = Fasta(os.path.join(self.datapath, 'annotations/hg19.genome.fa'))
        self.sequence_length = 200

        self.train_length = 51676736
        self.ladder_length = 8843011
        self.test_length = 60519747
        self.chunck_size = 1000000
        self.num_channels = 4
        self.num_tfs = len(self.get_tfs())
        self.num_celltypes = len(self.get_celltypes())
        self.save_dir = os.path.join(self.datapath, 'preprocess/features')

    ## UTILITIES
    def get_tfs(self):
        '''
        Returns the list of all TFs
        :return: list of tfs
        '''
        tfs = []
        files = [f for f in os.listdir(self.label_path)]
        tfs = []
        for f in files:
            tfs.append(f.split('.')[0])
        tfs = list(set(tfs))
        return tfs

    def get_celltypes(self):
        celltypes = []
        for f in [f for f in os.listdir(self.datapath+self.dna_peak_c_path)]:
            celltypes.append(f.split('.')[1])
        celltypes = list(set(celltypes))
        return celltypes

    def sequence_to_one_hot(self, sequence):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoding = np.zeros((len(sequence), 4), dtype=np.float32)
            # Process A
            encoding[(sequence == 'A') | (sequence == 'a'), 0] = 1
            # Process C
            encoding[(sequence == 'C') | (sequence == 'c'), 1] = 1
            # Process G
            encoding[(sequence == 'G') | (sequence == 'g'), 2] = 1
            # Process T
            encoding[(sequence == 'T') | (sequence == 't'), 3] = 1
            return encoding

    # FEATURE GENERATION
    def generate_sequence(self, segment, bin_size):
        if segment not in ['train', 'ladder', 'test']:
            raise Exception('Please specify the segment')
        bin_correction = max(0, (bin_size - 200) / 2)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        l_idx = 0
        with gzip.open('../data/annotations/%s_regions.blacklistfiltered.bed.gz' % segment) as fin:
            for line in fin:
                if l_idx % self.chunck_size == 0:
                    if l_idx != 0:
                        # save X
                        save_path = os.path.join(self.save_dir,
                                                 'segment_' + segment +
                                                 '_bin_size_' + str(bin_size) +
                                                 '_chunk_id_' + str(l_idx-X.shape[0]) + '.npy')
                        np.save(save_path, X)
                        print "Chunk", l_idx, "saved"

                    if segment == 'train':
                        length = min(self.chunck_size, self.train_length-l_idx)
                    if segment == 'ladder':
                        length = min(self.chunck_size, self.ladder_length-l_idx)
                    if segment == 'test':
                        length = min(self.chunck_size, self.test_length-l_idx)
                    X = np.zeros((length, bin_size, self.num_channels), dtype=np.float16)

                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                sequence = self.hg19[chromosome][start-bin_correction:start + self.sequence_length + bin_correction]
                sequence_one_hot = self.sequence_to_one_hot(sequence)
                X[l_idx % self.chunck_size] = sequence_one_hot
                l_idx += 1
        # save remainder X
        save_path = os.path.join(self.save_dir,
                                 'segment_' + segment +
                                 '_bin_size_' + str(bin_size) +
                                 '_chunk_id_' + str(l_idx-X.shape[0]) + '.npy')
        if not os.path.exists(save_path):
            np.save(save_path, X)

    def generate_y(self):
        tf_lookup = self.get_tf_lookup()
        celltype_lookup = self.get_celltype_lookup()
        y = -1 * np.ones((self.train_length, self.num_tfs, self.num_celltypes), dtype=np.float16)

        for transcription_factor in self.get_tfs():
            path = os.path.join(self.label_path, '%s.train.labels.tsv.gz' % transcription_factor)
            labels = pd.read_csv(path, delimiter='\t')
            celltype_names = list(labels.columns[3:])

            for celltype in celltype_names:
                pdb.set_trace()
                celltype_labels = np.array(labels.celltype)
                unbound_indices = np.where(celltype_labels == 'U')
                y[unbound_indices, tf_lookup[transcription_factor], celltype_lookup[celltype]] = 0
                ambiguous_indices = np.where(celltype_labels == 'A')
                y[ambiguous_indices, tf_lookup[transcription_factor], celltype_lookup[celltype]] = 0
                bound_indices = np.where(celltype_labels == 'C')
                y[bound_indices, tf_lookup[transcription_factor], celltype_lookup[celltype]] = 1
            '''
            with gzip.open(path) as fin:
                celltype_names = fin.readline().split()[3:]
                for l_idx, line in enumerate(fin):
                    tokens = line.split()
                    bound_states = tokens[3:]

                    for i in range(len(bound_states)):
                        if bound_states[i] == 'B':
                            y[l_idx, tf_lookup[transcription_factor], celltype_lookup[celltype_names[i]]] = 1
                        else:
                            y[l_idx, tf_lookup[transcription_factor], celltype_lookup[celltype_names[i]]] = 0
            '''
            pdb.set_trace()

        np.save(os.path.join(self.save_dir, 'y.npy'), y)

    def get_bound_lookup(self):
        lookup = {}
        tf_lookup = self.get_tf_lookup()
        celltype_lookup = self.get_celltype_lookup()
        y = np.load(os.path.join(self.save_dir, 'y.npy'))
        for tf in self.get_tfs():
            for celltype in self.get_celltypes():
                lookup[(tf, celltype)] = np.where(y[:, tf_lookup[tf], celltype_lookup[celltype]])
        return lookup

    def get_tf_lookup(self):
        lookup = {}
        for idx, tf in enumerate(self.get_tfs()):
            lookup[tf] = idx
        return lookup

    def get_celltype_lookup(self):
        lookup = {}
        for idx, celltype in enumerate(self.get_celltypes()):
            lookup[celltype] = idx
        return lookup

    def get_sequece_from_ids(self, ids, segment):
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_sequence', action='store_true', required=False)
    parser.add_argument('--gen_y', action='store_true', required=False)
    parser.add_argument('--segment', required=True)
    parser.add_argument('--bin_size', required=True)

    args = parser.parse_args()
    bin_size = int(200 if args.bin_size is None else args.bin_size)
    bin_size -= bin_size % 2

    datagen = DataGenerator()

    if args.gen_sequence:
        datagen.generate_sequence(args.segment, bin_size)
    if args.gen_y:
        datagen.generate_y()
