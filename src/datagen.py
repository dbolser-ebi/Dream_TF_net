import numpy as np
from pyfasta import Fasta
import os
import warnings
import argparse
import gzip
import pdb
import pandas as pd
from random import shuffle
import random


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
        self.chunk_size = 1000000
        self.num_channels = 4
        self.num_trans_fs = len(self.get_trans_fs())
        self.num_celltypes = len(self.get_celltypes())
        self.save_dir = os.path.join(self.datapath, 'preprocess/features')
        self.bin_size = 600
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def get_celltypes_for_trans_f(self, transcription_factor):
        '''
        Returns the list of celltypes for that particular transcription factor
        :param datapath: Path to the label directory
        :param transcription_factor: TF to Crossval
        :return: celltypes, a list of celltype for that TF
        '''
        files = [f for f in os.listdir(os.path.join(self.datapath, self.label_path))]
        celltypes = []
        for f_name in files:
            if transcription_factor in f_name:
                fpath = os.path.join(self.datapath, os.path.join(self.label_path, f_name))
                with gzip.open(fpath) as fin:
                    header_tokens = fin.readline().split()
                    celltypes = header_tokens[3:]
                break
        celltypes = list(set(celltypes))
        return celltypes

    ## UTILITIES
    def get_trans_fs(self):
        '''
        Returns the list of all TFs
        :return: list of trans_fs
        '''
        trans_fs = []
        files = [f for f in os.listdir(self.label_path)]
        trans_fs = []
        for f in files:
            trans_fs.append(f.split('.')[0])
        trans_fs = list(set(trans_fs))
        return trans_fs

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
    def generate_sequence(self, segment):
        if segment not in ['train', 'ladder', 'test']:
            raise Exception('Please specify the segment')
        bin_correction = max(0, (self.bin_size - 200) / 2)

        l_idx = 0
        with gzip.open('../data/annotations/%s_regions.blacklistfiltered.bed.gz' % segment) as fin:
            for line in fin:
                if l_idx % self.chunk_size == 0:
                    if l_idx != 0:
                        # save X
                        save_path = os.path.join(self.save_dir,
                                                 'segment_' + segment +
                                                 '_bin_size_' + str(self.bin_size) +
                                                 '_chunk_id_' + str(l_idx-X.shape[0]) + '.npy')
                        np.save(save_path, X)
                        print "Chunk", l_idx, "saved"

                    if segment == 'train':
                        length = min(self.chunk_size, self.train_length-l_idx)
                    if segment == 'ladder':
                        length = min(self.chunk_size, self.ladder_length-l_idx)
                    if segment == 'test':
                        length = min(self.chunk_size, self.test_length-l_idx)
                    X = np.zeros((length, self.bin_size, self.num_channels), dtype=np.float16)

                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                sequence = self.hg19[chromosome][start-bin_correction:start + self.sequence_length + bin_correction]
                sequence_one_hot = self.sequence_to_one_hot(np.array(list(sequence)))
                X[l_idx % self.chunk_size, :, :] = sequence_one_hot
                l_idx += 1
        # save remainder X
        save_path = os.path.join(self.save_dir,
                                 'segment_' + segment +
                                 '_bin_size_' + str(self.bin_size) +
                                 '_chunk_id_' + str(l_idx-X.shape[0]) + '.npy')
        if not os.path.exists(save_path):
            np.save(save_path, X)

    def generate_y(self):
        trans_f_lookup = self.get_trans_f_lookup()
        celltype_lookup = self.get_celltype_lookup()
        y = -1 * np.ones((self.train_length, self.num_trans_fs, self.num_celltypes), dtype=np.float16)

        for transcription_factor in self.get_trans_fs():
            path = os.path.join(self.label_path, '%s.train.labels.tsv.gz' % transcription_factor)
            labels = pd.read_csv(path, delimiter='\t')
            celltype_names = list(labels.columns[3:])

            for celltype in celltype_names:
                celltype_labels = np.array(labels[celltype])
                unbound_indices = np.where(celltype_labels == 'U')
                y[unbound_indices, trans_f_lookup[transcription_factor], celltype_lookup[celltype]] = 0
                ambiguous_indices = np.where(celltype_labels == 'A')
                y[ambiguous_indices, trans_f_lookup[transcription_factor], celltype_lookup[celltype]] = 0
                bound_indices = np.where(celltype_labels == 'B')
                y[bound_indices, trans_f_lookup[transcription_factor], celltype_lookup[celltype]] = 1

        for celltype in self.get_celltypes():
            np.save(os.path.join(self.save_dir, 'y_%s.npy' % celltype),
                    np.reshape(y[:, :, celltype_lookup[celltype]], (self.train_length, self.num_trans_fs))
                    )

    def get_bound_lookup(self):
        lookup = {}
        trans_f_lookup = self.get_trans_f_lookup()

        for celltype in self.get_celltypes():
            for trans_f in self.get_trans_fs():
                print celltype, trans_f, self.get_celltypes_for_trans_f(trans_f)
                if celltype not in self.get_celltypes_for_trans_f(trans_f):
                    continue
                y = np.load(os.path.join(self.save_dir, 'y_%s.npy' % celltype))
                lookup[(trans_f, celltype)] = np.where(y[:, trans_f_lookup[trans_f]] == 1)
        return lookup

    def get_bound_for_celltype(self, celltype):
        save_path = os.path.join(self.save_dir, 'bound_positions_%s.npy' % celltype)
        if os.path.exists(save_path):
            positions = np.load(save_path)
        else:
            print "Getting bound locations for celltype", celltype
            y = np.load(os.path.join(self.save_dir, 'y_%s.npy' % celltype))
            y = np.max(y, axis=1)
            positions = np.where(y == 1)[0]
            np.save(save_path, positions)
        return positions

    def get_bound_for_trans_f(self, trans_f):
        trans_f_lookup = self.get_trans_f_lookup()
        save_path = os.path.join(self.save_dir, 'bound_positions_%s.npy' % trans_f)
        if os.path.exists(save_path):
            positions = np.load(save_path)
        else:
            positions = []
            print "Getting bound locations for transcription factor", trans_f
            for celltype in self.get_celltypes_for_trans_f(trans_f):
                y = np.load(os.path.join(self.save_dir, 'y_%s.npy' % celltype))
                bound_locations = list(np.where(y[:, trans_f_lookup[trans_f]] == 1)[0])
                positions.extend(bound_locations)
            positions = list(set(positions))
            np.save(save_path, np.array(positions, dtype=np.int32))

        return positions

    def get_trans_f_lookup(self):
        lookup = {}
        for idx, trans_f in enumerate(self.get_trans_fs()):
            lookup[trans_f] = idx
        return lookup

    def get_celltype_lookup(self):
        lookup = {}
        for idx, celltype in enumerate(self.get_celltypes()):
            lookup[celltype] = idx
        return lookup

    def get_sequece_from_ids(self, ids, segment):
        id_to_position = {}
        for i, id in enumerate(ids):
            id_to_position[id] = i

        X = np.zeros((len(ids), self.bin_size, self.num_channels), dtype=np.float16)
        chunk_lookup = {}
        for id in ids:
            chunk_id = (id/self.chunk_size)*self.chunk_size
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = []
            chunk_lookup[chunk_id].append(id)
        for chunk_id in chunk_lookup.keys():
            chunk = np.load(os.path.join(self.save_dir,
                                             'segment_%s_bin_size_%d_chunk_id_%d.npy')
                                % (segment, self.bin_size, chunk_id))
            ids = chunk_lookup[chunk_id]
            x_idxs = map(lambda x: id_to_position[x], ids)
            ids_corrected = map(lambda x: x-chunk_id, ids)
            X[x_idxs] = chunk[ids_corrected]

        return X

    def get_bound_positions(self):
        save_path = os.path.join(self.save_dir, 'bound_positions.npy')
        if os.path.exists(save_path):
            bound_positions = np.load(save_path)
        else:
            bound_positions = []
            print "Getting bound locations"
            for celltype in self.get_celltypes():
                y = np.load(os.path.join(self.save_dir, 'y_%s.npy' % celltype))
                y = np.max(y, axis=1)
                locations = list(np.where(y == 1)[0])
                bound_positions.extend(locations)
            bound_positions = np.array(list(set(bound_positions)), dtype=np.int32)
            np.save(save_path, bound_positions)
        return bound_positions


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_sequence', action='store_true', required=False)
    parser.add_argument('--gen_y', action='store_true', required=False)
    parser.add_argument('--segment', required=True)

    args = parser.parse_args()
    datagen = DataGenerator()

    if args.gen_sequence:
        datagen.generate_sequence(args.segment)
    if args.gen_y:
        datagen.generate_y()
    '''

    datagen = DataGenerator()
    datagen.get_sequece_from_ids(random.sample(xrange(51676736), 100), 'train')




