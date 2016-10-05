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
from enum import Enum
from wiggleReader import *
from multiprocessing import Process


class CrossvalOptions(Enum):
    filter_on_DNase_peaks = 1
    balance_peaks = 2
    random_shuffle_10_percent = 3


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
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


    ###################### UTILITIES #############################################################

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

    def get_celltypes_for_tf(self, transcription_factor):
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

    def get_trans_fs_for_celltype_train(self, celltype):
        '''
        Returns the list of transcription factors for that particular celltype
        :param datapath: Path to the label directory
        :param celltype: celltype to Crossval
        :return:
        '''
        files = [f for f in os.listdir(self.label_path)]
        tfs = []
        for f in files:
            fpath = os.path.join(self.label_path, f)
            with gzip.open(fpath) as fin:
                header = fin.readline()
                if celltype in header:
                    tf = f.split('.')[0]
                    tfs.append(tf)
        return list(set(tfs))

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

    def get_sequece_from_ids(self, ids, segment, bin_size=600):
        id_to_position = {}
        for i, id in enumerate(ids):
            id_to_position[id] = i

        X = np.zeros((len(ids), bin_size, self.num_channels), dtype=np.float16)
        chunk_lookup = {}
        for id in ids:
            chunk_id = (id / self.chunk_size) * self.chunk_size
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = []
            chunk_lookup[chunk_id].append(id)
        for chunk_id in chunk_lookup.keys():
            chunk = np.load(os.path.join(self.save_dir,
                                         'segment_%s_bin_size_%d_chunk_id_%d.npy')
                            % (segment, bin_size, chunk_id))
            ids = chunk_lookup[chunk_id]
            x_idxs = map(lambda x: id_to_position[x], ids)
            ids_corrected = map(lambda x: x - chunk_id, ids)
            X[x_idxs] = chunk[ids_corrected]

        return X

    def get_dnase_features_from_ids(self, ids, segment, celltype, dnase_bin_size=600):
        return np.load(os.path.join(self.save_dir, 'dnase_fold_%s_%s_%d.npy' % (segment, celltype, dnase_bin_size)))[ids]

    def get_dnase_features(self, segment, celltype, dnase_bin_size=600):
        return np.load(os.path.join(self.save_dir, 'dnase_fold_%s_%s_%d.npy' % (segment, celltype, dnase_bin_size)))

    def get_y(self, celltype):
        return np.load('../data/preprocess/features/y_%s.npy' % celltype)

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

    def generate_position_tree(self, transcription_factor, celltypes, options=CrossvalOptions.balance_peaks,
                                unbound_fraction=1):
        position_tree = set()  # keeps track of which lines (chr, start) to include

        if options == CrossvalOptions.balance_peaks:
            with gzip.open(self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
                bound_lines = []
                unbound_lines = []
                # only the train set
                for line_idx, line in enumerate(fin):
                    if 'B' in line:
                        bound_lines.append(line_idx)
                    else:
                        unbound_lines.append(line_idx)
                random.shuffle(unbound_lines)
                if unbound_fraction > 0:
                    bound_lines.extend(
                        unbound_lines[:int(min(len(bound_lines) * unbound_fraction, len(unbound_lines)))])
                position_tree.update(set(bound_lines))

        elif options == CrossvalOptions.random_shuffle_10_percent:
            with gzip.open(self.datapath + self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
                fin.readline()
                a = range(len(fin.read().splitlines()))
                random.shuffle(a)
                position_tree.update(a[:int(0.1 * len(a))])

        print "len position treee", len(position_tree)

        return position_tree

    def get_train_data(self, part, transcription_factor, celltypes, options=CrossvalOptions.balance_peaks,
                                unbound_fraction=1, bin_size=600, dnase_bin_size=600):
        position_tree = self.generate_position_tree(transcription_factor, celltypes, options, unbound_fraction)
        ids = list(position_tree)

        trans_f_lookup = self.get_trans_f_lookup()
        X = self.get_sequece_from_ids(ids, part, bin_size)
        dnase_features = np.zeros((len(ids), 1+3+dnase_bin_size/10-1, len(celltypes)), dtype=np.float32)
        labels = np.zeros((len(ids), len(celltypes)), dtype=np.float32)

        for c_idx, celltype in enumerate(celltypes):
            dnase_features[:, :, c_idx] = np.load('../data/preprocess/DNASE_FEATURES_NORM/%s_%s_%d.gz_non_norm.npy' % (celltype, part, dnase_bin_size))[ids]
            labels[:, c_idx] = \
                np.load('../data/preprocess/features/y_%s.npy' % celltype)[ids, trans_f_lookup[transcription_factor]]

        return X, dnase_features, labels

    def get_motifs_h(self, transcription_factor, verbose=False):
        motifs = []

        def get_motif(directory, unpack=True, skiprows=0, calc_pssm=False):
            for f in os.listdir(directory):
                if transcription_factor.upper() == f.split('_')[0].upper():
                    motif = np.loadtxt(os.path.join(directory, f), dtype=np.float32, unpack=unpack, skiprows=skiprows)
                    if calc_pssm:
                        motif = self.calc_pssm(motif)
                    if verbose:
                        print "motif found in", directory
                        print "motif:", f, motif.shape
                        print motif
                    motifs.append(motif)

        # Try Jaspar
        JASPAR_dir = os.path.join(self.datapath, 'preprocess/JASPAR/')
        get_motif(JASPAR_dir, calc_pssm=True)

        # Try SELEX
        SELEX_dir = os.path.join(self.datapath, 'preprocess/SELEX_PWMs_for_Ensembl_1511_representatives/')
        get_motif(SELEX_dir, calc_pssm=True)

        # Try Autosome mono
        AUTOSOME_mono_dir = os.path.join(self.datapath, 'preprocess/autosome/mono_pwm')
        get_motif(AUTOSOME_mono_dir, unpack=False, skiprows=1)

        return motifs

    def calc_pssm(self, pfm, pseudocounts=0.001):
        pfm += pseudocounts
        norm_pwm = pfm / pfm.sum(axis=1)[:, np.newaxis]
        return np.log2(norm_pwm / 0.25)


    ##### DATA GENERATION ##############################################################################
    def generate_sequence(self, segment, bin_size):
        if segment not in ['train', 'ladder', 'test']:
            raise Exception('Please specify the segment')
        bin_correction = max(0, (bin_size - 200) / 2)

        l_idx = 0
        with gzip.open('../data/annotations/%s_regions.blacklistfiltered.bed.gz' % segment) as fin:
            for line in fin:
                if l_idx % self.chunk_size == 0:
                    if l_idx != 0:
                        # save X
                        save_path = os.path.join(self.save_dir,
                                                 'segment_' + segment +
                                                 '_bin_size_' + str(bin_size) +
                                                 '_chunk_id_' + str(l_idx-X.shape[0]) + '.npy')
                        np.save(save_path, X)
                        print "Chunk", l_idx, "saved"

                    if segment == 'train':
                        length = min(self.chunk_size, self.train_length-l_idx)
                    if segment == 'ladder':
                        length = min(self.chunk_size, self.ladder_length-l_idx)
                    if segment == 'test':
                        length = min(self.chunk_size, self.test_length-l_idx)
                    X = np.zeros((length, bin_size, self.num_channels), dtype=np.float16)

                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                sequence = self.hg19[chromosome][start-bin_correction:start + self.sequence_length + bin_correction]
                sequence_one_hot = self.sequence_to_one_hot(np.array(list(sequence)))
                X[l_idx % self.chunk_size, :, :] = sequence_one_hot
                l_idx += 1
        # save remainder X
        save_path = os.path.join(self.save_dir,
                                 'segment_' + segment +
                                 '_bin_size_' + str(bin_size) +
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

    def generate_dnase(self, segment, bin_size, num_processes):

        def get_DNAse_fold_track(celltype, chromosome, left, right):
            fpath = os.path.join(self.datapath, 'dnase_fold_coverage/DNASE.%s.fc.signal.bigwig' % celltype)
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
                    length = end - start + 1
                    value = float(tokens[3])
                    track[position:position + length] = value
                    position += length
                else:
                    value = float(tokens[0])
                    track[position] = value
                    position += 1
            return track

        def DNAseSignalProcessor(lines, segment, celltype, bin_size):
            print "parsing dnase for", segment, celltype, bin_size
            bin_correction = max(0, (bin_size - 200) / 2)
            if segment == 'train':
                length = self.train_length
            elif segment == 'ladder':
                length = self.ladder_length
            elif segment == 'test':
                length = self.test_length
            dnase_features = np.zeros((length, bin_size), dtype=np.float16)
            idx = 0
            for line in split_iter(lines):
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                track = get_DNAse_fold_track(celltype, chromosome, start - bin_correction, end + bin_correction)
                for pos in range(start, end - 200 + 1, 50):
                    sbin = np.log(track[pos - start:pos - start + bin_size]+1)
                    dnase_features[idx, :] = sbin
                    idx += 1
            print idx
            np.save(os.path.join(self.save_dir, 'dnase_fold_%s_%s_%d' % (segment, celltype, bin_size)), dnase_features)

        with open('../data/annotations/%s_regions.blacklistfiltered.merged.bed' % segment) as fin:
            lines = fin.read()
        processes = []
        for celltype in self.get_celltypes():
            processes.append(Process(
                target=DNAseSignalProcessor,
                args=(lines, segment, celltype, bin_size))
                )

        for i in range(0, len(processes), num_processes):
            map(lambda x: x.start(), processes[i:i + num_processes])
            map(lambda x: x.join(), processes[i:i + num_processes])

    def generate_shape(self):
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_sequence', action='store_true', required=False)
    parser.add_argument('--gen_y', action='store_true', required=False)
    parser.add_argument('--gen_dnase', action='store_true', required=False)
    parser.add_argument('--segment', required=True)
    parser.add_argument('--bin_size', type=int, required=True)
    parser.add_argument('--num_jobs', type=int, required=False)

    args = parser.parse_args()
    datagen = DataGenerator()

    assert(args.bin_size >= 200 and args.bin_size % 2 == 0)

    if args.gen_sequence:
        datagen.generate_sequence(args.segment, args.bin_size)
    if args.gen_y:
        datagen.generate_y()
    if args.gen_dnase:
        datagen.generate_dnase(args.segment, args.bin_size, args.num_jobs)





