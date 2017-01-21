import numpy as np
from pyfasta import Fasta
import os
import warnings
import argparse
import gzip
import pdb
import pandas as pd
import random
from enum import Enum
from wiggleReader import *
from multiprocessing import Process, Queue
import bisect
from sequence import *


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
        self.bin_size = 200
        self.correction = 400

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
        if 'SK-N-SH' in celltypes:
            celltypes.remove('SK-N-SH')
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
        if 'SK-N-SH' in celltypes:
            celltypes.remove('SK-N-SH')
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
        return positions.tolist()

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

        return positions.tolist()

    def get_bound_positions_for_trans_f_celltype(self, trans_f, celltype):
        trans_f_lookup = self.get_trans_f_lookup()
        save_path = os.path.join(self.save_dir, 'bound_positions_%s_%s.npy' % (trans_f, celltype))
        if os.path.exists(save_path):
            positions = np.load(save_path)
        else:
            print "Getting bound locations for transcription factor %s, celltype %s" % (trans_f, celltype)
            y = np.load(os.path.join(self.save_dir, 'y_%s.npy' % celltype))
            positions = (np.where(y[:, trans_f_lookup[trans_f]] == 1)[0]).flatten().astype(np.int32)
            np.save(save_path, positions)
        return positions.tolist()


    def get_trans_f_lookup(self):
        lookup = {}
        for idx, trans_f in enumerate(self.get_trans_fs()):
            lookup[trans_f] = idx
        return lookup

    def get_reverse_trans_f_lookup(self):
        trans_f_lookup = self.get_trans_f_lookup()
        lookup = {}
        for trans_f in trans_f_lookup.keys():
            lookup[trans_f_lookup[trans_f]] = trans_f
        return lookup

    def get_celltype_lookup(self):
        lookup = {}
        for idx, celltype in enumerate(self.get_celltypes()):
            lookup[celltype] = idx
        return lookup

    def get_reverse_celltype_lookup(self):
        inv_map = {v: k for k, v in self.get_celltype_lookup().iteritems()}
        return inv_map

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
        return norm_pwm
        #return np.log2(norm_pwm / 0.25)z

    def get_DNAse_peak_list(self, celltype, conservative=True):

        peak_type = 'conservative' if conservative else 'relaxed'

        with gzip.open(os.path.join(self.datapath,
                                    'dnase_peaks_{0}/DNASE.{1}.{0}.narrowPeak.gz'.format(peak_type, celltype))) as f_handler:
            l = []
            for line in f_handler:
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                stop = int(tokens[2])
                l.append((chromosome, start, stop))
            l.sort()
            return l

    def get_dnase_accesible_ids_for_celltype(self, celltype, conservative=True):
        peak_type = 'conservative' if conservative else 'relaxed'
        save_path = os.path.join(self.save_dir, 'dnase_%s_peaks_%s.npy' % (peak_type, celltype))
        if os.path.exists(save_path):
            ids = np.load(save_path)
        else:
            dnase_list = self.get_DNAse_peak_list(celltype, conservative)
            ids = []

            with gzip.open(os.path.join(self.datapath,
                                        'annotations/train_regions.blacklistfiltered.bed.gz')) as fin:
                for id, line in enumerate(fin):
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])

                    dnase_pos = bisect.bisect_left(dnase_list, (chromosome, start, start + self.bin_size))
                    # check left
                    if dnase_pos < len(dnase_list):
                        dnase_chr, dnase_start, dnase_end = dnase_list[dnase_pos]
                        if dnase_start <= start + self.bin_size and start <= dnase_end:
                            ids.append(id)
                    # check right
                    if dnase_pos + 1 < len(dnase_list):
                        dnase_chr, dnase_start, dnase_end = dnase_list[dnase_pos + 1]
                        if dnase_start <= start + self.bin_size and start <= dnase_end:
                            ids.append(id)
            ids = np.array(ids)
            np.save(save_path, ids.astype(np.int32))

        return ids.tolist()

    def get_dnase_accesible_ids(self, celltypes, conservative=True):
        ids = []
        for celltype in celltypes:
            ids.extend(self.get_dnase_accesible_ids_for_celltype(celltype, conservative))
        ids = list(set(ids))

        return ids

    def get_dnase(self, segment, celltype):
        return np.load(os.path.join(self.save_dir, 'dnase_fold_%s_%s.npy' % (segment, celltype)))

    def get_labels(self, celltype):
        return np.load(os.path.join(self.save_dir, 'y_%s.npy' % celltype))

    def get_chipseq_signal(self, celltype):
        return np.load(os.path.join(self.save_dir, 'y_quant_%s.npy' % celltype))

    def get_positions_for_ids(self, ids, segment):
        start_positions = []
        offset = 0
        with open(os.path.join(self.datapath, 'annotations/%s_regions.blacklistfiltered.merged.bed' % segment)) as fin:
            for line in fin:
                tokens = line.split()
                start = int(tokens[1])
                end = int(tokens[2])
                start_positions.append(offset)
                offset += ((end-start)-150)/50
        positions = []
        for id in ids:
            jumps = bisect.bisect(start_positions, id)-1
            positions.append(self.correction+jumps*(150+self.correction*2)+id*50)
        return positions

    def get_complement(self, one_hot_sequence):
        complement = np.zeros(one_hot_sequence.shape, dtype=np.float32)
        a_idx = np.where(one_hot_sequence[:, 0] == 1)[0]
        c_idx = np.where(one_hot_sequence[:, 1] == 1)[0]
        g_idx = np.where(one_hot_sequence[:, 2] == 1)[0]
        t_idx = np.where(one_hot_sequence[:, 3] == 1)[0]

        complement[a_idx, 3] = 1
        complement[c_idx, 2] = 1
        complement[g_idx, 1] = 1
        complement[t_idx, 0] = 1
        return complement

    ##### DATA GENERATION ##############################################################################

    def generate_sequence(self, segment):
        if segment not in ['train', 'ladder', 'test']:
            raise Exception('Please specify the segment')
        num_positions = 0
        with open(os.path.join(self.datapath, 'annotations/%s_regions.blacklistfiltered.merged.bed' % segment)) as fin:
            for line in fin:
                tokens = line.split()
                start = int(tokens[1])
                end = int(tokens[2])
                num_positions += end-start+self.correction*2
        X = np.zeros((num_positions, 4), dtype=np.bool)
        offset = 0
        with open(os.path.join(self.datapath, 'annotations/%s_regions.blacklistfiltered.merged.bed' % segment)) as fin:
            for line in fin:
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                sequence = self.hg19[chromosome][start-self.correction:end+self.correction]
                print "len sequence", len(sequence), end-start+self.correction*2
                #sequence_encoding = one_hot_encode_sequence(sequence)
                sequence_encoding = self.sequence_to_one_hot(np.array(list(sequence)))
                X[offset:offset+end-start+self.correction*2] = sequence_encoding
                offset += end - start + self.correction*2
        save_path = os.path.join(self.save_dir, 'sequence_' + segment + '.npy')
        np.save(save_path, X)

    def generate_shape(self, segment):
        if segment not in ['train', 'ladder', 'test']:
            raise Exception('Please specify the segment')
        num_positions = 0
        with open(os.path.join(self.datapath, 'annotations/%s_regions.blacklistfiltered.merged.bed' % segment)) as fin:
            for line in fin:
                tokens = line.split()
                start = int(tokens[1])
                end = int(tokens[2])
                num_positions += end-start+self.correction*2
        X = np.zeros((num_positions, 4), dtype=np.float16)
        offset = 0
        f_mgw = open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.MGW2'))
        f_helt = open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.HelT2'))
        f_roll = open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.Roll2'))
        f_prot = open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.ProT2'))

        mgw = f_mgw.read()
        helt = f_helt.read()
        roll = f_roll.read()
        prot = f_prot.read()

        with open(os.path.join(self.datapath,
                                   'annotations/%s_regions.blacklistfiltered.merged.bed' % segment)) as fin:

            curr_chr = '-1'

            for line in fin:
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])

                if curr_chr != chromosome:
                    curr_chr = chromosome
                    chr_order = ['chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr1',
                     'chr20', 'chr21', 'chr22', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrM',
                     'chrX', 'chrY']
                    idx = chr_order.index(curr_chr)*2+1
                    mgw_chunk = np.array(mgw.split()[idx].split(',')[:-1], dtype=np.float16)
                    helt_chunk = np.array(['0']+helt.split()[idx].split(',')[:-1], dtype=np.float16)
                    roll_chunk = np.array(['0']+roll.split()[idx].split(',')[:-1], dtype=np.float16)
                    prot_chunk = np.array(prot.split()[idx].split(',')[:-1], dtype=np.float16)

                print "len sequence", len(mgw_chunk), len(helt_chunk), len(roll_chunk), len(prot_chunk), end - start + self.correction * 2

                X[offset:offset + end - start + self.correction * 2, 0] = mgw_chunk[start-self.correction:end+self.correction]
                X[offset:offset + end - start + self.correction * 2, 1] = helt_chunk[start-self.correction:end+self.correction]
                X[offset:offset + end - start + self.correction * 2, 2] = roll_chunk[start-self.correction:end+self.correction]
                X[offset:offset + end - start + self.correction * 2, 3] = prot_chunk[start-self.correction:end+self.correction]
                offset += end - start + self.correction * 2

        save_path = os.path.join(self.save_dir, 'shape_' + segment + '.npy')
        np.save(save_path, X)

        f_mgw.close()
        f_helt.close()
        f_roll.close()
        f_prot.close()

    def generate_y(self):
        trans_f_lookup = self.get_trans_f_lookup()
        celltype_lookup = self.get_celltype_lookup()
        y = np.ones((self.train_length, self.num_trans_fs, self.num_celltypes), dtype=np.int8)
        y *= -1

        for transcription_factor in self.get_trans_fs():
            path = os.path.join(self.label_path, '%s.train.labels.tsv.gz' % transcription_factor)
            labels = pd.read_csv(path, delimiter='\t')
            celltype_names = list(labels.columns[3:])
            if 'SK-N-SH' in celltype_names:
                celltype_names.remove('SK-N-SH')

            for celltype in celltype_names:
                celltype_labels = np.array(labels[celltype])
                unbound_indices = np.where(celltype_labels == 'U')
                y[unbound_indices, trans_f_lookup[transcription_factor], celltype_lookup[celltype]] = 0
                ambiguous_indices = np.where(celltype_labels == 'A')
                y[ambiguous_indices, trans_f_lookup[transcription_factor], celltype_lookup[celltype]] = 0
                bound_indices = np.where(celltype_labels == 'B')
                y[bound_indices, trans_f_lookup[transcription_factor], celltype_lookup[celltype]] = 1

        np.save(os.path.join(self.save_dir, 'y_full.npy'), y)

        for celltype in self.get_celltypes():
            np.save(os.path.join(self.save_dir, 'y_%s.npy' % celltype),
                    np.reshape(y[:, :, celltype_lookup[celltype]], (self.train_length, self.num_trans_fs))
                    )

    def generate_dnase(self, segment, num_processes):

        def get_DNAse_fold_track(celltype, chromosome, left, right):
            fpath = os.path.join(self.datapath, 'dnase_fold_coverage/DNASE.%s.fc.signal.bigwig' % celltype)
            process = Popen(["wiggletools", "seek", chromosome, str(left), str(right), fpath],
                            stdout=PIPE)
            (wiggle_output, _) = process.communicate()
            track = np.zeros((right - left,), dtype=np.float32)
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

        def DNAseSignalProcessor(segment, celltype, length):
            print "generating", celltype, segment
            dnase_features = np.zeros((length,), dtype=np.float32)
            offset = 0
            with open('../data/annotations/%s_regions.blacklistfiltered.merged.bed' % segment) as fin:
                lines = fin.read()

            for line in split_iter(lines):
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                end = int(tokens[2])
                track = get_DNAse_fold_track(celltype, chromosome, start-self.correction, end+self.correction)
                dnase_features[offset:offset+end-start+self.correction*2] = track
                offset += end-start+self.correction*2

            np.save(os.path.join(self.save_dir, 'dnase_fold_%s_%s' % (segment, celltype)),
                    dnase_features.astype(np.float16))

        with open('../data/annotations/%s_regions.blacklistfiltered.merged.bed' % segment) as fin:
            lines = fin.read()
        length = 0
        for line in split_iter(lines):
            tokens = line.split()
            start = int(tokens[1])
            end = int(tokens[2])
            length += end-start+self.correction*2

        if num_processes == 1:
            for celltype in self.get_celltypes():
                DNAseSignalProcessor(segment, celltype, length)
        else:
            processes = []
            for celltype in self.get_celltypes():
                processes.append(Process(
                    target=DNAseSignalProcessor,
                    args=(segment, celltype, length))
                    )

            for i in range(0, len(processes), num_processes):
                map(lambda x: x.start(), processes[i:i + num_processes])
                map(lambda x: x.join(), processes[i:i + num_processes])

    def generate_chipseq(self, num_processes, num_bins=10):
        def get_ChIPseq_fold_track(transcription_factor, celltype, chromosome, left, right):
            fpath = os.path.join(self.datapath, 'chipseq_fold_change_signal/ChIPseq.%s.%s.fc.signal.train.bw' %
                                 (celltype, transcription_factor))
            process = Popen(["wiggletools", "seek", chromosome, str(left), str(right), fpath],
                            stdout=PIPE)
            (wiggle_output, _) = process.communicate()
            track = np.zeros((right - left,), dtype=np.float32)
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

        chipseq_dir = os.path.join(self.save_dir, 'chipseq')

        if not os.path.exists(chipseq_dir):
            os.mkdir(chipseq_dir)

        def ChIPseqSignalProcessor(transcription_factor, celltype):
            print "generating", transcription_factor, celltype
            chipseq_features = np.zeros((self.train_length, num_bins), dtype=np.float16)
            idx = 0
            with open('../data/annotations/%s_regions.blacklistfiltered.merged.bed' % 'train') as fin:
                for line in fin:
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    end = int(tokens[2])
                    track = get_ChIPseq_fold_track(transcription_factor, celltype, chromosome, start,
                                                   end)
                    for i in range(start, end - 200 + 1, 50):
                        sbin = track[i - start:i - start + 200]
                        bins = np.split(sbin, num_bins)
                        chipseq_features[idx] = np.mean(bins, axis=1)
                        idx += 1
                np.save(os.path.join(chipseq_dir, '%s_%s.npy' % (transcription_factor, celltype)), chipseq_features)

        trans_f_lookup = self.get_trans_f_lookup()
        celltype_lookup = self.get_celltype_lookup()

        processes = []
        for transcription_factor in self.get_trans_fs():
            for celltype in self.get_celltypes():
                if celltype not in self.get_celltypes_for_tf(transcription_factor):
                    continue
                p = Process(target=ChIPseqSignalProcessor, args=(transcription_factor, celltype))
                processes.append(p)

        for i in range(0, len(processes), num_processes):
            map(lambda x: x.start(), processes[i:i + num_processes])
            map(lambda x: x.join(), processes[i:i + num_processes])

        for celltype in self.get_celltypes():
            y = np.ones((self.train_length, self.num_trans_fs, num_bins), dtype=np.float16)
            y *= -1

            for fname in os.listdir(chipseq_dir):
                tokens = fname.split('.')[0].split('_')
                transcription_factor = tokens[0]
                celltype_ = tokens[1]
                if celltype == celltype_:
                    features = np.load(os.path.join(chipseq_dir, fname))
                    y[:, trans_f_lookup[transcription_factor]] = features

            np.save(os.path.join(self.save_dir, 'y_quant_%s.npy' % celltype), y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_sequence', action='store_true', required=False)
    parser.add_argument('--gen_y', action='store_true', required=False)
    parser.add_argument('--gen_dnase', action='store_true', required=False)
    parser.add_argument('--gen_shape', action='store_true', required=False)
    parser.add_argument('--gen_chipseq', action='store_true', required=False)

    parser.add_argument('--segment', required=True)
    parser.add_argument('--num_jobs', type=int, default=1, required=False)

    args = parser.parse_args()
    datagen = DataGenerator()

    if args.gen_sequence:
        datagen.generate_sequence(args.segment)
    if args.gen_y:
        datagen.generate_y()
    if args.gen_dnase:
        datagen.generate_dnase(args.segment, args.num_jobs)
    if args.gen_shape:
        datagen.generate_shape(args.segment)
    if args.gen_chipseq:
        datagen.generate_chipseq(args.num_jobs)
