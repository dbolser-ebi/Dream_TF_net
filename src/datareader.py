import gzip
from pyfasta import Fasta
import numpy as np
import os
from enum import Enum
import math
import random
import warnings


class CrossvalOptions(Enum):
    filter_on_DNase_peaks = 1
    balance_peaks = 2

class DNasePeakEntry:
    def __init__(self, chromosome, start):
        self.chromosome = chromosome
        self.start = str(start)

    def __eq__(self, other):
        return self.chromosome == other.chromosome \
               and self.start == other.start

    def __hash__(self):
        return hash(self.chromosome + self.start)


class TFPeakEntry:
    def __init__(self, celltype, chromosome, start):
        self.celltype = celltype
        self.chromosome = chromosome
        self.start = str(start)

    def __eq__(self, other):
        return self.celltype == other.celltype \
               and self.chromosome == other.chromosome \
               and self.start == other.start

    def __hash__(self):
        return hash(self.celltype+self.chromosome+self.start)


class PeakEntry:
    def __init__(self, transcription_factor, celltype, chromosome, start):
        self.transcription_factor = transcription_factor
        self.celltype = celltype
        self.chromosome = chromosome
        self.start = str(start)

    def __eq__(self, other):
        return self.transcription_factor == other.transcription_factor \
               and self.celltype == other.celltype \
               and self.chromosome == other.chromosome \
               and self.start == other.start

    def __hash__(self):
        return hash(self.transcription_factor + self.celltype + self.chromosome + self.start)


class DataReader:
    def __init__(self, datapath):
        self.datapath = datapath
        self.dna_peak_c_path = 'dnase_peaks_conservative/'
        self.label_path = 'chipseq_labels/'
        self.hg19 = Fasta(datapath+'annotations/hg19.genome.fa')
        self.preprocess_path = 'preprocess/'

        # constants
        self.stride = 50
        self.sequence_length = 200

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

    def get_celltypes_for_tf(self, transcription_factor):
        '''
        Returns the list of celltypes for that particular transcription factor
        :param datapath: Path to the label directory
        :param transcription_factor: TF to Crossval
        :return: celltypes, a list of celltype for that TF
        '''
        files = [f for f in os.listdir(os.path.join(self.datapath, self.label_path))]
        celltypes = []
        for f in files:
            if transcription_factor in f:
                fpath = os.path.join(self.datapath, os.path.join(self.label_path, f))
                with gzip.open(fpath) as fin:
                    header_tokens = fin.readline().split()
                    celltypes = header_tokens[3:]
                break
        return celltypes

    def get_tfs_for_celltype(self, datapath, celltype):
        '''
        Returns the list of transcription factors for that particular celltype
        :param datapath: Path to the label directory
        :param celltype: celltype to Crossval
        :return:
        '''
        files = [f for f in os.listdir(datapath)]
        tfs = []
        for f in files:
            fpath = os.path.join(datapath, f)
            with gzip.open(fpath) as fin:
                header = fin.readline()
                if celltype in header:
                    tf = f.split('.')[0]
                    tfs.append(tf)
        return tfs

    def get_tfs(self):
        '''
        Returns the list of all TFs
        :return: list of tfs
        '''
        tfs = []
        files = [f for f in os.listdir(self.datapath+self.label_path)]
        tfs = []
        for f in files:
            tfs.append(f.split('.')[0])
        return tfs

    def get_celltypes(self):
        celltypes = []
        for f in [f for f in os.listdir(self.datapath+self.dna_peak_c_path)]:
            celltypes.append(f.split('.')[1])
        return celltypes

    def get_chromosome_ordering(self):
        ordering = {'chr10': 0,
                    'chr1': 1,
                    'chr11': 2,
                    'chr12': 3,
                    'chr13': 4,
                    'chr14': 5,
                    'chr15': 6,
                    'chr16': 7,
                    'chr17': 8,
                    'chr18': 9,
                    'chr19': 10,
                    'chr20': 11,
                    'chr2': 12,
                    'chr21': 13,
                    'chr22': 14,
                    'chr3': 15,
                    'chr4': 16,
                    'chr5': 17,
                    'chr6': 18,
                    'chr7': 19,
                    'chr8': 20,
                    'chr9': 21,
                    'chrX': 22,
                    'chrY': 23}
        return ordering

    def get_tf_celltype_combinations(self):
        '''
        Provides consistency in the indexing of tf, celltype combinations
        :return:
        '''
        tfs = self.get_tfs()
        celltypes = self.get_celltypes()
        combinations = []
        for tf in tfs:
            for celltype in celltypes:
                combinations.append((tf, celltype))
        return combinations

    def get_combination_to_idx_mapping(self):
        mp = {}
        combinations = self.get_tf_celltype_combinations()
        for i, comb in enumerate(combinations):
            mp[comb] = i
        return mp

    def get_idx_to_combination_mapping(self):
        mp = {}
        combinations = self.get_tf_celltype_combinations()
        for i, comb in enumerate(combinations):
            mp[i] = comb
        return mp

    def get_DNAse_peaks(self, celltypes, conservative=True):
        '''
        Merges peaks from different celltypes
        :param celltypes: The celltypes to use
        :return: Merged peaks
        '''
        intervals = []
        for celltype in celltypes:
            with gzip.open(self.datapath+self.dna_peak_c_path+'DNASE.'+celltype+'.conservative.narrowPeak.gz') as fin:
                for line in fin.read().splitlines():
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    stop = int(tokens[2])
                    intervals.append((chromosome, start, stop))
        sorted_intervals = sorted(intervals)
        merged_intervals = list()
        merged_intervals.append(sorted_intervals[0])
        for i, (chr_, start_, stop_) in enumerate(sorted_intervals):
            (chr_top, start_top, stop_top) = merged_intervals[-1]
            if chr_ == chr_top and start_ <= stop_top:
                merged_intervals.pop()
                merged_intervals.append((chr_, start_top, max(stop_, stop_top)))
            else:
                merged_intervals.append((chr_, start_, stop_))
        return merged_intervals

    def get_DNAse_peaks_tree(self, celltype, conservative=True):
        dnase_peaks = self.get_DNAse_peaks([celltype])
        dnase_peaks_tree = set()
        for (chromosome, start, stop) in dnase_peaks:
            dnase_peaks_tree.add(DNasePeakEntry(chromosome, start))
        return dnase_peaks_tree

    def get_chipseq_peaks_tree_for_tf(self, tf):
        '''
        Provides datastructure to do amortized O(1) lookup of peaks
        for the given transcription factor
        @param tf Specifies the transcription factor to use
        :return: A hash set containing keys in format: (celltype, chromosome, start)
        '''
        chipseq_peaks = set()
        with gzip.open(self.datapath + self.label_path + tf + '.train.labels.tsv.gz') as fin:
            lines = fin.read().splitlines()
            celltype_names = lines[0].split()[3:]
            lines = filter(lambda x: 'B' in x, lines)
            for line in lines:
                tokens = line.split()
                bound_states = tokens[3:]
                start = tokens[1]
                chromosome = tokens[0]
                for idx, state in enumerate(bound_states):
                    if state == 'B':
                        chipseq_peaks.add(TFPeakEntry(celltype_names[idx], chromosome, start))
        return chipseq_peaks

    def get_chipseq_peaks_tree(self):
        if not os.path.exists(self.datapath+self.preprocess_path+'chipseq_peaks'):
            self.write_chipseq_peaks_tree()
        # read
        chipseq_peaks = set()
        with open(self.datapath+self.preprocess_path+'chipseq_peaks') as fin:
            for line in fin:
                tokens = line.split()
                chipseq_peaks.add(PeakEntry(tokens[0], tokens[1], tokens[2], tokens[3]))
        return chipseq_peaks

    def write_chipseq_peaks_tree(self):
        '''
        Preprocess peak locations
        Writes a file to preprocess called chipseq_peaks
        each line in file is formatted as follows:
        tf celltype chromosome start
        :return:
        '''
        tfs = self.get_tfs()
        with open(self.datapath+self.preprocess_path+'chipseq_peaks', 'w') as fout:
            for idx, tf in enumerate(self.get_tfs()):
                print tf, str(idx+1)+'/'+str(len(tfs))
                with gzip.open(self.datapath + self.label_path + tf + '.train.labels.tsv.gz') as fin:
                    lines = fin.read().splitlines()
                    celltype_names = lines[0].split()[3:]
                    lines = filter(lambda x: 'B' in x, lines[1:])
                    for line in lines:
                        tokens = line.split()
                        bound_states = tokens[3:]
                        start = tokens[1]
                        chromosome = tokens[0]
                        for idx, state in enumerate(bound_states):
                            if state == 'B':
                                print >>fout, tf, celltype_names[idx], chromosome, start

    def calc_num_instances_train_given_dnase(self, celltypes):
        dna_peaks = self.get_DNAse_peaks(celltypes)

        def calculate_instances(tot, (_, start, stop)):
            return tot+int(math.ceil((stop-start+1-self.sequence_length+1)/float(self.stride)))

        return reduce(calculate_instances, dna_peaks, 0)

    def get_num_instances(self, chromosome):
        chr_num = {'chr7': 3151146, 'chr6': 3419293, 'chr5': 3602250, 'chr4': 3812501, 'chr3': 3955446, 'chr2': 4857748,
         'chr9': 2817543, 'chrX': 3097086, 'chr13': 2303355, 'chr12': 2659379, 'chr11': 2672380, 'chr10': 2702470,
         'chr17': 1622319, 'chr16': 1798121, 'chr15': 2049716, 'chr14': 2145762, 'chr20': 1258450, 'chr22': 1025608,
         'chr19': 1165049, 'chr18': 1561114}
        return chr_num[chromosome]

    def generate_within_celltype(self, transcription_factor, celltype, chromosomes, options=None):
        if options == CrossvalOptions.filter_on_DNase_peaks:
            chipseq_peaks = self.get_chipseq_peaks_tree_for_tf(transcription_factor)
            dnase_peaks = self.get_DNAse_peaks([celltype])
            for (chromosome, start, stop) in dnase_peaks:
                if chromosome in chromosomes:
                    self.hg19[chromosome].as_string = False
                    for i in range(start, stop - self.sequence_length, self.stride):
                        features = self.sequence_to_one_hot(self.hg19[chromosome][i:i + self.sequence_length])
                        labels = np.zeros((1, 1), dtype=np.float32)
                        if TFPeakEntry(celltype, chromosome, i) in chipseq_peaks:
                            labels[0] = 1
                        yield features, labels
        else:
            with gzip.open(self.datapath + self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
                celltype_names = fin.readline().split()[3:]
                idxs = []
                for i, celltype in enumerate(celltype_names):
                    if celltype in celltypes:
                        idxs.append(i)
                for position, line in enumerate(fin):
                    tokens = line.split()
                    bound_states = tokens[3:]
                    start = int(tokens[1])
                    chromosome = tokens[0]
                    features = self.hg19[chromosome][start:start + self.sequence_length]
                    labels = np.zeros((1, len(celltypes)), dtype=np.float32)
                    for i, idx in enumerate(idxs):
                        if bound_states[idx] == 'B':
                            labels[:, i] = 1
                    yield (chromosome, start), features, labels

    def generate_cross_celltype(self, transcription_factor, celltypes, options=[]):
        position_tree = set()  # keeps track of which lines (chr, start) to include

        if CrossvalOptions.filter_on_DNase_peaks in options:
            with gzip.open(self.datapath + self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
                fin.readline()
                dnase_peaks = self.get_DNAse_peaks(celltypes)
                ordering = self.get_chromosome_ordering()
                d_idx = 0
                bin_length = 200
                chr_dnase, start_dnase, _ = dnase_peaks[d_idx]
                for line_idx, line in enumerate(fin):
                    tokens = line.split()
                    chromosome = tokens[0]
                    start = int(tokens[1])
                    while ordering[chr_dnase] < ordering[chromosome] \
                            and d_idx < len(dnase_peaks) - 1:
                        d_idx += 1
                        (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]
                    while chr_dnase == chromosome and start_dnase + bin_length < start \
                            and d_idx < len(dnase_peaks) - 1:
                        d_idx += 1

                    (chr_dnase, start_dnase, _) = dnase_peaks[d_idx]

                    if chr_dnase == chromosome \
                            and (start <= start_dnase + bin_length
                                 and start_dnase <= start + bin_length):
                        position_tree.add(line_idx)

        elif CrossvalOptions.balance_peaks in options:
            with gzip.open(self.datapath + self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
                fin.readline()
                bound_lines = []
                unbound_lines = []
                # only the train set
                for line_idx, line in enumerate(fin):
                    if 'B' in line[:-2]:
                        bound_lines.append(line_idx)
                    else:
                        unbound_lines.append(line_idx)

                print "num bound lines", len(bound_lines)
                random.shuffle(unbound_lines)
                bound_lines.extend(unbound_lines[:len(bound_lines)])
                position_tree.update(set(bound_lines))

        with gzip.open(self.datapath + self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin, \
                open('../data/preprocess/SEQUENCE_FEATURES/' + transcription_factor + '.txt') as f_seqfeat:
            celltype_names = fin.readline().split()[3:]
            sequence_features = [] #TODO
            idxs = []
            for i, celltype in enumerate(celltype_names):
                if celltype in celltypes:
                    idxs.append(i)
            for lidx, line in enumerate(fin):
                if len(position_tree) == 0 or lidx in position_tree:
                    tokens = line.split()
                    bound_states = tokens[3:]
                    start = int(tokens[1])
                    chromosome = tokens[0]
                    sequence = self.hg19[chromosome][start:start + self.sequence_length]
                    labels = np.zeros((1, len(celltypes)), dtype=np.float32)
                    for i, idx in enumerate(idxs):
                        if bound_states[idx] == 'B':
                            labels[:, i] = 1
                    yield (chromosome, start), sequence, sequence_features, labels

    def generate_multi_task(self, tf_mapping, options=None):
        if options == CrossvalOptions.filter_on_DNase_peaks:
            celltypes = set()
            for tf in tf_mapping:
                for celltype in tf_mapping[tf]:
                    celltypes.add(celltype)

            celltypes = list(celltypes)
            dnase_peaks = self.get_DNAse_peaks(celltypes)
            chipseq_peaks = self.get_chipseq_peaks_tree()

            combination_to_idx = self.get_combination_to_idx_mapping()

            for (chromosome, start, stop) in dnase_peaks:
                self.hg19[chromosome].as_string = False
                for i in range(start, stop - self.sequence_length, self.stride):
                    features = self.sequence_to_one_hot(self.hg19[chromosome][i:i+self.sequence_length])
                    labels = np.zeros((1, len(self.get_tf_celltype_combinations())), dtype=np.float32)
                    for tf in tf_mapping:
                        for celltype in tf_mapping[tf]:
                            if PeakEntry(tf, celltype, chromosome, i) in chipseq_peaks:
                                print combination_to_idx[(tf, celltype)]
                                labels[combination_to_idx[(tf, celltype)]] = 1
                    yield features, labels

