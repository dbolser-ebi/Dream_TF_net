import gzip
from pyfasta import Fasta
import numpy as np
import os
from enum import Enum
import math
import random
import warnings
from sklearn.decomposition import PCA
import bisect


def optional_gzip_open(fname):
    return gzip.open(fname) if fname.endswith(".gz") else open(fname)


def conditional_open(fname):
    if os.path.exists(fname):
        return optional_gzip_open(fname)
    else:
        return None


class CrossvalOptions(Enum):
    filter_on_DNase_peaks = 1
    balance_peaks = 2
    random_shuffle_10_percent = 3


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

    def sequence_to_one_hot_transpose(self, sequence):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoding = np.zeros((4, len(sequence)), dtype=np.float32)
            # Process A
            encoding[0, (sequence == 'A') | (sequence == 'a')] = 1
            # Process C
            encoding[1, (sequence == 'C') | (sequence == 'c')] = 1
            # Process G
            encoding[2, (sequence == 'G') | (sequence == 'g')] = 1
            # Process T
            encoding[3, (sequence == 'T') | (sequence == 't')] = 1
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
        for f_name in files:
            if transcription_factor in f_name:
                fpath = os.path.join(self.datapath, os.path.join(self.label_path, f_name))
                with gzip.open(fpath) as fin:
                    header_tokens = fin.readline().split()
                    celltypes = header_tokens[3:]
                break
        return celltypes

    def get_num_sequence_features(self, transcription_factor):
        with open(os.path.join(self.datapath, 'preprocess/SEQUENCE_FEATURES/'+transcription_factor+'.txt')) as fin:
            return len(fin.readline().split())

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

    def get_num_bound_lines(self, transcription_factor, ambiguous_as_bound=False):
        if ambiguous_as_bound:
            mp = {'YY1': 3108428, 'JUND': 3818996, 'MAFK': 2549072, 'TEAD4': 3861733, 'CEBPB': 4625934,
                  'ZNF143': 1843626, 'SPI1': 832029, 'NANOG': 448269, 'EGR1': 2082234, 'GATA3': 2237461,
                  'MYC': 2502483, 'SRF': 2021623, 'REST': 2828169, 'CREB1': 3056214, 'STAT3': 676551,
                  'EP300': 3783998, 'ATF7': 3609714, 'ARID3A': 786946, 'E2F6': 2382647, 'E2F1': 1437776,
                  'TAF1': 2269991, 'RFX5': 2219658, 'MAX': 5691432, 'HNF4A': 496317, 'TCF7L2': 2271997,
                  'FOXA1': 1083820, 'FOXA2': 1581858, 'ATF3': 2814860, 'TCF12': 2875805, 'GABPA': 3469748,
                  'CTCF': 2083719, 'ATF2': 3275622}
        else:
            mp = {'YY1': 414733, 'JUND': 734552, 'MAFK': 442379, 'TEAD4': 576098, 'CEBPB': 984566, 'ZNF143': 312498,
         'SPI1': 200845, 'NANOG': 32918, 'EGR1': 267537, 'GATA3': 496617, 'MYC': 347756, 'SRF': 229617, 'REST': 255624,
         'CREB1': 391070, 'STAT3': 108787, 'EP300': 789926, 'ATF7': 732843, 'ARID3A': 132702, 'E2F6': 303125,
         'E2F1': 127319, 'TAF1': 342390, 'RFX5': 237118, 'MAX': 1115468, 'HNF4A': 106308, 'TCF7L2': 322078,
         'FOXA1': 256632, 'FOXA2': 374750, 'ATF3': 455348, 'TCF12': 568606, 'GABPA': 219328, 'CTCF': 497961,
         'ATF2': 380831}
        return mp[transcription_factor]

    def get_gene_expression_tpm(self, celltypes):
        idxs = []
        features = None
        keep_lines = []
        with open(os.path.join(self.datapath, 'preprocess/gene_ids.data')) as fin:
            tfs = self.get_tfs()
            for idx, line in enumerate(fin):
                tokens = line.split()
                tf = 'NONE' if len(tokens) < 2 else tokens[1]
                if tf in tfs:
                    keep_lines.append(idx)
        for idx, celltype in enumerate(self.get_celltypes()):
            with open(os.path.join(self.datapath, 'rnaseq/gene_expression.{}.biorep1.tsv'.format(celltype))) as fin1,\
                    open(os.path.join(self.datapath, 'rnaseq/gene_expression.{}.biorep2.tsv'.format(celltype))) as fin2:
                if celltype in celltypes:
                    idxs.append(idx)
                tpm1 = []
                tpm2 = []
                fin1.readline()
                fin2.readline()
                for l_idx, line in enumerate(fin1):
                    if l_idx not in keep_lines:
                        continue
                    tokens = line.split()
                    tpm1.append(float(tokens[5]))
                for l_idx, line in enumerate(fin2):
                    if l_idx not in keep_lines:
                        continue
                    tokens = line.split()
                    tpm2.append(float(tokens[5]))

                tpm1 = np.array(tpm1, dtype=np.float32)
                tpm2 = np.array(tpm2, dtype=np.float32)

                tpm1 = (tpm1 + tpm2) / 2

                if idx == 0:
                    features = tpm1
                else:
                    features = np.vstack((features, tpm1))
        shiftscaled = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1)
        return shiftscaled[idxs]
        #pca = PCA(whiten=True)
        #features = pca.fit_transform(features)
        #return features[idxs]

    def get_chromosomes(self):
        chromosomes =['chr7', 'chr6', 'chr5', 'chr4', 'chr3', 'chr2',
                   'chr9', 'chrX', 'chr13', 'chr12', 'chr11',
                   'chr10', 'chr17', 'chr16', 'chr15', 'chr14', 'chr20',
                   'chr22','chr19', 'chr18']
        return chromosomes

    def get_num_instances(self, chromosome):
        chr_num = {'chr7': 3151146, 'chr6': 3419293, 'chr5': 3602250, 'chr4': 3812501, 'chr3': 3955446, 'chr2': 4857748,
         'chr9': 2817543, 'chrX': 3097086, 'chr13': 2303355, 'chr12': 2659379, 'chr11': 2672380, 'chr10': 2702470,
         'chr17': 1622319, 'chr16': 1798121, 'chr15': 2049716, 'chr14': 2145762, 'chr20': 1258450, 'chr22': 1025608,
         'chr19': 1165049, 'chr18': 1561114}
        return chr_num[chromosome]

    def get_motifs_h(self, transcription_factor):
        motifs = []
        # Try Jaspar
        JASPAR_dir = os.path.join(self.datapath, 'preprocess/JASPAR/')
        for f in os.listdir(JASPAR_dir):
            if transcription_factor.upper() in f.upper():
                # print "motif found in JASPAR"
                motifs.append(np.loadtxt(os.path.join(JASPAR_dir, f), dtype=np.float32, unpack=True))

        # Try SELEX
        SELEX_dir = os.path.join(self.datapath, 'preprocess/SELEX_PWMs_for_Ensembl_1511_representatives/')
        for f in os.listdir(SELEX_dir):
            if f.upper().startswith(transcription_factor.upper()):
                # print "motif found in SELEX"
                motifs.append(np.loadtxt(os.path.join(SELEX_dir, f), dtype=np.float32, unpack=True))

        return motifs

    def calc_pssm(self, pfm, pseudocounts=0.001):
        pfm += pseudocounts
        norm_pwm = pfm / pfm.sum(axis=1)[:, np.newaxis]
        return np.log2(norm_pwm / 0.25)

    def get_shape_features(self, chromosome):
        savepath = os.path.join(self.datapath, 'preprocess/SHAPE_FEATURES/chromosome.npy')
        if os.path.exists(savepath):
            features = np.load(savepath)
            return features

        with open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.MGW')) as fmgw,\
             open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.HelT')) as fhelt,\
             open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.Roll')) as froll,\
             open(os.path.join(self.datapath, 'annotations/hg19.genome.fa.ProT')) as fprot:

            def get_features(f_handler, add_zero):
                out = [0] if add_zero else []
                skip = True
                for line in f_handler:
                    if line.strip() == '>' + chromosome:
                        skip = False
                    elif skip:
                        continue
                    elif line.startswith('>chr'):
                        break
                    else:
                        out.extend(map(float, line.strip().replace('NA', '0').split(',')))
                return out

            outmgw = get_features(fmgw, False)
            outhelt = get_features(fhelt, True)
            outroll = get_features(froll, True)
            outprot = get_features(fprot, False)

            features = np.array([outmgw, outhelt, outroll, outprot], dtype=np.float32).transpose()
            np.save(savepath, features)

            return features

    def get_DNAse_conservative_peak_lists(self, celltypes):
        dnase_files = []
        dnase_lists = []
        for celltype in celltypes:
            dnase_files.append(gzip.open(os.path.join(self.datapath,
                                                 'dnase_peaks_conservative/DNASE.{}.conservative.narrowPeak.gz'.format(
                                                     celltype))))
        for f_handler in dnase_files:
            l = []
            for line in f_handler:
                tokens = line.split()
                chromosome = tokens[0]
                start = int(tokens[1])
                stop = int(tokens[2])
                l.append((chromosome, start, stop))
            l.sort()
            dnase_lists.append(l)
        return dnase_lists

    def generate_cross_celltype(self, transcription_factor, celltypes, options=[], unbound_fraction=1, ambiguous_as_bound=False):
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
                    if 'B' in line or (ambiguous_as_bound and 'A' in line):
                        bound_lines.append(line_idx)
                    else:
                        unbound_lines.append(line_idx)
                random.shuffle(unbound_lines)
                if unbound_fraction > 0:
                    bound_lines.extend(unbound_lines[:int(min(len(bound_lines)*unbound_fraction, len(unbound_lines)))])
                position_tree.update(set(bound_lines))

        elif CrossvalOptions.random_shuffle_10_percent in options:
            with gzip.open(self.datapath + self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
                fin.readline()
                a = range(len(fin.read().splitlines()))
                random.shuffle(a)
                position_tree.update(a[:int(0.1*len(a))])

        with gzip.open(self.datapath + self.label_path + transcription_factor + '.train.labels.tsv.gz') as fin:
            dnase_lists = self.get_DNAse_conservative_peak_lists(celltypes)

            curr_chromosome = 'chr10'
            shape_features = self.get_shape_features(curr_chromosome)

            celltype_names = fin.readline().split()[3:]
            idxs = []
            for i, celltype in enumerate(celltype_names):
                if celltype in celltypes:
                    idxs.append(i)
            for l_idx, line in enumerate(fin):
                if len(position_tree) == 0 or l_idx in position_tree:
                    tokens = line.split()
                    bound_states = tokens[3:]
                    start = int(tokens[1])
                    chromosome = tokens[0]
                    sequence = self.hg19[chromosome][start:start + self.sequence_length]
                    labels = np.zeros((1, len(celltypes)), dtype=np.float32)

                    # find position in dnase on the left in sorted order
                    dnase_labels = np.zeros((1, len(celltypes)), dtype=np.float32)
                    for c_idx, celltype in enumerate(celltypes):
                        dnase_pos = bisect.bisect_left(dnase_lists[c_idx], (chromosome, start, start+200))
                        # check left
                        if dnase_pos < len(dnase_lists[c_idx]):
                            dnase_chr, dnase_start, dnase_end = dnase_lists[c_idx][dnase_pos]
                            if dnase_start <= start+200 and start <= dnase_end:
                                dnase_labels[:, c_idx] = 1
                        # check right
                        if dnase_pos + 1 < len(dnase_lists[c_idx]):
                            dnase_chr, dnase_start, dnase_end = dnase_lists[c_idx][dnase_pos+1]
                            if dnase_start <= start + 200 and start <= dnase_end:
                                dnase_labels[:, c_idx] = 1

                    if chromosome != curr_chromosome:
                        curr_chromosome = chromosome
                        shape_features = self.get_shape_features(curr_chromosome)

                    for i, idx in enumerate(idxs):
                        if bound_states[idx] == 'B' or (ambiguous_as_bound and bound_states[idx] == 'A'):
                            labels[:, i] = 1

                    yield (chromosome, start), sequence, shape_features[start:start+self.sequence_length], \
                        dnase_labels, labels

if __name__ == '__main__':
    datareader = DataReader('../data/')
    out = datareader.get_shape_features('chr10')
    print out.shape

