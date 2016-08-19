from jasparclient import *
from Bio.motifs import read
from datareader import *
from wiggleReader import get_wiggle_output, wiggleToBedGraph, split_iter
import argparse
import os


class MotifProcessor:

    def __init__(self, datapath):
        self.datareader = DataReader(datapath)
        np.set_printoptions(suppress=True)
        self.datapath = datapath
        self.preprocesspath = 'preprocess/'

    def dump_jaspar_motifs(self):
        count = 0
        passwd = getpass.getpass()
        missing = []
        for tf in self.datareader.get_tfs():
            motifs = get_motifs_for_tf(tf, passwd)
            if len(motifs) > 0:
                count += 1
            else:
                missing.append(tf)
            for motif in motifs:
                print motif.name
                with open(os.path.join(self.datapath, self.preprocesspath+'JASPAR/'+motif.name+'_'+motif.matrix_id+'.pfm'), 'w') as fout:
                    print >>fout, motif.format('pfm')
        print "Found motifs for", count, "of", len(self.datareader.get_tfs()), "transcription factors"
        print "Missing", missing
    '''
    def get_motifs_h(self, transcription_factor):
        motifs = []
        # Try Jaspar
        JASPAR_dir = '../data/preprocess/JASPAR/'
        for f in os.listdir(JASPAR_dir):
            if transcription_factor.upper() in f.upper():
                print "motif found in JASPAR"
                motifs.append(np.loadtxt(os.path.join(JASPAR_dir, f), dtype=np.float32, unpack=True))

        # Try SELEX
        SELEX_dir = '../data/preprocess/SELEX_PWMs_for_Ensembl_1511_representatives/'
        for f in os.listdir(SELEX_dir):
            if f.upper().startswith(transcription_factor.upper()):
                print "motif found in SELEX"
                motifs.append(np.loadtxt(os.path.join(SELEX_dir, f), dtype=np.float32, unpack=True))

        return motifs

    def calc_pssm(self, pfm, pseudocounts=0.001):
        pfm += pseudocounts
        norm_pwm = pfm / pfm.sum(axis=1)[:, np.newaxis]
        return np.log2(norm_pwm/0.25)

    def calc_scores(self, pssm, sequence):
        scores = []
        for i in xrange(0, sequence.shape[0]-pssm.shape[0]+1):
            scores.append(np.prod(pssm*sequence[i:i+pssm.shape[0], :]))
        return scores
    '''

    def get_motifs(self, transcription_factor):
        motifs = []
        # Try Jaspar
        JASPAR_dir = '../data/preprocess/JASPAR/'
        for f in os.listdir(JASPAR_dir):
            if transcription_factor.upper() in f.upper():
                with open(os.path.join(JASPAR_dir, f)) as handle:
                    motif = read(handle, 'pfm')
                    print "motif found in JASPAR", f
                    motifs.append(motif)

        # Try SELEX
        SELEX_dir = '../data/preprocess/SELEX_PWMs_for_Ensembl_1511_representatives/'
        for f in os.listdir(SELEX_dir):
            if f.upper().startswith(transcription_factor.upper()):
                with open(os.path.join(SELEX_dir, f)) as handle:
                    motif = read(handle, 'pfm')
                    print "motif found in SELEX", f
                    motifs.append(motif)

        # Try Factorbook

        return motifs
    '''
    def benchmark_scoring(self):
        num_examples = 1000000
        tf = 'CTCF'
        celltypes = self.datareader.get_celltypes_for_tf(tf)
        motifs = self.get_motifs(tf)
        motifs_h = self.get_motifs_h(tf)
        pssms = map(lambda x: x.pssm, motifs)
        pssms_h = map(lambda x: self.calc_pssm(x), motifs_h)
        for idx, instance in enumerate(self.datareader.generate_cross_celltype(tf,
                                                                          celltypes,
                                                                          CrossvalOptions.filter_on_DNase_peaks)):
            if idx >= num_examples:
                break
            (features, labels) = instance

            #scores1 = pssms[0].calculate(Seq(features, alphabet=pssms[0].alphabet))
            scores2 = self.calc_scores(pssms_h[0], self.datareader.sequence_to_one_hot(features))
    '''

    '''
    def print_dnaseq_signal(self):
        lines = get_wiggle_output('../data/dnase_fold_coverage/DNASE.A549.fc.signal.bigwig').splitlines()
        for line in lines[:30]:
            print line
    '''

    def featurize_sequence(self, transcription_factor):
        pssms = map(lambda x: x.counts.normalize(pseudocounts=0.5).log_odds(), self.get_motifs(transcription_factor))
        with gzip.open(self.datapath + 'chipseq_labels/' + transcription_factor + '.train.labels.tsv.gz') as fin,\
                open('../data/preprocess/SEQUENCE_FEATURES/' + transcription_factor + '.txt', 'w') as fout:
            hg19 = Fasta('../data/annotations/hg19.genome.fa')
            fin.readline()
            for lidx, line in enumerate(fin):
                tokens = line.split()
                start = int(tokens[1])
                chromosome = tokens[0]
                sequence = hg19[chromosome][start:start + 200]
                for i in xrange(len(pssms)):
                    score = pssms[i].calculate(Seq(sequence, pssms[i].alphabet))
                    arr = np.array([np.max(score), np.percentile(score, 90), np.percentile(score, 80),
                                    np.percentile(score, 70), np.percentile(score, 50), np.mean(score),
                                    np.sum(score)],
                                   dtype=np.float32)
                    arr[np.isnan(arr)] = -50000.0
                    for e in np.nditer(arr):
                        print>> fout, e,
                    print>>fout

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', help='transcription factor', required=True)
    args = parser.parse_args()
    mp = MotifProcessor('../data/')
    mp.featurize_sequence(transcription_factor=args.tf)


