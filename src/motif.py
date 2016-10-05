from jasparclient import *
from Bio.motifs import read
from datagen import *
from wiggleReader import get_wiggle_output, wiggleToBedGraph, split_iter
import argparse
import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import *
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as ro
import matplotlib.pyplot as plt


class MotifProcessor:

    def __init__(self, datapath):
        self.datagen = DataGenerator()
        np.set_printoptions(suppress=True)
        self.datapath = datapath
        self.preprocesspath = 'preprocess/'

    def dump_jaspar_motifs(self):
        count = 0
        passwd = getpass.getpass()
        missing = []
        for tf in self.datagen.get_trans_fs():
            motifs = get_motifs_for_tf(tf, passwd)
            if len(motifs) > 0:
                count += 1
            else:
                missing.append(tf)
            for motif in motifs:
                print motif.name
                with open(os.path.join(self.datapath, self.preprocesspath+'JASPAR/'+motif.name+'_'+motif.matrix_id+'.pfm'), 'w') as fout:
                    print >>fout, motif.format('pfm')
        print "Found motifs for", count, "of", len(self.datagen.get_trans_fs()), "transcription factors"
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


def calc_pssm(pfm, pseudocounts=0.001):
    pfm += pseudocounts
    norm_pwm = pfm / pfm.sum(axis=0)
    #return np.log2(norm_pwm / 0.25)
    return norm_pwm

def plot_motif():
    dir_path = '../data/preprocess/SELEX_PWMs_for_Ensembl_1511_representatives'
    f_path = 'HOXB2_RFX5_AY_TGCTCT40NTTA_SYMATTANNNNNNRGCAACN_m1_c2.pfm'
    pfm = np.loadtxt(os.path.join(dir_path, f_path))
    pssm = calc_pssm(pfm)
    ro.globalenv['pwm'] = pssm
    ro.r('library(seqLogo); seqLogo(pwm); Sys.sleep(20)')

if __name__ == '__main__':
    plot_motif()
