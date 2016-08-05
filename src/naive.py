import numpy as np
from jasparclient import *
from Bio.Seq import Seq
from motif import *


class AlwaysFalse:
    def __init__(self):
        # number of outputs per input
        self.num_outputs = 1

    def fit(self, X, y):
        # y ~ [instance, output]
        return

    def predict(self, X):
        # X ~ [instance, feature]
        return np.zeros((X.shape[0], self.num_outputs))


class AlwaysTrue:
    def __init__(self):
        # number of outputs per input
        self.num_outputs = 1

    def fit(self, X, y):
        # y ~ [instance, output]
        return

    def predict(self, X):
        # X ~ [instance, feature]
        return np.ones((X.shape[0], self.num_outputs))


class RandomPredictor:
    def __init__(self, seed=12345):
        np.random.seed(seed)
        self.num_outputs = 1

    def fit(self, X, y):
        return

    def predict(self, X):
        return np.random.uniform(0.0, 1.0, (X.shape[0], self.num_outputs))


class TFMotifScanner:
    def __init__(self, transcription_factor=None, threshold=3.0, datapath='../data/'):
        self.tf = transcription_factor
        self.threshold = threshold
        self.motifprocessor = MotifProcessor(datapath)

    def set_transcription_factor(self, transcription_factor):
        self.tf = transcription_factor

    def fit(self, X, y):
        return

    def predict(self, sequence):
        motifs = self.motifprocessor.get_motifs(self.tf)
        for motif in motifs:
            pssm = motif.pssm
            for _, _ in pssm.search(Seq(sequence.upper(), alphabet=pssm.alphabet), threshold=self.threshold):
                return 1
        return 0





