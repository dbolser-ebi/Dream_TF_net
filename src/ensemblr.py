from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
import numpy as np
import pdb


class RidgeEnsembler:
    def __init__(self, models, alpha=1.0):
        self.cls = Ridge(alpha=alpha)
        self.models = models
        self.transcription_factor = 'CTCF'

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def fit(self, X, y, S=None, gene_expression=None, da=None, chipseq_fold_coverage=None):
        for model in self.models:
            model.set_transcription_factor(self.transcription_factor)
            model.fit(X, y, S, gene_expression, da, chipseq_fold_coverage)
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X, S, gene_expression, da))
        X_ens = np.array(predictions).transpose()
        self.cls.fit(X_ens, np.max(y, axis=1))

    def predict(self, X, S=None, gene_expression=None, da=None):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X, S, gene_expression, da))
        X_ens = np.array(predictions).transpose()
        out = np.clip(np.array(self.cls.predict(X_ens)), 0, 1)
        return out


class RFEnsembler:

    def __init__(self, models, n_estimators=1000, max_features="auto"):
        self.cls = RandomForestClassifier(n_estimators, max_features=max_features)
        self.models = models
        self.transcription_factor = 'CTCF'

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def fit(self, X, y, S=None, gene_expression=None, da=None, chipseq_fold_coverage=None):
        for model in self.models:
            model.set_transcription_factor(self.transcription_factor)
            model.fit(X, y, S, gene_expression, da, chipseq_fold_coverage)
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X, S, gene_expression, da))
        X_ens = np.array(predictions).transpose()
        self.cls.fit(X_ens, np.max(y, axis=1))

    def predict(self, X, S=None, gene_expression=None, da=None):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X, S, gene_expression, da))
        X_ens = np.array(predictions).transpose()
        out = np.array(self.cls.predict_proba(X_ens))[:, 1]
        return out


class AvgEnsembler:
    def __init__(self, models):
        self.models = models
        self.transcription_factor = 'CTCF'

    def set_transcription_factor(self, transcription_factor):
        self.transcription_factor = transcription_factor

    def fit(self, *args):
        for model in self.models:
            model.set_transcription_factor(self.transcription_factor)
            model.fit(*args)

    def predict(self, *args):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(*args))
        y = np.mean(np.array(predictions).transpose(), axis=1)
        return y
