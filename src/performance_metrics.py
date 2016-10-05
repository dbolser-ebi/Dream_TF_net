from sklearn.metrics import *
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as ro


def auroc(y_true, y_model):
    try:
        auc = roc_auc_score(y_true, y_model)
    except:
        auc = 0
    return auc


def auprc(y_true, y_model):
    ro.globalenv['pred'] = y_model
    ro.globalenv['labels'] = y_true
    return ro.r('library(PRROC); pr.curve(scores.class0=pred, weights.class0=labels)$auc.davis.goadrich')[0]


def recall_at_fdr(y_true, y_model, fdr_cutoff):
    '''
    Computes the recall at given FDR cutoff
    :param y_true: true labels, 1 = Bound, 0 = Unbound
    :param y_model: model probabilities, 1 = Bound, 0 = Unbound
    :param fdr_cutoff: FDR cutoff as a fraction
    :return: recall at specified fdr cutoff
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_model)
    fdr = 1 - np.array(precision)
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]
