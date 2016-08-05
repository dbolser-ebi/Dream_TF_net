from sklearn.metrics import *
import numpy as np
from sklearn.metrics import precision_recall_curve, auc



def auroc(y_true, y_model):
    return roc_auc_score(y_true, y_model)


def auprc(y_true, y_model):
    y_true = np.array(y_true, dtype=np.float32)
    y_model = np.array(y_model, dtype=np.float32)
    assert (y_true.size == y_model.size)
    prec_, rec, _ = precision_recall_curve(y_true, y_model)

    #interpolate curve
    prec = prec_
    p_temp = prec[0]
    n = len(prec)

    for i in xrange(n):
        if prec[i] < p_temp:
            prec[i] = p_temp
        else:
            p_temp = prec[i]

    # calculate Area under Curve
    result = auc(rec, prec)

    return result


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
