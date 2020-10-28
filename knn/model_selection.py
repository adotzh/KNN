from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import KNNClassifier


def knn_cross_val_score(X, y, k_list, scoring="accuracy", cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    k_list = sorted(k_list)
    clf = KNNClassifier(k_list[-1], **kwargs)
    res = {i: [] for i in k_list}
    for KFold_cv in cv.split(X, y):
        clf.fit(X[KFold_cv[0]], y[KFold_cv[0]])
        distances, indices = clf.kneighbors(X[KFold_cv[1]], True)
        for i in k_list:
            labels = clf._predict_precomputed(indices[:, :i], distances[:, :i])
            score = scorer(y[KFold_cv[1]], labels)
            res[i].append(score)
    return res
