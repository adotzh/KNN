import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    top_slice = (slice(None), ) * axis + (slice(None, top), )

    if top < ranks.shape[axis]:
        indices = np.argpartition(ranks, top, axis=axis)[top_slice]
        ranks_top = np.take_along_axis(ranks, indices, axis=axis)
        indices = np.take_along_axis(indices, ranks_top.argsort(axis=axis), axis=axis)
    else:
        indices = np.argsort(ranks, axis=axis)[top_slice]

    result = (indices, )

    if return_ranks:
        ranks = np.take_along_axis(ranks, indices, axis=axis)
        result = (ranks, ) + result

    return result if len(result) > 1 else result[0]


class NearestNeighborsFinder:
    """Unsupervised learner for implementing nearest neighbor finder.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    metric : {'euclidean', 'cosine'}, default='euclidean'
        Metric used to compute distances to neighbors. The default metric is
        the standard Euclidean metric. See the documentation of :module:
        `distances` for a list of available metrics.
    """
    def __init__(self, n_neighbors, metric="euclidean"):
        """Initialized class
        """
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            Input training data matrix.
        y : np.ndarray (n_samples)
            Input target values matrix.
        """
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        """Find the k nearest neighbors. Distance between vectors is computed with
        self.metric

        Parameters
        ----------
        X : np.ndarray (n_queries, n_features)
            Test samples.

        return_distance: bool
            Boolean flag for whether to return distances for objects.
        Returns
        -------
        distances: np.ndarray (n_queries, n_neighbors). If return_distance=True array is returned.
            distances[i, j] - distance from the i-th object to its j-th nearest neighbor.

        indices: np.ndarray (n_queries, n_neighbors)
            indices[i, j] - index of the j-th nearest neighbor from the training sample to the object with index i

        If return_distance=False, only the second of the specified arrays is returned.
        """
        xx_true = self._metric_func(X, self._X)
        return get_best_ranks(xx_true, top=self.n_neighbors, axis=1, return_ranks=return_distance)
