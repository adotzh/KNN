import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    """Classifier implementing the k-nearest neighbors vote.

    Parameters
    ----------
    n_neighbors: int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    algorithm: {'my_own', 'brute', ’kd_tree’, ’ball_tree’}, default=’my_own’
        Algorithm used to compute the nearest neighbors:
        - 'my_own' will use knn.nearest_neighbors.NearestNeighborsFinder();
        - ’brute’ - sklearn.neighbors.NearestNeighbors(algorithm=’brute’);
        – ’kd_tree’ - sklearn.neighbors.NearestNeighbors(algorithm=’kd_tree’);
        – ’ball_tree’ sklearn.neighbors.NearestNeighbors(algorithm=’ball_tree’)

    metric : {'euclidean', 'cosine'}, default='euclidean'
        Metric used to compute distances to neighbors. The default metric is
        the standard Euclidean metric. See the documentation of :module:
        `distances` for a list of available metrics.

    weights: {'uniform','distance'} , default=’uniform’
        weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally, and equal 1.
        - 'distance' : weighted nearest neighbor method, where the weight of
        each object is equal to weight = 1/(distance + eps), where eps=10^{-5}
    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier
    _labels : array of shape as train data.
        Labels of train data
    _finder: function()
        Model for finding the nearest neighbors

    """
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)
        
        self.n_neighbors = n_neighbors
        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        """Fit the model using X as training data and y as target values.
        Trains the algorithm based on the strategy specified in the
        parameter self.algorithm.
        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            Input training data matrix.
        y : np.ndarray (n_samples)
            Input target values matrix.
        """
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def kneighbors(self, X, return_distance=False):
        """Read description of method in _finder documentation.
        """
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        """Predict the class labels for the provided data
        Parametres
        ----------
        X: np.ndarray (n_samples, n_features)
            Test sample
        Returns
        -------
        see _predict_precomputed.
        """
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)

    def _predict_precomputed(self, indices, distances):
        """Predict the class labels for the provided data
        Parametres
        ----------
        distances: np.ndarray (n_queries, n_neighbors).
            If return_distance=True array is returned.
            distances[i, j] - distance from the i-th object to its j-th nearest neighbor.

        indices: np.ndarray (n_queries, n_neighbors)
            indices[i, j] - index of the j-th nearest neighbor from the training sample
            to the object with index i
        Returns
        -------
        pred: np.ndarray (n_queries, )
            The predicted classes for the set of nearest neighbors and distances
        """
        n_queries = distances.shape[0]
        neighbor_labels = self._labels[indices]
        # pred = np.zeros((n_queries, self.n_classes))
        if self._weights == 'uniform':
            weights = np.ones(distances.shape)
        elif self._weights == 'distance':
            weights = 1/(self.EPS + distances)
        pred = [(np.argmax(np.bincount(neighbor_labels[i], weights[i]))) for i in range(n_queries)]
        return pred


class BatchedMixin:
    def __init__(self):
        self.batch_size = None

    def kneighbors(self, X, return_distance=False):
        if not hasattr(self,  'batch_size'):
            self.batch_size = None

        batch_size = self.batch_size or X.shape[0]

        distances, indices = [], []

        for i_min in range(0, X.shape[0], batch_size):
            i_max = min(i_min + batch_size, X.shape[0])
            X_batch = X[i_min:i_max]

            indices_ = super().kneighbors(X_batch, return_distance=return_distance)
            if return_distance:
                distances_, indices_ = indices_
            else:
                distances_ = None

            indices.append(indices_)
            if distances_ is not None:
                distances.append(distances_)

        indices = np.vstack(indices)
        distances = np.vstack(distances) if distances else None

        result = (indices,)
        if return_distance:
            result = (distances,) + result
        return result if len(result) > 1 else result[0]

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


class BatchedKNNClassifier(BatchedMixin, KNNClassifier):
    def __init__(self, *args, **kwargs):
        KNNClassifier.__init__(self, *args, **kwargs)
        BatchedMixin.__init__(self)
