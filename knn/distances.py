import numpy as np


def euclidean_distance(X, Y):
    """
    Computes the Euclidean distance between two (N-D) and (M-D) arrays.

    Parameters
    ----------
    X: np.ndarray (N,D) input array,
    Y: np.ndarray (M,D) input array.

    Returns
    -------
    euclidean : np.ndarray (N,M)
        The Euclidean distance between the corresponding pair of vectors from
        arrays X and Y
    """
    dist = (-2*X.dot(Y.T) + (X**2).sum(axis=1).reshape(-1, 1) + (Y**2).sum(axis=1))**(1/2)
    return dist


def cosine_distance(X, Y):
    """Computes the Cosine distance between the corresponding pair of vectors
    from two (N-D) and (M-D) matrix.

    Parameters
    ----------
    X: np.ndarray (N,D)
        Input matrix.
    Y: np.ndarray (M,D)
        Input matrix.

    Returns
    -------
    cosine : np.ndarray (N,M)
        The Cosine distance between the corresponding pair of vectors from
        arrays X and Y
    """
    xx = ((X**2).sum(axis=1))**(1/2)
    yy = ((Y**2).sum(axis=1))**(1/2)
    dist = (xx.reshape(-1, 1)*yy - X.dot(Y.T))/(xx.reshape(-1, 1)*yy)
    return dist
