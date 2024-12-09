import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Cosine distance for each point in y.

    """

    x = np.array(x)
    y = np.array(y)

    dot_product = np.dot(y, x)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y, axis=1)

    similarity = dot_product / (norm_x * norm_y)

    distance = 1 - similarity

    return distance


if __name__ == '__main__':
    # test cosine_distance
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    our_distance = cosine_distance(x, y)
    # using sklearn
    # to test this snippet, you need to install sklearn (pip install -U scikit-learn)
    from sklearn.metrics.pairwise import euclidean_distances
    sklearn_distance = cosine_distance(x.reshape(1, -1), y)
    assert np.allclose(our_distance, sklearn_distance)
    print(our_distance, sklearn_distance)
