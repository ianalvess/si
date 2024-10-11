import numpy as np

def cosine_distance(x, y):
    x = np.array(x)
    y = np.array(y)

    dot_product = np.dot(y, x)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y, axis=1)

    similarity = dot_product / (norm_x * norm_y)

    distance = 1 - similarity

    return distance