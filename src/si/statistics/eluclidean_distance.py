import numpy as np

def euclidean_distance(x,y):

    distances = []

    for  sample in y:
        distance = np.sqrt(np.sum(x - sample)**2)
        distances.append(distance)
    return np.array(distances)
