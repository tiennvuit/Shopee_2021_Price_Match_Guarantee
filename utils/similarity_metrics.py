import numpy as np
from scipy.spatial.distance import cdist
import os


def manhattan(vector1: np.array, vector2: np.array):
    return cdist(vector1, vector2, metric='cityblock')


def euclid(vector1: np.array, vector2: np.array):
    return np.linalg.norm(vector1-vector2)


def cosine(vector1: np.array, vector2: np.array):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

