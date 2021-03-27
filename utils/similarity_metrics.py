import numpy as np
from scipy.spatial.distance import cdist
import os


def manhattan(vector1: np.array, vector2: np.array):
    return cdist(vector1, vector2, metric='cityblock', axis=1)


def euclid(vector1: np.array, vector2: np.array):
    return np.linalg.norm(vector1-vector2, axis=1)


def cosine(vector1: np.array, vector2: np.array):
    return np.dot(vector1, vector2, axis=1)/(np.linalg.norm(vector1, axis=1)*np.linalg.norm(vector2, axis=1))


