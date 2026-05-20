import numpy as np


class EigenFace:

    def __init__(self, n_components=100):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None

    def fit(self, X):
        self.mean_face = np.mean(X, axis=0)

        A = X - self.mean_face

        U, S, VT = np.linalg.svd(A, full_matrices=False)

        self.eigenfaces = VT[:self.n_components]

        self.projections = np.dot(A, self.eigenfaces.T)

    def transform(self, face):
        face = face - self.mean_face
        return np.dot(face, self.eigenfaces.T)
