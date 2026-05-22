import numpy as np

class Distance:
    @staticmethod
    def euclidean_distance(v1, v2):
        """Menghitung jarak lurus antar koordinat wajah menggunakan NumPy."""
        return np.sqrt(np.sum((v1 - v2) ** 2))