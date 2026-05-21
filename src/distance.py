import numpy as np

class Distance:
    @staticmethod
    def cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.sqrt(np.sum(a ** 2))
        norm_b = np.sqrt(np.sum(b ** 2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)