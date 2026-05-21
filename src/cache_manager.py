import os
import numpy as np

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

class CacheManager:
    @staticmethod
    def save(name, data):
        path = os.path.join(CACHE_DIR, f'{name}.npy')
        np.save(path, data)

    @staticmethod
    def load(name):
        path = os.path.join(CACHE_DIR, f'{name}.npy')
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)
        return None