import numpy as np  # <--- Ini baris yang kurang yang menyebabkan np undefined!
from src.dataset_loader import DatasetLoader
from src.eigenface import EigenFace
from src.cache_manager import CacheManager

class Trainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def train(self, update_callback=None):
        if update_callback:
            update_callback(2, "Menginisialisasi pustaka dataset...")

        loader = DatasetLoader(self.dataset_path)
        faces, labels, image_paths = loader.load_dataset(update_callback=update_callback)

        if len(faces) == 0:
            raise ValueError("Dataset kosong atau struktur direktori salah!")

        # Menggunakan 12 komponen utama agar komputasi manual Power Iteration lebih efisien
        model = EigenFace(n_components=12)
        model.fit(faces, update_callback=update_callback)

        if update_callback:
            update_callback(95, "Mengarsip data model ke penyimpanan cache...")

        # Simpan semua komponen matriks dan path gambar ke dalam cache (.npy)
        CacheManager.save('faces', faces)
        CacheManager.save('labels', labels)
        CacheManager.save('image_paths', np.array(image_paths, dtype=object)) # Membutuhkan np di sini
        CacheManager.save('mean_face', model.mean_face)
        CacheManager.save('eigenfaces', model.eigenfaces)
        CacheManager.save('projections', model.projections)

        if update_callback:
            update_callback(100, "Selesai!")

        return model, labels, image_paths