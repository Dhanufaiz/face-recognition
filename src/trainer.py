import numpy as np 
import torch
from src.dataset_loader import DatasetLoader
from src.eigenface import EigenFace
from src.cache_manager import CacheManager

class Trainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def train(self, update_callback=None):
        if update_callback:
            update_callback(2, "Menginisialisasi pustaka dataset...")

        # 1. Memuat seluruh gambar dari folder dataset
        loader = DatasetLoader(self.dataset_path)
        faces, labels, image_paths = loader.load_dataset(update_callback=update_callback)

        if len(faces) == 0:
            raise ValueError("Dataset kosong atau struktur direktori salah! Pastikan wajah terdeteksi.")

        # 2. Proses perhitungan Aljabar Linier dengan matriks kolom vertikal murni temanmu
        model = EigenFace(n_components=50)
        model.fit(faces, update_callback=update_callback)

        if update_callback:
            update_callback(95, "Mengarsip data model ke penyimpanan cache...")
            
        # 3. Amankan data ke numpy array biasa agar CacheManager milikmu tidak eror/crash
        # Kita pastikan semua objek bukan berupa objek Tensor PyTorch saat disimpan
        faces_np = np.array(faces, dtype=np.float32)
        labels_np = np.array(labels, dtype=object)
        paths_np = np.array(image_paths, dtype=object)

        # Ekstraksi isi matriks model ke NumPy array murni
        mean_face_save = model.mean_face if isinstance(model.mean_face, np.ndarray) else model.mean_face.cpu().numpy()
        eigenfaces_save = model.eigenfaces if isinstance(model.eigenfaces, np.ndarray) else model.eigenfaces.cpu().numpy()
        projections_save = model.projections if isinstance(model.projections, np.ndarray) else model.projections.cpu().numpy()

        # 4. Simpan ke folder cache/.npy
        CacheManager.save('faces', faces_np)
        CacheManager.save('labels', labels_np)
        CacheManager.save('image_paths', paths_np)
        CacheManager.save('mean_face', mean_face_save)
        CacheManager.save('eigenfaces', eigenfaces_save)
        CacheManager.save('projections', projections_save)

        if update_callback:
            update_callback(100, "Selesai!")