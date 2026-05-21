import os
import numpy as np
from src.preprocessing import preprocess_image

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_dataset(self, update_callback=None):
        faces = []
        labels = []
        image_paths = [] # Kita simpan path-nya untuk menampilkan output gambar mirip di GUI

        if not os.path.exists(self.dataset_path):
            return np.array([]), np.array([]), []

        subfolders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]
        total_folders = len(subfolders)

        for folder_idx, person_name in enumerate(subfolders):
            person_folder = os.path.join(self.dataset_path, person_name)
            
            # Sinyal progress awal pembacaan folder (alokasi rentang kemajuan 5% s.d 35%)
            if update_callback:
                p_baca = int(5 + (folder_idx / total_folders) * 30)
                update_callback(p_baca, f"Membaca dataset subfolder: {person_name}...")

            for filename in os.listdir(person_folder):
                if filename.lower().endswith(VALID_EXTENSIONS):
                    path = os.path.join(person_folder, filename)
                    img = preprocess_image(path)

                    if img is not None:
                        faces.append(img.flatten())
                        labels.append(person_name)
                        image_paths.append(path)

        faces = np.array(faces, dtype=np.float32)
        labels = np.array(labels)

        return faces, labels, image_paths