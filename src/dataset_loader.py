import os
import numpy as np
from src.preprocessing import extract_and_preprocess_face

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_dataset(self, update_callback=None):
        faces = []
        labels = []
        image_paths = []

        if not os.path.exists(self.dataset_path):
            return np.array([]), np.array([]), []

        subfolders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]
        total_folders = len(subfolders)

        for folder_idx, person_name in enumerate(subfolders):
            person_folder = os.path.join(self.dataset_path, person_name)
            
            if update_callback:
                p_baca = int(5 + (folder_idx / total_folders) * 30)
                update_callback(p_baca, f"Memotong wajah dataset: {person_name}...")

            for filename in os.listdir(person_folder):
                if filename.lower().endswith(VALID_EXTENSIONS):
                    path = os.path.join(person_folder, filename)
                    img_vector = extract_and_preprocess_face(path)

                    if img_vector is not None:
                        faces.append(img_vector)
                        labels.append(person_name)
                        image_paths.append(path)
                        
        return np.array(faces), np.array(labels), image_paths