import os
import numpy as np

from src.preprocessing import preprocess_image

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


class DatasetLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_dataset(self):
        faces = []
        labels = []

        for person_name in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_folder):
                continue

            for filename in os.listdir(person_folder):
                if filename.lower().endswith(VALID_EXTENSIONS):
                    path = os.path.join(person_folder, filename)

                    img = preprocess_image(path)

                    if img is not None:
                        faces.append(img.flatten())
                        labels.append(person_name)

        faces = np.array(faces, dtype=np.float32)
        labels = np.array(labels)

        return faces, labels
