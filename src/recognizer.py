import numpy as np
from src.distance import Distance

class Recognizer:
    def __init__(self, model, labels, image_paths):
        self.model = model
        self.labels = labels
        self.image_paths = image_paths

    def predict(self, face_vector):
        query_projection = self.model.transform(face_vector)

        best_score = -1.0
        best_label = 'Unknown'
        best_img_path = None

        for idx, train_projection in enumerate(self.model.projections):
            score = Distance.cosine_similarity(query_projection, train_projection)

            if score > best_score:
                best_score = score
                best_label = self.labels[idx]
                best_img_path = self.image_paths[idx]

        return best_label, best_score, best_img_path