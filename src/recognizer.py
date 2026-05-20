import numpy as np

from src.distance import Distance


class Recognizer:

    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def predict(self, face_vector):
        query_projection = self.model.transform(face_vector)

        best_score = -1
        best_label = 'Unknown'

        for idx, train_projection in enumerate(self.model.projections):
            score = Distance.cosine_similarity(
                query_projection,
                train_projection
            )

            if score > best_score:
                best_score = score
                best_label = self.labels[idx]

        return best_label, best_score
