from src.dataset_loader import DatasetLoader
from src.eigenface import EigenFace
from src.cache_manager import CacheManager


class Trainer:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def train(self):
        print('Loading dataset...')

        loader = DatasetLoader(self.dataset_path)

        faces, labels = loader.load_dataset()

        print('Training EigenFace...')

        model = EigenFace(n_components=100)
        model.fit(faces)

        CacheManager.save('faces', faces)
        CacheManager.save('labels', labels)
        CacheManager.save('mean_face', model.mean_face)
        CacheManager.save('eigenfaces', model.eigenfaces)
        CacheManager.save('projections', model.projections)

        print('Training completed.')
