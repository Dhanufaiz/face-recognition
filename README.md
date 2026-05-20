# FULL FACE RECOGNITION PROJECT (OPTIMIZED)

## Struktur Project

```text
project/
│
├── main.py
├── gui.py
│
├── src/
│   ├── preprocessing.py
│   ├── dataset_loader.py
│   ├── eigenface.py
│   ├── distance.py
│   ├── recognizer.py
│   ├── cache_manager.py
│   └── trainer.py
│
├── dataset/
│   ├── person1/
│   ├── person2/
│   └── ...
│
├── cache/
│
└── assets/
```

---

# requirements.txt

```txt
numpy
opencv-python
Pillow
scikit-learn
```

---

# src/preprocessing.py

```python
import cv2
import numpy as np

IMAGE_SIZE = (100, 100)


def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, IMAGE_SIZE)

    img = img.astype(np.float32) / 255.0

    return img



def preprocess_uploaded_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, IMAGE_SIZE)

    img = img.astype(np.float32) / 255.0

    return img.flatten()
```

---

# src/dataset_loader.py

```python
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
```

---

# src/cache_manager.py

```python
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
```

---

# src/eigenface.py

```python
import numpy as np


class EigenFace:

    def __init__(self, n_components=100):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None

    def fit(self, X):
        self.mean_face = np.mean(X, axis=0)

        A = X - self.mean_face

        U, S, VT = np.linalg.svd(A, full_matrices=False)

        self.eigenfaces = VT[:self.n_components]

        self.projections = np.dot(A, self.eigenfaces.T)

    def transform(self, face):
        face = face - self.mean_face
        return np.dot(face, self.eigenfaces.T)
```

---

# src/distance.py

```python
import numpy as np


class Distance:

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )
```

---

# src/recognizer.py

```python
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
```

---

# src/trainer.py

```python
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
```

---

# gui.py

```python
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from PIL import Image
from PIL import ImageTk

import threading
import numpy as np

from src.preprocessing import preprocess_uploaded_image
from src.cache_manager import CacheManager
from src.eigenface import EigenFace
from src.recognizer import Recognizer
from src.trainer import Trainer


class FaceRecognitionGUI:

    def __init__(self, root):
        self.root = root

        self.root.title('Face Recognition')
        self.root.geometry('1000x700')
        self.root.configure(bg='#1e1e1e')

        self.model = None
        self.labels = None

        self.setup_ui()

        self.load_model()

    def setup_ui(self):
        title = tk.Label(
            self.root,
            text='FACE RECOGNITION',
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#1e1e1e'
        )
        title.pack(pady=20)

        self.image_label = tk.Label(self.root, bg='#1e1e1e')
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(
            self.root,
            text='No Image Selected',
            font=('Arial', 16),
            fg='white',
            bg='#1e1e1e'
        )
        self.result_label.pack(pady=10)

        self.score_label = tk.Label(
            self.root,
            text='',
            font=('Arial', 12),
            fg='lightgreen',
            bg='#1e1e1e'
        )
        self.score_label.pack(pady=5)

        button_frame = tk.Frame(self.root, bg='#1e1e1e')
        button_frame.pack(pady=20)

        upload_button = tk.Button(
            button_frame,
            text='Upload Image',
            command=self.upload_image,
            width=20,
            height=2
        )
        upload_button.grid(row=0, column=0, padx=10)

        train_button = tk.Button(
            button_frame,
            text='Train Model',
            command=self.start_training,
            width=20,
            height=2
        )
        train_button.grid(row=0, column=1, padx=10)

        self.progress = ttk.Progressbar(
            self.root,
            orient='horizontal',
            length=400,
            mode='indeterminate'
        )
        self.progress.pack(pady=10)

    def start_training(self):
        thread = threading.Thread(target=self.train_model)
        thread.start()

    def train_model(self):
        self.progress.start()

        trainer = Trainer('dataset')
        trainer.train()

        self.load_model()

        self.progress.stop()

        self.result_label.config(text='Training Completed')

    def load_model(self):
        mean_face = CacheManager.load('mean_face')
        eigenfaces = CacheManager.load('eigenfaces')
        projections = CacheManager.load('projections')
        labels = CacheManager.load('labels')

        if mean_face is None:
            return

        model = EigenFace()

        model.mean_face = mean_face
        model.eigenfaces = eigenfaces
        model.projections = projections

        self.model = model
        self.labels = labels

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[('Image Files', '*.jpg *.jpeg *.png')]
        )

        if not path:
            return

        self.show_image(path)

        if self.model is None:
            self.result_label.config(text='Model not trained yet')
            return

        face_vector = preprocess_uploaded_image(path)

        recognizer = Recognizer(self.model, self.labels)

        label, score = recognizer.predict(face_vector)

        self.result_label.config(text=f'Person: {label}')

        self.score_label.config(
            text=f'Similarity: {score:.4f}'
        )

    def show_image(self, path):
        image = Image.open(path)

        image.thumbnail((300, 300))

        photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=photo)
        self.image_label.image = photo
```

---

# main.py

```python
import tkinter as tk

from gui import FaceRecognitionGUI


root = tk.Tk()

app = FaceRecognitionGUI(root)

root.mainloop()
```

---

# FORMAT DATASET

```text
dataset/
│
├── Dhanu/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
│
├── Budi/
│   ├── 001.jpg
│   └── ...
│
└── Sinta/
    ├── 001.jpg
    └── ...
```

---

# INSTALL

```bash
pip install -r requirements.txt
```

---

# RUN

```bash
python main.py
```

---

# FITUR YANG SUDAH DIOPTIMALKAN

✅ grayscale

✅ resize otomatis

✅ float32

✅ cosine similarity

✅ threading GUI

✅ cache model

✅ SVD PCA

✅ hemat RAM

✅ scalable dataset besar

✅ GUI tidak freeze

✅ startup cepat

---

# LANGKAH BERIKUTNYA

Kalau nanti ingin lebih modern:

* webcam realtime
* multi face detection
* anti spoofing
* deep learning
* GPU acceleration
* FaceNet
* ArcFace
* PyTorch CNN

maka architecture ini sudah siap untuk dikembangkan.
