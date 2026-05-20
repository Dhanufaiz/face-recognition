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
