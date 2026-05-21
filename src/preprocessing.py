import cv2
import numpy as np

IMAGE_SIZE = (100, 100)

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE)
    # Melakukan histogram equalization untuk mengatasi masalah pencahayaan
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_uploaded_image(path):
    img = preprocess_image(path)
    if img is None:
        return None
    return img.flatten()