import cv2
import numpy as np

IMAGE_SIZE = (50, 50)

# Load cascade ganda dari OpenCV
CASCADE_FRONTAL = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CASCADE_PROFILE = cv2.data.haarcascades + 'haarcascade_profileface.xml'

face_cascade_frontal = cv2.CascadeClassifier(CASCADE_FRONTAL)
face_cascade_profile = cv2.CascadeClassifier(CASCADE_PROFILE)

def extract_and_preprocess_face(path):
    img = cv2.imread(path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # KELONGGARAN 1: Menurunkan scaleFactor ke 1.05 dan minNeighbors ke 3 (Lebih sensitif mendeteksi wajah)
    faces = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    
    # 2. Jika gagal, coba profile samping dengan detektor yang dilonggarkan
    if len(faces) == 0:
        faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        if len(faces) == 0:
            gray_flipped = cv2.flip(gray, 1)
            faces = face_cascade_profile.detectMultiScale(gray_flipped, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                gray_cropped = gray_flipped[y:y+h, x:x+w]
                gray_cropped = cv2.flip(gray_cropped, 1)
                return finalize_vector(gray_cropped)
                
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        gray_cropped = gray[y:y+h, x:x+w]
        return finalize_vector(gray_cropped)
        
    # KELONGGARAN 2 (PENTING): Jika Haar Cascade temanmu tetap gagal mendeteksi wajah sama sekali,
    # jangan langsung buang gambarnya (None). Sebagai gantinya, potong bagian tengah gambar secara otomatis 
    # agar proses training dataset tidak mogok dan tetap bisa jalan.
    h_img, w_img = gray.shape
    min_dim = min(h_img, w_img)
    start_x = (w_img - min_dim) // 2
    start_y = (h_img - min_dim) // 2
    center_cropped = gray[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    return finalize_vector(center_cropped)

def finalize_vector(gray_img):
    resized = cv2.resize(gray_img, IMAGE_SIZE)
    equalized = cv2.equalizeHist(resized)
    return equalized.flatten().astype(np.float32) / 255.0

def preprocess_uploaded_image(path):
    return extract_and_preprocess_face(path)