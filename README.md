<p align="center">
  <img src="assets/logo_uns.png" alt="Logo UNS" width="120"/>
</p>

<h1 align="center"> Full Face Recognition — Eigenface Method</h1>

<p align="center">
  <strong>Kelompok 8 <br> informatika D <br> Universitas Sebelas Maret</strong>
</p>

---
## 👥 Anggota Kelompok
* **Dhanu Fa'iz Sugara**        - L0125008
* **M. Juan Fernando Aziz A.**  - L0125052
* **Andra Satria Ardiansyah**   - L0125072

**Dosen Pengampu:** Drs. Bambang Harjito, M.App.Sc., Ph.D.

---

## Arsitektur Program

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
    # Melakukan histogram equalization untuk mengatasi masalah pencahayaan
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_uploaded_image(path):
    img = preprocess_image(path)
    if img is None:
        return None
    return img.flatten()
```

---

# src/dataset_loader.py

```python
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
```python
import numpy as np

class EigenFace:
    def __init__(self, n_components=12):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None

    def power_iteration(self, A, num_simulations=25):
        """Mencari matriks eigen terbesar secara manual tanpa library."""
        b_k = np.random.rand(A.shape[1])
        for _ in range(num_simulations):
            b_k1 = np.dot(A, b_k)
            norm = np.sqrt(np.sum(b_k1 ** 2))
            if norm == 0:
                break
            b_k = b_k1 / norm
        
        Ab = np.dot(A, b_k)
        nilai_eigen = np.sum(b_k * Ab) / np.sum(b_k ** 2)
        return nilai_eigen, b_k

    def fit(self, X, update_callback=None):
        self.mean_face = np.mean(X, axis=0)
        A = X - self.mean_face

        if update_callback:
            update_callback(40, "Menghitung Matriks Kovarian (L = A * A^T)...")
        
        # Matriks Kovarian bentuk kecil (M x M) agar komputasi ringan
        L = np.dot(A, A.T)
        
        A_eff = L.copy().astype(float)
        n = A_eff.shape[0]
        k = min(self.n_components, n)
        
        eigen_vectors_L = []
        
        for step in range(k):
            val, vec = self.power_iteration(A_eff)
            eigen_vectors_L.append(vec)
            
            # Deflation: kurangi porsi komponen matematika yang telah didapatkan
            A_eff -= val * np.outer(vec, vec)
            
            if update_callback:
                p_eigen = int(45 + ((step + 1) / k) * 45)
                update_callback(p_eigen, f"Mengekstrak fitur matriks eigen ({step + 1}/{k})...")

        eigen_vectors_L = np.array(eigen_vectors_L).T
        
        # Kembalikan ke dimensi asli: U = A^T * V
        self.eigenfaces = np.dot(A.T, eigen_vectors_L).T
        
        # Normalisasi panjang vektor fitur agar bernilai 1
        for i in range(self.eigenfaces.shape[0]):
            norm_val = np.sqrt(np.sum(self.eigenfaces[i] ** 2))
            if norm_val != 0:
                self.eigenfaces[i] /= norm_val

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
        dot_product = np.dot(a, b)
        norm_a = np.sqrt(np.sum(a ** 2))
        norm_b = np.sqrt(np.sum(b ** 2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
```

---

# src/recognizer.py

```python
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
```

---

# src/trainer.py

```python
import numpy as np  # <--- Ini baris yang kurang yang menyebabkan np undefined!
from src.dataset_loader import DatasetLoader
from src.eigenface import EigenFace
from src.cache_manager import CacheManager

class Trainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def train(self, update_callback=None):
        if update_callback:
            update_callback(2, "Menginisialisasi pustaka dataset...")

        loader = DatasetLoader(self.dataset_path)
        faces, labels, image_paths = loader.load_dataset(update_callback=update_callback)

        if len(faces) == 0:
            raise ValueError("Dataset kosong atau struktur direktori salah!")

        # Menggunakan 12 komponen utama agar komputasi manual Power Iteration lebih efisien
        model = EigenFace(n_components=12)
        model.fit(faces, update_callback=update_callback)

        if update_callback:
            update_callback(95, "Mengarsip data model ke penyimpanan cache...")

        # Simpan semua komponen matriks dan path gambar ke dalam cache (.npy)
        CacheManager.save('faces', faces)
        CacheManager.save('labels', labels)
        CacheManager.save('image_paths', np.array(image_paths, dtype=object)) # Membutuhkan np di sini
        CacheManager.save('mean_face', model.mean_face)
        CacheManager.save('eigenfaces', model.eigenfaces)
        CacheManager.save('projections', model.projections)

        if update_callback:
            update_callback(100, "Selesai!")

        return model, labels, image_paths
```

---

# gui.py

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import time
import threading
import os
from PIL import Image, ImageTk
import numpy as np

from src.trainer import Trainer
from src.recognizer import Recognizer
from src.eigenface import EigenFace
from src.cache_manager import CacheManager
from src.preprocessing import preprocess_uploaded_image

class FaceRecognitionGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition System")
        self.window.geometry("820x540")
        self.window.configure(bg="#F8F9FA")  # Background abu-abu sangat muda (modern)
        
        # Menggunakan tema ttk yang lebih bersih
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Konfigurasi style Progress Bar agar minimalis
        self.style.configure("Horizontal.TProgressbar", 
                             background="#4A90E2", 
                             troughcolor="#E9ECEF", 
                             thickness=8)
        
        self.model = None
        self.labels = None
        self.image_paths = None
        
        # Otomatis muat data dari cache jika ada
        self.muat_data_cache_startup()
        
        # --- TOP HEADER ---
        header_frame = tk.Frame(window, bg="#FFFFFF", highlightbackground="#E9ECEF", highlightthickness=1)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        lbl_title = tk.Label(header_frame, text="FACE RECOGNITION", font=("Helvetica", 16, "bold"), fg="#212529", bg="#FFFFFF", padx=20, pady=15)
        lbl_title.pack(side=tk.LEFT)
        
        status_awal = "Model Ready (Loaded from cache)" if self.model is not None else "Model Status: Please train first!"
        self.lbl_status = tk.Label(header_frame, text=status_awal, font=("Helvetica", 9), fg="#6C757D", bg="#FFFFFF", padx=20)
        self.lbl_status.pack(side=tk.RIGHT, pady=15)

        # --- MAIN BODY ---
        body_frame = tk.Frame(window, bg="#F8F9FA", padx=20, pady=20)
        body_frame.pack(expand=True, fill=tk.BOTH)

        # --- LEFT PANEL (CONTROL & RESULTS) ---
        left_panel = tk.Frame(body_frame, bg="#FFFFFF", padx=20, pady=20, highlightbackground="#E9ECEF", highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Tombol-tombol dengan gaya flat modern
        self.btn_load_dataset = tk.Button(left_panel, text="Insert Your Dataset", font=("Helvetica", 10, "bold"), 
                                          fg="#FFFFFF", bg="#4A90E2", activebackground="#357ABD",
                                          relief=tk.FLAT, width=22, height=2, command=self.start_training_thread, cursor="hand2")
        self.btn_load_dataset.pack(pady=(0, 10))
        
        self.btn_upload_test = tk.Button(left_panel, text="Insert Your Image", font=("Helvetica", 10, "bold"), 
                                         fg="#FFFFFF", bg="#343A40", activebackground="#212529",
                                         relief=tk.FLAT, width=22, height=2, command=self.click_upload_foto_uji, cursor="hand2")
        self.btn_upload_test.pack(pady=(0, 15))
        
        # Progress Bar Minimalis
        self.progress_bar = ttk.Progressbar(left_panel, orient="horizontal", length=180, mode="determinate", style="Horizontal.TProgressbar")
        self.progress_bar.pack(pady=(0, 20))
        
        # Garis Pembatas Tipis
        separator = tk.Frame(left_panel, height=1, bg="#E9ECEF")
        separator.pack(fill=tk.X, pady=(0, 15))
        
        # Panel Informasi & Hasil
        self.lbl_execution_time = tk.Label(left_panel, text="Execution Time\n-", font=("Helvetica", 9), fg="#6C757D", bg="#FFFFFF", justify=tk.LEFT, anchor="w")
        self.lbl_execution_time.pack(fill=tk.X, pady=(0, 12))
        
        self.lbl_score = tk.Label(left_panel, text="Similarity Score\n-", font=("Helvetica", 9), fg="#6C757D", bg="#FFFFFF", justify=tk.LEFT, anchor="w")
        self.lbl_score.pack(fill=tk.X, pady=(0, 15))
        
        self.lbl_name_result = tk.Label(left_panel, text="RESULT\n-", font=("Helvetica", 11, "bold"), fg="#4A90E2", bg="#FFFFFF", justify=tk.LEFT, anchor="w")
        self.lbl_name_result.pack(fill=tk.X, pady=(5, 0))
        
        # --- RIGHT PANEL (DISPLAY IMAGE PLACES) ---
        right_panel = tk.Frame(body_frame, bg="#F8F9FA")
        right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(10, 0))
        
        # Box Foto Uji
        frame_uji = tk.LabelFrame(right_panel, text=" Test Image ", font=("Helvetica", 9, "bold"), fg="#495057", bg="#FFFFFF", relief=tk.GROOVE, padx=10, pady=10)
        frame_uji.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))
        self.canvas_uji = tk.Label(frame_uji, text="No Image", font=("Helvetica", 9), fg="#ADB5BD", bg="#F8F9FA")
        self.canvas_uji.pack(expand=True, fill=tk.BOTH)
        
        # Box Hasil Cocok
        frame_mirip = tk.LabelFrame(right_panel, text=" Closest Result ", font=("Helvetica", 9, "bold"), fg="#495057", bg="#FFFFFF", relief=tk.GROOVE, padx=10, pady=10)
        frame_mirip.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(5, 0))
        self.canvas_mirip = tk.Label(frame_mirip, text="No Match", font=("Helvetica", 9), fg="#ADB5BD", bg="#F8F9FA")
        self.canvas_mirip.pack(expand=True, fill=tk.BOTH)

    def muat_data_cache_startup(self):
        mean_face = CacheManager.load('mean_face')
        eigenfaces = CacheManager.load('eigenfaces')
        projections = CacheManager.load('projections')
        labels = CacheManager.load('labels')
        image_paths = CacheManager.load('image_paths')

        if mean_face is not None and eigenfaces is not None:
            self.model = EigenFace()
            self.model.mean_face = mean_face
            self.model.eigenfaces = eigenfaces
            self.model.projections = projections
            self.labels = labels
            self.image_paths = image_paths.tolist() if isinstance(image_paths, np.ndarray) else image_paths

    def start_training_thread(self):
        folder = filedialog.askdirectory(title="Pilih Folder Dataset Utama")
        if folder:
            self.btn_load_dataset.config(state=tk.DISABLED, bg="#ADB5BD")
            self.progress_bar["value"] = 0
            self.lbl_status.config(text="Status: Scanning directory...", fg="#FF9F43")
            
            t = threading.Thread(target=self.worker_training, args=(folder,))
            t.daemon = True
            t.start()

    def trigger_update_progress(self, nilai, status_teks):
        def update_gui():
            self.progress_bar.config(value=nilai)
            self.lbl_status.config(text=f"{status_teks} ({nilai}%)")
            self.window.update_idletasks()
        self.window.after(0, update_gui)

    def worker_training(self, folder):
        start_t = time.time()
        try:
            trainer_obj = Trainer(folder)
            model, labels, image_paths = trainer_obj.train(update_callback=self.trigger_update_progress)
            
            self.model = model
            self.labels = labels
            self.image_paths = image_paths
            
            end_t = time.time()
            self.window.after(0, lambda: self.training_sukses(end_t - start_t))
        except Exception as e:
            self.window.after(0, lambda: self.training_gagal(str(e)))

    def training_sukses(self, durasi):
        self.btn_load_dataset.config(state=tk.NORMAL, bg="#4A90E2")
        self.progress_bar["value"] = 100
        self.lbl_execution_time.config(text=f"Execution Time\n{durasi:.2f} s (Offline)")
        self.lbl_status.config(text="Model Status: Ready (Cache Updated)", fg="#28C76F")
        messagebox.showinfo("Success", "Training completed successfully and cache updated!")

    def training_gagal(self, error_msg):
        self.btn_load_dataset.config(state=tk.NORMAL, bg="#4A90E2")
        self.progress_bar["value"] = 0
        self.lbl_status.config(text="Status: Training Failed!", fg="#EA5455")
        messagebox.showerror("Error", error_msg)

    def click_upload_foto_uji(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please insert and train a dataset first!")
            return
            
        file_path = filedialog.askopenfilename(title="Pilih Citra Uji Wajah", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.render_image(file_path, self.canvas_uji)
            
            face_vector = preprocess_uploaded_image(file_path)
            if face_vector is None:
                messagebox.showerror("Error", "Failed to preprocess test image!")
                return
                
            start_t = time.time()
            recognizer = Recognizer(self.model, self.labels, self.image_paths)
            nama_cocok, kemiripan_skor, path_gambar_mirip = recognizer.predict(face_vector)
            end_t = time.time()
            
            # Mengubah skor similarity ke persen (0.0 - 1.0 menjadi 0% - 100%)
            similarity_percentage = max(0.0, min(100.0, kemiripan_skor * 100))
            
            self.lbl_execution_time.config(text=f"Execution Time\n{end_t - start_t:.4f} seconds")
            self.lbl_score.config(text=f"Similarity Score\n{similarity_percentage:.2f}%")
            
            # Batas Threshold Penerimaan (misal > 65%)
            if similarity_percentage > 65.0:
                self.lbl_name_result.config(text=f"RESULT\n{nama_cocok}", fg="#28C76F")
                if path_gambar_mirip and os.path.exists(path_gambar_mirip):
                    self.render_image(path_gambar_mirip, self.canvas_mirip)
                else:
                    self.canvas_mirip.config(image='', text="[ Image Classified ]", fg="#6C757D")
            else:
                self.lbl_name_result.config(text="RESULT\nUnknown", fg="#EA5455")
                self.canvas_mirip.config(image='', text="[ Below Threshold ]", fg="#EA5455")

    def render_image(self, img_path, label_widget):
        img = Image.open(img_path)
        # Menyesuaikan ukuran box gambar agar pas dengan layout baru yang rapi
        img = img.resize((210, 210), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk
```

---

# main.py

```python
import tkinter as tk
from gui import FaceRecognitionGUI

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
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
├── Juan/
│   ├── 001.jpg
│   └── ...
│
└── Andra/
    ├── 001.jpg
    └── ...
```

---

# CARA INSTALL
Clone Repository ke local
```txt
git clone https://github.com/dhanufaiz/face-recognition
```
install requirement yang ada di dalam repository
```bash
pip install -r requirements.txt
```
run main.py 
```bash
python main.py
```