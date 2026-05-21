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
        self.window.title("Face Recognition Dashboard - Eigenface + PCA")
        self.window.geometry("780x490")
        
        self.model = None
        self.labels = None
        self.image_paths = None
        
        # Otomatis load model dari Cache saat aplikasi baru dibuka
        self.muat_data_cache_startup()
        
        # --- Desain Komponen Antarmuka GUI ---
        left_panel = tk.Frame(window, padx=15, pady=15)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.btn_load_dataset = tk.Button(left_panel, text="Insert Your Dataset", width=22, command=self.start_training_thread, bg="#E1E1E1")
        self.btn_load_dataset.pack(pady=5)
        
        self.btn_upload_test = tk.Button(left_panel, text="Insert Your Image", width=22, command=self.click_upload_foto_uji, bg="#E1E1E1")
        self.btn_upload_test.pack(pady=5)
        
        # Progress Bar Loading Indicator
        self.progress_bar = ttk.Progressbar(left_panel, orient="horizontal", length=180, mode="determinate")
        self.progress_bar.pack(pady=12)
        
        status_awal = "Model Status: Ready (Loaded from cache)" if self.model is not None else "Model Status: Please train first!"
        self.lbl_status = tk.Label(left_panel, text=status_awal, fg="purple", justify=tk.LEFT, wraplength=180)
        self.lbl_status.pack(pady=5)
        
        self.lbl_execution_time = tk.Label(left_panel, text="Execution time: -", fg="green", anchor="w", justify=tk.LEFT)
        self.lbl_execution_time.pack(fill=tk.X, pady=5)
        
        self.lbl_score = tk.Label(left_panel, text="Skor Similarity: -", fg="green", anchor="w", justify=tk.LEFT)
        self.lbl_score.pack(fill=tk.X, pady=5)
        
        self.lbl_name_result = tk.Label(left_panel, text="Result: -", font=("Arial", 11, "bold"), fg="blue", anchor="w")
        self.lbl_name_result.pack(fill=tk.X, pady=15)
        
        # --- Panel Kanan Display Visualisasi Citra Wajah ---
        right_panel = tk.Frame(window, padx=10, pady=10)
        right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        frame_uji = tk.LabelFrame(right_panel, text="Test Image")
        frame_uji.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.canvas_uji = tk.Label(frame_uji, text="[ Foto Uji ]")
        self.canvas_uji.pack(expand=True)
        
        frame_mirip = tk.LabelFrame(right_panel, text="Closest Result")
        frame_mirip.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.canvas_mirip = tk.Label(frame_mirip, text="[ Hasil Mirip ]")
        self.canvas_mirip.pack(expand=True)

    def muat_data_cache_startup(self):
        """Mengecek apakah hasil training terdahulu ada di folder cache."""
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
            self.btn_load_dataset.config(state=tk.DISABLED)
            self.progress_bar["value"] = 0
            self.lbl_status.config(text="Status: Memulai pindaian direktori...", fg="orange")
            
            # Melempar kalkulasi objek ke Background Worker Thread agar GUI tidak Not Responding
            t = threading.Thread(target=self.worker_training, args=(folder,))
            t.daemon = True
            t.start()

    def trigger_update_progress(self, nilai, status_teks):
        """Metode jembatan sinkronisasi berkala untuk menggerakkan progress bar."""
        def update_gui():
            self.progress_bar.config(value=nilai)
            self.lbl_status.config(text=f"Status: {status_teks} ({nilai}%)")
            self.window.update_idletasks() # Refresh grafis jendela secara instan
            
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
        self.btn_load_dataset.config(state=tk.NORMAL)
        self.progress_bar["value"] = 100
        self.lbl_execution_time.config(text=f"Execution time: {durasi:.2f} s (Offline)")
        self.lbl_status.config(text="Model Status: Ready (Updated cache!)", fg="green")
        messagebox.showinfo("Sukses", "Proses pengenalan matriks wajah selesai & cache diperbarui!")

    def training_gagal(self, error_msg):
        self.btn_load_dataset.config(state=tk.NORMAL)
        self.progress_bar["value"] = 0
        self.lbl_status.config(text="Status: Training Gagal!", fg="red")
        messagebox.showerror("Error", error_msg)

    def click_upload_foto_uji(self):
        if self.model is None:
            messagebox.showwarning("Peringatan", "Silakan masukkan dataset training terlebih dahulu!")
            return
            
        file_path = filedialog.askopenfilename(title="Pilih Citra Uji Wajah", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.render_image(file_path, self.canvas_uji)
            
            # Pra-proses vektor uji
            face_vector = preprocess_uploaded_image(file_path)
            if face_vector is None:
                messagebox.showerror("Error", "Gagal memproses file citra uji!")
                return
                
            start_t = time.time()
            # Inisialisasi proses pencocokan online lewat Cosine Similarity terbesar
            recognizer = Recognizer(self.model, self.labels, self.image_paths)
            nama_cocok, kemiripan_skor, path_gambar_mirip = recognizer.predict(face_vector)
            end_t = time.time()
            
            self.lbl_execution_time.config(text=f"Execution time: {end_t - start_t:.4f} detik")
            self.lbl_score.config(text=f"Skor Similarity: {kemiripan_skor * 100:.2f}%")
            
            # Batas threshold pengenalan (misal minimal 65% kemiripan untuk validitas data)
            if kemiripan_skor > 0.65:
                self.lbl_name_result.config(text=f"Result: {nama_cocok}")
                if path_gambar_mirip and os.path.exists(path_gambar_mirip):
                    self.render_image(path_gambar_mirip, self.canvas_mirip)
                else:
                    self.canvas_mirip.config(image='', text="[ Gambar Terklasifikasi ]")
            else:
                self.lbl_name_result.config(text="Result: Unknown (Tidak Dikenali)")
                self.canvas_mirip.config(image='', text="[ Di bawah Batas Threshold ]")

    def render_image(self, img_path, label_widget):
        img = Image.open(img_path)
        img = img.resize((185, 185), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk