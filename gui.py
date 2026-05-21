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
        self.window.configure(bg="#0877E6")  
        
        try:
            icon_img = Image.open("assets/logo_uns.png")
            icon_tk = ImageTk.PhotoImage(icon_img)
            self.window.iconphoto(False, icon_tk)
        except Exception as e:
            print(f"Ikon gagal dimuat: {e}")
            
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Horizontal.TProgressbar", 
                             background="#4A90E2", 
                             troughcolor="#E9ECEF", 
                             thickness=8)
        
        self.model = None
        self.labels = None
        self.image_paths = None
        self.threshold_var = tk.DoubleVar(value=50.0)
        
        # Load cache
        self.muat_data_cache_startup()
        
    # Header
        header_frame = tk.Frame(window, bg="#FFFFFF", highlightbackground="#E9ECEF", highlightthickness=1)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        left_header_container = tk.Frame(header_frame, bg="#FFFFFF")
        left_header_container.pack(side=tk.LEFT, padx=20, pady=10)
        
        try:
            logo_img = Image.open("img/logo_uns.png") 
            logo_img = logo_img.resize((40, 40), Image.Resampling.LANCZOS)
            self.logo_tk = ImageTk.PhotoImage(logo_img)
            
            lbl_logo = tk.Label(left_header_container, image=self.logo_tk, bg="#FFFFFF")
            lbl_logo.pack(side=tk.LEFT, padx=(0, 10)) 
        except Exception as e:
            print(f"Logo header opsional tidak ditemukan: {e}")
            
        lbl_title = tk.Label(left_header_container, text="FACE RECOGNITION | KELOMPOK 8", font=("Helvetica", 16, "bold"), fg="#212529", bg="#FFFFFF")
        lbl_title.pack(side=tk.LEFT)
        
        status_awal = "Model sudah siap (dimuat dari cache)" if self.model is not None else "Model Status: Silahkan Training dataset terlebih dahulu!"
        self.lbl_status = tk.Label(header_frame, text=status_awal, font=("Helvetica", 9), fg="#6C757D", bg="#FFFFFF", padx=20)
        self.lbl_status.pack(side=tk.RIGHT, pady=15)
        
    # Body
        body_frame = tk.Frame(window, bg="#F8F9FA", padx=20, pady=20)
        body_frame.pack(expand=True, fill=tk.BOTH)

        # Left Panel 
        left_panel = tk.Frame(body_frame, bg="#FFFFFF", padx=20, pady=20, highlightbackground="#E9ECEF", highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Tombol Aksi
        self.btn_load_dataset = tk.Button(left_panel, text="Upload Dataset", font=("Helvetica", 10, "bold"), 
                                          fg="#FFFFFF", bg="#4A90E2", activebackground="#357ABD",
                                          relief=tk.FLAT, width=22, height=2, command=self.start_training_thread, cursor="hand2")
        self.btn_load_dataset.pack(pady=(0, 10))
        
        self.btn_upload_test = tk.Button(left_panel, text="Upload Foto", font=("Helvetica", 10, "bold"), 
                                         fg="#FFFFFF", bg="#343A40", activebackground="#212529",
                                         relief=tk.FLAT, width=22, height=2, command=self.click_upload_foto_uji, cursor="hand2")
        self.btn_upload_test.pack(pady=(0, 15))
        
        # Progress Bar
        self.progress_bar = ttk.Progressbar(left_panel, orient="horizontal", length=180, mode="determinate", style="Horizontal.TProgressbar")
        self.progress_bar.pack(pady=(0, 20))
        
        # Garis Pembatas 
        separator = tk.Frame(left_panel, height=1, bg="#E9ECEF")
        separator.pack(fill=tk.X, pady=(0, 15))
        
        # Slider Threshold
        lbl_slider_title = tk.Label(left_panel, text="Ubah Threshold (%)", font=("Helvetica", 9, "bold"), fg="#495057", bg="#FFFFFF", anchor="w")
        lbl_slider_title.pack(fill=tk.X, pady=(0, 2))
        
        def on_slider_scroll(val):
            self.lbl_slider_val.config(text=f"{float(val):.1f}%")

        self.slider_threshold = tk.Scale(left_panel, from_=0.0, to=100.0, resolution=0.5, 
                                         orient=tk.HORIZONTAL, variable=self.threshold_var,
                                         showvalue=False, bg="#FFFFFF", fg="#4A90E2",
                                         troughcolor="#E9ECEF", activebackground="#4A90E2", 
                                         relief=tk.FLAT, highlightthickness=0, command=on_slider_scroll)
        self.slider_threshold.pack(fill=tk.X, pady=(0, 2))
        
        self.lbl_slider_val = tk.Label(left_panel, text="50.0%", font=("Helvetica", 10, "bold"), fg="#4A90E2", bg="#FFFFFF", anchor="w")
        self.lbl_slider_val.pack(fill=tk.X, pady=(0, 15))
        
        # Informasi Hasil Komputasi 
        self.lbl_execution_time = tk.Label(left_panel, text="Waktu Eksekusi\n-", font=("Helvetica", 9), fg="#6C757D", bg="#FFFFFF", justify=tk.LEFT, anchor="w")
        self.lbl_execution_time.pack(fill=tk.X, pady=(0, 12))
        
        self.lbl_score = tk.Label(left_panel, text="Similarity Score\n-", font=("Helvetica", 9), fg="#6C757D", bg="#FFFFFF", justify=tk.LEFT, anchor="w")
        self.lbl_score.pack(fill=tk.X, pady=(0, 15))
        
        self.lbl_name_result = tk.Label(left_panel, text="HASIL\n-", font=("Helvetica", 11, "bold"), fg="#4A90E2", bg="#FFFFFF", justify=tk.LEFT, anchor="w")
        self.lbl_name_result.pack(fill=tk.X, pady=(5, 0))
        
        
        # Right Panel
        right_panel = tk.Frame(body_frame, bg="#F8F9FA")
        right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(10, 0))
        
        # Box Foto Uji
        frame_uji = tk.LabelFrame(right_panel, text=" Gambar Test ", font=("Helvetica", 9, "bold"), fg="#495057", bg="#FFFFFF", relief=tk.GROOVE, padx=10, pady=10)
        frame_uji.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))
        self.canvas_uji = tk.Label(frame_uji, text="Tidak ada gambar yang diupload", font=("Helvetica", 9), fg="#ADB5BD", bg="#F8F9FA")
        self.canvas_uji.pack(expand=True, fill=tk.BOTH)
        
        # Box Hasil Cocok
        frame_mirip = tk.LabelFrame(right_panel, text="Hasil Paling Mirip", font=("Helvetica", 9, "bold"), fg="#495057", bg="#FFFFFF", relief=tk.GROOVE, padx=10, pady=10)
        frame_mirip.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(5, 0))
        self.canvas_mirip = tk.Label(frame_mirip, text="Tidak ada yang cocok", font=("Helvetica", 9), fg="#ADB5BD", bg="#F8F9FA")
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
            
            nama_cocok, kemiripan_skor, path_gambar_mirip, is_valid_face = recognizer.predict(face_vector)
            
            end_t = time.time()
            
            similarity_percentage = max(0.0, min(100.0, kemiripan_skor * 100))
            
            self.lbl_execution_time.config(text=f"Execution Time\n{end_t - start_t:.4f} seconds")
            self.lbl_score.config(text=f"Similarity Score\n{similarity_percentage:.2f}%")
            
            current_threshold = self.threshold_var.get()
            
            if is_valid_face and (similarity_percentage >= current_threshold):
                self.lbl_name_result.config(text=f"RESULT\n{nama_cocok}", fg="#28C76F")
                if path_gambar_mirip and os.path.exists(path_gambar_mirip):
                    self.render_image(path_gambar_mirip, self.canvas_mirip)
                else:
                    self.canvas_mirip.config(image='', text="Gambar Terklasifikasi", fg="#6C757D")
            else:
                self.lbl_name_result.config(text="RESULT\nTidak Cocok", fg="#EA5455")
                alasan_teks = "Wajah Tidak Dikenali" if is_valid_face else "Objek Ditolak\nBukan Struktur Wajah Valid"
                
                self.canvas_mirip.config(
                    image='', 
                    text=alasan_teks, 
                    font=("Helvetica", 9, "italic"), 
                    fg="#EA5455"
                )
                                                
    def render_image(self, img_path, label_widget):
        img = Image.open(img_path)
        orig_w, orig_h = img.size
        
        target_w, target_h = 210, 210
        
        ratio_w = target_w / orig_w
        ratio_h = target_h / orig_h
        scale = min(ratio_w, ratio_h)  
        
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        background = Image.new("RGB", (target_w, target_h), "#F8F9FA")
        
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        background.paste(img_resized, (offset_x, offset_y))
        
        img_tk = ImageTk.PhotoImage(background)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk