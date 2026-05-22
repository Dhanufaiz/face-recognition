import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import time
import threading
import os
from PIL import Image, ImageTk
import numpy as np

# Import malas (lazy import) dipindahkan ke dalam fungsi loading agar jendela langsung muncul

class FaceRecognitionGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("FACE RECOGNITION | KELOMPOK 8")
        self.window.geometry("820x540")
        self.window.configure(bg="#FFFFFF")  
        
        # Sembunyikan jendela utama pas awal-awal loading
        self.window.withdraw()
        
        self.model = None
        self.labels = None
        self.image_paths = None
        self.threshold_var = tk.DoubleVar(value=10.0) 
        
        # Munculkan Splash Screen duluan
        self.show_splash_screen()
        
    def show_splash_screen(self):
        # 1. Membuat jendela loading tanpa bingkai tepat di tengah layar
        self.splash = tk.Toplevel()
        self.splash.title("Loading...")
        self.splash.geometry("450x250")
        self.splash.configure(bg="#1064B4")
        self.splash.overrideredirect(True) # Menghilangkan tombol close/minimize bawaan windows
        
        screen_width = self.splash.winfo_screenwidth()
        screen_height = self.splash.winfo_screenheight()
        x = (screen_width // 2) - (450 // 2)
        y = (screen_height // 2) - (250 // 2)
        self.splash.geometry(f"450x250+{x}+{y}")
        
        # Komponen teks visual loading
        lbl_splash_title = tk.Label(self.splash, text="FACE RECOGNITION | KELOMPOK 8", font=("Helvetica", 14, "bold"), fg="white", bg="#1064B4")
        lbl_splash_title.pack(pady=(40, 10))
        
        lbl_uni = tk.Label(self.splash, text="Universitas Sebelas Maret", font=("Helvetica", 10), fg="#1064B4", bg="#D1E7DD")
        lbl_uni.pack(pady=(0, 20))
        
        self.lbl_splash_status = tk.Label(self.splash, text="Memulai sistem, mohon tunggu...", font=("Helvetica", 9, "italic"), fg="white", bg="#1064B4")
        self.lbl_splash_status.pack(pady=5)
        
        self.splash_progress = ttk.Progressbar(self.splash, orient=tk.HORIZONTAL, length=350, mode='determinate')
        self.splash_progress.pack(pady=10)
        
        # Jalankan background thread agar aplikasi tidak "Not Responding" saat loading import yang berat
        threading.Thread(target=self.initialization_background, daemon=True).start()

    def initialization_background(self):
        # Tahapan progress loading
        steps = [
            (20, "Memuat modul antarmuka komponen..."),
            (40, "Menginisialisasi pustaka PyTorch Tensor (import torch)..."),
            (70, "Membuka berkas arsip aljabar model (Cache)..."),
            (90, "Menyelaraskan visualisasi ruang matriks..."),
            (100, "Sistem siap dibuka!")
        ]
        
        for percent, msg in steps:
            # Karena ini hanya merubah teks label, kita aman pakai safe config atau update_idletasks
            try:
                self.lbl_splash_status.config(text=msg)
                self.splash_progress['value'] = percent
                self.splash.update_idletasks()
            except Exception:
                pass
            
            # Trik lazy-load: import modul berat dilakukan secara internal SAAT loading bar berjalan
            if percent == 40:
                global Trainer, Recognizer, EigenFace, CacheManager, preprocess_uploaded_image
                from src.trainer import Trainer
                from src.recognizer import Recognizer
                from src.eigenface import EigenFace
                from src.cache_manager import CacheManager
                from src.preprocessing import preprocess_uploaded_image
                
            elif percent == 70:
                self.load_model_on_start_silent()
                
            time.sleep(0.4) # Mengatur ritme jeda kedipan bar agar nyaman dilihat
            
        # SOLUSI: Lempar perintah pembuatan UI utama ke Main Loop Thread via .after() agar tidak memicu RuntimeError
        self.window.after(0, self.buka_aplikasi_utama)

    def buka_aplikasi_utama(self):
        # Tutup jendela splash murni dari Main Thread
        if hasattr(self, 'splash') and self.splash.winfo_exists():
            self.splash.destroy()
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Horizontal.TProgressbar", 
                             background="#FFFFFF", \
                             troughcolor="#E9ECEF", \
                             thickness=8)
        
        # Panggil susunan UI modifikasi milikmu
        self.setup_ui()
        
        # Munculkan kembali jendela utama ke layar laptop
        self.window.deiconify()

    def load_model_on_start_silent(self):
        # Membaca data cache lama di latar belakang tanpa mengganggu jalannya GUI utama
        mean_face = CacheManager.load('mean_face')
        eigenfaces = CacheManager.load('eigenfaces')
        projections = CacheManager.load('projections')
        labels = CacheManager.load('labels')
        image_paths = CacheManager.load('image_paths')
        
        if mean_face is not None and eigenfaces is not None and projections is not None:
            class DummyModel: pass
            self.model = DummyModel()
            self.model.mean_face = mean_face
            self.model.eigenfaces = eigenfaces
            self.model.projections = projections
            self.labels = labels
            self.image_paths = image_paths

    def setup_ui(self):
        # Header / Judul
        lbl_title = tk.Label(self.window, text="FACE RECOGNITION | KELOMPOK 8", font=("Helvetica", 16, "bold"), fg="#0877E6", bg="#FFFFFF")
        lbl_title.pack(pady=12)
        
        # Main Layout Frame
        main_frame = tk.Frame(self.window, bg="#FFFFFF")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # --- LEFT PANEL: CONTROL & INPUT ---
        left_panel = tk.Frame(main_frame, bg="#FFFFFF", width=260)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        left_panel.pack_propagate(False)
        
        btn_train = tk.Button(left_panel, text="📁 Upload Dataset / Train", command=self.start_training_thread,
                              font=("Helvetica", 10, "bold"), bg="#28A745", fg="white", activebackground="#218838", activeforeground="white", bd=0, pady=8)
        btn_train.pack(fill=tk.X, pady=8)
        
        btn_browse = tk.Button(left_panel, text="📸 Pilih Foto Uji", command=self.browse_test_image,
                               font=("Helvetica", 10, "bold"), bg="#0877E6", fg="#FFFFFF", activebackground="#E6E6E6", bd=0, pady=8)
        btn_browse.pack(fill=tk.X, pady=8)
        
        lbl_slider = tk.Label(left_panel, text="Ambang Batas Jarak (Threshold):", font=("Helvetica", 9, "bold"), fg="#0877E6", bg="#FFFFFF")
        lbl_slider.pack(anchor=tk.W, pady=(15, 2))
        
        # Slider Euclidean
        self.slider = tk.Scale(left_panel, from_=0.0, to=30.0, resolution=0.1, variable=self.threshold_var,
                               orient=tk.HORIZONTAL, bg="#FFFFFF", fg="#0877E6", highlightthickness=0, troughcolor="#4A90E2")
        self.slider.pack(fill=tk.X, pady=2)
        
        # Status Loading Bar
        self.lbl_status = tk.Label(left_panel, text="Sistem Siap", font=("Helvetica", 9, "italic"), fg="#0877E6", bg="#B2B4B6", wraplength=240, justify=tk.LEFT)
        self.lbl_status.pack(anchor=tk.W, pady=(20, 2))
        
        # Set teks status awal berdasarkan ketersediaan cache model
        if self.model is not None:
            self.lbl_status.config(text="Mode Terbaca dari Cache. Sistem Siap Saji.")
        else:
            self.lbl_status.config(text="Cache Kosong. Silakan Training Dataset Terlebih Dahulu.")
            
        self.progress = ttk.Progressbar(left_panel, orient=tk.HORIZONTAL, mode='determinate', style="Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=2)
        
        # --- CENTER PANEL: TEST IMAGE PREVIEW ---
        center_panel = tk.Frame(main_frame, bg="#FFFFFF")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        lbl_input_title = tk.Label(center_panel, text="Foto Input Uji", font=("Helvetica", 10, "bold"), fg="#1064B4", bg="#FFFFFF")
        lbl_input_title.pack(pady=(0, 4))
        
        self.canvas_uji = tk.Label(center_panel, text="Belum Ada Foto", bg="#A0A1A2", fg="#1064B4", width=25, height=12, relief=tk.RIDGE, bd=2)
        self.canvas_uji.pack(fill=tk.BOTH, expand=True)
        
        # --- RIGHT PANEL: MATCH RESULT PREVIEW ---
        right_panel = tk.Frame(main_frame, bg="#FFFFFF")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        lbl_output_title = tk.Label(right_panel, text="Hasil Pencocokan Database", font=("Helvetica", 10, "bold"), fg="#1064B4", bg="#FFFFFF")
        lbl_output_title.pack(pady=(0, 4))
        
        self.canvas_mirip = tk.Label(right_panel, text="Menunggu Pengujian", bg="#B2B4B6", fg="#1064B4", width=25, height=12, relief=tk.RIDGE, bd=2)
        self.canvas_mirip.pack(fill=tk.BOTH, expand=True)
        
        # Kotak Label Hasil Rekognisi
        self.lbl_name_result = tk.Label(self.window, text="RESULT\n-", font=("Helvetica", 12, "bold"), bg="#FFFFFF", fg="#333333", height=2, relief=tk.GROOVE, bd=1)
        self.lbl_name_result.pack(fill=tk.X, side=tk.BOTTOM, padx=30, pady=15)
        
    def load_model_on_start(self):
        mean_face = CacheManager.load('mean_face')
        eigenfaces = CacheManager.load('eigenfaces')
        projections = CacheManager.load('projections')
        labels = CacheManager.load('labels')
        image_paths = CacheManager.load('image_paths')
        
        if mean_face is not None and eigenfaces is not None and projections is not None:
            class DummyModel: pass
            self.model = DummyModel()
            self.model.mean_face = mean_face
            self.model.eigenfaces = eigenfaces
            self.model.projections = projections
            self.labels = labels
            self.image_paths = image_paths
            self.lbl_status.config(text="Mode Terbaca dari Cache. Sistem Siap Saji.")
        else:
            self.lbl_status.config(text="Cache Kosong. Silakan Training Dataset Terlebih Dahulu.")

    def start_training_thread(self):
        selected_dir = filedialog.askdirectory(title="Pilih Folder Dataset Wajah")
        if not selected_dir:
            self.lbl_status.config(text="Training dibatalkan (Folder tidak dipilih).")
            return
            
        if not os.listdir(selected_dir):
            messagebox.showerror("Error", "Folder yang kamu pilih kosong! Pastikan di dalamnya ada subfolder nama orang.")
            return
            
        self.progress['value'] = 0
        threading.Thread(target=self.run_training, args=(selected_dir,), daemon=True).start()
    
    def run_training(self, path):
        try:
            trainer = Trainer(path)
            trainer.train(update_callback=self.update_progress)
            self.load_model_on_start()
            messagebox.showinfo("Sukses", "Training Ruang EigenFace Berhasil Disimpan!")
        except Exception as e:
            messagebox.showerror("Error", f"Proses Gagal: {str(e)}")
            self.update_progress(0, "Sistem Eror.")

    def update_progress(self, percent, message):
        self.progress['value'] = percent
        self.lbl_status.config(text=message)
        self.window.update_idletasks()

    def browse_test_image(self):
        if self.model is None:
            messagebox.showwarning("Peringatan", "Lakukan training dataset terlebih dahulu!")
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.render_image(file_path, self.canvas_uji)
            vector = preprocess_uploaded_image(file_path)
            
            recognizer = Recognizer(self.model, self.labels, self.image_paths)
            name, distance, match_path, is_valid_face = recognizer.predict(vector)
            
            current_threshold = self.threshold_var.get()
            
            if is_valid_face and distance <= current_threshold and name != "Tidak Dikenali":
                self.lbl_name_result.config(text=f"RESULT\nTERDETEKSI: {name.upper()} (Jarak Euclidean: {distance:.2f})", fg="#28A745")
                if match_path and os.path.exists(match_path):
                    self.render_image(match_path, self.canvas_mirip)
                else:
                    self.canvas_mirip.config(image='', text="Gambar Terklasifikasi", fg="#6C757D")
            else:
                self.lbl_name_result.config(text=f"RESULT\nTIDAK COCOK (Jarak Terdekat: {distance:.2f})", fg="#EA5455")
                alasan_teks = "Wajah Tidak Dikenali (Di Luar Batas Threshold)" if is_valid_face else "Objek Ditolak\nBukan Struktur Wajah Valid"
                self.canvas_mirip.config(image='', text=alasan_teks, font=("Helvetica", 9, "italic"), fg="#EA5455")
                                                
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
        background = Image.new("RGB", (target_w, target_h), "#B2B4B6")
        
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        background.paste(img_resized, (offset_x, offset_y))
        
        img_tk = ImageTk.PhotoImage(background)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk 

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()