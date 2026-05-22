import numpy as np
import torch

class Recognizer:
    def __init__(self, model, labels, image_paths):
        self.model = model
        self.labels = labels
        self.image_paths = image_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, face_vector):
        if face_vector is None:
            return "Ditolak", float('inf'), None, False

        # Load parameter cache model numpy ke bentuk Tensor kolom PyTorch
        mean_face = torch.tensor(self.model.mean_face, dtype=torch.float32, device=self.device).view(-1, 1)
        eigenfaces = torch.tensor(self.model.eigenfaces, dtype=torch.float32, device=self.device)
        projections = torch.tensor(self.model.projections, dtype=torch.float32, device=self.device)
        
        # Ubah gambar uji menjadi bentuk vektor kolom vertikal (N_piksel x 1)
        test_tensor = torch.tensor(face_vector, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # 1. Sentralisasi dan Proyeksi Wajah Uji (Rumus murni main.py temanmu)
        Phi_test = test_tensor - mean_face
        projected_test_face = torch.matmul(eigenfaces.T, Phi_test) # Hasil: (K_komponen x 1)
        
        # 2. Validasi Struktur Objek Wajah via Galat Rekonstruksi Citra
        gambar_rekonstruksi = torch.matmul(eigenfaces, projected_test_face) + mean_face
        galat_rekonstruksi = torch.norm(test_tensor - gambar_rekonstruksi).item()
        
        # Ambang batas penyaringan objek non-wajah untuk resolusi citra 50x50
        is_valid_face = galat_rekonstruksi < 18.0 
        
        min_dist = float('inf')
        min_idx = -1
        
        # 3. Cari Jarak Euclidean Paling Minimum (Paling Mirip)
        num_images = projections.shape[1] # Indeks 1 menyatakan jumlah sampel database vertikal
        for i in range(num_images):
            train_vector = projections[:, i].view(-1, 1)
            
            # Rumus manual_euclidean_distance dari berkas temanmu
            diff = projected_test_face - train_vector
            dist = torch.sqrt(torch.sum(diff ** 2)).item()
            
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        if min_idx == -1:
            return "Tidak Dikenali", min_dist, None, is_valid_face
            
        nama_cocok = self.labels[min_idx]
        path_gambar_mirip = self.image_paths[min_idx]
        
        return nama_cocok, min_dist, path_gambar_mirip, is_valid_face