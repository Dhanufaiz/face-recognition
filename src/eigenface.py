import numpy as np
import torch

class EigenFace:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, update_callback=None):
        # 1. Ubah matriks input menjadi orientasi kolom vertikal (N_piksel x M_sampel)
        # Di kodemu X berbentuk (M x N), kita transpose agar sama dengan struktur temanmu
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).T
        
        # 2. Hitung wajah rata-rata (Mean Face) berupa vektor kolom (N_piksel x 1)
        mean_face_tensor = torch.mean(X_tensor, dim=1, keepdim=True)
        
        # 3. Sentralisasi Matriks A
        A = X_tensor - mean_face_tensor  # Ukuran: (N_piksel x M_sampel)

        if update_callback:
            update_callback(40, "Menghitung Matriks Kovarian Sederhana (L = A^T * A)...")
        
        # 4. Hitung matriks surplus L (M_sampel x M_sampel) sesuai berkas train.py temanmu
        L = torch.matmul(A.T, A)
        
        A_k = L.clone()
        n = A_k.shape[0]
        num_components = min(self.n_components, n)
        
        eigenvectors_L = torch.zeros((n, num_components), dtype=torch.float32, device=self.device)
        
        # 5. Algoritma Murni Power Iteration + Deflation dari Temanmu
        max_iter = 100
        tol = 1e-5
        
        for i in range(num_components):
            v = torch.rand(n, 1, dtype=torch.float32, device=self.device)
            v = v / torch.norm(v)
            
            for _ in range(max_iter):
                v_new = torch.matmul(A_k, v)
                v_norm = torch.norm(v_new)
                if v_norm == 0:
                    break
                v_new = v_new / v_norm
                
                # Cek konvergensi arah vektor eigen
                if torch.norm(v_new - v) < tol or torch.norm(v_new + v) < tol:
                    v = v_new
                    break
                v = v_new
            
            eigenvectors_L[:, i] = v.view(-1)
            
            # Deflation: Bersihkan nilai eigen terhitung dari matriks utama
            val = torch.matmul(v.T, torch.matmul(A_k, v))[0, 0]
            A_k -= val * torch.matmul(v, v.T)
            
            if update_callback:
                p_eigen = int(45 + ((i + 1) / num_components) * 45)
                update_callback(p_eigen, f"Mengekstrak ruang komponen eigen ({i + 1}/{num_components})...")

        # 6. Transformasi balik ke dimensi piksel asli: Eigenfaces = A * V_L (N_piksel x K_komponen)
        eigenfaces_tensor = torch.matmul(A, eigenvectors_L)
        
        # Normalisasi panjang vektor fitur agar bernilai 1 (Kunci Akurasi Jarak)
        norms = torch.norm(eigenfaces_tensor, dim=0, keepdim=True)
        norms[norms == 0] = 1.0
        eigenfaces_tensor = eigenfaces_tensor / norms
        
        # 7. Proyeksikan database latihan ke ruang koordinat bobot (K_komponen x M_sampel)
        # Rumus murni temanmu: projected = eigenfaces.T * A
        projections_tensor = torch.matmul(eigenfaces_tensor.T, A)
        
        # 8. Kembalikan ke array NumPy agar aman masuk ke CacheManager milikmu
        self.mean_face = mean_face_tensor.cpu().numpy().flatten() # Disimpan flat agar mudah dikurangi di recognizer
        self.eigenfaces = eigenfaces_tensor.cpu().numpy()
        self.projections = projections_tensor.cpu().numpy()