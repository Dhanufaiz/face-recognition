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
        
        # memperkecil ukuran (MxM)
        L = np.dot(A, A.T)
        
        A_eff = L.copy().astype(float)
        n = A_eff.shape[0]
        k = min(self.n_components, n)
        
        eigen_vectors_L = []
        
        for step in range(k):
            val, vec = self.power_iteration(A_eff)
            eigen_vectors_L.append(vec)
            
            # mengurangi komponen matematika yang telah didapat
            A_eff -= val * np.outer(vec, vec)
            
            if update_callback:
                p_eigen = int(45 + ((step + 1) / k) * 45)
                update_callback(p_eigen, f"Mengekstrak fitur matriks eigen ({step + 1}/{k})...")

        eigen_vectors_L = np.array(eigen_vectors_L).T
        
        # mengembalikan bentuk asli U = A^T * V
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