import numpy as np

class Recognizer:
    def __init__(self, model, labels, image_paths):
        self.model = model
        self.labels = labels
        self.image_paths = image_paths

    def predict(self, face_vector):
        wajah_terpusat = face_vector - self.model.mean_face
        proyeksi_uji = np.dot(wajah_terpusat, self.model.eigenfaces.T)
        
        gambar_rekonstruksi = np.dot(proyeksi_uji, self.model.eigenfaces) + self.model.mean_face
        
        galat_rekonstruksi = np.linalg.norm(face_vector - gambar_rekonstruksi)
        
        # threshold absolut objek wajah
        is_valid_face = galat_rekonstruksi < 3500.0 
        
        # 4. Cosine Similarity 
        skor_tertinggi = -1
        indeks_tercocok = -1
        
        for i, proyeksi_data_latih in enumerate(self.model.projections):
            dot_product = np.dot(proyeksi_uji, proyeksi_data_latih)
            norm_uji = np.linalg.norm(proyeksi_uji)
            norm_latih = np.linalg.norm(proyeksi_data_latih)
            
            if norm_uji == 0 or norm_latih == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_uji * norm_latih)
                
            if similarity > skor_tertinggi:
                skor_tertinggi = similarity
                indeks_tercocok = i
                
        nama_cocok = self.labels[indeks_tercocok]
        path_gambar = self.image_paths[indeks_tercocok]
        
        # Kembalikan status validasi wajah ke GUI
        return nama_cocok, skor_tertinggi, path_gambar, is_valid_face