<p align="center">
  <img src="assets/logo_uns.png" alt="Logo UNS" width="120"/>
</p>

<h1 align="center"> Full Face Recognition — Eigenface Method</h1>

<p align="center">
  <strong>Kelompok 8 <br> informatika D <br> Universitas Sebelas Maret</strong>
</p>

---
## 👥 Anggota Kelompok
| NIM        | NAMA                           |
|:-----------|:-----------------              |
| L0125008   | Dhanu Fa'iz Sugara             |
| L0125052   | Muhammad Juan Fernando Aziz A. |
| L0125072   | Andra Satria Ardiansyah        |

**Dosen Pengampu:** Drs. Bambang Harjito, M.App.Sc., Ph.D.

---
## 🧑‍💻 Deskripsi Program
Sistem Pengenalan Wajah (*Face Recognition*) modern yang dibangun menggunakan implementasi murni konsep **Aljabar Linear**, khususnya **Principal Component Analysis (PCA)** dan **Eigenface**. Proyek ini dilengkapi dengan antarmuka grafis (GUI) berbasis `tkinter` yang interaktif, performa pencocokan berbasis *Cosine Similarity*, sistem *caching* data latih, serta fitur **Threshold Dinamis** untuk menguji akurasi klasifikasi secara *real-time*.

## 📌 Fitur Utama
1. **Modern GUI Dashboard**: Antarmuka grafis flat minimalis menggunakan kustomisasi tema `ttk` dengan skema warna profesional, indikator progress bar *Power Iteration*, dan visualisasi komputasi.
2. **Proporsional Image Matting**: Menampilkan citra uji (*Test Image*) dan citra hasil pencocokan (*Closest Result*) secara tajam menggunakan algoritma interpolasi `LANCZOS` tanpa mengalami distorsi/efek melar (*stretch*).
3. **Interactive Threshold Slider**: Komponen slider interaktif ($0.0\%$ - $100.0\%$) untuk menyesuaikan sensitivitas batas kelulusan pencocokan wajah secara langsung saat demo pengujian.
4. **Fast Startup Cache System**: Fitur otomatis memuat matriks transformasi wajah (`mean_face`, `eigenfaces`, `projections`, dll.) dari penyimpanan lokal `.npy` saat aplikasi dibuka, sehingga tidak perlu melakukan *training* ulang setiap kali aplikasi dijalankan.
5. **Background Thread Processing**: Proses ekstraksi basis data wajah dijalankan pada *worker thread* terpisah (`threading.Thread`) agar GUI tidak mengalami pembekuan (*freeze/not responding*).

---

## 📘 BAB II: LANDASAN TEORI

### 1. Perkalian Matriks
Perkalian matriks memegang peran sentral dalam setiap tahap metode *eigenface*, mulai dari pembentukan matriks kovarians hingga proses klasifikasi wajah. 

* **Matriks Kovarians:** Untuk mengetahui bagaimana fitur-fitur wajah saling berkaitan secara statistik, matriks data wajah ($A$) dikalikan dengan matriks transposnya ($A^T$). Jika terdapat $M$ gambar wajah berukuran $N \times N$, perkalian langsung $A \cdot A^T$ akan menghasilkan matriks berukuran $N^2 \times N^2$ yang secara komputasi sangat tidak efisien.
* **Proyeksi Ruang Eigenface:** Ketika sebuah gambar wajah baru yang direpresentasikan sebagai vektor $U$ ingin dikenali oleh sistem, gambar tersebut harus diproyeksikan ke dalam ruang *eigenface*. Proses ini dilakukan melalui perkalian matriks antara matriks *eigenface* ($E^T$) dengan vektor wajah baru ($U$):

  $$\Omega = E^T \cdot U$$

  Hasil perkalian ini adalah vektor kolom $\Omega$ yang memuat koefisien bobot $(w_1, w_2, w_3, \dots, w_k)$. Angka-angka inilah yang menjadi representasi unik atau *"tanda tangan digital"* dari wajah tersebut.
* **Klasifikasi Wajah:** Langkah terakhir adalah membandingkan vektor bobot $\Omega$ dengan vektor bobot wajah-wajah yang telah tersimpan di dalam database ($\Omega_k$). Perbandingan ini dilakukan menggunakan konsep norma dalam aljabar linear yang dikenal sebagai **Jarak Euclidean**:

  $$\text{Jarak} = \|\Omega - \Omega_k\|$$

  Wajah di dalam database yang menghasilkan nilai jarak terkecil akan dinyatakan sebagai identitas yang paling sesuai.

---

### 2. Nilai Eigen (*Eigenvalues*)
Dalam analisis komponen utama (*Principal Component Analysis* / PCA) untuk pengenalan wajah, nilai eigen diperoleh dari perhitungan matriks kovarians citra wajah. 

* **Representasi Varians:** Secara statistik, nilai eigen ($\lambda$) berbanding lurus dengan jumlah varians data yang ditangkap oleh vektor eigen pasangannya. 
* **Fitur Dominan vs Noise:** Nilai eigen yang besar menandakan bahwa vektor eigen tersebut mewakili fitur dominan yang membedakan antarwajah (seperti variasi pencahayaan, kontras, atau pola simetri). Sebaliknya, nilai eigen yang mendekati nol menandakan detail minor atau *noise* digital.
* **Urutan Prioritas:** Nilai-nilai eigen diurutkan dari yang terbesar hingga terkecil:

  $$\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \dots \ge \lambda_n$$

  Urutan ini sangat krusial karena menjadi dasar bagi sistem untuk memprioritaskan *eigenface* yang paling informatif.
* **Reduksi Dimensi:** Total varians dihitung dengan menjumlahkan semua nilai eigen. Untuk menghemat memori, sistem hanya menggunakan $k$ buah *eigenface* pertama hingga akumulasi nilai eigennya memenuhi target informasi tertentu (misalnya 95% dari total varians):

  $$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{n} \lambda_i} \ge \text{Target (misal 0.95)}$$

---

### 3. Vektor Eigen (*Eigenvectors*)
Dalam aljabar linear, vektor eigen dari sebuah matriks bujursangkar adalah vektor tak-nol yang arahnya tidak berubah ketika transformasi linear diterapkan padanya; vektor tersebut hanya mengalami penskalaan sebesar nilai eigen ($\lambda$).

* **Normalisasi Data:** Setiap gambar wajah dalam database pelatihan ($N \times N$ piksel) diratakan menjadi vektor kolom berdimensi tinggi ($N^2$). Karena struktur wajah manusia serupa, titik data ini membentuk subruang yang terstruktur. 
* **Proses Eliminasi Rata-Rata:** Sistem menghitung wajah rata-rata dari seluruh dataset, kemudian mengurangkannya dari setiap gambar wajah individual untuk memperoleh data yang ternormalisasi sebelum matriks kovarians dihitung.
* **Komponen Utama:** Vektor eigen yang diekstrak dari matriks kovarians mewakili arah variasi terbesar. Sistem hanya menyimpan sejumlah kecil vektor eigen dengan nilai eigen tertinggi—disebut komponen utama (*principal components*)—sebagai basis dari ruang wajah.

---

### 4. Eigenface
Metode *eigenface* merupakan algoritma pengenalan wajah berbasis PCA yang digunakan untuk mereduksi data multidimensi menjadi dimensi yang lebih kecil dengan tetap mempertahankan karakteristik utamanya.

* **Visualisasi Fisik:** Vektor eigen memiliki dimensi yang sama dengan gambar asli (misal: panjang 10.000 elemen untuk foto $100 \times 100$ piksel). Jika elemen ini disusun kembali menjadi matriks dan ditampilkan sebagai gambar, akan terlihat siluet wajah abstrak berwarna kelabu. Area terang mewakili variasi tinggi (seperti bayangan mata atau batas rahang), sedangkan area gelap mewakili area yang cenderung seragam.
* **Basis Ortogonal:** Kumpulan *eigenface* ($E_1, E_2, E_3, \dots, E_k$) bertindak sebagai basis ortogonal (seperti sumbu X, Y, Z pada ruang 3D) yang mendefinisikan karakteristik unik setiap wajah di dalam **Ruang Wajah**.
* **Kompresi Data:** Gambar berdimensi tinggi dapat direpresentasikan hanya dengan puluhan angka koefisien proyeksi saja (misal dari 10.000 piksel diringkas menjadi 20 atau 50 angka). Informasi yang dibuang umumnya berupa *noise* atau detail latar belakang yang tidak relevan.
* **Deteksi Non-Wajah (*Reconstruction Error*):** *Eigenface* dapat mendeteksi objek non-wajah melalui proses rekonstruksi. Jika gambar wajah diproyeksikan dan dibangun kembali, hasilnya akan sangat mirip dengan aslinya. Namun, jika objek lain (seperti kucing atau mobil) direkonstruksi menggunakan basis *eigenface*, hasilnya akan mengalami distorsi signifikan.

```text
[ Dataset Latih ] ──> [ Preprocessing ] ──> [ PCA / Training ] ──> [ Matriks Ruang Eigen (.npy) ]
                                                                             │
[ Foto Uji (GUI) ] ──> [ Preprocessing ] ──> [ Proyeksi Ruang Eigen ] <──────┘
                                                    │
                                         [ Perhitungan Kedekatan ]
                                                    │
                                          [ Evaluasi Threshold ] ──> [ RESULT: Cocok / Tidak ]
```
---
## 📊 Cara Kerja Program
---
## 📂 Struktur Direktori Proyek

```text
FACE_RECOGNITION/
│
├── .venv/                  # Virtual Environment Python
├── cache/                  # Penyimpanan biner hasil ekstraksi matriks wajah
│
├── dataset/                # Folder penyimpanan gambar latih (dikelompokkan per nama)
├── img/                    # Aset grafis aplikasi
│   └── logo_uns.png        # Logo Universitas Sebelas Maret
│
├── src/                    # Berkas kode sumber backend logika aljabar
│   ├── __init__.py         # Inisialisasi modul src
│   ├── cache_manager.py    # Manajemen penyimpanan dan pembacaan berkas .npy
│   ├── dataset_loader.py   # Skrip pembacaan file citra di dalam direktori
│   ├── distance.py         # Logika matematika komparasi kedekatan vektor
│   ├── eigenface.py        # Struktur data model objek matematika Eigenface
│   ├── preprocessing.py    # Standardisasi ukuran gambar (grayscale & flatten)
│   ├── recognizer.py       # Logika proyeksi citra uji & prediksi klasifikasi
│   └── trainer.py          # Implementasi kalkulasi PCA offline
│
├── gui.py                  # Kelas utama antarmuka grafik (Tkinter Dashboard)
├── main.py                 # File gerbang utama untuk menjalankan aplikasi
├── README.md               # Dokumentasi petunjuk proyek
└── requirements.txt        # Daftar pustaka Python
```
---

## 🖥️ Device Requirement
| Komponen        | Versi Minimum    | Keterangan                                    |
|:----------------|:-----------------|:----------------------------------------------|
| Python          | 3.8+             | Direkomendasikan 3.10 atau 3.11               |

## 🚀 CARA INSTALL

### 1. Clone Repository 
unduh source code dari repository git menggunakan   git clone  . buka terminal dan jalankan perintah berikut:
```txt
git clone https://github.com/dhanufaiz/face-recognition
```

### 2. Masuk ke dalam direktori root proyek
```txt
cd Face_Recognition_Alin
```
### 3. Membuat Virtual Environment (.venv)
Virtual Environment digunakan untuk mengisolasi semua pustaka matematika Python agar tidak mengganggu atau bentrok dengan proyek Python lain yang ada di komputer.
```txt
python -m venv .venv
```
### 4. Aktivasi Virtual Environment
* Windows (Command Prompt):
  ```txt
  .venv\Scripts\activate
  ```
* Windows (Powershell):
  ```txt
  .venv\Scripts\activate.ps1
  ```
* Linux/ MacOS:
  ```txt
  source .venv/bin/activate
  ```
Tanda jika `.venv` sudah aktof adalah anda akan melihat teks `(.venv)` muncul pada bagian paling kiri baris ketik di terminal.

### 5. install dependensi dari `requirements.txt`
```txt
pip install --upgrade pip
pip install -r requirements.txt
```
### 6. Struktur penataan dataset
Sebelum menjalankan aplikasi untuk pertama kali, pastikan Anda telah menata foto wajah anggota kelompok atau target di dalam direktori `dataset/`. Sistem akan membaca sub-folder sebagai label nama kelas secara otomatis:
```txt
Face_Recognition_Alin/
└── dataset/
    ├── Juan/
    │   ├── juan1.jpg
    │   ├── juan2.png
    │   └── juan3.jpeg
    ├── Dhanu_Faiz/
    │   ├── dhanu1.jpg
    │   └── dhanu2.png
    └── Andra/
        ├── andra1.jpg
        └── andra2.jpg
```

### 7. Run Aplikasi
Jika seluruh dependensi telah terinstal dan folder dataset telah diatur dengan benar, jalankan file gerbang utama `(main.py)` untuk membuka dashboard GUI:
```txt
python main.py
```
