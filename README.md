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

## 📐 Landasan Teori Aljabar Linear

Mengimplementasikan reduksi dimensi citra digital berukuran $100 \times 100$ piksel (vektor berdimensi $10.000$) menjadi ruang eigen berdimensi rendah menggunakan prinsip-prinsip matriks berikut:

1. **Vektor Rata-Rata (Mean Face / $\Psi$)**:
   $$\Psi = \frac{1}{M} \sum_{i=1}^{M} \Gamma_i$$
   Menghitung wajah rata-rata dari seluruh variasi dataset latih untuk proses sentralisasi data.
2. **Matriks Selisih Wajah Terpusat ($\Phi_i$)**:
   $$\Phi_i = \Gamma_i - \Psi$$
   Mengurangkan setiap komponen piksel citra asli dengan wajah rata-rata agar fokus pada komponen variasi unik.
3. **Matriks Kovarian dan Ruang Eigen (Eigenfaces / $u_i$)**:
   Menghitung matriks proyeksi melalui dekomposisi nilai eigen (*Eigenvalue Decomposition*) terhadap matriks kovarian data, menghasilkan arah variansi maksimum (Vektor Eigen) yang merepresentasikan fitur bayangan wajah (*Eigenface*).
4. **Metrik Kedekatan (Cosine Similarity)**:
   $$\text{Score} = \frac{A \cdot B}{\|A\| \|B\|}$$
   Mengukur sudut kosinus antara vektor proyeksi citra uji dengan vektor proyeksi database. Bernilai $1.0$ ($100\%$) jika arah vektor identik, dan mengecil jika struktur wajah berbeda.
---
## 🔄 Flowchart Program
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
