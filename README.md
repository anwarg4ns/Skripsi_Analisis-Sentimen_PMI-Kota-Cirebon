# Sistem Analisis Sentimen PMI Kota Cirebon

Aplikasi web untuk menganalisis sentimen ulasan masyarakat terhadap pelayanan PMI Kota Cirebon. Aplikasi ini dibangun dengan **Streamlit**, menggunakan **TF-IDF** untuk ekstraksi fitur dan **Multinomial Naive Bayes** untuk klasifikasi sentimen `Positif`, `Netral`, atau `Negatif`.

## Fitur aplikasi

- **Dashboard**: menampilkan jumlah total ulasan dan ringkasan jumlah sentimen.
- **Pengujian sentimen**: menerima satu ulasan baru, melakukan preprocessing, lalu menampilkan hasil prediksi.
- **Visualisasi sentimen**: menampilkan diagram pie dan bar untuk distribusi sentimen.
- **Analisis keluhan**: memfilter dan menampilkan seluruh ulasan dengan sentimen negatif.

## Cara kerja aplikasi

```text
CSV ulasan
    │
    ▼
Preprocessing teks ──► dataset_pmi_clean.csv ──► TF-IDF
    ▲                                              │
    │                                              ▼
Input ulasan pengguna                    SMOTE + Multinomial Naive Bayes
                                                   │
                              ┌────────────────────┴────────────────────┐
                              ▼                                         ▼
                    Model + vectorizer (.pkl)                 MySQL tabel_ulasan
                              └────────────────────┬────────────────────┘
                                                   ▼
                                      Dashboard Streamlit
```

Secara ringkas, data ulasan dibersihkan melalui *cleaning*, *case folding*, tokenisasi, penghapusan stopword, dan stemming dengan Sastrawi. Dataset bersih digunakan untuk melatih model. Aplikasi Streamlit membaca data ulasan dari MySQL dan menggunakan model tersimpan untuk memprediksi ulasan baru.

## Struktur proyek

```text
Skripsi_PMI/
├── app.py
├── preprocessing.py
├── model_training.py
├── setup_db.py
├── dataset/
├── models/
├── archive/
├── .gitignore
├── .venv/                 # lokal, tidak diunggah ke GitHub
├── __pycache__/            # cache Python, lokal
└── .git/                   # metadata Git, lokal
```

| Lokasi | Fungsi |
|---|---|
| `app.py` | Titik masuk aplikasi Streamlit. Menghubungkan MySQL, model, dan empat menu aplikasi. |
| `preprocessing.py` | Fungsi preprocessing teks dan pembuatan dataset bersih. Diimpor langsung oleh `app.py`, jadi harus tetap di folder utama. |
| `model_training.py` | Melatih dan mengevaluasi model TF-IDF + SMOTE + Multinomial Naive Bayes, lalu menyimpan artefak model. |
| `setup_db.py` | Membuat database/tabel MySQL dan mengimpor `dataset_pmi_clean.csv`. Hanya dipakai saat inisialisasi atau pemulihan database. |
| `dataset/dataset_pmi_master.csv` | Dataset utama sebelum preprocessing. |
| `dataset/dataset_pmi_clean.csv` | Dataset setelah preprocessing; dipakai untuk pelatihan dan impor database. |
| `dataset/triwulan_*.csv` | Arsip data ulasan per triwulan sebagai sumber penelitian. Tidak dibaca langsung saat aplikasi berjalan. |
| `models/nb_model.pkl` | Model Multinomial Naive Bayes yang dipakai untuk prediksi. |
| `models/tfidf_vectorizer.pkl` | Vectorizer TF-IDF yang dipakai bersama model saat prediksi. |
| `archive/cek_data.py` | Skrip pemeriksaan jumlah data historis; tidak diperlukan untuk menjalankan aplikasi. |
| `archive/gabung_data.py` | Skrip lama untuk menggabungkan CSV triwulan; tidak diperlukan untuk operasi normal. |
| `archive/inject_vocab.py` | Skrip eksperimen untuk mengubah sebagian ulasan sebagai *vocabulary anchor*. Jangan jalankan pada data produksi tanpa backup. |
| `.gitignore` | Daftar file/folder lokal yang tidak boleh dikirim ke GitHub. |
| `.venv/` | Virtual environment Python lokal. Buat ulang di setiap komputer, jangan diunggah. |
| `__pycache__/` | Cache otomatis Python; boleh dihapus dan akan dibuat kembali bila diperlukan. |

## Prasyarat

- Windows dengan Python 3.14 (versi yang digunakan proyek saat ini).
- MySQL yang aktif, misalnya melalui Laragon atau XAMPP.
- Git, jika ingin mengunggah proyek ke GitHub.

## Instalasi dari awal

Jalankan PowerShell di folder proyek.

### 1. Buat virtual environment

```powershell
py -3.14 -m venv .venv
```

### 2. Instal dependensi

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install streamlit==1.64.0 streamlit-option-menu==0.4.0 pandas==3.0.6 plotly==7.1.0 mysql-connector-python==26.7.0 Sastrawi==1.0.1 scikit-learn==1.9.1 imbalanced-learn==0.14.2 joblib==1.6.0
```

> Jika Python 3.14 belum terpasang, gunakan versi Python yang tersedia untuk membuat virtual environment. Pastikan seluruh dependensi berhasil dipasang sebelum menjalankan aplikasi.

## Konfigurasi MySQL

Secara default aplikasi memakai konfigurasi berikut:

| Pengaturan | Nilai default |
|---|---|
| Host | `localhost` |
| Port | `3306` |
| User | `root` |
| Password | kosong |
| Database | `db_pmi_cirebon` |
| Tabel | `tabel_ulasan` |

Apabila konfigurasi MySQL Anda berbeda, atur environment variable berikut di PowerShell sebelum menjalankan aplikasi:

```powershell
$env:PMI_DB_HOST = "localhost"
$env:PMI_DB_PORT = "3306"
$env:PMI_DB_USER = "root"
$env:PMI_DB_PASSWORD = "password-MySQL-Anda"
```

Jangan menyimpan password pada `app.py`, `setup_db.py`, atau GitHub.

## Menyiapkan database

Pastikan layanan MySQL sudah berjalan. Jika database `db_pmi_cirebon` dan tabel `tabel_ulasan` belum ada, jalankan:

```powershell
.\.venv\Scripts\python.exe setup_db.py
```

> **Peringatan:** `setup_db.py` mengosongkan (`TRUNCATE`) tabel `tabel_ulasan` sebelum mengimpor ulang data dari `dataset/dataset_pmi_clean.csv`. Jalankan hanya saat inisialisasi atau saat memang ingin membangun ulang isi tabel.

## Menjalankan aplikasi

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Setelah terminal menampilkan alamat lokal, buka biasanya `http://localhost:8501` di browser. Hentikan aplikasi dengan `Ctrl+C` di terminal.

## Pengujian singkat

Setelah aplikasi terbuka, lakukan pemeriksaan berikut:

1. Pada **Dashboard**, pastikan total ulasan tampil sebanyak 1.205 dan jumlah tiga kelas sentimen sesuai dengan totalnya.
2. Pada **Pengujian Sentimen**, coba masukkan `Pelayanan sangat cepat dan petugasnya ramah.`; hasilnya seharusnya `Positif`.
3. Coba `Antrean panjang, petugas lambat, dan ruang tunggu kotor.`; hasilnya seharusnya `Negatif`.
4. Pada **Visualisasi Sentimen**, pastikan diagram pie dan bar tampil.
5. Pada **Analisis Keluhan**, pastikan tabel hanya menampilkan ulasan berlabel `Negatif`.

Untuk memeriksa sintaks file Python tanpa menjalankan aplikasi:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py preprocessing.py model_training.py setup_db.py
```

## Melatih ulang model

Apabila dataset bersih berubah, latih ulang model dengan:

```powershell
.\.venv\Scripts\python.exe model_training.py
```

Skrip ini akan menimpa `models/nb_model.pkl` dan `models/tfidf_vectorizer.pkl` dengan artefak baru. Jalankan ulang aplikasi setelah pelatihan selesai.

## Mengunggah proyek ke GitHub

### 1. Buat repository di GitHub

Buat repository baru di GitHub. Disarankan memilih **Private** apabila dataset penelitian tidak boleh dibuka ke publik. Saat membuat repository, jangan tambahkan README, `.gitignore`, atau license dari halaman GitHub karena proyek lokal sudah memilikinya.

### 2. Jalankan perintah Git

Ganti `USERNAME` dan `NAMA-REPO` dengan akun serta nama repository Anda:

```powershell
git status
git add .
git diff --cached --name-only
git commit -m "Initial commit: aplikasi analisis sentimen PMI"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Sebelum menjalankan `git commit`, pastikan daftar dari `git diff --cached --name-only` **tidak** berisi `.venv/`, `__pycache__/`, file `.env`, atau kredensial database.

Jika Git meminta identitas, atur sekali saja:

```powershell
git config --global user.name "Nama Anda"
git config --global user.email "email-anda@example.com"
```

Untuk mengirim perubahan berikutnya:

```powershell
git add .
git commit -m "Jelaskan perubahan"
git push
```

## Catatan penting

- Database MySQL tidak ikut terunggah ke GitHub. Di komputer lain, buat database kembali dengan `setup_db.py` setelah data dan dependensi tersedia.
- Jangan mengunggah password, `.env`, atau `.streamlit/secrets.toml`.
- Simpan backup dataset dan model sebelum menjalankan skrip yang mengubah data atau melatih ulang model.
