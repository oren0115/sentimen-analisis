# Analisis Sentimen Opini Publik Terhadap Pilpres

Proyek analisis sentimen menggunakan Naive Bayes untuk menganalisis opini publik terhadap Pilpres.

## 📁 Struktur Proyek

```
tugas/
├── scripts/                 # Script-script executable
│   ├── main.py              # Entry point untuk menjalankan analisis
│   ├── dashboard.py         # Dashboard interaktif Streamlit
│   ├── generate_presentasi.py # Script untuk generate hasil analisis
│   ├── run_dashboard.bat     # Helper script Windows (batch)
│   └── run_dashboard.ps1     # Helper script Windows (PowerShell)
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── config.py            # Konfigurasi dan konstanta
│   ├── text_cleaning.py     # Fungsi-fungsi untuk pembersihan teks
│   ├── data_processing.py   # Fungsi-fungsi untuk pemrosesan data
│   ├── visualization.py     # Fungsi-fungsi untuk visualisasi
│   └── models.py            # Fungsi-fungsi untuk training model
├── data/                    # Data files
│   └── Sentiment1.csv       # Data utama (akan di-split menjadi train & test)
├── models/                  # Saved models (optional)
├── outputs/                 # Output files (plots, results, etc.)
│   └── HASIL_ANALISIS.txt   # Hasil analisis untuk presentasi
├── docs/                    # Dokumentasi
│   ├── DASHBOARD.md         # Panduan dashboard
│   ├── PRESENTASI_HASIL.md   # Template presentasi
│   ├── PRESENTASI.md         # Template presentasi (alternatif)
│   ├── PANDUAN_PRESENTASI.md # Panduan membuat presentasi
│   └── TROUBLESHOOTING.md    # Troubleshooting guide
├── requirements.txt         # Python dependencies
└── README.md                # Dokumentasi proyek (file ini)
```

## 📚 Deskripsi Modul

### `src/config.py`
Berisi semua konfigurasi dan konstanta yang digunakan di seluruh proyek:
- Konstanta model (SEED, MAX_TOKEN_LENGTH, dll)
- Mapping sentimen
- Path file data
- Konfigurasi style untuk plotting

### `src/text_cleaning.py`
Modul untuk pembersihan dan preprocessing teks:
- `strip_emoji()` - Menghapus emoji
- `strip_all_entities()` - Menghapus punctuation, links, mentions
- `clean_hashtags()` - Membersihkan hashtag
- `filter_chars()` - Filter karakter khusus
- `clean_text()` - Fungsi utama untuk membersihkan teks

### `src/data_processing.py`
Modul untuk pemrosesan data:
- `load_data()` - Memuat data dari file CSV tunggal
- `split_data()` - Membagi data menjadi training dan testing dengan stratified split
- `preprocess_data()` - Preprocessing data (cleaning, filtering)
- `process_token_lengths()` - Memproses dan filter berdasarkan panjang token
- `encode_sentiments()` - Encode label sentimen ke nilai numerik
- `prepare_data_for_training()` - Menyiapkan data untuk training

### `src/visualization.py`
Modul untuk visualisasi data:
- `plot_confusion_matrix()` - Plot confusion matrix
- `plot_tweets_by_date()` - Plot jumlah tweet per tanggal
- `plot_text_length_distribution()` - Plot distribusi panjang teks

### `src/models.py`
Modul untuk training dan evaluasi model:
- `train_naive_bayes()` - Training model Naive Bayes

### `scripts/main.py`
File utama yang mengintegrasikan semua modul dan menjalankan pipeline lengkap.

### `scripts/dashboard.py`
Dashboard interaktif menggunakan Streamlit untuk visualisasi dan eksplorasi data.

### `scripts/generate_presentasi.py`
Script untuk mengumpulkan hasil analisis dan membuat file hasil untuk presentasi.

## 🚀 Cara Menggunakan

### 1. Install Dependencies

**Instalasi Dasar (Direkomendasikan):**
```bash
pip install -r requirements.txt
```

**Catatan:** TensorFlow tidak diperlukan untuk menjalankan proyek ini. Jika Anda mengalami masalah instalasi, lihat file `docs/TROUBLESHOOTING.md` untuk panduan lengkap.

**Penting:** Jika menggunakan Windows dan mengalami error "Long Path support", lihat `docs/TROUBLESHOOTING.md` untuk solusinya.

### 2. Siapkan Data

Letakkan file data CSV di folder `data/`:
- `data/Sentiment1.csv` - Data utama (akan otomatis di-split menjadi training 80% dan testing 20%)

**Catatan:** Data akan otomatis dibagi menjadi training dan testing dengan stratified split untuk menjaga distribusi kelas sentimen yang seimbang.

### 3. Konfigurasi (Opsional)

Edit file `src/config.py` jika perlu mengubah konfigurasi:

```python
USE_COLAB = False  # Set True jika menggunakan Google Colab
SEED = 42
MAX_TOKEN_LENGTH = 80
MIN_TEXT_LENGTH = 4
```

### 4. Jalankan Program

**A. Menjalankan Analisis (Command Line):**
```bash
python scripts/main.py
```

**B. Menjalankan Dashboard Interaktif (Direkomendasikan untuk Presentasi):**

**Cara 1 (Direkomendasikan):**
```bash
python -m streamlit run scripts/dashboard.py
```

**Cara 2 (Windows - Double-click):**
Double-click file `scripts/run_dashboard.bat`

**Cara 3 (Windows - PowerShell):**
```powershell
.\scripts\run_dashboard.ps1
```

**Cara 4 (Jika streamlit ada di PATH):**
```bash
streamlit run scripts/dashboard.py
```

Dashboard akan terbuka di browser di `http://localhost:8501`

**Catatan:** Jika mendapat error "streamlit is not recognized", gunakan Cara 1, 2, atau 3.

**C. Generate Hasil untuk Presentasi:**
```bash
python scripts/generate_presentasi.py
```

File hasil akan disimpan di `outputs/HASIL_ANALISIS.txt`

**Penting:** 
- Untuk presentasi dan visualisasi, gunakan dashboard interaktif
- Lihat `docs/DASHBOARD.md` untuk panduan lengkap dashboard
- Lihat `docs/PANDUAN_PRESENTASI.md` untuk panduan membuat presentasi

## ⚙️ Konfigurasi

Semua konfigurasi dapat diubah di file `src/config.py`:

- `SEED`: Random seed untuk reproducibility (default: 42)
- `MAX_TOKEN_LENGTH`: Maksimum panjang token (default: 80)
- `MIN_TEXT_LENGTH`: Minimum panjang teks (default: 4)
- `TEST_SIZE`: Ukuran validation set (default: 0.1)
- `USE_COLAB`: Gunakan Google Colab atau tidak (default: True)

## 📖 Dokumentasi

- **Dashboard**: Lihat `docs/DASHBOARD.md` untuk panduan lengkap menggunakan dashboard
- **Presentasi**: Lihat `docs/PANDUAN_PRESENTASI.md` untuk panduan membuat presentasi
- **Troubleshooting**: Lihat `docs/TROUBLESHOOTING.md` untuk solusi masalah umum

## 📝 Catatan

- Pastikan file data CSV sudah tersedia di path yang dikonfigurasi
- Untuk menggunakan Google Colab, set `USE_COLAB = True` di `src/config.py`
- Model BERT memerlukan koneksi internet untuk download pretrained model pertama kali
- Semua output (plot, hasil analisis) akan disimpan di folder `outputs/`

## 🎯 Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Pastikan data ada di folder `data/`
3. Jalankan dashboard: `python -m streamlit run scripts/dashboard.py`
4. Atau jalankan analisis: `python scripts/main.py`
5. Generate hasil presentasi: `python scripts/generate_presentasi.py`

## 📧 Kontak

Jika ada pertanyaan atau masalah, silakan buka issue di repository atau hubungi maintainer.
