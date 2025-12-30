# 📊 PRESENTASI HASIL ANALISIS DATA
## Analisis Sentimen Opini Publik Terhadap Pilpres

---

## SLIDE 1: JUDUL

# Analisis Sentimen Opini Publik Terhadap Pilpres
### Menggunakan Machine Learning dengan Naive Bayes Classifier

**Disusun oleh:** [Nama Anda]  
**Tanggal:** [Tanggal Presentasi]  
**Mata Kuliah:** Artificial Intelligence

---

## SLIDE 2: LATAR BELAKANG

### Mengapa Analisis Sentimen Penting?

- **Media Sosial** menjadi sumber data yang kaya untuk memahami opini publik
- **Pilpres** adalah momen penting yang memicu banyak diskusi di media sosial
- **Analisis Sentimen** membantu memahami:
  - Tren opini publik secara real-time
  - Sentimen positif, negatif, dan netral
  - Pola dan tema yang muncul dalam diskusi

### Tantangan
- Volume data yang besar
- Bahasa informal dan slang
- Emoji dan karakter khusus
- Perlu preprocessing yang tepat

---

## SLIDE 3: TUJUAN PENELITIAN

### Tujuan Umum
Mengembangkan sistem analisis sentimen untuk mengklasifikasikan opini publik terhadap Pilpres menggunakan Machine Learning.

### Tujuan Khusus
1. **Preprocessing Data**: Membersihkan dan memproses data tweet dari noise
2. **Klasifikasi Sentimen**: Mengklasifikasikan sentimen menjadi 3 kategori:
   - **Negative** (Negatif)
   - **Neutral** (Netral)
   - **Positive** (Positif)
3. **Evaluasi Model**: Mengukur performa model menggunakan metrik akurasi, precision, recall, dan F1-score
4. **Visualisasi**: Membuat dashboard interaktif untuk presentasi hasil

---

## SLIDE 4: METODOLOGI

### Alur Penelitian

```
1. PENGUMPULAN DATA
   ↓
2. PREPROCESSING & CLEANING
   ↓
3. FEATURE EXTRACTION (TF-IDF)
   ↓
4. TRAINING MODEL (Naive Bayes)
   ↓
5. EVALUASI & VALIDASI
   ↓
6. VISUALISASI HASIL
```

### Teknologi yang Digunakan
- **Python 3.x**
- **Scikit-learn** (Naive Bayes, TF-IDF)
- **Streamlit** (Dashboard)
- **Plotly** (Visualisasi interaktif)

---

## SLIDE 5: PREPROCESSING DATA

### Langkah-langkah Preprocessing

1. **Pembersihan Teks**
   - Menghapus emoji
   - Menghapus URL dan mention (@username)
   - Menghapus punctuation yang tidak perlu
   - Normalisasi teks (lowercase)

2. **Filter Data**
   - Menghapus duplikat
   - Filter berdasarkan panjang teks minimum (4 kata)
   - Filter berdasarkan panjang token maksimum (80 token)

3. **Oversampling**
   - Menggunakan Random Over-Sampling untuk menyeimbangkan distribusi kelas

### Hasil Preprocessing
- Data lebih bersih dan siap untuk analisis
- Mengurangi noise yang dapat mengganggu klasifikasi
- Distribusi kelas menjadi seimbang

---

## SLIDE 6: STATISTIK DATASET

### Dataset Overview

| Metrik | Training Data | Test Data |
|--------|--------------|-----------|
| **Jumlah Data Awal** | [ISI DARI OUTPUT] | [ISI DARI OUTPUT] |
| **Setelah Preprocessing** | [ISI DARI OUTPUT] | [ISI DARI OUTPUT] |
| **Setelah Filtering** | [ISI DARI OUTPUT] | [ISI DARI OUTPUT] |
| **Setelah Oversampling** | [ISI DARI OUTPUT] | - |

### Distribusi Sentimen (Training - Setelah Oversampling)

- **Negative**: [X]% ([jumlah] data)
- **Neutral**: [Y]% ([jumlah] data)
- **Positive**: [Z]% ([jumlah] data)

### Karakteristik Data
- Rata-rata panjang teks: [N] kata
- Median panjang teks: [M] kata
- Rentang panjang teks: [Min] - [Max] kata

**Cara mendapatkan data:**
Jalankan `python main.py` dan lihat output di console, atau buka dashboard dan lihat di halaman Overview.

---

## SLIDE 7: DISTRIBUSI DATA

### Visualisasi Distribusi

**1. Distribusi Sentimen**
- Pie chart menunjukkan proporsi setiap kelas sentimen
- Setelah oversampling, distribusi menjadi seimbang
- **Sumber**: Dashboard → Analisis Data → Distribusi Sentimen

**2. Jumlah Tweet per Tanggal** (jika ada kolom Date)
- Bar chart menunjukkan aktivitas diskusi seiring waktu
- Mencerminkan momen-momen penting dalam periode Pilpres
- **Sumber**: Dashboard → Analisis Data → Jumlah Tweet per Tanggal

**3. Distribusi Panjang Teks**
- Histogram menunjukkan sebaran panjang tweet
- Kebanyakan tweet memiliki panjang [X-Y] kata
- **Sumber**: Dashboard → Analisis Data → Distribusi Panjang Teks

**Cara melihat:**
Jalankan dashboard dengan `python -m streamlit run dashboard.py` dan navigasi ke halaman "Analisis Data"

---

## SLIDE 8: PERFORMANSI MODEL

### Evaluasi Model Naive Bayes

| Metrik | Negative | Neutral | Positive | Average |
|--------|----------|---------|----------|---------|
| **Precision** | [X]% | [Y]% | [Z]% | [Avg]% |
| **Recall** | [X]% | [Y]% | [Z]% | [Avg]% |
| **F1-Score** | [X]% | [Y]% | [Z]% | [Avg]% |
| **Support** | [N] | [M] | [O] | [Total] |

### Akurasi Keseluruhan
**Akurasi Model: [X]%**

### Interpretasi
- Model menunjukkan performa yang [baik/cukup baik]
- [Kelas tertentu] memiliki performa terbaik karena [alasan]
- [Kelas tertentu] lebih sulit diprediksi karena [alasan]

**Cara mendapatkan data:**
Jalankan `python main.py` dan lihat output "Classification Report for Naive Bayes" di console.

---

## SLIDE 9: CONFUSION MATRIX

### Confusion Matrix

```
                Predicted
            Neg    Neu    Pos
Actual Neg  [A]    [B]    [C]
       Neu  [D]    [E]    [F]
       Pos  [G]    [H]    [I]
```

### Analisis
- **Diagonal (A, E, I)**: Prediksi benar
- **Off-diagonal**: Prediksi salah
- **Insight**: 
  - Model paling baik memprediksi [kelas]
  - Kesalahan paling sering terjadi antara [kelas X] dan [kelas Y]

**Cara mendapatkan:**
Jalankan `python main.py` dan lihat plot confusion matrix yang muncul, atau screenshot dari output.

---

## SLIDE 10: CONTOH HASIL ANALISIS

### Contoh Teks per Sentimen

**Contoh Negative:**
- **Teks**: "[Contoh dari dashboard]"
- **Prediksi**: Negative ✅

**Contoh Neutral:**
- **Teks**: "[Contoh dari dashboard]"
- **Prediksi**: Neutral ✅

**Contoh Positive:**
- **Teks**: "[Contoh dari dashboard]"
- **Prediksi**: Positive ✅

**Cara mendapatkan:**
Buka dashboard → Analisis Sentimen → Contoh Teks per Sentimen, pilih sentimen dan ambil screenshot.

---

## SLIDE 11: WORD CLOUD

### Analisis Kata-kata Populer

**Word Cloud per Sentimen:**

1. **Negative Sentiment**
   - Kata-kata dominan: [lihat dari word cloud]
   - Tema: [tema yang muncul]

2. **Neutral Sentiment**
   - Kata-kata dominan: [lihat dari word cloud]
   - Tema: [tema yang muncul]

3. **Positive Sentiment**
   - Kata-kata dominan: [lihat dari word cloud]
   - Tema: [tema yang muncul]

### Insight
- Kata-kata tertentu lebih sering muncul dalam sentimen tertentu
- Dapat digunakan untuk memahami tema diskusi

**Cara mendapatkan:**
Buka dashboard → Analisis Sentimen → Word Cloud, pilih sentimen dan ambil screenshot.

---

## SLIDE 12: DASHBOARD INTERAKTIF

### Fitur Dashboard

**4 Halaman Utama:**

1. **Overview**
   - Statistik dataset
   - Preview data
   - Informasi umum

2. **Analisis Data**
   - Distribusi sentimen
   - Visualisasi temporal
   - Statistik panjang teks

3. **Analisis Sentimen**
   - Perbandingan sentimen
   - Contoh teks per kategori
   - Word cloud interaktif

4. **Visualisasi Detail**
   - Filter interaktif
   - Box plot dan violin plot
   - Tabel data detail

### Keuntungan Dashboard
- ✅ Interaktif dan mudah digunakan
- ✅ Cocok untuk presentasi
- ✅ Real-time filtering
- ✅ Visualisasi yang menarik

**Cara menjalankan:**
```bash
python -m streamlit run dashboard.py
```

---

## SLIDE 13: KESIMPULAN

### Kesimpulan Penelitian

1. **Preprocessing Berhasil**
   - Teknik preprocessing yang digunakan efektif membersihkan data
   - Data siap untuk analisis machine learning

2. **Model Naive Bayes Efektif**
   - Mencapai akurasi [X]% (isi dari hasil)
   - Cocok untuk klasifikasi sentimen teks
   - Cepat dalam training dan prediction

3. **Dashboard Interaktif**
   - Memudahkan visualisasi dan presentasi
   - User-friendly dan informatif

4. **Insight Opini Publik**
   - Berhasil mengidentifikasi pola sentimen
   - Dapat digunakan untuk analisis lebih lanjut

---

## SLIDE 14: KETERBATASAN & SARAN

### Keterbatasan Penelitian

1. **Dataset**
   - Terbatas pada periode tertentu
   - Mungkin tidak representatif untuk semua konteks

2. **Model**
   - Naive Bayes memiliki asumsi independensi yang mungkin tidak selalu benar
   - Tidak menggunakan deep learning yang lebih kompleks

3. **Preprocessing**
   - Beberapa konteks mungkin hilang dalam proses cleaning
   - Slang dan bahasa informal mungkin tidak tertangani sempurna

### Saran Pengembangan

1. **Model yang Lebih Canggih**
   - Mencoba model deep learning (LSTM, BERT)
   - Ensemble methods untuk meningkatkan akurasi

2. **Dataset yang Lebih Besar**
   - Mengumpulkan lebih banyak data
   - Menambahkan data dari berbagai sumber

3. **Feature Engineering**
   - Menambahkan fitur n-gram
   - Menggunakan word embeddings (Word2Vec, GloVe)

---

## SLIDE 15: TERIMA KASIH

### Terima Kasih

**Pertanyaan?**

**Kontak:**
- Email: [email@example.com]
- GitHub: [github.com/username]

---

## 📝 CATATAN UNTUK PRESENTASI

### Tips Presentasi

1. **Slide 1-3**: Pengenalan (5 menit)
   - Fokus pada latar belakang dan tujuan

2. **Slide 4-5**: Metodologi (5 menit)
   - Jelaskan alur penelitian secara detail
   - Highlight teknik preprocessing

3. **Slide 6-11**: Hasil (10 menit)
   - Tampilkan visualisasi yang menarik
   - Jelaskan insight dari setiap hasil
   - **Demo Dashboard** (5 menit)

4. **Slide 12-14**: Kesimpulan (5 menit)
   - Ringkas temuan utama
   - Diskusikan keterbatasan dan saran

5. **Slide 15**: Penutup (5 menit)
   - Q&A session

### Total Waktu: ~30-35 menit

### Key Points untuk Presentasi

✅ **Highlight:**
- Preprocessing yang komprehensif
- Model Naive Bayes yang efektif
- Dashboard interaktif yang user-friendly
- Insight yang dapat ditindaklanjuti

✅ **Visualisasi:**
- Gunakan chart yang jelas dan mudah dibaca
- Highlight angka-angka penting
- Gunakan warna yang konsisten
- Ambil screenshot dari dashboard untuk slide

✅ **Storytelling:**
- Mulai dengan masalah
- Jelaskan solusi
- Tunjukkan hasil
- Diskusikan dampak

---

## 🔧 CARA MENGISI DATA HASIL

### Langkah-langkah:

1. **Jalankan Analisis:**
   ```bash
   python main.py
   ```
   - Catat jumlah data training dan test
   - Catat distribusi sentimen
   - Catat hasil classification report
   - Screenshot confusion matrix

2. **Jalankan Dashboard:**
   ```bash
   python -m streamlit run dashboard.py
   ```
   - Ambil screenshot dari setiap halaman
   - Catat statistik yang ditampilkan
   - Ambil screenshot word cloud untuk setiap sentimen
   - Ambil contoh teks untuk setiap sentimen

3. **Isi Template:**
   - Ganti semua `[ISI DARI OUTPUT]` dengan data aktual
   - Ganti `[X]`, `[Y]`, `[Z]` dengan angka hasil
   - Ganti `[Nama Anda]` dengan nama Anda
   - Ganti `[Tanggal Presentasi]` dengan tanggal presentasi

4. **Siapkan Visualisasi:**
   - Screenshot dari dashboard
   - Screenshot confusion matrix
   - Screenshot word cloud
   - Screenshot contoh teks

---

**SELAMAT PRESENTASI! 🎉**

