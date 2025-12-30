# 📊 PRESENTASI: ANALISIS SENTIMEN OPINI PUBLIK TERHADAP PILPRES

**Menggunakan Machine Learning dengan Naive Bayes Classifier**

---

## SLIDE 1: HALAMAN JUDUL

### Analisis Sentimen Opini Publik Terhadap Pilpres
**Menggunakan Machine Learning dengan Naive Bayes Classifier**

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

## SLIDE 4: METODOLOGI - ALUR PENELITIAN

### Tahapan Penelitian

```
1. PENGUMPULAN DATA
   ↓
2. PREPROCESSING & CLEANING
   ↓
3. FEATURE EXTRACTION
   ↓
4. TRAINING MODEL (Naive Bayes)
   ↓
5. EVALUASI & VALIDASI
   ↓
6. VISUALISASI HASIL
```

---

## SLIDE 5: METODOLOGI - PREPROCESSING DATA

### Langkah-langkah Preprocessing

1. **Pembersihan Teks**
   - Menghapus emoji
   - Menghapus URL dan mention (@username)
   - Menghapus punctuation yang tidak perlu
   - Normalisasi teks (lowercase)

2. **Pembersihan Hashtag**
   - Menghapus hashtag di akhir kalimat
   - Mempertahankan kata dalam hashtag (tanpa simbol #)

3. **Filter Karakter Khusus**
   - Menghapus karakter non-ASCII
   - Menghapus karakter khusus ($, &)

4. **Normalisasi Spasi**
   - Menghapus multiple spaces
   - Menjaga konsistensi format

### Hasil Preprocessing
- Data lebih bersih dan siap untuk analisis
- Mengurangi noise yang dapat mengganggu klasifikasi

---

## SLIDE 6: METODOLOGI - FEATURE EXTRACTION

### Teknik Feature Extraction

1. **Count Vectorization**
   - Mengubah teks menjadi vektor berdasarkan frekuensi kata
   - Membuat vocabulary dari seluruh dataset

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Memberikan bobot pada kata berdasarkan:
     - **TF**: Seberapa sering kata muncul dalam dokumen
     - **IDF**: Seberapa jarang kata muncul di seluruh dokumen
   - Mengurangi bobot kata yang terlalu umum

### Keuntungan TF-IDF
- Kata yang jarang muncul mendapat bobot lebih tinggi
- Kata umum (stopwords) mendapat bobot lebih rendah
- Meningkatkan kualitas fitur untuk klasifikasi

---

## SLIDE 7: METODOLOGI - MODEL MACHINE LEARNING

### Naive Bayes Classifier

**Alasan Pemilihan:**
- ✅ Efektif untuk klasifikasi teks
- ✅ Cepat dalam training dan prediction
- ✅ Tidak memerlukan banyak data
- ✅ Mudah diinterpretasikan

### Cara Kerja Naive Bayes
1. Menghitung probabilitas setiap kata untuk setiap kelas sentimen
2. Menggunakan asumsi "naive" bahwa kata-kata independen
3. Mengalikan probabilitas untuk mendapatkan prediksi kelas

### Rumus Dasar
```
P(Sentimen | Kata) = P(Kata | Sentimen) × P(Sentimen) / P(Kata)
```

---

## SLIDE 8: METODOLOGI - HANDLING IMBALANCED DATA

### Masalah Data Tidak Seimbang

Dataset sering memiliki distribusi sentimen yang tidak seimbang:
- Satu kelas mungkin lebih banyak dari yang lain
- Dapat menyebabkan bias pada model

### Solusi: Random Over-Sampling

- **Teknik**: Menambah sampel kelas minoritas secara acak
- **Tujuan**: Menyeimbangkan distribusi kelas
- **Hasil**: Model lebih baik dalam memprediksi semua kelas

### Sebelum vs Sesudah Oversampling
- **Sebelum**: Distribusi tidak seimbang
- **Sesudah**: Setiap kelas memiliki jumlah yang seimbang

---

## SLIDE 9: HASIL - STATISTIK DATASET

### Dataset Overview

| Metrik | Training Data | Test Data |
|--------|--------------|-----------|
| **Jumlah Data Awal** | [X] | [Y] |
| **Setelah Preprocessing** | [X'] | [Y'] |
| **Setelah Filtering** | [X''] | [Y''] |

### Distribusi Sentimen (Training)

- **Negative**: [X]% ([jumlah] data)
- **Neutral**: [Y]% ([jumlah] data)
- **Positive**: [Z]% ([jumlah] data)

### Karakteristik Data
- Rata-rata panjang teks: [N] kata
- Median panjang teks: [M] kata
- Rentang panjang teks: [Min] - [Max] kata

---

## SLIDE 10: HASIL - DISTRIBUSI DATA

### Visualisasi Distribusi

**1. Distribusi Sentimen**
- Pie chart menunjukkan proporsi setiap kelas sentimen
- Setelah oversampling, distribusi menjadi seimbang

**2. Jumlah Tweet per Tanggal**
- Bar chart menunjukkan aktivitas diskusi seiring waktu
- Mencerminkan momen-momen penting dalam periode Pilpres

**3. Distribusi Panjang Teks**
- Histogram menunjukkan sebaran panjang tweet
- Kebanyakan tweet memiliki panjang [X-Y] kata

---

## SLIDE 11: HASIL - PERFORMANSI MODEL

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

---

## SLIDE 12: HASIL - CONFUSION MATRIX

### Confusion Matrix

```
                Predicted
            Neg  Neu  Pos
Actual Neg  [A]  [B]  [C]
       Neu  [D]  [E]  [F]
       Pos  [G]  [H]  [I]
```

### Analisis
- **Diagonal (A, E, I)**: Prediksi benar
- **Off-diagonal**: Prediksi salah
- **Insight**: 
  - Model paling baik memprediksi [kelas]
  - Kesalahan paling sering terjadi antara [kelas X] dan [kelas Y]

---

## SLIDE 13: HASIL - CONTOH PREDIKSI

### Contoh Prediksi yang Benar

**Contoh 1: Negative**
- **Teks**: "[Contoh tweet negatif]"
- **Prediksi**: Negative ✅
- **Probabilitas**: [X]%

**Contoh 2: Positive**
- **Teks**: "[Contoh tweet positif]"
- **Prediksi**: Positive ✅
- **Probabilitas**: [X]%

### Contoh Prediksi yang Salah

**Contoh 3: Misclassified**
- **Teks**: "[Contoh tweet]"
- **Prediksi**: [Kelas salah]
- **Sebenarnya**: [Kelas benar]
- **Alasan**: [Analisis mengapa salah]

---

## SLIDE 14: HASIL - WORD CLOUD

### Analisis Kata-kata Populer

**Word Cloud per Sentimen:**

1. **Negative Sentiment**
   - Kata-kata dominan: [kata1, kata2, kata3]
   - Tema: [tema yang muncul]

2. **Neutral Sentiment**
   - Kata-kata dominan: [kata1, kata2, kata3]
   - Tema: [tema yang muncul]

3. **Positive Sentiment**
   - Kata-kata dominan: [kata1, kata2, kata3]
   - Tema: [tema yang muncul]

### Insight
- Kata-kata tertentu lebih sering muncul dalam sentimen tertentu
- Dapat digunakan untuk memahami tema diskusi

---

## SLIDE 15: DASHBOARD INTERAKTIF

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

---

## SLIDE 16: KESIMPULAN

### Kesimpulan Penelitian

1. **Preprocessing Berhasil**
   - Teknik preprocessing yang digunakan efektif membersihkan data
   - Data siap untuk analisis machine learning

2. **Model Naive Bayes Efektif**
   - Mencapai akurasi [X]%
   - Cocok untuk klasifikasi sentimen teks
   - Cepat dalam training dan prediction

3. **Dashboard Interaktif**
   - Memudahkan visualisasi dan presentasi
   - User-friendly dan informatif

4. **Insight Opini Publik**
   - Berhasil mengidentifikasi pola sentimen
   - Dapat digunakan untuk analisis lebih lanjut

---

## SLIDE 17: KETERBATASAN

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

4. **Bahasa**
   - Fokus pada bahasa Indonesia
   - Campuran bahasa (code-switching) mungkin kurang optimal

---

## SLIDE 18: SARAN PENGEMBANGAN

### Saran untuk Pengembangan Selanjutnya

1. **Model yang Lebih Canggih**
   - Mencoba model deep learning (LSTM, BERT)
   - Ensemble methods untuk meningkatkan akurasi

2. **Dataset yang Lebih Besar**
   - Mengumpulkan lebih banyak data
   - Menambahkan data dari berbagai sumber

3. **Feature Engineering**
   - Menambahkan fitur n-gram
   - Menggunakan word embeddings (Word2Vec, GloVe)

4. **Analisis Lanjutan**
   - Analisis temporal (trend over time)
   - Analisis topik (topic modeling)
   - Analisis emosi yang lebih detail

5. **Deployment**
   - Membuat API untuk prediksi real-time
   - Integrasi dengan platform media sosial

---

## SLIDE 19: IMPLEMENTASI TEKNIS

### Struktur Proyek

```
tugas/
├── main.py              # Entry point analisis
├── dashboard.py         # Dashboard interaktif
├── src/                 # Source code modules
│   ├── config.py        # Konfigurasi
│   ├── text_cleaning.py # Preprocessing
│   ├── data_processing.py
│   ├── visualization.py
│   └── models.py
├── data/                # Dataset
└── outputs/             # Hasil analisis
```

### Teknologi yang Digunakan
- **Python 3.x**
- **Scikit-learn** (Naive Bayes, TF-IDF)
- **Streamlit** (Dashboard)
- **Plotly** (Visualisasi interaktif)
- **Pandas, NumPy** (Data processing)
- **Transformers** (BERT tokenizer untuk analisis)

---

## SLIDE 20: DEMONSTRASI

### Demo Dashboard

**Cara Menjalankan:**
```bash
python -m streamlit run dashboard.py
```

**Fitur yang Ditampilkan:**
1. Overview dataset
2. Distribusi sentimen interaktif
3. Analisis temporal
4. Word cloud per sentimen
5. Filter dan eksplorasi data

**Akses Dashboard:**
- URL: `http://localhost:8501`
- Interaktif dan real-time
- Cocok untuk presentasi

---

## SLIDE 21: REFERENSI

### Referensi

1. **Text Classification**
   - Jurafsky, D., & Martin, J. H. (2020). Speech and Language Processing

2. **Naive Bayes**
   - Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval

3. **Sentiment Analysis**
   - Liu, B. (2012). Sentiment Analysis and Opinion Mining

4. **Preprocessing Text**
   - Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python

5. **Streamlit Documentation**
   - https://docs.streamlit.io/

---

## SLIDE 22: TERIMA KASIH

### Terima Kasih

**Pertanyaan?**

**Kontak:**
- Email: [email@example.com]
- GitHub: [github.com/username]

**Sumber Kode:**
- Repository: [link repository jika ada]

---

## CATATAN UNTUK PRESENTASI

### Tips Presentasi

1. **Slide 1-3**: Pengenalan (5 menit)
   - Fokus pada latar belakang dan tujuan

2. **Slide 4-8**: Metodologi (10 menit)
   - Jelaskan alur penelitian secara detail
   - Highlight teknik preprocessing dan feature extraction

3. **Slide 9-14**: Hasil (10 menit)
   - Tampilkan visualisasi yang menarik
   - Jelaskan insight dari setiap hasil

4. **Slide 15**: Demo Dashboard (5 menit)
   - Live demonstration dashboard
   - Interaktif dengan audience

5. **Slide 16-18**: Kesimpulan (5 menit)
   - Ringkas temuan utama
   - Diskusikan keterbatasan dan saran

6. **Slide 19-22**: Penutup (5 menit)
   - Q&A session

### Total Waktu: ~40 menit

---

## LAMPIRAN: POIN-POIN PENTING

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

✅ **Storytelling:**
- Mulai dengan masalah
- Jelaskan solusi
- Tunjukkan hasil
- Diskusikan dampak

---

**SELAMAT PRESENTASI! 🎉**

