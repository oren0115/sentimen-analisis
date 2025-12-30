# 📋 Panduan Membuat Presentasi Hasil Analisis

## 🎯 Langkah-langkah

### 1. Jalankan Analisis Data
```bash
python main.py
```
Ini akan menjalankan analisis lengkap dan menampilkan hasil di console.

### 2. Generate Hasil untuk Presentasi
```bash
python generate_presentasi.py
```
Script ini akan:
- Mengumpulkan semua hasil analisis
- Menghitung statistik dataset
- Menghitung performa model
- Menyimpan hasil ke file `HASIL_ANALISIS.txt`

### 3. Buka File Hasil
Buka file `HASIL_ANALISIS.txt` yang baru dibuat. File ini berisi:
- Statistik dataset
- Distribusi sentimen
- Statistik panjang teks
- Performa model (akurasi, precision, recall, F1-score)
- Confusion matrix

### 4. Isi Template Presentasi
Buka file `PRESENTASI_HASIL.md` dan:
- Ganti semua `[ISI DARI OUTPUT]` dengan data dari `HASIL_ANALISIS.txt`
- Ganti `[Nama Anda]` dengan nama Anda
- Ganti `[Tanggal Presentasi]` dengan tanggal presentasi

### 5. Ambil Screenshot dari Dashboard
```bash
python -m streamlit run dashboard.py
```

Buka dashboard dan ambil screenshot untuk:
- **Slide 7**: Distribusi Sentimen (Pie Chart)
- **Slide 7**: Jumlah Tweet per Tanggal (Bar Chart)
- **Slide 7**: Distribusi Panjang Teks (Histogram)
- **Slide 10**: Contoh Teks per Sentimen
- **Slide 11**: Word Cloud untuk setiap sentimen

### 6. Siapkan Visualisasi
- Screenshot dari dashboard
- Screenshot confusion matrix (dari output `main.py`)
- Screenshot word cloud
- Screenshot contoh teks

## 📊 Struktur File Presentasi

File `PRESENTASI_HASIL.md` berisi 15 slide:

1. **Slide 1**: Judul
2. **Slide 2**: Latar Belakang
3. **Slide 3**: Tujuan Penelitian
4. **Slide 4**: Metodologi
5. **Slide 5**: Preprocessing Data
6. **Slide 6**: Statistik Dataset ← **ISI DARI HASIL_ANALISIS.txt**
7. **Slide 7**: Distribusi Data ← **SCREENSHOT DARI DASHBOARD**
8. **Slide 8**: Performansi Model ← **ISI DARI HASIL_ANALISIS.txt**
9. **Slide 9**: Confusion Matrix ← **ISI DARI HASIL_ANALISIS.txt**
10. **Slide 10**: Contoh Hasil Analisis ← **SCREENSHOT DARI DASHBOARD**
11. **Slide 11**: Word Cloud ← **SCREENSHOT DARI DASHBOARD**
12. **Slide 12**: Dashboard Interaktif
13. **Slide 13**: Kesimpulan
14. **Slide 14**: Keterbatasan & Saran
15. **Slide 15**: Terima Kasih

## 💡 Tips

### Untuk Presentasi PowerPoint/Google Slides:
1. Copy isi dari `PRESENTASI_HASIL.md` ke slide presentasi
2. Sisipkan screenshot yang sudah diambil
3. Format tabel dengan rapi
4. Gunakan warna yang konsisten

### Untuk Presentasi Langsung dari Markdown:
- Gunakan tools seperti:
  - [Marp](https://marp.app/) - Convert markdown to slides
  - [Reveal.js](https://revealjs.com/) - HTML presentation framework
  - [Slidev](https://sli.dev/) - Presentation slides for developers

## ⚠️ Troubleshooting

### Error saat menjalankan generate_presentasi.py
- Pastikan sudah menjalankan `main.py` terlebih dahulu
- Pastikan semua dependencies terinstall
- Pastikan file data ada di folder `data/`

### Dashboard tidak muncul
- Pastikan Streamlit terinstall: `pip install streamlit`
- Pastikan Plotly terinstall: `pip install plotly`
- Coba refresh browser

### Hasil tidak sesuai
- Pastikan data sudah di-preprocess dengan benar
- Periksa output console untuk error messages
- Pastikan model sudah di-train dengan benar

## 📝 Checklist Sebelum Presentasi

- [ ] Sudah menjalankan `python main.py`
- [ ] Sudah menjalankan `python generate_presentasi.py`
- [ ] Sudah mengisi semua data di `PRESENTASI_HASIL.md`
- [ ] Sudah mengambil screenshot dari dashboard
- [ ] Sudah mengambil screenshot confusion matrix
- [ ] Sudah mengambil screenshot word cloud
- [ ] Sudah mengambil screenshot contoh teks
- [ ] Sudah memformat presentasi dengan rapi
- [ ] Sudah memeriksa semua angka dan statistik
- [ ] Sudah mempersiapkan demo dashboard (jika perlu)

## 🎉 Selamat Presentasi!

Jika ada pertanyaan, periksa file:
- `PRESENTASI_HASIL.md` - Template presentasi
- `HASIL_ANALISIS.txt` - Hasil analisis (dibuat setelah menjalankan generate_presentasi.py)
- `DASHBOARD.md` - Panduan dashboard

