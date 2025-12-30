# Troubleshooting Dashboard

Panduan mengatasi masalah umum saat menjalankan dashboard.

## ⚠️ Warning yang Bisa Diabaikan

### 1. "missing ScriptRunContext" Warning

**Pesan:**
```
WARNING streamlit.runtime.scriptrunner_utils.script_run_context: 
Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
```

**Penyebab:**
Dashboard dijalankan langsung dengan Python (`python dashboard.py`) bukan dengan Streamlit.

**Solusi:**
Jalankan dashboard dengan perintah yang benar:
```bash
python -m streamlit run dashboard.py
```

**Catatan:** Warning ini bisa diabaikan jika dashboard tetap berfungsi, tapi lebih baik menggunakan perintah yang benar.

---

### 2. "use_container_width" Deprecation Warning

**Pesan:**
```
Please replace `use_container_width` with `width`.
`use_container_width` will be removed after 2025-12-31.
```

**Status:** ✅ **SUDAH DIPERBAIKI**

File `dashboard.py` sudah diperbarui untuk menggunakan `width='stretch'` sebagai pengganti `use_container_width=True`.

---

### 3. PyTorch/TensorFlow Warning

**Pesan:**
```
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. 
Models won't be available and only tokenizers, configuration and file/data utilities can be used.
```

**Penyebab:**
Kode hanya menggunakan BERT tokenizer, tidak memerlukan PyTorch atau TensorFlow untuk model.

**Solusi:**
**TIDAK PERLU DIPERBAIKI** - Ini adalah warning normal dan tidak mempengaruhi fungsi dashboard. Tokenizer tetap berfungsi tanpa framework deep learning.

---

## ❌ Error yang Perlu Diperbaiki

### 1. "streamlit is not recognized"

**Pesan:**
```
streamlit : The term 'streamlit' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

**Solusi:**
Gunakan Python module:
```bash
python -m streamlit run dashboard.py
```

Atau install streamlit:
```bash
pip install streamlit
```

---

### 2. "No module named 'src'"

**Pesan:**
```
ModuleNotFoundError: No module named 'src'
```

**Penyebab:**
Jalankan dari root directory proyek, bukan dari dalam folder `src/`.

**Solusi:**
```bash
# Pastikan Anda di root directory
cd "C:\Users\USER\Desktop\Nyoman\smester 5\AI\tugas"

# Kemudian jalankan
python -m streamlit run dashboard.py
```

---

### 3. "File not found" untuk Data CSV

**Pesan:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/Sentiment1.csv'
```

**Solusi:**
1. Pastikan file CSV ada di folder `data/`:
   - `data/Sentiment1.csv`
   - `data/Train1.csv`

2. Periksa path di `src/config.py`:
   ```python
   DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
   TRAIN_PATH = os.path.join(DATA_DIR, 'Sentiment1.csv')
   ```

---

### 4. Dashboard Tidak Terbuka di Browser

**Penyebab:**
- Firewall memblokir port 8501
- Browser tidak terbuka otomatis

**Solusi:**
1. Buka browser manual dan kunjungi: `http://localhost:8501`
2. Periksa firewall settings
3. Coba port lain:
   ```bash
   streamlit run dashboard.py --server.port 8502
   ```

---

### 5. Error Loading Data

**Pesan:**
```
Error loading data: [error message]
```

**Kemungkinan Penyebab:**
1. Encoding file salah
2. Format CSV tidak sesuai
3. File corrupt

**Solusi:**
1. Periksa encoding file (harus ISO-8859-1 atau UTF-8)
2. Buka file CSV dengan text editor untuk memastikan format benar
3. Pastikan kolom 'Text' dan 'Sentiment' ada

---

## 🔧 Tips Troubleshooting

### 1. Cek Dependencies

Pastikan semua dependencies terinstall:
```bash
pip install -r requirements.txt
```

### 2. Cek Python Version

Dashboard memerlukan Python 3.7+:
```bash
python --version
```

### 3. Clear Cache

Jika ada masalah dengan cache:
```bash
# Hapus folder __pycache__
rm -r src/__pycache__  # Linux/Mac
rmdir /s src\__pycache__  # Windows
```

### 4. Restart Dashboard

Jika dashboard freeze atau error:
1. Tekan `Ctrl+C` untuk stop
2. Jalankan lagi dengan `python -m streamlit run dashboard.py`

### 5. Check Logs

Periksa output terminal untuk error messages yang lebih detail.

---

## 📞 Bantuan Lebih Lanjut

Jika masalah masih terjadi:

1. **Periksa Logs**: Lihat output terminal untuk error messages
2. **Cek Dokumentasi**: Lihat `README.md` dan `DASHBOARD.md`
3. **Verifikasi Setup**: Pastikan semua file ada di tempat yang benar
4. **Test Dependencies**: Pastikan semua package terinstall dengan benar

---

## ✅ Checklist Sebelum Menjalankan Dashboard

- [ ] Python 3.7+ terinstall
- [ ] Semua dependencies terinstall (`pip install -r requirements.txt`)
- [ ] File CSV ada di folder `data/`
- [ ] Menjalankan dari root directory proyek
- [ ] Menggunakan perintah yang benar: `python -m streamlit run dashboard.py`

---

**Selamat menggunakan dashboard! 🎉**

