# Panduan Deploy ke Streamlit Cloud

## ⚠️ PENTING: Pastikan File Sudah Di-Push ke GitHub

Error yang terjadi menunjukkan bahwa Streamlit Cloud masih membaca versi lama dari requirements.txt yang memiliki versi exact (`pandas==2.0.3`, `numpy==1.24.3`, `wordcloud==1.9.2`).

## Langkah-langkah

### 1. Pastikan Semua Perubahan Sudah Di-Commit dan Push

```bash
git add requirements.txt
git add dashboard.py
git add scripts/dashboard.py
git add runtime.txt
git add .streamlit/config.toml
git commit -m "Fix requirements.txt for Python 3.13 compatibility"
git push origin main
```

### 2. Verifikasi requirements.txt di GitHub

Pastikan file `requirements.txt` di GitHub memiliki:
- ✅ Versi minimum (menggunakan `>=`), BUKAN versi exact (`==`)
- ✅ Tidak ada `wordcloud` di requirements.txt
- ✅ Versi yang kompatibel dengan Python 3.11/3.13

### 3. Di Streamlit Cloud

1. Buka aplikasi di Streamlit Cloud
2. Klik "Settings" atau "⚙️"
3. Pilih "Reboot app" untuk memaksa rebuild
4. Atau hapus dan buat ulang aplikasi

### 4. Jika Masih Error

**Opsi A: Gunakan Python 3.11**
- Pastikan `runtime.txt` berisi: `3.11`
- Streamlit Cloud akan menggunakan Python 3.11 yang lebih stabil

**Opsi B: Hapus Cache**
- Di Streamlit Cloud, coba "Clear cache" atau "Reboot app"

## File yang Harus Ada

1. ✅ `requirements.txt` - dengan versi minimum (>=)
2. ✅ `runtime.txt` - berisi `3.11` untuk Python version
3. ✅ `dashboard.py` - dengan wordcloud optional
4. ✅ `.streamlit/config.toml` - konfigurasi Streamlit

## Verifikasi requirements.txt

File `requirements.txt` yang benar harus seperti ini:

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
transformers>=4.30.0
nltk>=3.8.0
emoji>=2.0.0
streamlit>=1.28.0
plotly>=5.17.0
```

**TIDAK BOLEH ada:**
- ❌ `pandas==2.0.3` (versi exact)
- ❌ `numpy==1.24.3` (versi exact)
- ❌ `wordcloud==1.9.2` atau `wordcloud>=1.9.0` (sudah optional)

## Troubleshooting

Jika masih error setelah push:
1. Cek file di GitHub web interface untuk memastikan perubahan sudah ter-push
2. Clear browser cache
3. Reboot app di Streamlit Cloud
4. Jika perlu, hapus dan buat ulang aplikasi

