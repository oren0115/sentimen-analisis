ANALISIS SENTIMEN OPINI PUBLIK TERHADAP PEMILIHAN PRESIDEN MENGGUNAKAN METODE NAIVE BAYES

DAFTAR GAMBAR

Gambar 1. Grafik Jumlah Tweet per Tanggal
Lokasi: `outputs/tweets_by_date.png`

Gambar 2. Confusion Matrix Model Naive Bayes
Lokasi: `outputs/confusion_matrix.png`

Gambar 3. Distribusi Panjang Teks - Data Training
Lokasi: `outputs/text_length_distribution_train.png`

Gambar 4. Distribusi Panjang Teks - Data Testing
Lokasi: `outputs/text_length_distribution_test.png`

Gambar 5. Distribusi Sentimen Sebelum dan Sesudah Oversampling
Lokasi: `outputs/sentiment_distribution.png`

________________________________________
BAB I
PENDAHULUAN
1.1 Latar Belakang
Perkembangan media sosial telah mengubah cara masyarakat menyampaikan opini dan berpartisipasi dalam diskursus publik. Twitter (kini X) menjadi salah satu platform yang paling aktif digunakan untuk mengekspresikan pandangan politik karena sifatnya yang real-time, terbuka, dan mudah diakses. Dalam konteks Pemilihan Presiden (Pilpres) 2024 di Indonesia, Twitter menjadi ruang publik digital tempat masyarakat menyampaikan dukungan, kritik, maupun pandangan netral terhadap pasangan calon presiden.
Volume data yang dihasilkan dari media sosial sangat besar dan terus bertambah setiap waktu. Oleh karena itu, diperlukan pendekatan komputasional yang mampu mengolah data teks dalam jumlah besar secara otomatis. Salah satu pendekatan yang umum digunakan adalah analisis sentimen, yaitu teknik untuk mengidentifikasi kecenderungan emosi atau opini dalam teks ke dalam kategori positif, negatif, atau netral.
Metode Naive Bayes merupakan algoritma klasifikasi yang banyak digunakan dalam analisis sentimen karena kesederhanaannya, efisiensi komputasi, serta performanya yang cukup baik pada data teks. Meskipun memiliki asumsi kemandirian antar fitur, Naive Bayes tetap relevan dan efektif untuk tugas klasifikasi opini publik di media sosial.
1.2 Rumusan Masalah
Rumusan masalah dalam penelitian ini adalah sebagai berikut: 
1. Bagaimana tahapan preprocessing data tweet yang efektif untuk analisis sentimen? 
2. Bagaimana penerapan metode Naive Bayes dalam mengklasifikasikan sentimen opini publik terhadap Pilpres? 
3. Seberapa baik performa model Naive Bayes dalam mengklasifikasikan sentimen positif, negatif, dan netral?
1.3 Tujuan Penelitian
Tujuan dari penelitian ini adalah: 
1. Membangun sistem analisis sentimen opini publik menggunakan metode Naive Bayes. 2. Mengevaluasi kinerja model dalam mengklasifikasikan sentimen menjadi tiga kategori. 3. Menyajikan hasil analisis sentimen dalam bentuk visualisasi yang mudah dipahami.
1.4 Manfaat Penelitian
Penelitian ini diharapkan memberikan manfaat sebagai berikut: 
- Manfaat Teoritis: Menambah referensi akademik terkait penerapan analisis sentimen berbasis Naive Bayes pada data berbahasa Indonesia. 
- Manfaat Praktis: Memberikan gambaran opini publik terhadap kandidat presiden yang dapat dimanfaatkan oleh peneliti, akademisi, maupun pemangku kepentingan.
1.5 Batasan Masalah
Batasan dalam penelitian ini meliputi: 
1. Data berasal dari Twitter dengan topik terkait calon presiden tertentu. 
2. Klasifikasi sentimen dibatasi pada tiga kelas: positif, negatif, dan netral. 
3. Model yang digunakan adalah Multinomial Naive Bayes dengan representasi fitur TF-IDF.
1.6 Sistematika Penulisan
Makalah ini disusun dalam lima bab, yaitu Pendahuluan, Tinjauan Pustaka, Metodologi Penelitian, Hasil dan Pembahasan, serta Kesimpulan dan Saran.
________________________________________
BAB II
TINJAUAN PUSTAKA
2.1 Analisis Sentimen
Analisis sentimen adalah proses untuk mengidentifikasi dan mengklasifikasikan opini atau emosi yang terkandung dalam teks (Liu, 2012). Analisis ini banyak digunakan dalam text mining dan NLP untuk memahami persepsi publik terhadap suatu isu, produk, atau tokoh tertentu.
2.2 Naive Bayes Classifier
Naive Bayes adalah algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi independensi antar fitur (Mitchell, 1997). Dalam klasifikasi teks, Multinomial Naive Bayes sering digunakan karena sesuai dengan data berbasis frekuensi kata.
2.3 Preprocessing Teks
Preprocessing data teks bertujuan untuk mengurangi noise dan meningkatkan kualitas data. Tahapan umum meliputi pembersihan teks, normalisasi, penghapusan stopword, serta tokenisasi.
2.4 Feature Extraction
Metode representasi teks yang digunakan dalam penelitian ini adalah Count Vectorizer dan TF-IDF. TF-IDF memberikan bobot lebih besar pada kata-kata yang memiliki tingkat kepentingan tinggi dalam dokumen.
2.5 Penelitian Terkait
Beberapa penelitian terdahulu menunjukkan bahwa Naive Bayes memiliki performa yang kompetitif dalam analisis sentimen Twitter (Pak & Paroubek, 2010; Agarwal et al., 2011).
________________________________________
BAB III
METODOLOGI PENELITIAN
3.1 Rancangan Penelitian
Penelitian ini menggunakan pendekatan kuantitatif dengan metode eksperimen. Pendekatan ini bertujuan untuk menguji performa algoritma Naive Bayes dalam mengklasifikasikan sentimen opini publik berdasarkan data teks dari media sosial Twitter. Alur penelitian dimulai dari pengumpulan data, preprocessing, ekstraksi fitur, pelatihan model, evaluasi performa, hingga visualisasi hasil.
Tahapan penelitian secara umum meliputi:
1.	Pengumpulan data tweet yang telah diberi label sentimen.
2.	Preprocessing data untuk mengurangi noise dan meningkatkan kualitas teks.
3.	Ekstraksi fitur menggunakan metode Count Vectorizer dan TF-IDF.
4.	Pembagian data menjadi data latih, validasi, dan uji.
5.	Pelatihan model Multinomial Naive Bayes.
6.	Evaluasi performa model menggunakan metrik evaluasi.
7.	Penyajian hasil dalam bentuk visualisasi.
3.2 Sumber Data
Dataset yang digunakan berasal dari data tweet yang telah dilabeli secara manual ke dalam tiga kelas sentimen, yaitu positif, negatif, dan netral. Dataset dibagi menjadi dua bagian, yaitu data latih dan data uji.
•	Data latih terdiri dari 1.336 tweet awal yang setelah proses deduplikasi dan filtering menjadi 1.225 tweet.
•	Data uji terdiri dari 1.336 tweet yang setelah proses filtering menjadi 1.224 tweet.
Setiap tweet memiliki atribut tanggal, teks tweet, nama pengguna, panjang teks, dan label sentimen.
3.3 Preprocessing Data
Preprocessing data merupakan tahap penting untuk memastikan teks yang digunakan bebas dari noise. Tahapan preprocessing meliputi deduplikasi data, pembersihan teks, serta filtering berdasarkan panjang teks dan token.
Deduplikasi dilakukan untuk menghapus tweet yang memiliki isi teks yang sama. Pembersihan teks meliputi penghapusan emoji, URL, mention, tanda baca, karakter non-ASCII, serta normalisasi huruf menjadi huruf kecil. Selain itu, hashtag dibersihkan dengan menghilangkan simbol pagar (#) tanpa menghapus kata kunci yang terkandung di dalamnya.
Filtering dilakukan dengan menghapus tweet yang memiliki panjang kurang dari empat kata karena dianggap tidak informatif. Selain itu, dilakukan filtering berdasarkan panjang token menggunakan tokenizer BERT, dimana tweet dengan panjang token lebih dari 80 dihapus untuk menghindari outlier.
3.4 Encoding dan Penyeimbangan Data
Label sentimen dikonversi ke dalam bentuk numerik untuk memudahkan proses komputasi, yaitu negatif (0), netral (1), dan positif (2). Karena distribusi kelas pada data latih tidak seimbang, dilakukan penyeimbangan data menggunakan metode Random Over Sampler agar setiap kelas memiliki jumlah data yang relatif sama.
3.5 Ekstraksi Fitur
Ekstraksi fitur dilakukan untuk mengubah data teks menjadi representasi numerik. Pada tahap ini digunakan Count Vectorizer untuk menghitung frekuensi kemunculan kata, kemudian dilakukan pembobotan menggunakan TF-IDF untuk menekankan kata-kata yang memiliki tingkat kepentingan tinggi dalam dokumen.
3.6 Pelatihan dan Evaluasi Model
Model yang digunakan adalah Multinomial Naive Bayes. Model dilatih menggunakan data latih yang telah melalui proses ekstraksi fitur. Evaluasi performa dilakukan menggunakan data uji dengan metrik akurasi, precision, recall, dan F1-score untuk setiap kelas sentimen serta rata-rata keseluruhan.

3.7 Visualisasi dan Penyimpanan Hasil
Semua visualisasi yang dihasilkan dari analisis disimpan dalam folder `outputs/` pada direktori proyek. Visualisasi yang dihasilkan meliputi:
- Confusion Matrix: `outputs/confusion_matrix.png`
- Grafik Jumlah Tweet per Tanggal: `outputs/tweets_by_date.png`
- Distribusi Panjang Teks Training: `outputs/text_length_distribution_train.png`
- Distribusi Panjang Teks Testing: `outputs/text_length_distribution_test.png`
- Distribusi Sentimen: `outputs/sentiment_distribution.png`

Selain itu, hasil analisis lengkap disimpan dalam file teks di `outputs/HASIL_ANALISIS.txt`. Semua gambar dan hasil analisis dapat diakses melalui dashboard interaktif yang dijalankan menggunakan Streamlit.
________________________________________
BAB IV
HASIL DAN PEMBAHASAN
4.1 Hasil Preprocessing Data
Hasil preprocessing menunjukkan bahwa tahapan pembersihan dan filtering data berhasil mengurangi jumlah data yang tidak relevan tanpa menghilangkan informasi penting. Setelah proses deduplikasi dan filtering, jumlah data latih menjadi sekitar 1.213 tweet, sedangkan data uji berjumlah sekitar 1.224 tweet.

Visualisasi distribusi jumlah tweet per tanggal dapat dilihat pada Gambar 1 (lokasi: `outputs/tweets_by_date.png`). Gambar ini menunjukkan pola distribusi tweet dari waktu ke waktu, memberikan gambaran tentang aktivitas diskusi publik terkait topik penelitian.

Contoh hasil pembersihan teks menunjukkan bahwa noise seperti emoji, mention, dan tanda baca berhasil dihilangkan sehingga teks menjadi lebih bersih dan siap digunakan dalam proses klasifikasi. Visualisasi distribusi panjang teks untuk data training dapat dilihat pada Gambar 3 (lokasi: `outputs/text_length_distribution_train.png`), sedangkan untuk data testing dapat dilihat pada Gambar 4 (lokasi: `outputs/text_length_distribution_test.png`).
4.2 Distribusi Sentimen
Distribusi sentimen pada data latih sebelum penyeimbangan menunjukkan ketidakseimbangan kelas, dengan kelas positif memiliki jumlah data terbanyak. Setelah dilakukan oversampling menggunakan Random Over Sampler, distribusi sentimen menjadi seimbang sehingga model dapat belajar secara adil dari setiap kelas.

Visualisasi distribusi sentimen sebelum dan sesudah oversampling dapat dilihat pada Gambar 5 (lokasi: `outputs/sentiment_distribution.png`). Gambar ini memperlihatkan perbandingan distribusi kelas sentimen yang menunjukkan efektivitas teknik oversampling dalam menyeimbangkan data.
4.3 Evaluasi Performa Model
Hasil pengujian model Naive Bayes menunjukkan performa yang sangat baik dengan tingkat akurasi sebesar 95%. Nilai precision, recall, dan F1-score untuk setiap kelas juga menunjukkan hasil yang tinggi, yang menandakan bahwa model mampu mengklasifikasikan sentimen secara konsisten.

Confusion Matrix yang menunjukkan detail prediksi model untuk setiap kelas dapat dilihat pada Gambar 2 (lokasi: `outputs/confusion_matrix.png`). Confusion matrix ini memperlihatkan jumlah prediksi benar dan salah untuk setiap kelas sentimen, memberikan wawasan tentang pola kesalahan klasifikasi yang terjadi.

Kelas positif memiliki nilai precision tertinggi, yang menunjukkan bahwa prediksi sentimen positif sangat akurat. Sementara itu, kelas negatif dan netral memiliki nilai recall yang tinggi, menandakan bahwa sebagian besar data pada kelas tersebut berhasil teridentifikasi dengan baik oleh model.
4.4 Analisis Hasil
Hasil evaluasi menunjukkan bahwa kombinasi preprocessing yang komprehensif, representasi fitur TF-IDF, serta penggunaan metode Naive Bayes memberikan kontribusi signifikan terhadap performa model. Penyeimbangan data juga berperan penting dalam meningkatkan kemampuan model dalam mengenali kelas minoritas.
Meskipun demikian, beberapa kesalahan klasifikasi masih terjadi, terutama pada tweet yang memiliki konteks ambigu atau menggunakan bahasa sindiran. Hal ini menunjukkan keterbatasan model berbasis statistik dalam memahami konteks semantik yang lebih dalam.
4.5 Pembahasan
Secara keseluruhan, hasil penelitian menunjukkan bahwa metode Naive Bayes efektif digunakan untuk analisis sentimen opini publik di media sosial. Dengan akurasi yang tinggi dan proses komputasi yang efisien, metode ini cocok digunakan sebagai baseline dalam penelitian analisis sentimen, khususnya pada data berbahasa Indonesia.


BAB V
KESIMPULAN DAN SARAN
5.1 Kesimpulan
Berdasarkan hasil penelitian, dapat disimpulkan bahwa metode Naive Bayes dengan representasi fitur TF-IDF mampu mengklasifikasikan sentimen opini publik dengan baik. Preprocessing yang tepat dan penanganan ketidakseimbangan data berperan penting dalam meningkatkan performa model.
5.2 Saran
Penelitian selanjutnya dapat mengembangkan model dengan dataset yang lebih besar, membandingkan dengan algoritma lain, serta menggunakan pendekatan deep learning untuk meningkatkan akurasi.
________________________________________
DAFTAR PUSTAKA
Agarwal, A., Xie, B., Vovsha, I., Rambow, O., & Passonneau, R. (2011). Sentiment Analysis of Twitter Data. Proceedings of the Workshop on Languages in Social Media.
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
Liu, B. (2012). Sentiment Analysis and Opinion Mining. Morgan & Claypool Publishers.
Mitchell, T. M. (1997). Machine Learning. McGraw-Hill.
Pak, A., & Paroubek, P. (2010). Twitter as a Corpus for Sentiment Analysis and Opinion Mining. LREC.
Sebastiani, F. (2002). Machine Learning in Automated Text Categorization. ACM Computing Surveys.

________________________________________
LAMPIRAN

Lampiran A: Lokasi File Gambar dan Hasil Analisis

Semua file gambar dan hasil analisis disimpan dalam folder `outputs/` pada direktori root proyek. Berikut adalah daftar lengkap lokasi file:

1. **Confusion Matrix**
   - Lokasi: `outputs/confusion_matrix.png`
   - Deskripsi: Matriks konfusi yang menunjukkan akurasi prediksi model untuk setiap kelas sentimen

2. **Grafik Jumlah Tweet per Tanggal**
   - Lokasi: `outputs/tweets_by_date.png`
   - Deskripsi: Bar chart yang menampilkan distribusi jumlah tweet berdasarkan tanggal

3. **Distribusi Panjang Teks - Data Training**
   - Lokasi: `outputs/text_length_distribution_train.png`
   - Deskripsi: Histogram distribusi panjang teks (dalam jumlah kata) untuk data training

4. **Distribusi Panjang Teks - Data Testing**
   - Lokasi: `outputs/text_length_distribution_test.png`
   - Deskripsi: Histogram distribusi panjang teks (dalam jumlah kata) untuk data testing

5. **Distribusi Sentimen**
   - Lokasi: `outputs/sentiment_distribution.png`
   - Deskripsi: Grafik yang menampilkan distribusi kelas sentimen sebelum dan sesudah oversampling

6. **Hasil Analisis Lengkap**
   - Lokasi: `outputs/HASIL_ANALISIS.txt`
   - Deskripsi: File teks yang berisi ringkasan lengkap hasil analisis, metrik evaluasi, dan statistik data

**Catatan**: 
- Untuk menghasilkan semua gambar, jalankan script `python scripts/main.py` atau gunakan dashboard interaktif dengan `python -m streamlit run scripts/dashboard.py`
- Dashboard interaktif juga menyediakan visualisasi real-time yang dapat diakses melalui browser di `http://localhost:8501`
- File gambar akan otomatis tersimpan di folder `outputs/` setelah menjalankan analisis

Lampiran B: Struktur Direktori Proyek

```
TUGAS BESAR/
├── data/                          # Folder data
│   ├── Sentiment1.csv            # Data training
│   └── Train1.csv                # Data testing
├── src/                          # Source code modules
│   ├── config.py                 # Konfigurasi
│   ├── text_cleaning.py          # Pembersihan teks
│   ├── data_processing.py        # Pemrosesan data
│   ├── visualization.py          # Visualisasi
│   └── models.py                 # Model machine learning
├── scripts/                      # Script executable
│   ├── main.py                  # Entry point analisis
│   ├── dashboard.py             # Dashboard Streamlit
│   └── generate_presentasi.py   # Generate hasil
├── outputs/                      # Folder output (LOKASI GAMBAR)
│   ├── confusion_matrix.png     # Gambar 2: Confusion Matrix
│   ├── tweets_by_date.png        # Gambar 1: Tweet per Tanggal
│   ├── text_length_distribution_train.png
│   ├── text_length_distribution_test.png
│   ├── sentiment_distribution.png
│   └── HASIL_ANALISIS.txt       # Hasil analisis lengkap
├── docs/                         # Dokumentasi
│   └── MAKALAH.md               # Makalah ini
├── jupyter lab/                  # Notebook Jupyter
│   └── analisis-sentimen-opini-publik-terhadap-pilpres-checkpoint.ipynb
├── models/                       # Model tersimpan (opsional)
├── requirements.txt             # Dependencies
└── README.md                     # Dokumentasi proyek
```

**Panduan Mengakses Gambar:**
1. Pastikan telah menjalankan analisis dengan `python scripts/main.py`
2. Buka folder `outputs/` di direktori proyek
3. Semua gambar akan tersedia dalam format PNG
4. Atau gunakan dashboard interaktif untuk melihat visualisasi secara real-time
