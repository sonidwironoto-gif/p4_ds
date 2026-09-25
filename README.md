# Segmentasi Nasabah Kartu Kredit dengan K-Means (CRISP-DM)

Project Data Science sederhana untuk menerapkan algoritma *clustering* dalam menemukan pola/segmen tersembunyi pada data nasabah kartu kredit, dikerjakan mengikuti metodologi **CRISP-DM** (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment).

**Mata Kuliah:** Data Science — Pertemuan 4, Semester 7
**Nama / NIM:** _(isi di sini)_

## Deskripsi Singkat

Penerbit kartu kredit memiliki nasabah dengan perilaku transaksi yang sangat beragam (ada yang aktif berbelanja, ada yang sering menarik tunai, ada yang jarang memakai kartu). Project ini mengelompokkan nasabah berdasarkan pola transaksinya menggunakan **K-Means Clustering** (dibandingkan dengan Agglomerative Clustering), agar tim bisnis dapat menyusun strategi pemasaran/risiko yang berbeda untuk tiap segmen.

## Dataset

- **Sumber:** [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) — Kaggle, oleh Arjun Bhasin
- **Berkas:** `CC_GENERAL.csv`
- **Alasan kesesuaian untuk clustering:**
  1. Tidak memiliki label target → cocok untuk pendekatan *unsupervised*.
  2. Memiliki banyak fitur numerik perilaku transaksi (saldo, pembelian, tarik tunai, frekuensi, dll).
  3. Secara bisnis lazim digunakan untuk segmentasi nasabah.
  4. Memenuhi ketentuan tugas: ≥1.000 baris, ≥8 kolom, ≥3 fitur numerik.

> Catatan: ukuran pasti dataset (jumlah baris/kolom) dapat dilihat langsung pada output `df.shape` di notebook setelah dataset diunduh.

## Struktur Repository

```
.
├── app.py                       # Aplikasi deployment (Streamlit)
├── requirements.txt             # Daftar dependensi Python
├── cluster_model.joblib         # Model/pipeline hasil training (dipakai app.py)
├── CC_GENERAL.csv               # Dataset mentah
├── clustering_crisp_dm.ipynb    # Notebook lengkap tahapan CRISP-DM
└── README.md
```

## Cara Menjalankan Notebook (Lokal)

```bash
pip install -r requirements.txt jupyter
jupyter notebook clustering_crisp_dm.ipynb
```
Jalankan seluruh sel (Run All). Notebook akan membaca `CC_GENERAL.csv` dan menghasilkan `cluster_model.joblib` di akhir proses.

## Cara Menjalankan Aplikasi (Lokal)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Link Deployment

🔗 **Streamlit App:** _(isi setelah deploy, contoh: https://nama-app-anda.streamlit.app)_

## Ringkasan Metode

- **Fitur yang digunakan:** `BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, PURCHASES_FREQUENCY, CASH_ADVANCE_FREQUENCY, PRC_FULL_PAYMENT`
- **Preprocessing:** imputasi median untuk missing value, transformasi `log1p` (data uang yang miring ke kanan), standardisasi (`StandardScaler`)
- **Algoritma:** K-Means (`k` ditentukan lewat Elbow Method + Silhouette Score), dibandingkan dengan Agglomerative Clustering (Ward linkage)
- **Evaluasi:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- **Hasil:** _(isi ringkasan jumlah klaster dan interpretasinya setelah melihat hasil akhir notebook)_

## Keterbatasan

- Evaluasi bersifat internal (tidak ada *ground truth*/label asli), sehingga kebenaran segmentasi perlu divalidasi lebih lanjut secara bisnis.
- K-Means berasumsi klaster berbentuk bulat/isotropik dan sensitif terhadap inisialisasi awal.
- Hasil bergantung pada pemilihan fitur, transformasi, dan nilai `k` yang dipakai.
