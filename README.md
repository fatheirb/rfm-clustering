# Segmentasi Pelanggan RFM

Aplikasi Streamlit untuk memprediksi segmen pelanggan secara manual atau dari file transaksi CSV/Excel.

## Struktur berkas

Pastikan semua berkas berikut berada di direktori utama repository:

```text
app.py
requirements.txt
scaler_rfm.pkl
model_kmeans_rfm.pkl
```

`scaler_rfm.pkl` dan `model_kmeans_rfm.pkl` adalah artefak model yang digunakan aplikasi. Keduanya harus berasal dari proses training yang sama. Dependensi mematok scikit-learn `1.6.1`, sesuai versi saat model dibuat.

## Deploy ke Streamlit Community Cloud

1. Tambahkan kedua berkas model ke direktori utama proyek.
2. Push proyek ke repository GitHub. Repository boleh public atau private, tetapi akun Streamlit harus memiliki akses ke repository private.
3. Buka [share.streamlit.io](https://share.streamlit.io/) dan masuk dengan akun GitHub.
4. Pilih **Create app**, lalu pilih repository, branch, dan `app.py` sebagai main file.
5. Pilih **Deploy**. Streamlit Community Cloud akan memasang dependensi dari `requirements.txt`.
6. Setelah aplikasi berjalan, coba prediksi manual dan unggah file transaksi untuk memastikan kedua alur bekerja.

Jika deployment gagal memuat model, periksa bahwa nama dan lokasi kedua berkas `.pkl` tepat seperti di atas serta versi scikit-learn di `requirements.txt` sesuai dengan versi yang digunakan saat model dibuat.

## Menjalankan secara lokal

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
