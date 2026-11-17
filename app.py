from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(page_title="Segmentasi Pelanggan RFM", layout="wide")
st.title("📊 Aplikasi Segmentasi Pelanggan Ritel (RFM Clustering)")
st.write(
    "Prediksi segmen pelanggan secara satuan atau unggah file transaksi untuk analisis massal."
)


# 1. Load Model & Scaler
@st.cache_resource
def load_models():
    app_dir = Path(__file__).resolve().parent
    scaler = joblib.load(app_dir / "scaler_rfm.pkl")
    model = joblib.load(app_dir / "model_kmeans_rfm.pkl")
    return scaler, model


try:
    scaler, model = load_models()
except Exception as e:
    st.error(
        f"Gagal memuat model. Pastikan file .pkl berada di folder yang sama. Error: {e}"
    )
    st.stop()

# Mapping nama cluster (Sesuaikan label ini dengan hasil interpretasi bisnis di notebook-mu)
cluster_labels = {
    0: "Pelanggan Reguler / Menengah",
    1: "Pelanggan Loyal / Sultan (High Value)",
    2: "Pelanggan Berisiko / Pasif (Churn Risk)",
    3: "Pelanggan Baru (Potensial)",
}

# Membuat 2 Tab: Prediksi Manual & Upload Batch Data
tab1, tab2 = st.tabs(
    ["🔍 Prediksi Pelanggan Tunggal", "📁 Upload Dataset Transaksi (Batch)"]
)

# --- TAB 1: PREDIKSI SATUAN ---
with tab1:
    st.subheader("Masukkan Data Perilaku Pelanggan")
    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (Hari sejak belanja terakhir)", min_value=0, value=30
        )
    with col2:
        frequency = st.number_input(
            "Frequency (Total transaksi/invoice)", min_value=1, value=5
        )
    with col3:
        monetary = st.number_input(
            "Monetary (Total uang dibelanjakan)", min_value=1.0, value=500.0
        )

    if st.button("Prediksi Segmen", type="primary"):
        # Preprocessing harus SAMA PERSIS dengan yang dilakukan di Colab
        # Di Praktikum 2: Frequency dan Monetary ditransformasi dengan log1p
        input_data = np.array([[recency, np.log1p(frequency), np.log1p(monetary)]])

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediksi
        cluster_pred = model.predict(input_scaled)[0]
        label_hasil = cluster_labels.get(cluster_pred, f"Cluster {cluster_pred}")

        st.success(f"### Hasil Segmentasi: **{label_hasil}** (Cluster {cluster_pred})")

# --- TAB 2: PREDIKSI BATCH DARI FILE ---
with tab2:
    st.subheader("Upload File Transaksi (Format Online Retail)")
    st.info(
        "File harus memiliki kolom: `InvoiceNo`, `Quantity`, `InvoiceDate`, `UnitPrice`, dan `CustomerID`."
    )

    uploaded_file = st.file_uploader(
        "Pilih file Excel (.xlsx) atau CSV (.csv)", type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        with st.spinner("Memproses data dan menghitung RFM..."):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Data Cleaning sesuai Praktikum 4 & 5
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(subset=["CustomerID"])
            df = df[
                (df["Quantity"] > 0)
                & (df["UnitPrice"] > 0)
                & (~df["InvoiceNo"].astype(str).str.startswith("C"))
            ]
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            df["LineValue"] = df["Quantity"] * df["UnitPrice"]

            # Agregasi RFM sesuai Praktikum 2
            ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
            rfm = (
                df.groupby("CustomerID")
                .agg(
                    Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
                    Frequency=("InvoiceNo", "nunique"),
                    Monetary=("LineValue", "sum"),
                )
                .reset_index()
            )

            # Transformasi & Scaling
            rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()
            rfm_features["Frequency"] = np.log1p(rfm_features["Frequency"])
            rfm_features["Monetary"] = np.log1p(rfm_features["Monetary"])

            rfm_scaled = scaler.transform(rfm_features)
            rfm["Cluster"] = model.predict(rfm_scaled)
            rfm["Segmen"] = rfm["Cluster"].map(cluster_labels)

            st.write("### Hasil Segmentasi Pelanggan")
            st.dataframe(rfm.head(20), use_container_width=True)

            # Tombol Download Hasil
            csv_output = rfm.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Hasil Segmentasi (CSV)",
                data=csv_output,
                file_name="hasil_segmentasi_rfm.csv",
                mime="text/csv",
            )