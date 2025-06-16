import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Prediksi Harga & Rekomendasi Mobil Berdasarkan Spesifikasi")

st.write("Aplikasi ini memprediksi harga mobil dari input, lalu menampilkan mobil-mobil dengan harga mendekati hasil prediksi.")

# Baca dataset dari file default (tidak perlu upload)
try:
    df = pd.read_csv("scrap price.csv")
except FileNotFoundError:
    st.error("File 'scrap price.csv' tidak ditemukan di direktori. Pastikan file tersedia.")
    st.stop()

# Tampilkan preview
st.subheader("Preview Dataset")
st.write(df.head())

# Validasi kolom
required_columns = ['name', 'enginesize', 'horsepower', 'citympg', 'curbweight', 'price']
if all(col in df.columns for col in required_columns):

    # Model training
    fitur = ['enginesize', 'horsepower', 'citympg', 'curbweight']
    X = df[fitur]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)

    # Prediksi seluruh harga mobil
    df['predicted_price'] = model.predict(X)

    st.subheader("Masukkan Spesifikasi Mobil")
    enginesize = st.number_input("Ukuran Mesin (enginesize)", min_value=50, max_value=400, value=150)
    horsepower = st.number_input("Tenaga Mesin (horsepower)", min_value=40, max_value=300, value=120)
    citympg = st.number_input("Konsumsi BBM Dalam Kota (citympg)", min_value=5, max_value=50, value=25)
    curbweight = st.number_input("Berat Mobil (curbweight)", min_value=1000, max_value=5000, value=2500)

    toleransi = st.slider("Toleransi Selisih Harga (USD)", min_value=100, max_value=10000, value=2000, step=100)

    if st.button("Prediksi dan Tampilkan Mobil Serupa"):
        # Prediksi harga dari input
        input_df = pd.DataFrame([{
            'enginesize': enginesize,
            'horsepower': horsepower,
            'citympg': citympg,
            'curbweight': curbweight
        }])
        prediksi_harga = model.predict(input_df)[0]

        st.success(f"Perkiraan Harga Mobil Anda: ${prediksi_harga:,.2f}")

        # Filter mobil dengan harga mendekati prediksi
        hasil = df[
            (df['predicted_price'] >= (prediksi_harga - toleransi)) &
            (df['predicted_price'] <= (prediksi_harga + toleransi))
        ][['name', 'predicted_price']].sort_values(by='predicted_price').reset_index(drop=True)

        if hasil.empty:
            st.warning("Tidak ada mobil yang cocok dalam kisaran harga tersebut.")
        else:
            st.subheader("Mobil Lain dengan Harga Mendekati Prediksi")
            st.write(hasil)
else:
    st.error("Dataset tidak memiliki kolom yang dibutuhkan.")
