import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentasi Nasabah Kartu Kredit", page_icon="💳", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("cluster_model.joblib")


bundle = load_bundle()
model, features, profile, ranges = (bundle["model"], bundle["features"],
                                    bundle["profile"], bundle["ranges"])

st.title("💳 Segmentasi Nasabah Kartu Kredit")
st.caption("K-Means clustering • Metodologi CRISP-DM • Dataset: Credit Card Dataset for Clustering (Kaggle)")

tab1, tab2 = st.tabs(["Prediksi Klaster", "Profil Klaster"])

with tab1:
    st.subheader("Masukkan perilaku nasabah")
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(features):
        r = ranges[f]
        with cols[i % 2]:
            values[f] = st.slider(f, 0.0, round(r["max"], 2), round(r["median"], 2),
                                  step=0.01 if r["max"] <= 1 else max(r["max"] / 500, 0.01))
    if st.button("Prediksi Klaster", type="primary"):
        cluster = int(model.predict(pd.DataFrame([values]))[0])
        st.success(f"Nasabah ini masuk ke **Klaster {cluster}**")
        st.write("Rata-rata klaster tersebut:")
        st.dataframe(profile.loc[[cluster]])
        st.caption("Catatan: slider dibatasi pada persentil ke-99 data; nilai di atasnya tidak tercakup.")

with tab2:
    st.subheader("Ringkasan tiap klaster (rata-rata nilai asli)")
    st.dataframe(profile)
    st.bar_chart(profile["Jumlah"])
