"""
Aplikasi Streamlit - Prediksi Kebahagiaan Warga (Somerville Happiness Survey 2015)
Model utama: Support Vector Machine (SVM)
Dibuat berdasarkan modul praktikum Tugas_DSFH_23220021
"""

import io
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Kebahagiaan - Somerville Survey",
    page_icon="😊",
    layout="wide",
)

DATASET_URL = "https://archive.ics.uci.edu/static/public/479/somerville+happiness+survey.zip"

FEATURE_INFO = {
    "X1": "Kepuasan terhadap ketersediaan (availability) layanan kota",
    "X2": "Kepuasan terhadap biaya (cost) layanan kota",
    "X3": "Kepuasan terhadap fasilitas jalan raya & trotoar",
    "X4": "Kepuasan terhadap kualitas pendidikan sekolah umum",
    "X5": "Kepuasan terhadap ruang hijau kota (taman/rekreasi)",
    "X6": "Rasa percaya diri terhadap kepolisian setempat",
}


# ----------------------------------------------------------------------------
# LOAD DATA (di-cache agar tidak download ulang setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Mengunduh dan memuat dataset...")
def load_data_from_url():
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            raw = f.read()
            try:
                decoded = raw.decode("utf-16")
            except UnicodeDecodeError:
                decoded = raw.decode("utf-8")
    df = pd.read_csv(io.StringIO(decoded), sep=",")
    return df


def load_data():
    """Coba unduh dataset dari UCI. Jika gagal (mis. tidak ada akses internet
    di lingkungan deploy), izinkan pengguna mengunggah file CSV secara manual."""
    try:
        return load_data_from_url()
    except Exception as e:
        st.warning(
            "Tidak dapat mengunduh dataset otomatis dari UCI Repository "
            f"({e}). Silakan unggah file "
            "`SomervilleHappinessSurvey2015.csv` secara manual."
        )
        uploaded = st.file_uploader("Unggah dataset CSV", type=["csv"])
        if uploaded is not None:
            raw = uploaded.read()
            try:
                decoded = raw.decode("utf-16")
            except UnicodeDecodeError:
                decoded = raw.decode("utf-8")
            return pd.read_csv(io.StringIO(decoded), sep=",")
        st.stop()


@st.cache_resource(show_spinner="Melatih model...")
def train_all_models(X_train, y_train, X_test, y_test):
    """Melatih 10 model klasifikasi dan mengembalikan hasil evaluasinya."""
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, solver="liblinear"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "SVM (Support Vector Machine)": SVC(probability=True, random_state=42, kernel="rbf"),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "MLP Classifier": MLPClassifier(random_state=42, max_iter=2000),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    }

    results = {}
    fitted_models = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        results[name] = {
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
        }
        fitted_models[name] = clf

    return results, fitted_models


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("😊 Somerville Happiness")
st.sidebar.markdown(
    "Aplikasi ini memprediksi apakah seorang warga **bahagia (1)** atau "
    "**tidak bahagia (0)** terhadap kotanya, berdasarkan **Somerville "
    "Happiness Survey 2015** (UCI Machine Learning Repository)."
)
page = st.sidebar.radio(
    "Navigasi",
    ["📊 Eksplorasi Data", "🤖 Perbandingan Model", "🔮 Prediksi (SVM)"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model utama aplikasi ini adalah **Support Vector Machine (SVM)**, "
    "namun 9 algoritma lain turut dibandingkan untuk menentukan model "
    "dengan performa terbaik pada dataset ini."
)

# ----------------------------------------------------------------------------
# LOAD DATA & SPLIT
# ----------------------------------------------------------------------------
df = load_data()
X = df.drop("D", axis=1)
y = df["D"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results, fitted_models = train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
best_model_name = max(results, key=lambda k: results[k]["cv_mean"])

# ============================================================================
# HALAMAN 1: EKSPLORASI DATA
# ============================================================================
if page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data (EDA)")
    st.markdown(
        "Dataset **Somerville Happiness Survey 2015** berisi jawaban survei "
        "warga kota Somerville, Massachusetts, AS."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Cuplikan Data")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("Ukuran Data")
        st.metric("Jumlah Baris", df.shape[0])
        st.metric("Jumlah Kolom", df.shape[1])
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Keterangan Kolom")
    st.markdown("**D** = Target (0 = Tidak Bahagia, 1 = Bahagia)")
    for k, v in FEATURE_INFO.items():
        st.markdown(f"- **{k}**: {v} (skala 1–5)")

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Distribusi Target (D)")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(x="D", data=df, ax=ax, palette="Blues")
    ax.set_xlabel("D (0 = Tidak Bahagia, 1 = Bahagia)")
    st.pyplot(fig)

    st.subheader("Korelasi Antar Fitur")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

# ============================================================================
# HALAMAN 2: PERBANDINGAN MODEL
# ============================================================================
elif page == "🤖 Perbandingan Model":
    st.title("🤖 Perbandingan 10 Algoritma Klasifikasi")
    st.markdown(
        "Setiap model dilatih pada 70% data (stratified) dan diuji pada 30% "
        "data sisanya. Fitur telah dinormalisasi dengan `StandardScaler` "
        "agar sesuai untuk model berbasis jarak/margin seperti SVM."
    )

    summary_rows = []
    for name, res in results.items():
        summary_rows.append(
            {
                "Model": name,
                "Akurasi (Test)": round(res["accuracy"], 3),
                "Rata-rata CV (5-fold)": round(res["cv_mean"], 3),
                "Std CV": round(res["cv_std"], 3),
                "AUC": round(res["roc_auc"], 3) if res["roc_auc"] else "-",
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "Rata-rata CV (5-fold)", ascending=False
    ).reset_index(drop=True)

    st.subheader("Tabel Ringkasan Performa")
    st.dataframe(summary_df, use_container_width=True)

    st.success(
        f"🏆 **Model terbaik berdasarkan rata-rata cross-validation**: "
        f"**{best_model_name}** "
        f"(CV Accuracy = {results[best_model_name]['cv_mean']:.3f})"
    )

    st.subheader("Grafik Perbandingan Akurasi")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    order = summary_df.sort_values("Rata-rata CV (5-fold)")
    ax3.barh(order["Model"], order["Rata-rata CV (5-fold)"], color="#4C72B0")
    ax3.set_xlabel("Rata-rata Akurasi CV (5-fold)")
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Detail per Model")
    chosen = st.selectbox("Pilih model untuk melihat detail:", list(results.keys()),
                           index=list(results.keys()).index("SVM (Support Vector Machine)"))
    res = results[chosen]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        fig4, ax4 = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax4)
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        st.pyplot(fig4)
    with c2:
        if res["fpr"] is not None:
            st.markdown("**Kurva ROC**")
            fig5, ax5 = plt.subplots(figsize=(4, 3.5))
            ax5.plot(res["fpr"], res["tpr"], color="darkorange",
                     label=f"AUC = {res['roc_auc']:.2f}")
            ax5.plot([0, 1], [0, 1], linestyle="--", color="navy")
            ax5.set_xlabel("False Positive Rate")
            ax5.set_ylabel("True Positive Rate")
            ax5.legend(loc="lower right")
            st.pyplot(fig5)
        else:
            st.info("Model ini tidak mendukung predict_proba, ROC tidak tersedia.")

    st.markdown("**Classification Report**")
    st.dataframe(pd.DataFrame(res["report"]).transpose().round(3), use_container_width=True)

# ============================================================================
# HALAMAN 3: PREDIKSI DENGAN SVM
# ============================================================================
else:
    st.title("🔮 Prediksi Kebahagiaan (Model: SVM)")
    st.markdown(
        "Isi nilai kepuasan (skala **1 - 5**) untuk tiap aspek di bawah ini, "
        "lalu klik **Prediksi** untuk melihat apakah profil ini diprediksi "
        "**bahagia** atau **tidak bahagia** oleh model SVM."
    )

    svm_model = fitted_models["SVM (Support Vector Machine)"]
    svm_res = results["SVM (Support Vector Machine)"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Akurasi SVM (data uji)", f"{svm_res['accuracy']*100:.1f}%")
    with col_b:
        st.metric("Rata-rata Akurasi CV (5-fold)", f"{svm_res['cv_mean']*100:.1f}%")

    st.markdown("---")
    inputs = {}
    cols = st.columns(3)
    for i, (feat, desc) in enumerate(FEATURE_INFO.items()):
        with cols[i % 3]:
            inputs[feat] = st.slider(f"{feat} — {desc}", 1, 5, 3)

    if st.button("🔍 Prediksi", type="primary", use_container_width=True):
        input_df = pd.DataFrame([inputs])[X.columns]
        input_scaled = scaler.transform(input_df)
        pred = svm_model.predict(input_scaled)[0]
        proba = svm_model.predict_proba(input_scaled)[0]

        if pred == 1:
            st.success(f"😊 Prediksi: **BAHAGIA** (probabilitas: {proba[1]*100:.1f}%)")
        else:
            st.error(f"☹️ Prediksi: **TIDAK BAHAGIA** (probabilitas: {proba[0]*100:.1f}%)")

        st.progress(float(proba[1]))
        st.caption(
            f"Probabilitas Bahagia: {proba[1]*100:.1f}% | "
            f"Probabilitas Tidak Bahagia: {proba[0]*100:.1f}%"
        )

st.markdown("---")
st.caption(
    "Dataset: Somerville Happiness Survey 2015 — UCI Machine Learning Repository. "
    "Dibangun dengan Streamlit & scikit-learn."
)