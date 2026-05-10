"""
automate_DamarPrayogaSetiawan.py
=================================
Script otomatisasi preprocessing dataset Titanic.
Mengkonversi seluruh tahapan eksperimen (notebook) menjadi fungsi
yang dapat dijalankan secara otomatis oleh GitHub Actions maupun secara manual.

Tahapan:
  1. Data Loading
  2. Drop Duplicates
  3. Handle Missing Values (median untuk numerik, modus untuk kategorikal)
  4. Label Encoding untuk fitur kategorikal
  5. Standarisasi fitur dengan StandardScaler (kecuali kolom target)
  6. Simpan hasil ke titanic_preprocessing.csv

Penggunaan:
  python automate_DamarPrayogaSetiawan.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ──────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
TARGET_COLUMN = "Survived"
OUTPUT_FILENAME = "titanic_preprocessing.csv"


# ──────────────────────────────────────────────
# Fungsi-fungsi Preprocessing
# ──────────────────────────────────────────────

def load_data(source: str) -> pd.DataFrame:
    """
    Memuat dataset dari URL atau path file lokal.

    Parameters
    ----------
    source : str
        URL atau path file CSV yang akan dimuat.

    Returns
    -------
    pd.DataFrame
        DataFrame mentah hasil loading.
    """
    df = pd.read_csv(source)
    print(f"[load_data] Dataset berhasil dimuat → {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[load_data] Kolom: {df.columns.tolist()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data mentah:
      - Menghapus baris duplikat
      - Imputasi missing values (median untuk numerik, modus untuk kategorikal)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mentah.

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah bersih dari duplikat dan missing values.
    """
    df_cleaned = df.copy()

    # ── Step 1: Hapus Duplikat ──
    before = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    removed = before - len(df_cleaned)
    print(f"[clean_data] Duplikat dihapus: {removed} baris")

    # ── Step 2: Identifikasi Tipe Kolom ──
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()

    # Keluarkan target dari preprocessing
    if TARGET_COLUMN in num_cols:
        num_cols.remove(TARGET_COLUMN)
    if TARGET_COLUMN in cat_cols:
        cat_cols.remove(TARGET_COLUMN)

    # ── Step 3: Imputasi Missing Values ──
    if num_cols:
        imputer_num = SimpleImputer(strategy="median")
        df_cleaned[num_cols] = imputer_num.fit_transform(df_cleaned[num_cols])
        print(f"[clean_data] Imputasi MEDIAN untuk kolom numerik: {num_cols}")

    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df_cleaned[cat_cols] = imputer_cat.fit_transform(df_cleaned[cat_cols])
        print(f"[clean_data] Imputasi MODUS untuk kolom kategorikal: {cat_cols}")

    sisa_missing = df_cleaned.isnull().sum().sum()
    print(f"[clean_data] Total missing values tersisa: {sisa_missing}")

    return df_cleaned


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding dan standarisasi:
      - Label Encoding untuk kolom kategorikal
      - StandardScaler untuk seluruh fitur (kecuali kolom target)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang sudah bersih (output dari clean_data).

    Returns
    -------
    pd.DataFrame
        DataFrame siap latih — semua kolom numerik dan terstandarisasi.
    """
    data_ready = df.copy()

    # ── Step 4: Label Encoding ──
    cat_cols = data_ready.select_dtypes(exclude=[np.number]).columns.tolist()
    if TARGET_COLUMN in cat_cols:
        cat_cols.remove(TARGET_COLUMN)

    le = LabelEncoder()
    for col in cat_cols:
        data_ready[col] = le.fit_transform(data_ready[col].astype(str))
    print(f"[preprocess_data] Label Encoding selesai untuk: {cat_cols}")

    # ── Step 5: Standarisasi (StandardScaler) ──
    features_to_scale = [col for col in data_ready.columns if col != TARGET_COLUMN]
    scaler = StandardScaler()
    data_ready[features_to_scale] = scaler.fit_transform(data_ready[features_to_scale])
    print(f"[preprocess_data] Standarisasi selesai → {len(features_to_scale)} fitur diskala")

    return data_ready


def save_result(df: pd.DataFrame, output_path: str) -> None:
    """
    Menyimpan DataFrame hasil preprocessing ke file CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset yang sudah diproses.
    output_path : str
        Path tujuan file CSV output.
    """
    df.to_csv(output_path, index=False)
    print(f"[save_result] Hasil disimpan ke: {os.path.abspath(output_path)}")
    print(f"[save_result] Dimensi akhir: {df.shape[0]} baris x {df.shape[1]} kolom")


def automate_preprocessing(source: str = DATASET_URL,
                            output_path: str = OUTPUT_FILENAME) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis:
      load_data → clean_data → preprocess_data → save_result

    Parameters
    ----------
    source : str
        URL atau path file dataset mentah.
    output_path : str
        Path file CSV output.

    Returns
    -------
    pd.DataFrame
        Dataset yang sudah siap untuk pelatihan model.
    """
    print("=" * 55)
    print("  AUTOMATE PREPROCESSING — DamarPrayogaSetiawan")
    print("=" * 55)

    # Pipeline
    df_raw      = load_data(source)
    df_cleaned  = clean_data(df_raw)
    df_ready    = preprocess_data(df_cleaned)
    save_result(df_ready, output_path)

    print("\n" + "=" * 55)
    print("  PREPROCESSING SELESAI!")
    print("=" * 55)
    print(df_ready.head())
    return df_ready


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Jika argumen pertama diberikan, gunakan sebagai source dataset
    source = sys.argv[1] if len(sys.argv) > 1 else DATASET_URL

    # Output ke folder yang sama dengan script ini
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILENAME)

    automate_preprocessing(source=source, output_path=output_path)
