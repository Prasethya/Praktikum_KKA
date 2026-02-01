# train_model.py
# Praktikum Deteksi Emosi dari Teks
# Pesantren Teknologi Majapahit

import os
from pathlib import Path
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def main():
    # Direktori file ini (agar path relatif tidak tergantung current working dir)
    script_dir = Path(__file__).resolve().parent

    # Cari file dataset (prioritaskan 'emotion.csv', fallback ke 'cth-emotion.csv')
    possible_files = [script_dir / "emotion.csv"]
    csv_path = None
    for p in possible_files:
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(f"Tidak menemukan 'emotion.csv'{script_dir}")

    # Load dataset
    data = pd.read_csv(csv_path)

    # Validasi kolom dataset
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Dataset harus memiliki kolom 'text' dan 'label'")

    # Hapus baris yang punya nilai kosong pada kolom penting
    data = data[["text", "label"]].dropna()
    if data.empty:
        raise ValueError("Dataset kosong setelah menghapus baris dengan nilai kosong pada 'text' atau 'label'")

    # Pastikan tipe yang sesuai
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str)

    if data["label"].nunique() < 2:
        raise ValueError("Dataset harus memiliki minimal 2 kelas pada kolom 'label'")

        # Jika ada file tambahan (contoh), gabungkan untuk menambah data
    extra = script_dir / "emotion_50.csv"
    if extra.exists():
        extra_df = pd.read_csv(extra)
        if "text" in extra_df.columns and "label" in extra_df.columns:
            data = pd.concat([data, extra_df[["text","label"]]], ignore_index=True)

    # Preprocessing sederhana: lowercase + hapus tanda baca
    import re
    def clean(s):
        s = str(s).lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    data["text"] = data["text"].astype(str).map(clean)
    y = data["label"].astype(str)

    # Vektorisasi: gunakan karakter n-gram (char_wb) agar lebih tahan kata baru
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=5000)
    X = vectorizer.fit_transform(data["text"])

    # Split (stratify jika memungkinkan)
    stratify = y if y.nunique() > 1 and len(y) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify, shuffle=True
    )

    # Gunakan model yang lebih robust untuk teks kecil
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Simpan model dan vectorizer
    model_dir = script_dir / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model_emosi.pkl"
    with open(model_path, "wb") as f:
        pickle.dump((model, vectorizer), f)

    print(f"Model tersimpan: {model_path}")

if __name__ == "__main__":
    main()