# app.py

import pickle
from pathlib import Path
import sys
import subprocess


def find_model_file():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "model" / "model_emosi.pkl",
        script_dir.parent / "model" / "model_emosi.pkl",
        Path.cwd() / "model" / "model_emosi.pkl",
        Path("model/model_emosi.pkl"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_model():
    p = Path(__file__).resolve().parent / "model" / "model_emosi.pkl"
    if not p.exists():
        p = Path("model/model_emosi.pkl")
    if not p.exists():
        raise FileNotFoundError("Model tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        raise RuntimeError("File model tampak korup. Hapus dan jalankan 'train_model.py' untuk membuat ulang model.")


def main():
    model, vectorizer = load_model()
    print("=== APLIKASI DETEKSI EMOSI ===")
    try:
        while True:
            s = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
            if s.strip().lower() == "exit":
                print("Keluar.")
                break
            print("Emosi terdeteksi:", model.predict(vectorizer.transform([s]))[0])
    except (KeyboardInterrupt, EOFError):
        print("\nKeluar.")
        sys.exit(0)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()