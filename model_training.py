from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "dataset" / "dataset_pmi_clean.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "nb_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"


def load_training_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {input_file}")

    dataframe = pd.read_csv(input_file)
    required_columns = {"ulasan_bersih", "sentimen"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    dataframe["ulasan_bersih"] = dataframe["ulasan_bersih"].fillna("").astype(str)
    dataframe["ulasan_bersih"] = dataframe["ulasan_bersih"].str.strip()
    dataframe = dataframe[dataframe["ulasan_bersih"].ne("")].copy()
    dataframe = dataframe.dropna(subset=["sentimen"])

    if dataframe.empty:
        raise ValueError("Tidak ada data dengan ulasan_bersih yang dapat dilatih.")

    return dataframe


def train_and_evaluate(dataframe: pd.DataFrame) -> tuple[MultinomialNB, TfidfVectorizer]:
    texts = dataframe["ulasan_bersih"]
    labels = dataframe["sentimen"]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_tfidf, y_train)

    model = MultinomialNB()
    model.fit(x_train_balanced, y_train_balanced)
    predictions = model.predict(x_test_tfidf)

    class_labels = sorted(labels.unique())
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    print("Confusion Matrix (baris = aktual, kolom = prediksi):")
    print(pd.DataFrame(matrix, index=class_labels, columns=class_labels))
    print(f"\nAccuracy : {accuracy_score(y_test, predictions):.4f}")
    print(
        "Precision: "
        f"{precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        "Recall   : "
        f"{recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        "F1-Score : "
        f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print("\nDistribusi kelas training setelah SMOTE:")
    print(pd.Series(y_train_balanced).value_counts().sort_index().to_string())

    return model, vectorizer


def save_artifacts(
    model: MultinomialNB,
    vectorizer: TfidfVectorizer,
    model_dir: Path = MODEL_DIR,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / MODEL_FILE.name)
    joblib.dump(vectorizer, model_dir / VECTORIZER_FILE.name)
    print(f"\nModel tersimpan: {model_dir / MODEL_FILE.name}")
    print(f"Vectorizer tersimpan: {model_dir / VECTORIZER_FILE.name}")


def main() -> None:
    dataframe = load_training_data()
    print(f"Data siap dilatih: {len(dataframe)} baris")
    model, vectorizer = train_and_evaluate(dataframe)
    save_artifacts(model, vectorizer)


if __name__ == "__main__":
    main()