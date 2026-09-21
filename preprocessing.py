from pathlib import Path
import re

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory,
)


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "dataset" / "dataset_pmi_master.csv"
OUTPUT_FILE = PROJECT_DIR / "dataset" / "dataset_pmi_clean.csv"
REQUIRED_COLUMNS = {"tanggal", "ulasan", "sentimen"}


stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()


def cleaning(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def case_folding(text: str) -> str:
    return text.lower()


def tokenization(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ]+", text)


def filtering(tokens: list[str]) -> list[str]:
    filtered_text = stopword_remover.remove(" ".join(tokens))
    return filtered_text.split()


def stemming(tokens: list[str]) -> str:
    return stemmer.stem(" ".join(tokens))


def preprocess_text(text: str) -> str:
    cleaned_text = cleaning(text)
    folded_text = case_folding(cleaned_text)
    tokens = tokenization(folded_text)
    filtered_tokens = filtering(tokens)
    return stemming(filtered_tokens)


def load_dataset(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {input_file}")

    dataframe = pd.read_csv(input_file)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    dataframe["ulasan"] = dataframe["ulasan"].fillna("").astype(str)
    return dataframe


def save_clean_dataset(
    dataframe: pd.DataFrame,
    output_file: Path = OUTPUT_FILE,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False)


def main() -> None:
    dataframe = load_dataset()
    dataframe["ulasan_bersih"] = dataframe["ulasan"].apply(preprocess_text)
    save_clean_dataset(dataframe)

    print(f"Dataset berhasil diproses: {len(dataframe)} baris")
    print(f"File hasil: {OUTPUT_FILE}")
    print("Distribusi sentimen:")
    print(dataframe["sentimen"].value_counts().to_string())


if __name__ == "__main__":
    main()