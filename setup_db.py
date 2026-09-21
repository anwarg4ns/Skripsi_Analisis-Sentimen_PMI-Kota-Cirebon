import os
from pathlib import Path

import mysql.connector
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
CSV_FILE = PROJECT_DIR / "dataset" / "dataset_pmi_clean.csv"
DATABASE_NAME = "db_pmi_cirebon"
TABLE_NAME = "tabel_ulasan"

DB_CONFIG = {
    "host": os.getenv("PMI_DB_HOST", "localhost"),
    "port": int(os.getenv("PMI_DB_PORT", "3306")),
    "user": os.getenv("PMI_DB_USER", "root"),
    "password": os.getenv("PMI_DB_PASSWORD", ""),
}


def create_database() -> None:
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DATABASE_NAME}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cursor.close()
    connection.close()


def create_table(connection: mysql.connector.MySQLConnection) -> None:
    cursor = connection.cursor()
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tanggal DATE NOT NULL,
            ulasan TEXT NOT NULL,
            sentimen VARCHAR(20) NOT NULL,
            ulasan_bersih TEXT NOT NULL
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """
    )
    connection.commit()
    cursor.close()


def import_dataset(connection: mysql.connector.MySQLConnection) -> int:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {CSV_FILE}")

    dataframe = pd.read_csv(CSV_FILE)
    required_columns = {"tanggal", "ulasan", "sentimen", "ulasan_bersih"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    dataframe = dataframe.dropna(subset=["ulasan", "sentimen", "ulasan_bersih"]).copy()
    dataframe["tanggal"] = pd.to_datetime(dataframe["tanggal"], errors="raise")

    rows = [
        (
            row.tanggal.date(),
            row.ulasan,
            row.sentimen,
            row.ulasan_bersih,
        )
        for row in dataframe.itertuples(index=False)
    ]

    cursor = connection.cursor()
    cursor.execute(f"TRUNCATE TABLE `{TABLE_NAME}`")
    cursor.executemany(
        f"""
        INSERT INTO `{TABLE_NAME}` (tanggal, ulasan, sentimen, ulasan_bersih)
        VALUES (%s, %s, %s, %s)
        """,
        rows,
    )
    connection.commit()
    cursor.close()
    return len(rows)


def main() -> None:
    create_database()
    connection = mysql.connector.connect(**DB_CONFIG, database=DATABASE_NAME)
    try:
        create_table(connection)
        imported_rows = import_dataset(connection)
    finally:
        connection.close()

    print(f"Database siap: {DATABASE_NAME}")
    print(f"Tabel siap: {TABLE_NAME}")
    print(f"Data berhasil diimpor: {imported_rows} baris")


if __name__ == "__main__":
    main()