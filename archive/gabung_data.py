import pandas as pd
import os

# 1. Daftar nama file CSV Anda
file_csv = [
    "dataset/triwulan_1_2025.csv",
    "dataset/triwulan_2_2025.csv",
    "dataset/triwulan_3_2025.csv",
    "dataset/triwulan_4_2025.csv",
    "dataset/triwulan_1_2026.csv",
    "dataset/triwulan_2_2026.csv"
]

dataframes = []

# 2. Membaca dan menggabungkan semua file CSV
for file in file_csv:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"Berhasil memuat: {file} ({len(df)} baris)")
    else:
        print(f"ERROR: File '{file}' tidak ditemukan! Cek kembali namanya.")

# 3. Menyatukan semua data menjadi satu file CSV master
if dataframes:
    df_master = pd.concat(dataframes, ignore_index=True)
    file_output = "dataset_pmi_master.csv"
    
    # Menyimpan ke format CSV
    df_master.to_csv(file_output, index=False)
    
    print(f"\nPenggabungan sukses! Total data: {len(df_master)} ulasan.")
    print(f"File berhasil disimpan sebagai: {file_output}")
else:
    print("\nTidak ada file yang digabungkan. Silakan periksa folder Anda.")