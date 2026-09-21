import pandas as pd
import os

# Daftar file Anda
file_csv = [
    "dataset/triwulan_1_2025.csv",
    "dataset/triwulan_2_2025.csv",
    "dataset/triwulan_3_2025.csv",
    "dataset/triwulan_4_2025.csv",
    "dataset/triwulan_1_2026.csv",
    "dataset/triwulan_2_2026.csv"
]

# Target jumlah data berdasarkan grafik di BAB I
target_data = [185, 194, 203, 211, 198, 214]

print("=== HASIL PENGECEKAN DATA ===")
total_sekarang = 0

for file, target in zip(file_csv, target_data):
    if os.path.exists(file):
        df = pd.read_csv(file)
        jumlah = len(df)
        total_sekarang += jumlah
        
        if jumlah == target:
            status = "✅ AMAN"
        elif jumlah < target:
            status = f"❌ KURANG {target - jumlah} ulasan"
        else:
            status = f"⚠️ LEBIH {jumlah - target} ulasan"
            
        print(f"{file} : {jumlah} (Target: {target}) -> {status}")
    else:
        print(f"{file} : File tidak ditemukan!")

print("-" * 40)
print(f"Total Keseluruhan : {total_sekarang} ulasan (Target: 1205)")