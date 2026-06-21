import os
import pandas as pd

# 1. Tentukan format nama file dan buat list untuk menampung dataframe
all_dataframes = []
output_combined_file = 'data/raw/soundings_96749_2021_new.xlsx'

print("Memulai proses penggabungan file Excel...")

# 2. Lakukan perulangan dari tahun 2020 sampai 2024
for month in range(1, 13):
    month_str = f"{month:02d}"  # Format bulan dengan dua digit
    file_name = f"sounding_indices_2021_{month_str}.xlsx"
    
    # 3. Periksa apakah file tersebut ada di dalam folder
    if os.path.exists(file_name):
        print(f"-> Membaca dan memproses: {file_name}")
        # Baca file Excel
        df = pd.read_excel(file_name)
        # Masukkan ke dalam list
        all_dataframes.append(df)
    else:
        print(f"-> [Lewat] File {file_name} tidak ditemukan.")

# 4. Gabungkan semua dataframe yang berhasil dikumpulkan
if all_dataframes:
    # ignore_index=True membuat nomor baris/indeks diurutkan ulang secara kontinu (0, 1, 2, dst)
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 5. Simpan hasil gabungan ke file Excel baru
    combined_df.to_excel(output_combined_file, index=False)
    
    print("\n" + "="*50)
    print(f"SUKSES! Semua file berhasil digabungkan.")
    print(f"File output rapi: {output_combined_file}")
    print(f"Total seluruh data: {len(combined_df)} baris.")
    print("="*50)
else:
    print("\n[Gagal] Tidak ada file Excel yang ditemukan untuk digabungkan.")