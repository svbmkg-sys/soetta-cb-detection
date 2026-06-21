import re
import pandas as pd
from datetime import datetime

# 1. Tentukan nama file HTML input dan file Excel output
bulan = '01'  # Ganti dengan bulan yang diinginkan (format MM)
html_file_path = f'sounding_2021_{bulan}.html'
excel_file_path = f'sounding_indices_2021_{bulan}.xlsx'

# 2. Baca isi keseluruhan file HTML
with open(html_file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# 3. Cari seluruh blok teks yang berada di dalam tag <PRE> ... </PRE>
# Menggunakan re.IGNORECASE karena tag HTML bisa berupa huruf besar maupun kecil
pre_blocks = re.findall(r'<pre>(.*?)</pre>', html_content, re.DOTALL | re.IGNORECASE)

# 4. Definisikan urutan kolom persis seperti yang Anda minta
columns_order = [
    'Station identifier', 'Station number', 'Observation time', 'Station latitude',
    'Station longitude', 'Station elevation', 'Showalter index', 'Lifted index',
    'LIFT computed using virtual temperature', 'SWEAT index', 'K index',
    'Cross totals index', 'Vertical totals index', 'Totals totals index',
    'Convective Available Potential Energy', 'CAPE using virtual temperature',
    'Convective Inhibition', 'CINS using virtual temperature', 'Equilibrum Level',
    'Equilibrum Level using virtual temperature', 'Level of Free Convection',
    'LFCT using virtual temperature', 'Bulk Richardson Number',
    'Bulk Richardson Number using CAPV', 'Temp [K] of the Lifted Condensation Level',
    'Pres [hPa] of the Lifted Condensation Level', 'Equivalent potential temp [K] of the LCL',
    'Mean mixed layer potential temperature', 'Mean mixed layer mixing ratio',
    '1000 hPa to 500 hPa thickness', 'Precipitable water [mm] for entire sounding',
    'Tanggal', 'Jam'
]

data_list = []

# 5. Iterasi dan filter setiap blok PRE untuk mengambil parameter stasiun
for block in pre_blocks:
    # Memastikan blok PRE ini berisi informasi stasiun & indeks (bukan tabel data mentah atasnya)
    if 'Station identifier:' in block:
        row_dict = {}
        
        # Ekstrak key dan value secara line-by-line menggunakan regex
        # Pola ini memisahkan teks sebelum titik dua (:) sebagai Key dan setelahnya sebagai Value
        matches = re.findall(r'^\s*([^:]+):\s*(.*)$', block, re.MULTILINE)
        
        for key, val in matches:
            key_clean = key.strip()
            val_clean = val.strip()
            
            if key_clean in columns_order:
                # Konversi otomatis teks angka menjadi tipe numerik (float/int) agar rapi di Excel
                try:
                    if '.' in val_clean:
                        row_dict[key_clean] = float(val_clean)
                    else:
                        row_dict[key_clean] = int(val_clean)
                except ValueError:
                    row_dict[key_clean] = val_clean
        
        # 6. Pecah 'Observation time' (Format Wyoming: YYMMDD/HHMM) menjadi 'Tanggal' dan 'Jam'
        if 'Observation time' in row_dict:
            try:
                obs_time = str(row_dict['Observation time']).strip()
                dt = datetime.strptime(obs_time, "%y%m%d/%H%M")
                row_dict['Tanggal'] = dt.strftime("%Y-%m-%d")  # Hasil: YYYY-MM-DD
                row_dict['Jam'] = dt.strftime("%H:%M")          # Hasil: HH:MM
            except Exception:
                row_dict['Tanggal'] = None
                row_dict['Jam'] = None
        
        # Masukkan dict baris ke list utama jika berhasil terisi
        if row_dict:
            data_list.append(row_dict)

# 7. Membuat DataFrame Pandas dan mengurutkan kolom sesuai request Anda
df = pd.DataFrame(data_list, columns=columns_order)

# 8. Ekspor DataFrame langsung menjadi file Excel (.xlsx)
df.to_excel(excel_file_path, index=False)

print(f"Selesai! Berhasil mengekstrak {len(df)} data sounding ke dalam file '{excel_file_path}'.")