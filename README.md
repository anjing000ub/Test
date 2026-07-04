# DupliGuard

Sistem profesional untuk mendeteksi dan mengelola file duplikat di Google Drive.

DupliGuard terdiri dari dua komponen Python terpisah yang bekerja bersama:

- DGV (DupliGuard Vision) — scanner & reporter: mendeteksi duplikat gambar dan video menggunakan teknik hashing dan analisis visual.
- DGE (DupliGuard Evictor) — executor: mengeksekusi rekomendasi laporan secara aman dengan cara non-destruktif (mengeluarkan file dari folder, bukan menghapus).

Ringkasan singkat
------------------
DGV melakukan deteksi melalui beberapa tingkatan: exact match (BLAKE3), perceptual hashing (pHash/dHash), analisis warna dan fitur lokal, serta SSIM sebagai keputusan akhir. Untuk video, DGV mensampling frame dan membandingkan frame-level hashes.

DGE membaca laporan yang dihasilkan oleh DGV dan mengeksekusi tindakan yang aman: mengeluarkan file dari folder target menggunakan Drive API (`files.update` dengan `removeParents`) tanpa memindahkan atau menghapus file dari Drive pemilik.

Fitur utama
-----------
- Deteksi level byte (BLAKE3) untuk duplikat persis.
- Perceptual hashing (pHash/dHash) untuk pra-filter visual.
- Analisis multi-gate: aspect ratio, histogram warna, color-grid per-region, blur & edge detection, SSIM.
- Sampling frame untuk deteksi duplikat video.
- Laporan terstruktur (TXT + PDF) dengan thumbnail dan metadata.
- Operasi idempoten dan checkpoint untuk resume aman.
- Audit log dan rollback manual — semua tindakan tercatat.
- Throttling adaptif & retry cerdas untuk aman terhadap rate-limit Drive API.

Struktur repo & file utama
--------------------------
- DGV.py       — scanner & reporter (jalankan untuk membuat laporan)
- DGEvictor.py — executor (jalankan untuk menerapkan laporan)
- README.md    — (dokumen ini)

Quick start (Google Colab)
--------------------------
1. Buka https://colab.research.google.com/ dan buat notebook baru.
2. Upload `DGV.py` atau `DGEvictor.py` ke workspace Colab (drag & drop) atau mount repo/GDrive.
3. Jalankan scanning:

```python
!python DGV.py --folder "My Drive/Photos" --outdir "DupliGuard Vision/My Drive/Photos (ID)/2_laporan/"
```

4. Ikuti instruksi autentikasi Google saat diminta.
5. Saat scan selesai, jika memilih menyimpan laporan akan dibuat file TXT (layak dibaca mesin) dan PDF (ringkasan + thumbnails).

Menerapkan laporan dengan DGE
----------------------------
1. Upload `DGEvictor.py` dan laporan TXT hasil DGV.
2. Jalankan:

```python
!python DGEvictor.py --report "dgv_report_NamaFolder (ID).txt" --mode 1
```

Mode yang tersedia:
- 1 — Terapkan rekomendasi laporan (keluarkan semua duplikat sesuai rekomendasi).
- 2 — Kecualikan file tertentu (berikan ID duplikat yang ingin dipertahankan).
- 3 — Eksekusi manual by ID (masukkan ID file yang ingin dikeluarkan).

DGE selalu meminta konfirmasi ganda sebelum menulis perubahan ke Drive.

Format laporan
--------------
Laporan TXT berisi entri per-file dengan format terstruktur: Original/Caller, Candidate ID, hash (BLAKE3), pHash/dHash, SSIM (jika tersedia), ukuran, path folder, dan rekomendasi (keep/evict). Nama laporan mengikuti pola: `dgv_report_<NamaFolder> (<IDFolder>).txt`.

Arsitektur penyimpanan hasil scan
---------------------------------
DupliGuard Vision/
└── <Nama Folder> (<id>)/
    ├── 1_database/     # database hash dan manifest (persisten)
    ├── 2_laporan/      # TXT + PDF output
    └── 3_cache/        # cache / journal (dapat dibangun ulang)

Keamanan & kebijakan data
-------------------------
- Tidak ada penghapusan permanen: semua tindakan DGE bersifat non-destruktif (mengeluarkan dari parent folder saja).
- Operasi idempoten: menjalankan ulang alat tidak menyebabkan kehilangan data.
- Audit trail lengkap: setiap perubahan dicatat untuk memungkinkan rollback manual.
- Proteksi TOCTOU: DGE memverifikasi parent dan metadata (size/MD5/BLAKE3 opsional) sebelum eksekusi.

Panduan kontribusi
------------------
1. Fork repo dan buat branch fitur: `git checkout -b feat/your-feature`.
2. Tambahkan test dan dokumentasi singkat.
3. Buat PR dengan deskripsi perubahan dan contoh penggunaan.

Lisensi
-------
Lisensi tidak ditentukan dalam repo. Jika Anda ingin menambahkan lisensi, tambahkan file `LICENSE` (mis. MIT atau Apache-2.0).

Catatan terakhir
---------------
README ini dirancang agar ringkas, rapi, dan mudah diikuti oleh pengguna yang menjalankan DGV/DGE di Google Colab atau lingkungan lokal Python. Jika mau, saya bisa:
- Menambahkan contoh output laporan (potongan TXT/PDF),
- Menyertakan badge CI atau status, atau
- Menambahkan instruksi instalasi dependency (requirements.txt) yang sesuai kode DGV/DGE di repo.
