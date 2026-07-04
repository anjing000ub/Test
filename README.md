# DupliGuard

> Sistem deteksi dan manajemen duplikat file untuk Google Drive

DupliGuard adalah sistem komprehensif untuk mendeteksi dan mengelola file duplikat di Google Drive. Terdiri dari dua komponen terintegrasi yang bekerja bersama untuk memberikan manajemen duplikat yang aman, efisien, dan reversibel.

---

## Ringkasan

### DGV (DupliGuard Vision)
**Scanner & Pelapor** - Mendeteksi duplikat foto dan video di Google Drive menggunakan algoritma hashing lanjutan dan pencocokan perseptual.

### DGE (DupliGuard Evictor)
**Eksekutor** - Mengeluarkan file duplikat dari folder secara aman berdasarkan laporan DGV tanpa menghapus file.

---

## Fitur

### DGV (DupliGuard Vision)

**Kemampuan Deteksi:**
- **Exact Match**: Hashing BLAKE3 untuk deteksi duplikat level byte
- **Visual Match**: Perceptual hashing (pHash, dHash H/V) dengan analisis color grid, histogram, aspect ratio, blur detection, edge detection, blockiness JPEG, dan SSIM
- **Video Match**: Sampling frame dengan durasi gating untuk duplikat video

**Fitur Keandalan:**
- Pemulihan otomatis setelah crash dengan journal LMDB
- Rate limiting adaptif dengan circuit breaker untuk Drive API
- Verifikasi integritas unduhan (BLAKE3/MD5/SHA-256/size)
- Laporan TXT dan PDF komprehensif dengan thumbnail

### DGE (DupliGuard Evictor)

**Fitur Keamanan:**
- Penghapusan non-destruktif (file dikeluarkan dari folder, bukan dihapus)
- Operasi reversibel - file tetap ada di My Drive pemilik
- Mode interaktif untuk batch kecil, mode batch untuk jutaan file
- Token-bucket throttling proaktif untuk mencegah rate limit
- Retry cerdas dengan backoff dan klasifikasi error
- Log audit persisten dan reversibel untuk rollback manual
- Sistem checkpoint untuk resume aman setelah interrupt
- Operasi idempoten - aman dijalankan ulang

**Prinsip Utama:**
1. Hanya mengeluarkan file dari folder menggunakan `files.update + removeParents`
2. Tidak pernah menghapus file atau memindah ke trash
3. Semua tindakan reversibel - tidak ada kehilangan data

---

## Cara Penggunaan

1. **Buka Google Colab**
   - Kunjungi: https://colab.research.google.com/
   - Klik "New Notebook" untuk membuat notebook baru

2. **Upload File**
   - Upload `DGV.py` atau `DGEvictor.py` ke Colab (drag & drop ke panel file di sebelah kiri)
   - Atau gunakan menu: `File` → `Upload notebook`

3. **Jalankan Script**
   - Buat cell baru di notebook
   - Jalankan dengan perintah:
     ```python
     !python DGV.py
     ```
     atau
     ```python
     %run DGEvictor.py
     ```
   - Klik tombol "Play" (▶) atau tekan `Shift + Enter`

4. **Autentikasi**
   - Saat diminta, klik link autentikasi Google
   - Copy verification code dan paste ke kolom yang tersedia
   - Tunggu proses mounting Google Drive selesai

5. **Scan dengan DGV**
   - Jalankan `DGV.py` di Google Colab
   - Pilih folder Drive yang ingin di-scan (untuk folder yang dibagikan orang lain, buat pintasan di My Drive agar bisa di-scan)
   - Tunggu proses scanning selesai
   - Jika duplikat ditemukan, akan muncul opsi "Simpan laporan? (y/n)"
   - Pilih 'y' untuk membuat laporan (TXT + PDF) di folder `DupliGuard Vision/<Nama Folder> (ID Folder)/2_laporan/`
   - Pilih 'n' untuk tidak membuat laporan

6. **Eksekusi dengan DGE**
   - Jalankan `DGEvictor.py` di Google Colab
   - Masukkan atau salin nama file laporan TXT (contoh: `dgv_report_Photos (1ABC2DEF3GHI).txt`)
   - Review informasi laporan (folder, jumlah duplikat, ukuran)
   - Pilih mode eksekusi:
     - **[1]** Terapkan rekomendasi laporan (keluarkan semua duplikat sesuai laporan)
     - **[2]** Kecualikan file pilihan (masukkan ID duplikat yang ingin DIPERTAHANKAN - hanya ID duplikat yang diterima, file asli otomatis dilindungi)
     - **[3]** Pengeluaran manual by ID (masukkan ID file yang ingin DIKELUARKAN, bisa asli atau duplikat)
   - Konfirmasi dua kali sebelum eksekusi

---

## Alur Kerja Deteksi Duplikat

### Proses Analisis DGV

```
┌──────────────────────────────────────────────────────────────┐
│                      FILE DRIVE                              │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│             [1] EXACT MATCH (BLAKE3)                         │
│             Cek identik byte per byte                        │
└──────────────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────┐
        ↓                                 ↓
    Identik?                         Tidak Identik
        ↓                                 ↓
┌───────────────┐          ┌──────────────────────────────────┐
│ DUPLIKAT      │          │ [2] VISUAL MATCH - FOTO          │
│ EXACT         │          └──────────────┬───────────────────┘
└───────────────┘                         ↓
                           ┌──────────────────────────────────┐
                           │ 2.1 pHash/dHash H/V (pra-filter) │
                           └──────────────┬───────────────────┘
                                          ↓
                           ┌──────────────────────────────────┐
                           │ 2.2 Aspect Ratio Gate            │
                           └──────────────┬───────────────────┘
                                          ↓
                           ┌──────────────────────────────────┐
                           │ 2.3 Histogram Warna Global       │
                           └──────────────┬───────────────────┘
                                          ↓
                           ┌──────────────────────────────────┐
                           │ 2.4 Color Grid (per-region)      │
                           └──────────────┬───────────────────┘
                                          ↓
                           ┌──────────────────────────────────┐
                           │ 2.5 Blur Detection (per-region)  │
                           └──────────────┬───────────────────┘
                                          ↓
                           ┌──────────────────────────────────┐
                           │ 2.6 Edge Detection (per-region)  │
                           └──────────────┬───────────────────┘
                                          ↓
                           ┌──────────────────────────────────┐
                           │ 2.7 SSIM (final decision)        │
                           └──────────────┬───────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              ↓                       ↓
                          SSIM > 0.94             SSIM ≤ 0.94
                              ↓                       ↓
                      ┌──────────────┐        ┌──────────────┐
                      │ DUPLIKAT     │        │ BUKAN        │
                      │ VISUAL       │        │ DUPLIKAT     │
                      └──────────────┘        └──────────────┘

┌──────────────────────────────────────────────────────────────┐
│             [3] VISUAL MATCH - VIDEO                         │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 3.1 Sampling Frame Grid                                      │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 3.2 Durasi Gate (toleransi 3%)                               │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 3.3 pHash/dHash per Frame                                    │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 3.4 Frame Match Ratio > 97%                                  │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
                    ┌──────────────┐
                    │ DUPLIKAT     │
                    │ VIDEO        │
                    └──────────────┘
```

**Penjelasan Gerbang:**
- **Exact Match**: Hashing BLAKE3 untuk deteksi duplikat level byte (file identik persis)
- **pHash/dHash H/V**: Perceptual hashing untuk pra-filter cepat kandidat visual
- **Aspect Ratio**: Filter awal untuk menolak crop/rotasi
- **Histogram**: Korelasi warna global untuk menolak filter warna/B&W/sepia
- **Color Grid**: Analisis warna per-blok untuk mendeteksi perubahan lokal
- **Blur Detection**: Mendeteksi blur editan/sensor (wajah, plat nomor)
- **Edge Detection**: Mendeteksi stiker/teks/watermark kecil
- **SSIM**: Pemutus akhir untuk memastikan kesamaan struktural

---

## Struktur Penyimpanan

```
DupliGuard Vision/
└── <Nama Folder> (<id>)/
    ├── 1_database/     # hash_<id>.mdb + manifest_<id>.mdb (permanen)
    ├── 2_laporan/      # TXT + PDF (output scan)
    └── 3_cache/        # journal_<id>.mdb (bisa dibuat ulang)
```

---

## Keamanan

- **Tidak ada penghapusan permanen** - file hanya dikeluarkan dari folder
- **Operasi idempoten** - aman dijalankan ulang
- **Audit trail** - semua tindakan tercatat untuk review
- **Sistem checkpoint** - resume aman setelah interrupt
- **Proteksi TOCTOU** - verifikasi parent sebelum eksekusi
- **Verifikasi anti-stale** - cek size + MD5 (opsional BLAKE3)
