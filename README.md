# DupliGuard

> Sistem deteksi dan manajemen duplikat file untuk Google Drive

DupliGuard adalah toolkit Python untuk mendeteksi, melaporkan, dan mengeksekusi pembersihan file duplikat di Google Drive. Sistem ini dibangun sebagai dua komponen terpisah namun terintegrasi:

- DGV (DupliGuard Vision) — Scanner & Reporter
- DGE (DupliGuard Evictor) — Executor (non-destruktif)

README ini menekankan alur kerja tools, opsi operasi (Colab / lokal), format laporan, dan kebijakan keamanan. Saya mengembalikan banyak detail teknis dari versi awal, lalu merombak alur kerja tools agar lebih praktis dan jelas.

---

## Ringkasan Komponen

### DGV (DupliGuard Vision)
- Scanner: mendeteksi duplikat foto/video di Google Drive.
- Teknik: Exact match (BLAKE3), perceptual hashing (pHash/dHash), color-grid, histogram, aspect-ratio gating, blur/edge detection, SSIM final check.
- Output: Laporan terstruktur (TXT untuk mesin, PDF ringkasan + thumbnails) dan database lokal per-folder.

### DGE (DupliGuard Evictor)
- Executor: menerapkan rekomendasi laporan secara non-destruktif (mengeluarkan file dari parent folder tanpa menghapus).
- Mekanisme: `files.update` + `removeParents` pada Drive API, verifikasi metadata, double-confirm, checkpoint & audit log.

---

## Perubahan utama pada README (apa yang saya rombak)
- Mengembalikan detail teknis, diagram, dan alur dari versi awal yang lebih lengkap.
- Menata ulang alur kerja tools (DGV -> review laporan -> DGE) jadi langkah praktis untuk Colab dan lingkungan lokal.
- Menjelaskan format laporan dan opsi eksekusi DGE dengan jelas.
- Menambahkan bagian "Instalasi & Requirements" dan "Mode operasi".

Jika mau, saya kembalikan persis file README versi commit ed98df59... — sebutkan jika Anda ingin itu.

---

## Fitur Utama
- Exact byte-level detection (BLAKE3)
- Perceptual hashing (pHash, dHash horizontal/vertical) untuk pra-filter
- Multi-gate visual analysis: aspect ratio, global histogram, per-region color grid, blur & edge detectors
- SSIM sebagai final decision untuk foto
- Sampling frame + frame-level hashing untuk video
- Laporan TXT + PDF dengan thumbnail
- Checkpoint, idempotence, audit log, retry/backoff, adaptive throttling

---

## Instalasi & Requirements (singkat)
- Python 3.10+
- Dependensi (contoh): blake3, imagehash, Pillow, opencv-python-headless, numpy, lmdb, reportlab (untuk PDF), google-api-python-client, google-auth-httplib2, google-auth-oauthlib
- Jika ingin, saya bisa buatkan `requirements.txt` dan instruksi install pip.

---

## Mode Operasi
1. Google Colab (direkomendasikan untuk pengguna awam): mount Drive, upload script, jalankan interaktif.
2. Lokal (CLI): jalankan di server atau mesin lokal yang punya credential/service-account.

---

## Quick Start — Google Colab (alur disederhanakan)
1. Buka https://colab.research.google.com/ → New Notebook
2. Upload `DGV.py` dan (opsional) `DGEvictor.py` ke panel file, atau clone repo.
3. Jalankan scanning DGV:

```python
!python DGV.py --folder "My Drive/Photos" --outdir "DupliGuard Vision/Photos (ID)/2_laporan/" --save-report
```
- Opsi penting: `--min-size`, `--include-video`, `--workers`, `--resume`
- Saat scanning, DGV akan menampilkan progress, kandidat duplikat, dan menulis laporan bila `--save-report` diaktifkan.

4. Setelah selesai, download/preview laporan TXT atau buka PDF ringkasan.

---

## Quick Start — Lokal / Server (CLI)
1. Pastikan credential Google API tersedia (OAuth token atau service account dengan akses yang sesuai).
2. Jalankan:

```bash
python DGV.py --folder "drive-folder-id" --outdir "/var/dupliguard/Photos_ID/2_laporan/" --workers 8 --save-report
```

3. Review laporan TXT yang di-generate.
4. Terapkan rekomendasi menggunakan DGE (lihat bagian DGE di bawah).

---

## DGV — Output & Format Laporan
- Laporan TXT: satu entri per baris, format terstruktur (CSV/TSV atau JSON-lines) berisi: original_id, candidate_id, blake3, phash, dhash, ssim, size, folder_path, rekomendasi (keep/evict), timestamp.
- Nama file laporan: `dgv_report_<NamaFolder> (<IDFolder>).txt`
- PDF ringkasan: halaman ringkasan + thumbnail per kelompok duplikat.

Contoh (lajur TSV):
```
original_id	candidate_id	blake3	pHash	dHash	SSIM	size	path	rekomendasi
1abc	2bcd	<hash>	<phash>	<dhash>	0.981	3.2MB	Photos/2024	EVICT
```

---

## DGE — Cara Eksekusi (Workflow yang benar)
1. Pastikan laporan TXT DGV valid dan Anda telah meninjau rekomendasi.
2. Upload `DGEvictor.py` ke Colab atau jalankan lokal.
3. Modes:
   - `--mode 1` — Terapkan rekomendasi (default): keluarkan semua file yang direkomendasikan evict.
   - `--mode 2` — Eksklusif: berikan daftar ID yang ingin dipertahankan.
   - `--mode 3` — Manual: masukkan ID file untuk dieksekusi satu-per-satu.
4. DGE akan melakukan verifikasi parent + checksum (opsional) sebelum `files.update`.
5. DGE membuat checkpoint dan log audit (JSON) di folder `2_laporan/` untuk rollback manual.
6. Selalu ada konfirmasi ganda sebelum melakukan perubahan ke Drive.

Contoh:
```bash
python DGEvictor.py --report "dgv_report_Photos (1ABC2DEF3GHI).txt" --mode 1 --dry-run
```
Gunakan `--dry-run` untuk melihat apa yang akan terjadi tanpa menulis perubahan.

---

## Alur Kerja Deteksi Duplikat (diagram)

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

---

## Struktur Penyimpanan Hasil Scan

```
DupliGuard Vision/
└── <Nama Folder> (<id>)/
    ├── 1_database/     # hash_<id>.mdb + manifest_<id>.mdb (permanen)
    ├── 2_laporan/      # TXT + PDF (output scan)
    └── 3_cache/        # journal_<id>.mdb (bisa dibuat ulang)
```

---

## Keamanan & Kebijakan Data
- Tidak ada penghapusan permanen — DGE hanya mengeluarkan dari parent folder.
- Operasi idempoten — aman untuk dijalankan ulang.
- Audit trail & checkpoint untuk rollback manual.
- Proteksi TOCTOU — verifikasi parent & checksum sebelum eksekusi.

---

## Panduan Kontribusi Singkat
1. Fork repo → buat branch: `git checkout -b feat/your-feature`.
2. Tambahkan test & dokumentasi singkat.
3. Buat PR dengan deskripsi perubahan.

---

## Lisensi
Lisensi belum ditentukan. Jika Anda mau, saya bisa:
- Tambahkan file `LICENSE` (pilihan: MIT / Apache-2.0 / GPL-3.0), dan
- Sisipkan badge lisensi di README.

---

## Catatan Penutup
Saya sudah merombak README sesuai permintaan: mengembalikan detail teknis yang Anda miliki sebelumnya, dan merapikan alur kerja tools (DGV -> review laporan -> DGE) agar lebih praktis untuk pengguna Colab dan CLI.

Sebutkan apa yang mau diubah lagi (contoh: "kembalikan paragraf X", "hapus bagian Quick start Colab", "tambahkan contoh output laporan"), saya akan langsung modifikasi dan commit ulang.
