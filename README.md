# DupliGuard

Sistem deteksi dan manajemen duplikat file untuk Google Drive.

Singkat:
- DGV (DupliGuard Vision): scanner & reporter — deteksi duplikat foto/video (BLAKE3 + perceptual hashing + checks).
- DGE (DupliGuard Evictor): executor — terapkan rekomendasi laporan secara non-destruktif (removeParents).

Quick start singkat:
- Scan (Colab/CLI):
  ```bash
  python DGV.py --folder "My Drive/Photos" --outdir "DupliGuard Vision/Photos (ID)/2_laporan/" --save-report
  ```
- Apply (dry-run dulu):
  ```bash
  python DGEvictor.py --report "dgv_report_Photos (ID).txt" --mode 1 --dry-run
  ```
- Jika hasil sesuai, jalankan tanpa --dry-run.

Format laporan: TXT (struktur: original_id, candidate_id, blake3, pHash, ssim, size, path, rekomendasi).

Catatan:
- Tidak ada diagram di README ini. Saya ringkas semua informasi agar jelas dan rapi.
- Lisensi belum ditentukan. Jika mau, beri tahu lisensi yang diinginkan (MIT / Apache-2.0 / GPL-3.0) dan saya tambahkan file LICENSE.
