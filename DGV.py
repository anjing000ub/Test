"""
DupliGuard Vision - Detektor duplikat foto/video di Google Drive.

Cara kerja:
  - Exact match : BLAKE3 (identik byte per byte)
  - Visual match: pHash + dHash + color_grid + SSIM (identik beda resolusi/kompresi)
  - Video match : sampling frame grid + gerbang durasi

Fitur:
  - Resume otomatis setelah crash (journal LMDB)
  - Rate limiter adaptif + circuit breaker untuk Drive API
  - Verifikasi integritas unduhan (BLAKE3/MD5/SHA-256/ukuran)
  - Output laporan TXT + PDF dengan thumbnail, tersimpan di My Drive/DupliGuard Vision/
"""
!pip install weasyprint imageio-ffmpeg blake3 lmdb -q
!apt-get install -qq fonts-noto-core fonts-noto-ui-core > /dev/null 2>&1
!pip install "Pillow>=10.1,<12.0" -q
!pip install imagehash --no-deps -q
!pip install pillow-heif pillow-avif-plugin -q
!pip install scikit-image -q

import os, io, base64, tempfile, time, threading, json, hashlib
import random, shutil, logging, warnings, contextlib, traceback, html
import urllib.request, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any

import lmdb
import blake3
import imagehash
import numpy as np
import imageio

from google.colab import auth, drive
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image, ImageOps
from dateutil import parser as dateparser
from weasyprint import HTML as WeasyHTML
from IPython.display import display, HTML
try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    _SSIM_AVAILABLE = True
except ImportError:
    _skimage_ssim = None
    _SSIM_AVAILABLE = False

# Register opener HEIC/HEIF/AVIF agar bisa di-decode Pillow.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception as _e_heif:
    pass
try:
    import pillow_avif  # noqa: F401  (auto-register AVIF plugin)
except Exception as _e_avif:
    pass

warnings.filterwarnings("ignore")
# imageio / imageio_ffmpeg memakai logger sendiri (bukan modul warnings),
# sehingga filterwarnings di atas tidak meredamnya. Saat memproses video
# portrait/rotated, imageio_ffmpeg memuntahkan WARNING per-frame ("frame size
# ... different from source", "We had to kill ffmpeg") yang membanjiri output
# dan memutus baris progress \r MEMPROSES sehingga tampak menumpuk ke bawah.
# Naikkan levelnya ke ERROR agar konsol tetap bersih dan progress bar menimpa
# satu baris seperti seharusnya.
for _lg in ("googleapiclient", "weasyprint", "google_auth_httplib2", "PIL",
            "imageio", "imageio_ffmpeg"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

# ───────────────────── LOGGER ─────────────────────
# Log hanya ke file, tidak ke console, agar output Colab tetap bersih.
_logger = logging.getLogger("dupliguardvision")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False
if not _logger.handlers:
    _fh = logging.FileHandler("/tmp/dupliguardvision_audit.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_fh)

def _log(level: str, msg: str):
    getattr(_logger, level)(msg)

def _scroll_to_input():
    """Scroll output Colab ke area input agar pengguna tidak perlu scroll manual.
    Bergantung pada DOM Colab (tidak resmi); kegagalan diabaikan diam-diam."""
    try:
        from IPython.display import Javascript, display as _display
        _display(Javascript('''
            (function(){
              try {
                document.querySelectorAll('.output, .output_scroll, .output_area')
                  .forEach(function(el){ el.scrollTop = el.scrollHeight; });
                window.scrollTo(0, document.body.scrollHeight);
                setTimeout(function(){
                  var inputs = document.querySelectorAll('input[type=text]');
                  if (inputs.length) {
                    var last = inputs[inputs.length - 1];
                    last.focus();
                    last.scrollIntoView({block: 'center'});
                  }
                }, 150);
              } catch (e) {}
            })();
        '''))
    except Exception as _e_scroll:
        _log('debug', f"_scroll_to_input gagal (diabaikan): {_e_scroll}")

# ───────────────────── KONSTANTA ─────────────────────
SAVE_INTERVAL       = 25
CONCURRENT_WORKERS  = 8
# Jumlah pending yang diproses per batch (batasi RAM & overhead scheduling).
PROCESS_BATCH_SIZE  = 2000
MAX_RETRY           = 5
RETRY_BACKOFF       = [1, 2, 4, 8, 16]

# Ambang pHash/dHash untuk FOTO. Ini hanya pra-filter cepat; kandidat yang lolos
# masih harus melewati histogram, color_grid, edge, blur, lalu SSIM. Khusus
# foto (video memakai konstanta VIDEO_* terpisah).
PHASH_THRESHOLD     = 20    # jarak pHash maks (dari 256 bit)
DHASH_THRESHOLD     = 9     # jarak dHash H maks (dari 64 bit)
PHASH_HASH_SIZE     = 16    # menghasilkan pHash 256-bit
DHASH_HASH_SIZE     = 8     # menghasilkan dHash 64-bit
DVHASH_THRESHOLD    = 9     # jarak dHash V (gradien atas-bawah) maks (dari 64 bit)
# Gerbang ASPECT RATIO (pra-filter). Foto identik beda resolusi/kompresi punya
# rasio aspek sama; crop/rotasi mengubahnya. Aktif bila kedua record punya
# 'aspect_ratio' (record lama dilewati).
ASPECT_RATIO_GATE       = True
ASPECT_RATIO_MAX_DELTA  = 0.04   # selisih rasio aspek maks
# Ambang dHash KHUSUS VIDEO (terpisah & tetap ketat, dipakai _frame_close).
VIDEO_DHASH_FRAME_THRESHOLD  = 2   # jarak dHash H per frame maks (dari 64 bit)
VIDEO_DVHASH_FRAME_THRESHOLD = 2   # jarak dHash V per frame maks (dari 64 bit)
# Gerbang WARNA PER-REGION (color_grid): menolak filter warna/B&W/sepia/tint
# yang lolos hash grayscale. Foto dibagi grid, tiap blok dicatat rata-rata RGB;
# bila >= COLOR_GRID_MIN_BLOCKS blok berubah warna > COLOR_GRID_MAX_DIST,
# dianggap versi beda. Resize/re-kompresi murni hanya menggeser warna sedikit.
# Gerbang HISTOGRAM warna GLOBAL: foto identik beda resolusi punya korelasi
# histogram ~1.0; perubahan warna/konten menurunkannya. Aktif bila kedua record
# punya 'color_hist'.
HIST_CORR_GATE       = True
HIST_BINS            = 32    # jumlah bin histogram per channel RGB
HIST_CORR_THRESHOLD  = 0.80  # korelasi histogram minimum agar dianggap sama
COLOR_GRID_GATE      = True
COLOR_GRID           = 8     # grid 8x8 = 64 blok warna per foto
COLOR_GRID_IMG_SIZE  = 64    # ukuran resize sebelum dibagi blok warna
COLOR_GRID_MAX_DIST  = 24.0  # selisih warna blok (jarak Euclid RGB) di atas ini = berubah
COLOR_GRID_MIN_BLOCKS = 3    # jumlah blok berubah warna minimum untuk dianggap versi beda
# Saklar gerbang color_grid. False = abaikan gerbang ini tanpa scan ulang.
STRICT_VISUAL       = True
# Saklar gerbang blur per-region (anti blur/sensor lokal).
STRICT_BLUR         = True
# Gerbang BLUR PER-REGION: ukur ketajaman (varians Laplacian) per blok, lalu
# bandingkan blok-ke-blok. Wilayah yang sengaja di-blur (wajah, plat) anjlok
# ketajamannya sementara blok lain tetap. Foto dianggap versi beda bila ada
# >= BLUR_REGION_MIN_BLOCKS blok yang kedua sisinya bertekstur tapi rasio
# ketajaman (lo/hi) < BLUR_REGION_RATIO_MIN. Resize/re-kompresi murni menggeser
# ketajaman seragam sehingga duplikat sah tetap lolos.
BLUR_REGION_GATE        = True
BLUR_GRID               = 16    # grid 16x16 = 256 blok per foto
BLUR_REGION_IMG_SIZE    = 256   # ukuran resize sebelum dibagi blok
BLUR_REGION_RATIO_MIN   = 0.68  # rasio ketajaman blok (lo/hi) di bawah ini = di-blur sepihak
BLUR_REGION_MIN_SHARP   = 8.0   # blok dengan ketajaman < ini dianggap rata (diabaikan)
BLUR_REGION_MIN_BLOCKS  = 2     # jumlah blok blur-sepihak minimum untuk dianggap versi beda
# Gerbang BLOCKINESS JPEG: PEMBELA re-kompresi, bukan penolak. Kompresi JPEG
# meninggalkan artefak grid 8x8; blur editan/sensor justru menghaluskannya.
# Dipakai HANYA untuk membatalkan (rescue) vonis tolak gerbang blur bila buramnya
# terbukti dari re-kompresi global (artefak blok tinggi & merata), bukan sensor
# lokal. Tidak pernah menolak kandidat sendirian.
BLOCKINESS_RESCUE_GATE   = True
BLOCKINESS_GRID          = 8     # peta blockiness 8x8 = 64 blok per foto
BLOCKINESS_IMG_SIZE      = 256   # ukuran resize (kelipatan 8 agar batas blok JPEG selaras)
# Skor blockiness positif = ada artefak kompresi (re-kompresi); negatif = halus
# (blur editan). Ambang berikut memisahkan keduanya.
BLOCKINESS_BLOCK_MIN     = 0.15  # blockiness mean sisi buram minimum
BLOCKINESS_RATIO_MIN     = 0.5   # rasio blockiness sisi buram vs sisi tajam minimum
BLOCKINESS_GLOBAL_FRAC   = 0.10  # fraksi blok blur minimum agar rescue dipertimbangkan
BLOCKINESS_DEEPEST_MIN   = 0.0   # 0 = nonaktif (kedalaman blur tidak membatalkan rescue)
BLOCKINESS_MIN_COVERAGE  = 0.05  # cakupan blok ber-artefak sisi buram minimum
# Gerbang EDGE PER-REGION: anti emoji/stiker/teks/watermark kecil yang menambah
# tepi baru pada blok lokal. Ukur kepadatan tepi (gradien Sobel) per blok. Foto
# dianggap versi beda hanya bila ada >= EDGE_REGION_MIN_BLOCKS blok di mana satu
# sisi punya tepi jauh lebih banyak (delta > EDGE_REGION_DELTA_MIN DAN rasio
# lo/hi < EDGE_REGION_RATIO_MAX). Keputusan satu arah: resize/re-kompresi hanya
# menurunkan tepi, tak pernah menambah. STRICT_EDGE=False mematikan tanpa scan ulang.
STRICT_EDGE             = True
EDGE_REGION_GATE        = True
EDGE_GRID               = 16    # grid 16x16 = 256 blok per foto
EDGE_REGION_IMG_SIZE    = 256   # ukuran resize sebelum dibagi blok
EDGE_BLOCK_MIN_DENSITY  = 6.0   # blok dengan kepadatan tepi < ini dianggap polos (diabaikan)
EDGE_REGION_DELTA_MIN   = 9.0   # selisih kepadatan tepi (hi-lo) minimum agar dianggap tepi baru
EDGE_REGION_RATIO_MAX   = 0.55  # rasio kepadatan tepi (lo/hi) di bawah ini = satu sisi punya tepi jauh lebih banyak
EDGE_REGION_MIN_BLOCKS  = 3     # jumlah blok tepi-baru minimum untuk dianggap versi beda
# Gerbang SSIM: pemutus akhir foto (mahal, dijalankan paling belakang hanya untuk
# kandidat yang lolos semua gerbang murah). Foto identik beda resolusi/kompresi
# punya SSIM tinggi; edit struktur (stiker, teks, blur, crop) menurunkannya.
# Citra kanonik grayscale disimpan di record LMDB (field 'ssim_thumb').
# STRICT_SSIM=False mematikan tanpa scan ulang.
STRICT_SSIM             = True   # saklar gerbang SSIM
SSIM_IMG_SIZE           = 256    # ukuran citra kanonik grayscale untuk SSIM
SSIM_THRESHOLD          = 0.94   # skor SSIM minimum untuk dianggap duplikat (0..1)
# Laporan diagnostik proses/keputusan tiap gerbang per pasangan kandidat (untuk
# kalibrasi ambang). Dirender sebagai halaman lanjutan di PDF. Tidak mengubah
# keputusan duplikat. False = tidak mengumpulkan jejak (tanpa biaya tambahan).
PROCESS_REPORT          = True
# Foto yang gugur di pintu pHash tapi jaraknya masih < ambang ini tetap dicatat
# di laporan proses, agar terlihat gugur di pHash atau di gerbang dalam.
PROCESS_REPORT_NEAR_PHASH = 40
# Konstanta video: sampling frame, gerbang durasi, dan pencocokan frame.
# Definisi duplikat video: HANYA identik byte (BLAKE3) atau konten identik
# beda resolusi/kualitas. Edit apa pun (potong, sisip, ubah konten) dianggap BEDA.
VIDEO_SAMPLE_SEC        = 1     # jarak antar-frame sampel (detik)
VIDEO_MIN_SAMPLES       = 12    # minimal frame yang diambil per video
VIDEO_MAX_SAMPLES       = 300   # batas atas frame sampel per video
VIDEO_FRAME_MATCH_RATIO = 0.97  # fraksi frame yang harus cocok
# Toleransi durasi: re-encode/downscale tidak mengubah durasi, tapi encoder
# berbeda bisa menggeser beberapa frame (GOP/padding). Toleransi relatif +
# mutlak min/maks menyerap jitter sah tanpa memberi celah untuk editan trim/sisip.
VIDEO_DURATION_TOLERANCE = 0.03   # toleransi relatif maks (3%)
VIDEO_DURATION_MIN_DELTA = 0.30   # toleransi mutlak min (detik)
VIDEO_DURATION_MAX_DELTA = 2.0    # toleransi mutlak maks (detik)
VIDEO_PHASH_FRAME_THRESHOLD = 10  # jarak pHash per frame maks (dari 256 bit)
VIDEO_MIN_VALID_POS         = 8   # jumlah posisi terukur minimum sebelum rasio match dipercaya
VIDEO_FFMPEG_TOTAL_TIMEOUT  = 240 # batas waktu total ekstraksi frame FFmpeg (detik)
# Kuantum durasi untuk menghitung jumlah titik grid sampel. Menstabilkan jumlah
# titik agar jitter durasi re-encode tidak menggeser keselarasan posisional.
VIDEO_GRID_DURATION_QUANTUM = 0.5
STREAM_CHUNK        = 10 * 1024 * 1024
MAX_EMBED_MB        = 60
MIN_FILE_BYTES      = 128
THUMBNAIL_SIZE      = (640, 640)
THUMBNAIL_QUALITY   = 83
# Ukuran awal map LMDB. Tumbuh otomatis saat penuh via _lmdb_write_with_grow.
LMDB_MAP_SIZE       = 256 * 1024 ** 2
LMDB_MAX_DBS        = 16
MAX_LIST_ATTEMPTS   = 8     # batas retry list per folder per pass
SCAN_DEFER_PASSES   = 3     # berapa kali folder gagal di-retry ulang
# Verifikasi double-listing: listing kedua independen dibandingkan dengan
# manifest untuk menemukan file yang lolos dari listing pertama. Nonaktifkan
# untuk mempercepat scan pada folder sangat besar.
VERIFY_DOUBLE_LISTING = True
# Multi-Index Hashing: pHash dibagi MIH_NUM_BANDS band. Berdasarkan prinsip
# pigeonhole, dua hash dengan jarak Hamming <= (MIH_NUM_BANDS-1) pasti identik
# di minimal satu band. MIH_NUM_BANDS=48 memberi batas recall 47 bit; karena
# PHASH_THRESHOLD=20 jauh di bawah 47, tidak ada kandidat duplikat yang terlewat.
MIH_NUM_BANDS       = 48
# Batas anggota per-bucket MIH khusus video. Bucket yang melampaui cap dilewati
# saat pencarian kandidat; pasangan duplikat sah tetap berpasangan lewat band
# lain yang lebih selektif. Menekan ledakan kombinatorial O(n^2) pada video
# dengan banyak frame statik/mirip (CCTV, layar, fade).
MIH_VIDEO_BUCKET_CAP = 400
# Rate limiter global (dibagi semua worker).
API_RATE_PER_SEC    = 8.0
API_BURST           = 16
# Batas thumbnail base64 di mem-cache (LRU) agar RAM tidak membengkak.
THUMB_MEM_MAX       = 256

# Placeholder base64 (gambar 1x1 abu-abu) untuk menggantikan src kosong.
_PLACEHOLDER_B64 = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mN8"
    "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)

# ───────────────────── COLOR CODES ─────────────────────
class Colors:
    RESET         = '\033[0m'
    BOLD          = '\033[1m'
    DIM           = '\033[2m'
    HEADER        = '\033[38;5;30m'
    SUCCESS       = '\033[38;5;29m'
    WARNING       = '\033[38;5;172m'
    ERROR         = '\033[38;5;124m'
    CYAN          = '\033[38;5;30m'
    VALUE_SUCCESS = '\033[38;5;71m'
    BORDER        = '\033[38;5;239m'
    # ── Skema warna ringkasan output (lihat spesifikasi prompt warna) ──
    # KEY per-kategori: tiap bagian laporan diberi warna key berbeda agar
    # hierarki visual jelas. VALUE di luar kurung abu-abu (fokus ke key),
    # value di dalam kurung kuning tua (penekanan informasi tambahan).
    CYAN_KEY       = '\033[38;5;30m'   # Folder, Total File/Foto/Video, Status Scan
    GREEN_DIM_KEY  = '\033[38;5;65m'   # Rekonsiliasi, integritas, dst (hijau tua redup)
    ORANGE_DIM_KEY = '\033[38;5;130m'  # blok duplikat (orange redup)
    GRAY_VALUE     = '\033[38;5;248m'  # value di luar kurung (abu-abu)
    YELLOW_DIM     = '\033[38;5;100m'  # value di dalam kurung (kuning tua)

# Proteksi decompression-bomb Pillow: batasi jumlah piksel agar gambar kecil di
# disk namun raksasa saat di-decode tidak meng-OOM runtime. Batas ini masih
# menampung foto resolusi sangat tinggi. Warning dipromosikan ke error agar
# tertangkap try/except per-file (file dilewati, bukan menjatuhkan proses).
Image.MAX_IMAGE_PIXELS = 512_000_000
warnings.simplefilter('error', Image.DecompressionBombWarning)

# ───────────────────── SETUP ─────────────────────
auth.authenticate_user()
drive_service = build('drive', 'v3')
with open(os.devnull, 'w') as _devnull:
    with contextlib.redirect_stdout(_devnull):
        drive.mount("/content/drive", force_remount=True)

display(HTML('''
<center style="border:4px double #008080;color:#008080;font-family:'Impact',sans-serif;
letter-spacing:1px;padding:15px;width:390px;text-transform:uppercase;">
DUPLIGUARD VISION<br>
<span style="font-size:13px;font-family:sans-serif;font-weight:bold;letter-spacing:.5px;">
Exact (BLAKE3) + Visual (perceptual hashing) duplicate detector untuk Google Drive</span></center>'''))

# Struktur penyimpanan per folder Drive:
#   DupliGuard Vision/<Nama> (<id>)/
#     1_database/  hash_<id>.mdb + manifest_<id>.mdb (permanen)
#     2_laporan/   TXT + PDF (output)
#     3_cache/     journal_<id>.mdb (bisa dibuat ulang)
#                  (thumbnail render PDF TIDAK di sini: disimpan di disk lokal
#                   Colab /content, ephemeral, lihat _local_thumb_dir)
# Tiap env LMDB adalah file .mdb tunggal dengan pendamping .mdb-lock dan .mdb.gen.
BASE_PATH     = "/content/drive/My Drive/DupliGuard Vision"

# Pastikan Drive sudah ter-mount sebelum membuat BASE_PATH. Bila mount gagal
# diam-diam, data akan tersimpan di filesystem ephemeral dan hilang saat sesi berakhir.
_DRIVE_ROOT = "/content/drive/My Drive"
if not os.path.isdir(_DRIVE_ROOT):
    raise RuntimeError(
        "Google Drive tidak ter-mount di /content/drive/My Drive. "
        "Jalankan ulang sel dan pastikan proses mount Drive selesai sebelum melanjutkan, "
        "agar data tidak tersimpan di penyimpanan sementara yang akan hilang.")

os.makedirs(BASE_PATH, exist_ok=True)

def _bundle_name(folder_name: str, folder_id: str) -> str:
    """Nama bundle unik: '<Nama Disanitasi> (<id>)'. Nama dipotong agar tidak
    melebihi batas 255 byte filesystem; folder_id selalu utuh."""
    base = _truncate_component(sanitize_filename(folder_name) or 'Folder')
    return f"{base} ({folder_id})"

def _bundle_id_of(entry: str) -> Optional[str]:
    """Ekstrak folder_id dari nama bundle '<Nama> (<id>)'. Validasi karakter
    Drive id (alfanumerik, '_', '-') mencegah salah-parse pada folder yang
    namanya mengandung kurung. Return id atau None bila format tidak sesuai."""
    if not entry.endswith(')'):
        return None
    open_idx = entry.rfind('(')
    if open_idx == -1:
        return None
    inner = entry[open_idx + 1:-1]
    if not inner:
        return None
    if not all(c.isalnum() or c in ('_', '-') for c in inner):
        return None
    return inner

def _find_existing_bundle(folder_id: str) -> Optional[str]:
    """Cari bundle yang sudah ada berdasarkan folder_id. Return path atau None."""
    try:
        for entry in os.listdir(BASE_PATH):
            full = os.path.join(BASE_PATH, entry)
            if os.path.isdir(full) and _bundle_id_of(entry) == folder_id:
                return full
    except Exception:
        pass
    return None

def folder_paths(folder_id: str, folder_name: str = "") -> Dict[str, str]:
    """Sumber tunggal seluruh path penyimpanan satu folder Drive. Bundle dengan
    id sama selalu dipakai ulang; bila nama berubah, bundle di-rename sekali
    tanpa kehilangan data."""
    existing = _find_existing_bundle(folder_id)
    desired  = os.path.join(BASE_PATH, _bundle_name(folder_name or folder_id, folder_id))
    if existing and existing != desired and folder_name:
        # Selaraskan nama bundle ke nama baru. Bila gagal, tetap pakai existing.
        try:
            if not os.path.exists(desired):
                os.rename(existing, desired)
                _fsync_dir(desired)
                existing = desired
        except Exception as e:
            _log('warning', f"rename bundle gagal {existing} -> {desired}: {e}")
    bundle = existing or desired
    db_dir     = os.path.join(bundle, "1_database")
    report_dir = os.path.join(bundle, "2_laporan")
    cache_dir  = os.path.join(bundle, "3_cache")
    # Thumbnail disimpan di disk lokal Colab (ext4), bukan Drive: hanya bahan
    # antara untuk render PDF, tidak perlu persisten.
    thumb_dir  = _local_thumb_dir(folder_id)
    return {
        'bundle':   bundle,
        'database': db_dir,
        'report':   report_dir,
        'cache':    cache_dir,
        'thumb':    thumb_dir,
        'hash_env':     os.path.join(db_dir,    f"hash_{folder_id}.mdb"),
        'manifest_env': os.path.join(db_dir,    f"manifest_{folder_id}.mdb"),
        'journal_env':  os.path.join(cache_dir, f"journal_{folder_id}.mdb"),
    }

# Umur minimum (detik) file scratch sebelum boleh dihapus otomatis.
# Mencegah menghapus .tmp/.snap yang mungkin sedang aktif ditulis.
SCRATCH_STALE_AGE_SEC = 600   # 10 menit

def _cleanup_stale_scratch(db_dir: str, cache_dir: str):
    """Hapus file scratch usang sisa crash (.mdb.tmp, .mdb.gen.tmp, .mdb.snap,
    .mdb-lock yatim) dari folder database/cache bundle. Hanya menyentuh akhiran
    scratch; file final .mdb dan sidecar .gen tidak pernah dihapus. File .tmp/.snap
    hanya dihapus bila file final pasangannya sudah ada. Hanya menghapus scratch
    yang lebih tua dari SCRATCH_STALE_AGE_SEC."""
    now = time.time()

    def _final_exists(scratch_path: str) -> bool:
        if scratch_path.endswith('.tmp'):
            final = scratch_path[:-len('.tmp')]
        elif scratch_path.endswith('.snap'):
            final = scratch_path[:-len('.snap')]
        else:
            return False
        return os.path.exists(final)

    def _is_orphan_lock(path: str) -> bool:
        # .mdb-lock yatim bila .mdb pasangannya tidak ada.
        if not path.endswith('.mdb-lock'):
            return False
        return not os.path.exists(path[:-len('-lock')])

    for d in (db_dir, cache_dir):
        try:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                full = os.path.join(d, entry)
                if not os.path.isfile(full):
                    continue
                low = entry.lower()
                is_scratch = low.endswith(('.mdb.tmp', '.mdb.gen.tmp', '.mdb.snap'))
                is_orphan_lock = _is_orphan_lock(full)
                if not (is_scratch or is_orphan_lock):
                    continue
                try:
                    age = now - os.path.getmtime(full)
                except OSError:
                    continue
                if age < SCRATCH_STALE_AGE_SEC:
                    continue
                # Untuk .tmp/.snap: hanya hapus bila file final ada (lihat docstring).
                if is_scratch and not is_orphan_lock and not _final_exists(full):
                    _log('warning',
                         f"scratch tanpa file final dipertahankan (mungkin satu-satunya data): {full}")
                    continue
                try:
                    os.unlink(full)
                    _log('info', f"scratch usang dihapus permanen: {full}")
                except OSError as e:
                    _log('debug', f"gagal hapus scratch {full}: {e}")
        except Exception as e:
            _log('debug', f"_cleanup_stale_scratch gagal di {d}: {e}")

def _cleanup_scratch_now(db_dir: str, cache_dir: str):
    """Hapus scratch sisa proses (.mdb.tmp, .mdb.gen.tmp, .mdb.snap) dan
    .mdb-lock yatim TANPA syarat umur. Dipanggil setelah semua env LMDB ditutup
    di akhir analisis, saat sudah pasti tidak ada tulis berjalan. Aturan
    keamanan data sama dengan _cleanup_stale_scratch."""
    def _final_exists(scratch_path: str) -> bool:
        if scratch_path.endswith('.tmp'):
            final = scratch_path[:-len('.tmp')]
        elif scratch_path.endswith('.snap'):
            final = scratch_path[:-len('.snap')]
        else:
            return False
        return os.path.exists(final)

    def _is_orphan_lock(path: str) -> bool:
        if not path.endswith('.mdb-lock'):
            return False
        return not os.path.exists(path[:-len('-lock')])

    removed = 0
    for d in (db_dir, cache_dir):
        try:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                full = os.path.join(d, entry)
                if not os.path.isfile(full):
                    continue
                low = entry.lower()
                is_scratch = low.endswith(('.mdb.tmp', '.mdb.gen.tmp', '.mdb.snap'))
                is_orphan_lock = _is_orphan_lock(full)
                if not (is_scratch or is_orphan_lock):
                    continue
                # Tanpa gerbang umur: titik panggil sudah pasti aman (env
                # ditutup, tak ada tulis berjalan). Tetap pertahankan .tmp/.snap
                # bila file final pasangannya belum ada (lihat docstring).
                if is_scratch and not is_orphan_lock and not _final_exists(full):
                    _log('warning',
                         f"scratch tanpa file final dipertahankan (mungkin satu-satunya data): {full}")
                    continue
                try:
                    os.unlink(full)
                    removed += 1
                    _log('info', f"scratch dihapus permanen (akhir analisis): {full}")
                except OSError as e:
                    _log('debug', f"gagal hapus scratch {full}: {e}")
        except Exception as e:
            _log('debug', f"_cleanup_scratch_now gagal di {d}: {e}")
    return removed

def _dedupe_active_drive_duplicates(folder_id: str) -> int:
    """Audit dan hapus file aktif (non-trash) yang berduplikat nama di Drive.

    Di Drive FUSE, os.replace kadang tidak atomik: membuat objek baru bernama
    sama tanpa menghapus yang lama, sehingga muncul dua file aktif dengan nama
    identik. Fungsi ini mencari duplikat tersebut, mempertahankan salinan yang
    valid (terbesar & terbaru), dan menghapus sisanya. Selalu menyisakan minimal
    satu salinan. Hanya berjalan bila PURGE_TRASHED_MDB aktif & sirkuit API sehat.
    Return jumlah file duplikat yang dihapus.
    """
    if not PURGE_TRASHED_MDB:
        return 0
    if _circuit_breaker.is_open():
        return 0
    deleted_total = 0
    try:
        db_dir = folder_paths(folder_id)['database']
        # Cari folder database bundle di Drive berdasarkan path untuk membatasi
        # query ke parent yang benar. Bila gagal, batalkan (jangan dedup global
        # tanpa batas parent -> berisiko menyentuh file bernama sama di tempat
        # lain). parent_id None -> lewati audit.
        parent_id = _resolve_drive_folder_id(db_dir)
        if not parent_id:
            _log('debug', f"dedupe: folder database Drive tak ditemukan untuk {folder_id}, dilewati")
            return 0
        bases = [
            f"hash_{folder_id}.mdb",     f"hash_{folder_id}.mdb.gen",
            f"manifest_{folder_id}.mdb", f"manifest_{folder_id}.mdb.gen",
            f"journal_{folder_id}.mdb",  f"journal_{folder_id}.mdb.gen",
        ]
        for base in bases:
            safe_name = base.replace("'", "\\'")
            q = (f"name = '{safe_name}' and trashed = false "
                 f"and '{parent_id}' in parents")
            files = []
            page_token = None
            while True:
                resp, err = _drive_execute(lambda: _thread_drive().files().list(
                    q=q, spaces='drive',
                    fields='nextPageToken, files(id,name,trashed,modifiedTime,size)',
                    pageSize=100, pageToken=page_token,
                    supportsAllDrives=False))
                if err or not resp:
                    break
                for f in resp.get('files', []):
                    if f.get('trashed'):
                        continue
                    if f.get('name') != base:
                        continue
                    if f.get('id'):
                        files.append(f)
                page_token = resp.get('nextPageToken')
                if not page_token:
                    break
            if len(files) <= 1:
                continue
            # Pilih salinan yang valid (bisa dibuka sebagai LMDB), bukan sekadar
            # yang terbaru. Bila tidak ada yang valid, tidak menghapus apa pun.
            is_mdb = base.endswith('.mdb')
            keep = _pick_valid_drive_copy(files, base, parent_id, is_mdb)
            if keep is None:
                _log('warning',
                     f"dedupe Drive: '{base}' punya {len(files)} salinan tapi "
                     f"tidak ada yang terbukti valid -> TIDAK menghapus apa pun "
                     f"(amankan data, periksa manual)")
                continue
            for f in files:
                fid = f.get('id')
                if not fid or fid == keep.get('id'):
                    continue
                _, derr = _drive_execute(
                    lambda fid=fid: _thread_drive().files().delete(fileId=fid))
                if not derr:
                    deleted_total += 1
                else:
                    _log('debug', f"dedupe hapus '{base}' id={fid} gagal: {derr}")
            _log('info',
                 f"dedupe Drive: '{base}' punya {len(files)} salinan aktif, "
                 f"dipertahankan id={keep.get('id')} "
                 f"(modifiedTime={keep.get('modifiedTime')})")
        if deleted_total:
            _log('info', f"dedupe Drive: {deleted_total} salinan aktif berduplikat-nama dihapus permanen ({folder_id})")
    except Exception as e:
        _log('warning', f"_dedupe_active_drive_duplicates gagal {folder_id}: {e}")
    return deleted_total

def _is_valid_lmdb_file(path: str) -> bool:
    """Cek apakah file .mdb bisa dibuka sebagai env LMDB (tidak korup).
    Membuka read-only tanpa lock, membaca stat, lalu menutup. Non-destruktif."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False
        env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                        max_dbs=LMDB_MAX_DBS, create=False)
        try:
            with env.begin() as txn:
                txn.stat()  # baca meta page -> gagal bila korup
            return True
        finally:
            env.close()
    except Exception as e:
        _log('debug', f"_is_valid_lmdb_file gagal {path}: {e}")
        return False

def _pick_valid_drive_copy(files: List[Dict], base: str, parent_id: str,
                           is_mdb: bool) -> Optional[Dict]:
    """Pilih salinan yang harus dipertahankan dari beberapa file Drive aktif
    bernama sama. Return record file yang dipilih, atau None bila tidak ada yang
    aman dipertahankan (pemanggil tidak menghapus apa pun).

    - Non-.mdb: pertahankan yang modifiedTime terbaru.
    - .mdb: uji validitas tiap kandidat via LMDB. Di antara yang valid, pilih
      ukuran terbesar lalu terbaru. Bila tidak ada yang valid, return None.
    """
    if not files:
        return None
    if not is_mdb:
        return sorted(files, key=lambda f: f.get('modifiedTime', ''), reverse=True)[0]
    # Untuk .mdb: validasi via path FUSE Drive. Bila valid, pertahankan record
    # kanonik (terbesar lalu terbaru). Bila tidak bisa dibuktikan valid, return None.
    try:
        db_dir_local = None
        # Cari path lokal db_dir dari salah satu pemanggil tidak tersedia di sini;
        # rekonstruksi dari folder_paths memerlukan folder_id. base berbentuk
        # 'hash_<id>.mdb' / 'manifest_<id>.mdb', jadi ekstrak <id>.
        stem = base
        for suf in ('.mdb.gen', '.mdb'):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        fid_part = stem.split('_', 1)[1] if '_' in stem else None
        if fid_part:
            db_dir_local = folder_paths(fid_part)['database']
        if db_dir_local:
            candidate_path = os.path.join(db_dir_local, base)
            if _is_valid_lmdb_file(candidate_path):
                # Minimal satu salinan valid terbukti -> pilih record kanonik
                # (terbesar lalu terbaru) untuk dipertahankan.
                return sorted(
                    files,
                    key=lambda f: (int(f.get('size', 0) or 0), f.get('modifiedTime', '')),
                    reverse=True)[0]
        # Tidak bisa membuktikan validitas -> jangan hapus apa pun.
        return None
    except Exception as e:
        _log('debug', f"_pick_valid_drive_copy gagal '{base}': {e}")
        return None

def _resolve_drive_folder_id(local_drive_path: str) -> Optional[str]:
    """Resolusi ID folder Drive dari path FUSE lokal dengan menelusuri hierarki
    nama folder dari root 'My Drive'. Return folder id atau None."""
    try:
        marker = "/content/drive/My Drive"
        if not local_drive_path.startswith(marker):
            return None
        rel = local_drive_path[len(marker):].strip("/")
        if not rel:
            return None
        parts = [seg for seg in rel.split("/") if seg]
        cur_parent = 'root'
        for seg in parts:
            safe = seg.replace("'", "\\'")
            q = (f"name = '{safe}' and trashed = false "
                 f"and mimeType = 'application/vnd.google-apps.folder' "
                 f"and '{cur_parent}' in parents")
            resp, err = _drive_execute(lambda: _thread_drive().files().list(
                q=q, spaces='drive', fields='files(id,name)',
                pageSize=10, supportsAllDrives=False))
            if err or not resp:
                return None
            found = resp.get('files', [])
            if not found:
                return None
            cur_parent = found[0].get('id')
            if not cur_parent:
                return None
        return cur_parent
    except Exception as e:
        _log('debug', f"_resolve_drive_folder_id gagal {local_drive_path}: {e}")
        return None

def _sweep_trashed_bundle_versions(folder_id: str):
    """Hapus versi lama .mdb/.gen bundle ini yang menumpuk di Trash Drive.
    Aman: hanya menyasar item trashed=true dengan nama eksak file bundle ini.
    Hanya berjalan bila PURGE_TRASHED_MDB aktif dan sirkuit API sehat.
    """
    if not PURGE_TRASHED_MDB:
        return
    try:
        if _circuit_breaker.is_open():
            return
        bases = [
            f"hash_{folder_id}.mdb",     f"hash_{folder_id}.mdb.gen",
            f"manifest_{folder_id}.mdb", f"manifest_{folder_id}.mdb.gen",
            f"journal_{folder_id}.mdb",  f"journal_{folder_id}.mdb.gen",
        ]
        total = 0
        for base in bases:
            total += _purge_trashed_by_name(base)
        if total:
            _log('info', f"sweep Trash startup: {total} versi lama bundle {folder_id} dihapus permanen")
    except Exception as e:
        _log('debug', f"_sweep_trashed_bundle_versions gagal {folder_id}: {e}")

def ensure_bundle(folder_id: str, folder_name: str) -> Dict[str, str]:
    """Buat seluruh subfolder bundle dan jalankan housekeeping awal:
    bersihkan scratch usang, purge Trash, dan audit duplikat aktif di Drive."""
    p = folder_paths(folder_id, folder_name)
    # Buat direktori bundle di Drive. Env LMDB berupa file tunggal (.mdb),
    # bukan direktori, jadi path env tidak boleh dibuatkan folder.
    for key in ('bundle', 'database', 'report', 'cache'):
        os.makedirs(p[key], exist_ok=True)
    # Direktori thumbnail lokal (ext4 Colab).
    try:
        os.makedirs(p['thumb'], exist_ok=True)
    except Exception as e:
        _log('debug', f"makedirs thumb dir lokal gagal: {e}")
    # 1) Bersihkan scratch .tmp/.snap/.gen.tmp + lock yatim sisa crash.
    try:
        _cleanup_stale_scratch(p['database'], p['cache'])
    except Exception as e:
        _log('debug', f"cleanup scratch saat ensure_bundle gagal: {e}")
    # 2) Hapus versi lama .mdb/.gen bundle yang menumpuk di Trash Drive.
    try:
        _sweep_trashed_bundle_versions(folder_id)
    except Exception as e:
        _log('debug', f"sweep trash saat ensure_bundle gagal: {e}")
    # 3) Audit duplikat aktif berduplikat-nama (sisa os.replace non-atomik di
    #    Drive FUSE): pertahankan salinan valid, hapus sisanya.
    try:
        _dedupe_active_drive_duplicates(folder_id)
    except Exception as e:
        _log('debug', f"dedupe aktif saat ensure_bundle gagal: {e}")
    return p

# ───────────────────── FILE TYPE ─────────────────────
IMG_EXTS = frozenset({'.jpg','.jpeg','.png','.bmp','.gif','.tiff','.tif','.webp','.heic','.heif','.avif'})
VID_EXTS = frozenset({'.mp4','.avi','.mkv','.mov','.flv','.wmv','.webm','.m4v','.3gp','.ts','.mts','.m2ts'})

def get_file_type(name: str, mime_type: Optional[str] = None) -> Optional[str]:
    if mime_type:
        if mime_type.startswith('image/'): return 'image'
        if mime_type.startswith('video/'): return 'video'
    ext = os.path.splitext((name or '').lower())[1]
    if ext in IMG_EXTS: return 'image'
    if ext in VID_EXTS: return 'video'
    return None

def is_media_file(name: str, mime_type: Optional[str] = None) -> bool:
    return get_file_type(name, mime_type) is not None

# ───────────────────── UTILS ─────────────────────
def _safe_int_size(value: Any) -> int:
    """Konversi field 'size' Drive ke int dengan aman. Return 0 untuk nilai None
    atau tidak valid."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

def format_size(mb: float) -> str:
    # Guard input ekstrem (NaN/inf/negatif) dari record korup.
    try:
        mb = float(mb)
    except (TypeError, ValueError):
        return "? KB"
    if not (mb == mb) or mb == float('inf') or mb == float('-inf'):
        return "? KB"
    if mb < 0:
        mb = 0.0
    if mb >= 1024: return f"{mb/1024:.2f} GB"
    if mb >= 1:    return f"{mb:.2f} MB"
    return f"{mb*1024:.0f} KB"

def format_timestamp(ts: str) -> str:
    try: return dateparser.isoparse(ts).strftime("%d %b %Y, %H:%M")
    except Exception: return ts or "—"

def sanitize_filename(filename: str) -> str:
    for c in r'<>:"/|?*\\':
        filename = filename.replace(c, '_')
    return filename.strip()

# Batas aman panjang satu komponen path (nama file/direktori) dalam byte.
# Disisakan margin untuk suffix '(<id>)'/'.tmp' dll.
MAX_PATH_COMPONENT_BYTES = 200

def _truncate_component(name: str, max_bytes: int = MAX_PATH_COMPONENT_BYTES) -> str:
    """Potong `name` agar representasi UTF-8-nya <= max_bytes tanpa memutus
    karakter multibyte. Mencegah ENAMETOOLONG saat nama folder Drive sangat panjang."""
    if name is None:
        return ''
    enc = name.encode('utf-8')
    if len(enc) <= max_bytes:
        return name
    return enc[:max_bytes].decode('utf-8', 'ignore')

def report_subdir_name(folder_name: str, folder_id: str) -> str:
    """Nama folder laporan: '<Nama Disanitasi>_DupliGuardVision.(<id>)'.

    Penyimpanan memakai JSON + LMDB (bukan pickle), jadi akhiran '.pkl' lama
    tidak lagi dipakai.
    """
    sanitized = _truncate_component(sanitize_filename(folder_name))
    return f"{sanitized}_DupliGuardVision.({folder_id})"

def _popcount(x: int) -> int:
    # Hamming weight. Python 3.10+: int.bit_count (C-speed).
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def _hex_to_int(h: str) -> Optional[int]:
    # Defensif: record korup/lama bisa punya hash non-hex. Kembalikan None
    # alih-alih melempar ValueError yang akan menjatuhkan seluruh analisis.
    if not h:
        return None
    try:
        return int(h, 16)
    except (ValueError, TypeError):
        _log('debug', f"hash bukan hex valid, dilewati: {h!r}")
        return None

# ───────────────────── SERIALISASI RECORD ─────────────────────
# Record diserialisasi sebagai JSON (bukan pickle) karena file .mdb tersimpan
# di Drive yang bisa dibagikan; pickle pada data eksternal adalah vektor RCE.
# Record yang gagal didekode JSON dianggap korup (None) dan diproses ulang.
def _record_dumps(obj: Any) -> bytes:
    # default=str: jaring pengaman bila record memuat tipe non-JSON di masa depan.
    return json.dumps(obj, default=str).encode('utf-8')

def _record_loads(data: bytes) -> Any:
    if data is None:
        return None
    try:
        return json.loads(data.decode('utf-8'))
    except Exception as e:
        _log('error', f"record decode JSON gagal (dianggap korup, di-skip): {e}")
        return None

# ───────────────────── INTEGRITY HELPERS ─────────────────────
def _blake3_file(path: str) -> str:
    h = blake3.blake3()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def _fsync_dir(path: str):
    """fsync direktori agar rename entry benar-benar persisten."""
    try:
        d = os.path.dirname(path) or '.'
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception as e:
        _log('debug', f"fsync dir failed {path}: {e}")

def _lmdb_write_with_grow(env, write_fn, lock=None, max_attempts: int = 6):
    """Jalankan transaksi tulis LMDB. Bila map penuh (MapFullError), gandakan
    map_size lalu ulangi. write_fn harus idempoten bila diulang."""
    attempt = 0
    while True:
        try:
            with env.begin(write=True) as txn:
                return write_fn(txn)
        except lmdb.MapFullError:
            attempt += 1
            if attempt > max_attempts:
                _log('error', "LMDB MapFullError: gagal memperbesar map setelah beberapa percobaan")
                raise
            info = env.info()
            new_size = info['map_size'] * 2
            _log('warning', f"LMDB map penuh, perbesar map_size -> {new_size} bytes")
            if lock is not None:
                with lock:
                    env.set_mapsize(new_size)
            else:
                env.set_mapsize(new_size)

# Direktori mirror lokal (ext4 di Colab). LMDB dikerjakan di lokal lalu
# disalin balik ke Drive secara atomik pada sync()/close().
LOCAL_LMDB_DIR = "/content/_dupliguardvision_lmdb"
# Direktori thumbnail lokal (ext4 di Colab). Thumbnail hanya bahan antara
# untuk render PDF; tidak perlu persisten di Drive.
LOCAL_THUMB_DIR = "/content/_dupliguardvision_thumb"

def _local_thumb_dir(folder_id: str) -> str:
    """Direktori thumbnail lokal unik per folder Drive (deterministik, tidak bentrok)."""
    safe = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in (folder_id or 'folder'))
    return os.path.join(LOCAL_THUMB_DIR, safe)

def _cleanup_local_thumb_dir(folder_id: str):
    """Hapus direktori thumbnail lokal setelah analisis selesai. Thumbnail hanya
    bahan antara render PDF; tanpa pembersihan, analisis banyak folder bisa
    memicu ENOSPC. Hanya menghapus path di dalam LOCAL_THUMB_DIR."""
    try:
        root = os.path.abspath(LOCAL_THUMB_DIR)
        tgt_abs = os.path.abspath(_local_thumb_dir(folder_id))
        if tgt_abs != root and os.path.commonpath([root, tgt_abs]) == root:
            if os.path.isdir(tgt_abs):
                shutil.rmtree(tgt_abs, ignore_errors=True)
    except Exception as e:
        _log('debug', f"cleanup thumb lokal gagal {folder_id}: {e}")

# Interval minimal copy-back .mdb lokal ke Drive. Flush LMDB lokal tetap tiap
# sync; penyalinan file penuh ke Drive dibatasi agar tidak terlalu sering.
# close() tetap memaksa mirror final (force=True).
MIRROR_MIN_INTERVAL_SEC = 120.0

# Manajemen storage Drive: os.replace di Drive FUSE memindahkan versi lama ke
# Trash alih-alih menghapusnya. Setelah snapshot baru terverifikasi, versi lama
# di Trash dihapus permanen via Drive API agar tidak menumpuk menghabiskan kuota.
PURGE_TRASHED_MDB        = True
# Prune record hash orphan: buang record di hash.mdb yang file_id-nya sudah
# tidak ada di manifest (file dihapus / keluar dari scope scan), agar hash.mdb
# tidak membengkak seiring waktu. Hanya dijalankan saat scan TUNTAS & sehat
# (lihat gerbang pengaman di _analyze_folder_body). False = nonaktif (hash lama
# disimpan selamanya, hemat kuota untuk keluar-masuk file tapi DB bengkak).
PRUNE_ORPHAN_HASH        = True
# Guard kuota: bila sisa ruang Drive di bawah ambang ini, mirror ke Drive dilewati.
DRIVE_MIN_FREE_BYTES     = 512 * 1024 ** 2   # 512 MB
# Interval minimal antar-purge Trash. close() selalu memurge.
PURGE_MIN_INTERVAL_SEC   = 300.0

def _local_mirror_path(drive_path: str) -> str:
    """Path mirror lokal unik & deterministik untuk sebuah .mdb di Drive.
    Memakai hash path Drive agar tidak bentrok antar bundle."""
    key = hashlib.blake2b(drive_path.encode('utf-8'), digest_size=16).hexdigest()
    base = sanitize_filename(os.path.basename(drive_path)) or "env.mdb"
    return os.path.join(LOCAL_LMDB_DIR, f"{key}_{base}")

def _write_text_atomic(path: str, text: str):
    """Tulis teks ke path secara atomik (.tmp lalu os.replace + fsync), agar
    file sidecar tidak pernah setengah tertulis bila proses mati."""
    tmp = path + ".tmp"
    try:
        with open(tmp, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        raise

def _read_text_file(path: str) -> Optional[str]:
    """Baca isi teks file; None bila tidak ada atau gagal dibaca."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return f.read().strip()
    except Exception:
        return None

# Cache sisa kuota Drive (thread-safe, TTL 60 detik) untuk menekan API about().get() berulang.
_DRIVE_QUOTA_CACHE_TTL = 60.0
_drive_quota_lock = threading.Lock()
_drive_quota_cache: Dict[str, float] = {'ts': 0.0, 'free': -1.0}  # free=-1 -> None

def _drive_free_bytes() -> Optional[int]:
    """Sisa ruang Google Drive (limit - usage) dalam byte. Return None bila tak
    terbaca atau akun unlimited (fail-open: tidak menghalangi mirror).
    Hasil di-cache selama _DRIVE_QUOTA_CACHE_TTL detik."""
    now = time.monotonic()
    with _drive_quota_lock:
        if (now - _drive_quota_cache['ts']) < _DRIVE_QUOTA_CACHE_TTL:
            cached = _drive_quota_cache['free']
            return None if cached < 0 else int(cached)
    free_val: Optional[int] = None
    try:
        about, err = _drive_execute(
            lambda: _thread_drive().about().get(fields="storageQuota"))
        if not err and about:
            q = about.get('storageQuota', {}) or {}
            limit = int(q.get('limit', 0) or 0)
            usage = int(q.get('usage', 0) or 0)
            if limit > 0:
                free_val = max(0, limit - usage)
    except Exception as e:
        _log('debug', f"_drive_free_bytes gagal: {e}")
        free_val = None
    with _drive_quota_lock:
        # Simpan ke cache bila query sukses. None karena unlimited tetap di-cache.
        _drive_quota_cache['ts'] = now
        _drive_quota_cache['free'] = -1.0 if free_val is None else float(free_val)
    return free_val

def _purge_trashed_by_name(basename: str, parent_id: Optional[str] = None) -> int:
    """Hapus permanen file di Trash Drive yang namanya tepat sama dengan
    `basename`. Hanya menyasar item trashed=true dengan nama eksak; salinan
    aktif tidak pernah terhapus. Return jumlah file yang berhasil dihapus."""
    deleted = 0
    try:
        safe_name = basename.replace("'", "\\'")
        q = f"name = '{safe_name}' and trashed = true"
        page_token = None
        while True:
            resp, err = _drive_execute(lambda: _thread_drive().files().list(
                q=q, spaces='drive',
                fields='nextPageToken, files(id,name,trashed,parents)',
                pageSize=100, pageToken=page_token,
                supportsAllDrives=False))
            if err or not resp:
                break
            for f in resp.get('files', []):
                # Defensif ganda: pastikan benar-benar trashed sebelum hapus.
                if not f.get('trashed'):
                    continue
                if f.get('name') != basename:
                    continue
                fid = f.get('id')
                if not fid:
                    continue
                _, derr = _drive_execute(
                    lambda fid=fid: _thread_drive().files().delete(fileId=fid))
                if not derr:
                    deleted += 1
                else:
                    _log('debug', f"hapus permanen gagal {fid}: {derr}")
            page_token = resp.get('nextPageToken')
            if not page_token:
                break
        if deleted:
            _log('info', f"purge Trash: {deleted} salinan lama '{basename}' dihapus permanen")
    except Exception as e:
        _log('warning', f"_purge_trashed_by_name gagal '{basename}': {e}")
    return deleted

def _sweep_scratch_in_dir(directory: str) -> int:
    """Hapus file scratch yatim sisa crash di satu direktori tanpa syarat umur.
    Menyasar: .mdb.tmp, .mdb.gen.tmp, .mdb.snap, dan .mdb-lock yatim.
    File final .mdb dan sidecar .gen tidak pernah disentuh. .tmp/.snap hanya
    dihapus bila file final pasangannya sudah ada. Return jumlah file dihapus."""
    def _final_exists(scratch_path: str) -> bool:
        if scratch_path.endswith('.tmp'):
            final = scratch_path[:-len('.tmp')]
        elif scratch_path.endswith('.snap'):
            final = scratch_path[:-len('.snap')]
        else:
            return False
        return os.path.exists(final)

    def _is_orphan_lock(path: str) -> bool:
        if not path.endswith('.mdb-lock'):
            return False
        return not os.path.exists(path[:-len('-lock')])

    removed = 0
    try:
        if not os.path.isdir(directory):
            return 0
        for entry in os.listdir(directory):
            full = os.path.join(directory, entry)
            if not os.path.isfile(full):
                continue
            low = entry.lower()
            is_scratch = low.endswith(('.mdb.tmp', '.mdb.gen.tmp', '.mdb.snap'))
            is_orphan_lock = _is_orphan_lock(full)
            if not (is_scratch or is_orphan_lock):
                continue
            if is_scratch and not is_orphan_lock and not _final_exists(full):
                _log('warning',
                     f"scratch tanpa file final dipertahankan (mungkin satu-satunya data): {full}")
                continue
            try:
                os.unlink(full)
                removed += 1
                _log('info', f"scratch yatim dihapus permanen: {full}")
            except OSError as e:
                _log('debug', f"gagal hapus scratch {full}: {e}")
    except Exception as e:
        _log('debug', f"_sweep_scratch_in_dir gagal {directory}: {e}")
    return removed

def _copy_file_atomic(src: str, dst: str):
    """Salin src -> dst secara aman untuk Google Drive FUSE.

    Di Drive FUSE, os.replace tidak atomik: membuat objek baru bernama sama
    tanpa menghapus yang lama, sehingga muncul duplikat nama. Untuk mencegahnya:
    - dst SUDAH ADA: timpa in-place (tulis langsung ke file/ID yang sama, tanpa
      rename) sehingga Drive tidak membuat objek baru.
    - dst BELUM ADA: pakai .tmp -> os.replace (pembuatan pertama, aman).

    Bila in-place gagal, fallback ke .tmp -> os.replace. Bila proses mati di
    tengah in-place, dst bisa setengah tertulis; run berikutnya menyalin ulang
    dari lokal (sumber kebenaran) sehingga Drive diperbaiki otomatis.
    """
    parent = os.path.dirname(dst) or '.'
    os.makedirs(parent, exist_ok=True)

    if os.path.exists(dst):
        # Timpa in-place: tulis langsung ke dst yang sudah ada agar Drive tidak
        # membuat salinan bernama sama yang kedua.
        try:
            with open(src, 'rb') as fsrc, open(dst, 'r+b') as fdst:
                while True:
                    chunk = fsrc.read(STREAM_CHUNK)
                    if not chunk:
                        break
                    fdst.write(chunk)
                fdst.flush()
                os.fsync(fdst.fileno())
                fdst.truncate()
            _fsync_dir(dst)
            return
        except Exception as e:
            # Bila in-place gagal, fallback ke .tmp -> os.replace.
            _log('warning', f"tulis in-place gagal {dst}, fallback replace: {e}")

    tmp = dst + ".tmp"
    # Buang sisa .tmp yatim dari penyalinan sebelumnya yang gagal di tengah jalan.
    try:
        if os.path.exists(tmp):
            os.unlink(tmp)
    except OSError as e:
        _log('debug', f"hapus sisa .tmp lama gagal {tmp}: {e}")
    try:
        with open(src, 'rb') as fsrc, open(tmp, 'wb') as fdst:
            while True:
                chunk = fsrc.read(STREAM_CHUNK)
                if not chunk:
                    break
                fdst.write(chunk)
            fdst.flush()
            os.fsync(fdst.fileno())
        os.replace(tmp, dst)
        _fsync_dir(dst)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        raise

class _DriveMirroredEnv:
    """Pembungkus env LMDB yang beroperasi di disk lokal dan menyalin balik
    file .mdb ke Drive secara atomik pada sync()/close(). Mendelegasikan semua
    atribut/metode lain ke env lokal asli."""
    def __init__(self, env, local_path: str, drive_path: str):
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, '_local_path', local_path)
        object.__setattr__(self, '_drive_path', drive_path)
        object.__setattr__(self, '_sync_lock', threading.Lock())
        object.__setattr__(self, '_last_mirror', 0.0)
        object.__setattr__(self, '_last_purge', 0.0)

    def _mirror_to_drive(self, force: bool = False):
        # Throttle penyalinan file penuh ke Drive.
        now = time.monotonic()
        if not force and (now - object.__getattribute__(self, '_last_mirror')) < MIRROR_MIN_INTERVAL_SEC:
            return
        try:
            if not os.path.exists(self._local_path):
                return
            # Guard kuota: bila ruang Drive kritis, lewati mirror (database tetap
            # aman di lokal). force=True tetap mencoba mirror final.
            if not force:
                free = _drive_free_bytes()
                if free is not None and free < DRIVE_MIN_FREE_BYTES:
                    _log('warning',
                         f"ruang Drive kritis ({free} byte tersisa < "
                         f"{DRIVE_MIN_FREE_BYTES}); mirror ke Drive dilewati, "
                         f"database tetap aman di lokal: {self._drive_path}")
                    return
            # env.copy() membuat snapshot LMDB yang konsisten secara transaksional.
            # Byte-copy mentah dari .mdb yang sedang ditulis bisa menangkap state
            # setengah jadi dan korup saat resume.
            tmp_local = self._local_path + ".snap"
            try:
                if os.path.exists(tmp_local):
                    os.unlink(tmp_local)
            except Exception:
                pass
            used_snapshot = False
            try:
                self._env.copy(tmp_local, compact=False)
                used_snapshot = True
            except TypeError:
                # Versi lmdb lama tanpa argumen compact.
                self._env.copy(tmp_local)
                used_snapshot = True
            except Exception as e:
                _log('warning', f"env.copy snapshot gagal, fallback byte-copy {self._drive_path}: {e}")
            # Sumber yang ditulis ke Drive: snapshot bila ada, selain itu file lokal.
            written_src = tmp_local if (used_snapshot and os.path.exists(tmp_local)) else self._local_path
            # Hapus sidecar .gen dulu sebelum menulis .mdb, agar .gen tidak pernah
            # lebih basi dari .mdb yang dideskripsikannya.
            gen_path = self._drive_path + ".gen"
            try:
                if os.path.exists(gen_path):
                    os.unlink(gen_path)
                    _fsync_dir(gen_path)
            except Exception as e:
                _log('debug', f"hapus sidecar .gen lama gagal {self._drive_path}: {e}")
            if used_snapshot and os.path.exists(tmp_local):
                _copy_file_atomic(tmp_local, self._drive_path)
            else:
                _copy_file_atomic(self._local_path, self._drive_path)
            # Catat generasi isi (BLAKE3) ke sidecar Drive setelah .mdb sukses ditulis.
            try:
                gen = _blake3_file(written_src)
                _write_text_atomic(gen_path, gen)
            except Exception as e:
                _log('debug', f"tulis sidecar .gen gagal {self._drive_path}: {e}")
            if used_snapshot and os.path.exists(tmp_local):
                try: os.unlink(tmp_local)
                except Exception: pass
            # Hapus permanen versi lama di Trash (sisa os.replace di Drive FUSE).
            # Throttle purge; force=True selalu purge.
            if PURGE_TRASHED_MDB:
                _now_purge = time.monotonic()
                if force or (_now_purge - object.__getattribute__(self, '_last_purge')) >= PURGE_MIN_INTERVAL_SEC:
                    try:
                        base = os.path.basename(self._drive_path)
                        _purge_trashed_by_name(base)
                        _purge_trashed_by_name(base + ".gen")
                        object.__setattr__(self, '_last_purge', time.monotonic())
                    except Exception as e:
                        _log('debug', f"purge trash gagal {self._drive_path}: {e}")
            object.__setattr__(self, '_last_mirror', time.monotonic())
        except Exception as e:
            _log('warning', f"mirror ke Drive gagal {self._drive_path}: {e}")

    def sync(self, force: bool = False):
        with self._sync_lock:
            try:
                self._env.sync(force)
            except TypeError:
                # Versi lmdb lama tidak menerima argumen posisional pada sync().
                self._env.sync()
            self._mirror_to_drive(force=force)

    def close(self):
        # Flush + mirror final sebelum menutup.
        with self._sync_lock:
            try:
                try: self._env.sync(True)
                except TypeError: self._env.sync()
            except Exception as e:
                _log('debug', f"sync sebelum close gagal: {e}")
            self._mirror_to_drive(force=True)
            self._env.close()
            # Setelah env ditutup, sapu scratch yatim di folder Drive.
            try:
                _sweep_scratch_in_dir(os.path.dirname(self._drive_path) or '.')
            except Exception as e:
                _log('debug', f"sweep scratch Drive saat close gagal: {e}")

    def set_mapsize(self, size):
        # env.copy() dan set_mapsize tidak boleh berjalan bersamaan; _sync_lock
        # memastikan keduanya saling eksklusif.
        with self._sync_lock:
            return self._env.set_mapsize(size)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_env'), name)

def _open_lmdb_env(env_path: str, max_dbs: int = LMDB_MAX_DBS, map_size: int = LMDB_MAP_SIZE):
    """Buka env LMDB sebagai file tunggal (subdir=False). Env dijalankan di
    mirror lokal (/content, ext4) lalu disalin balik ke Drive pada sync()/close().
    Bila lokal tidak tersedia, fallback membuka langsung di Drive."""
    drive_parent = os.path.dirname(env_path) or '.'
    os.makedirs(drive_parent, exist_ok=True)

    def _open_at(path: str):
        return lmdb.open(
            path, subdir=False, map_size=map_size, max_dbs=max_dbs,
            sync=True, metasync=True, writemap=False, meminit=False,
        )

    # Bersihkan scratch yatim di folder Drive sebelum env dibuka.
    try:
        _sweep_scratch_in_dir(drive_parent)
    except Exception as e:
        _log('debug', f"sweep scratch Drive saat open gagal {drive_parent}: {e}")

    try:
        os.makedirs(LOCAL_LMDB_DIR, exist_ok=True)
        local_path = _local_mirror_path(env_path)
        # Drive adalah sumber kebenaran saat resume. Keputusan copy-in berbasis
        # isi (via sidecar .gen), bukan ukuran, karena .mdb sering berukuran
        # tetap walau isinya berbeda.
        try:
            if os.path.exists(env_path):
                need_copy = True
                if os.path.exists(local_path):
                    same_size = (os.path.getsize(local_path) == os.path.getsize(env_path))
                    if same_size:
                        drive_gen = _read_text_file(env_path + ".gen")
                        if drive_gen:
                            try:
                                need_copy = (_blake3_file(local_path) != drive_gen)
                            except Exception:
                                need_copy = True
                        else:
                            need_copy = True
                if need_copy:
                    # Sebelum copy-in, pastikan .mdb Drive tidak korup (sisa
                    # in-place crash). Bila Drive korup tapi lokal valid, pakai
                    # lokal; mirror akan memperbaiki Drive saat sync berikutnya.
                    drive_valid = _is_valid_lmdb_file(env_path)
                    if drive_valid:
                        _copy_file_atomic(env_path, local_path)
                    else:
                        local_valid = (os.path.exists(local_path)
                                       and _is_valid_lmdb_file(local_path))
                        if local_valid:
                            _log('warning',
                                 f"'.mdb' Drive tampak KORUP (mungkin sisa in-place "
                                 f"crash) -> mempertahankan mirror lokal VALID, "
                                 f"copy-in dilewati: {env_path}")
                        else:
                            _log('warning',
                                 f"'.mdb' Drive tampak KORUP dan mirror lokal tidak "
                                 f"valid/ tidak ada -> tetap copy-in apa adanya "
                                 f"(satu-satunya salinan); periksa manual: {env_path}")
                            _copy_file_atomic(env_path, local_path)
        except Exception as e:
            _log('warning', f"copy-in dari Drive gagal {env_path}: {e}")
        env = _open_at(local_path)
        return _DriveMirroredEnv(env, local_path, env_path)
    except Exception as e:
        _log('warning', f"mirror lokal tidak tersedia, buka langsung di Drive {env_path}: {e}")
        return _open_at(env_path)

# ═════════════════════════════════════════════════════════════════
# Rate limiter adaptif (token bucket, dibagi semua worker).
# Saat kena 403/429 rate turun 50%; saat sukses rate naik perlahan.
# ═════════════════════════════════════════════════════════════════
class AdaptiveRateLimiter:
    def __init__(self, rate: float = API_RATE_PER_SEC, burst: int = API_BURST):
        self._lock     = threading.Lock()
        self._max_rate = rate
        self._min_rate = 0.5
        self._rate     = rate
        self._burst    = float(burst)
        self._tokens   = float(burst)
        self._last     = time.monotonic()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                # Clamp ke [0, burst]: cegah token negatif (akibat perubahan
                # rate konkuren) yang membuat perhitungan wait melebihi target.
                self._tokens = min(self._burst, max(0.0, self._tokens + (now - self._last) * self._rate))
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / max(self._min_rate, self._rate)
            time.sleep(min(wait, 5.0))

    def penalize(self):
        with self._lock:
            self._rate = max(self._min_rate, self._rate * 0.5)
            _log('info', f"rate limiter penalized -> {self._rate:.2f}/s")

    def reward(self):
        with self._lock:
            # Additive increase (AIMD): naik pelan +2% dari rate maksimum per
            # sukses, bukan kelipatan dari rate saat ini. Mencegah rate cepat
            # melonjak balik ke max sesaat setelah penalize dan langsung memicu
            # 429 lagi (osilasi penalize/reward).
            self._rate = min(self._max_rate, self._rate + self._max_rate * 0.02)

_rate_limiter = AdaptiveRateLimiter()

# ───────────────────── HTTP ERROR CLASSIFICATION ─────────────────────
def _classify_http_error(code: int, reason: str = '') -> str:
    # Kembalikan salah satu: 'recoverable', 'fatal', atau 'fatal_global'.
    #  - recoverable : sesaat, aman di-retry dengan backoff (rate-limit, 5xx).
    #  - fatal       : gagal permanen tapi spesifik per-file (mis. 404 file
    #                  hilang). Tidak membuka circuit breaker.
    #  - fatal_global: penyebab mempengaruhi semua request (kuota harian/proyek
    #                  habis, storage quota, auth gagal). Membuka circuit
    #                  breaker agar sisa request berhenti cepat.
    # 403 bisa berarti rate-limit sesaat (recoverable) atau penolakan/kuota
    # habis (fatal). Substring 'limit' polos terlalu longgar (ikut menangkap
    # 'Daily Limit Exceeded', 'sharing limit'), jadi dipakai pola spesifik.
    if code == 403:
        r = (reason or '').lower()
        # Quota harian / limit yang fatal: berhenti, retry tidak akan menolong.
        # Catatan: 'limitexceeded' sebagai substring juga cocok dengan
        # 'userRateLimitExceeded' (karena mengandung 'limitexceeded'). Namun
        # guard di bawah ('ratelimitexceeded' / 'userratelimitexceeded') diperiksa
        # SEBELUM return 'fatal_global', sehingga rate-limit sesaat tetap
        # dikembalikan sebagai 'recoverable' meski cocok dengan fatal_markers.
        # Urutan pengecekan ini kritis: jangan ubah tanpa mempertimbangkan
        # interaksi antar-marker.
        fatal_markers = ('dailylimitexceeded', 'daily limit', 'quotaexceeded',
                         'quota exceeded', 'limitexceeded', 'storagequota',
                         'sharing limit')
        if any(m in r for m in fatal_markers):
            if 'ratelimitexceeded' in r or 'userratelimitexceeded' in r:
                return 'recoverable'
            return 'fatal_global'
        if 'ratelimit' in r or 'userratelimit' in r:
            return 'recoverable'
        return 'fatal'
    if code == 401: return 'fatal_global'
    if code == 404: return 'fatal'
    # 408 transien (timeout koneksi lambat), aman di-retry. 409 tidak disertakan.
    if code in (408, 429, 500, 502, 503, 504): return 'recoverable'
    return 'fatal'

class _CircuitBreaker:
    """Circuit breaker global thread-safe. Saat dibuka (kuota/auth habis),
    seluruh API call berikutnya gagal cepat tanpa menyentuh jaringan.
    State in-memory; run berikutnya dimulai dengan sirkuit tertutup."""
    def __init__(self):
        self._lock = threading.Lock()
        self._open = False
        self._reason = ''

    def open(self, reason: str):
        with self._lock:
            if not self._open:
                self._open = True
                self._reason = reason
                _log('error', f"CIRCUIT BREAKER terbuka (API dihentikan): {reason}")

    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def reset(self):
        with self._lock:
            self._open = False
            self._reason = ''

_circuit_breaker = _CircuitBreaker()

def _http_error_reason(e: HttpError) -> str:
    # Gabungkan reason dari content + error_details + str(e) untuk klasifikasi akurat.
    parts = []
    try:
        content = getattr(e, 'content', None)
        if content:
            parts.append(content.decode('utf-8', 'ignore') if isinstance(content, bytes) else str(content))
    except Exception:
        pass
    try:
        details = getattr(e, 'error_details', None)
        if details:
            parts.append(str(details))
    except Exception:
        pass
    parts.append(str(e))
    return ' '.join(parts)

def _backoff_sleep(attempt: int):
    base = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
    time.sleep(base + random.uniform(0, base * 0.5))

def _drive_execute(request_factory, max_retry: int = MAX_RETRY):
    """Eksekusi Drive API call dengan rate limit, retry, dan backoff.
    request_factory membuat request baru tiap percobaan (tidak boleh dipakai ulang).
    Return (result, None) atau (None, error_string).
    Di jalur konkuren, gunakan _thread_drive() agar tiap worker punya service sendiri.
    """
    last_err = 'unknown'
    for attempt in range(max_retry):
        # Fail-fast bila sirkuit terbuka (kuota/auth global): jangan kirim
        # request yang pasti ditolak.
        if _circuit_breaker.is_open():
            return None, f"circuit_open:{_circuit_breaker.reason()}"
        _rate_limiter.acquire()
        try:
            result = request_factory().execute()
            _rate_limiter.reward()
            return result, None
        except HttpError as e:
            code = int(e.resp.status)
            kind = _classify_http_error(code, _http_error_reason(e))
            last_err = f"http_{code}"
            if kind == 'fatal_global':
                _circuit_breaker.open(f"http_{code}: {_http_error_reason(e)[:160]}")
                return None, f"fatal_global:{code}"
            if kind == 'fatal':
                return None, f"fatal:{code}"
            _rate_limiter.penalize()
            if attempt < max_retry - 1:
                _backoff_sleep(attempt)
        except Exception as ex:
            last_err = str(ex)
            _log('warning', f"_drive_execute exception attempt={attempt}: {ex}")
            if attempt < max_retry - 1:
                _backoff_sleep(attempt)
    return None, f"retry_exhausted:{last_err}"

# ═════════════════════════════════════════════════════════════════
# Manifest file persisten berbasis LMDB (transaksional, crash-safe).
# Tiap operasi adalah transaksi ACID.
# ═════════════════════════════════════════════════════════════════
class LMDBIdentityError(RuntimeError):
    """Dilempar bila database LMDB bertipe atau milik folder lain dari yang
    diharapkan (mis. file .mdb tertukar antar bundle). File env kini diberi
    nama deskriptif (hash_<id>.mdb dll.) untuk meminimalkan risiko ini."""

def _stamp_or_verify_identity(env, meta_db, db_type: str, folder_id: str, lock: threading.RLock):
    """Tanam stempel identitas (db_type + folder_id) bila DB baru/kosong; bila
    sudah ada, verifikasi cocok untuk mencegah tertukar antar bundle."""
    with lock, env.begin(write=True) as txn:
        cur_type = txn.get(b"__db_type", db=meta_db)
        cur_fid  = txn.get(b"__folder_id", db=meta_db)
        if cur_type is None and cur_fid is None:
            txn.put(b"__db_type", db_type.encode(), db=meta_db)
            txn.put(b"__folder_id", folder_id.encode(), db=meta_db)
            return
        ct = cur_type.decode() if cur_type else None
        cf = cur_fid.decode() if cur_fid else None
        if (ct and ct != db_type) or (cf and cf != folder_id):
            raise LMDBIdentityError(
                f"Identitas database tidak cocok: ditemukan db_type={ct} folder_id={cf}, "
                f"diharapkan db_type={db_type} folder_id={folder_id}. "
                f"Kemungkinan file .mdb tertukar antar bundle.")

class PersistentFileManifest:
    def __init__(self, folder_id: str, env_path: Optional[str] = None):
        self.folder_id = folder_id
        self.env_path = env_path or folder_paths(folder_id)['manifest_env']
        self.lock = threading.RLock()
        self.env = _open_lmdb_env(self.env_path)
        self._files_db     = self.env.open_db(b"files")
        self._pending_db   = self.env.open_db(b"pending")
        self._processed_db = self.env.open_db(b"processed")
        self._failed_db    = self.env.open_db(b"failed")
        self._meta_db      = self.env.open_db(b"meta")
        _stamp_or_verify_identity(self.env, self._meta_db, 'manifest', folder_id, self.lock)
        # Pesan status yang di-defer dari _post_scan_integrity agar dicetak
        # setelah garis batas di _analyze_folder_body.
        self._deferred_status_msg: Optional[str] = None

    # ---- internal helpers ----
    def _meta_get(self, key: str, default=None):
        with self.env.begin() as txn:
            v = txn.get(key.encode(), db=self._meta_db)
        return json.loads(v) if v is not None else default

    def _meta_put_txn(self, txn, key: str, value):
        txn.put(key.encode(), json.dumps(value).encode(), db=self._meta_db)

    # ---- API publik ----
    def add_files(self, files: List[Dict]):
        def _do(txn):
            for f in files:
                fid = f['id']; key = fid.encode()
                old = txn.get(key, db=self._files_db)
                txn.put(key, _record_dumps(f), db=self._files_db)
                if txn.get(key, db=self._processed_db) is None:
                    txn.put(key, b"1", db=self._pending_db)
                else:
                    # Isi file berubah (md5 beda) -> re-pending agar hash dihitung ulang.
                    try:
                        old_md5 = (_record_loads(old) or {}).get('md5Checksum') if old else None
                    except Exception:
                        old_md5 = None
                    new_md5 = f.get('md5Checksum')
                    if new_md5 and old_md5 and new_md5 != old_md5:
                        txn.delete(key, db=self._processed_db)
                        txn.delete(key, db=self._failed_db)
                        txn.put(key, b"1", db=self._pending_db)
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def mark_scan_complete(self, failed_folders: Dict[str, str]):
        def _do(txn):
            self._meta_put_txn(txn, 'scan_complete', True)
            self._meta_put_txn(txn, 'failed_folders', dict(failed_folders))
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def invalidate_scan(self):
        def _do(txn):
            self._meta_put_txn(txn, 'scan_complete', False)
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def get_pending_files(self):
        """Generator (record) streaming atas seluruh entri pending dalam satu
        transaksi/cursor. Lock dipegang selama iterasi agar tidak bentrok dengan
        set_mapsize() di thread lain."""
        with self.lock, self.env.begin() as txn:
            cur = txn.cursor(db=self._pending_db)
            for k, _ in cur:
                data = txn.get(k, db=self._files_db)
                if data:
                    rec = _record_loads(data)
                    if rec is not None:
                        yield rec

    def get_all_file_ids(self) -> Set[str]:
        with self.lock, self.env.begin() as txn:
            return {k.decode() for k, _ in txn.cursor(db=self._files_db)}

    def iter_files(self):
        """Generator (file_id, record) streaming dalam satu transaksi/cursor.

        Pengganti pola get_all_file_ids() + get_file() per file, yang membuka
        transaksi LMDB read dan mengambil lock untuk setiap file -> jutaan
        transaksi pada folder besar. Di sini seluruh files_db dilintasi sekali
        dengan satu cursor, tanpa menahan seluruh id/record di RAM. Lock
        dipegang selama iterasi (sama seperti LMDBStorage.iterate) agar tidak
        bentrok dengan set_mapsize() saat map tumbuh di thread lain.
        """
        with self.lock, self.env.begin() as txn:
            cur = txn.cursor(db=self._files_db)
            if cur.first():
                while True:
                    yield cur.key().decode(), _record_loads(cur.value())
                    if not cur.next():
                        break

    def get_file(self, file_id: str) -> Optional[Dict]:
        with self.lock, self.env.begin() as txn:
            data = txn.get(file_id.encode(), db=self._files_db)
            return _record_loads(data) if data else None

    def mark_processed(self, file_id: str):
        key = file_id.encode()
        def _do(txn):
            txn.delete(key, db=self._pending_db)
            txn.put(key, b"1", db=self._processed_db)
            txn.delete(key, db=self._failed_db)
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def mark_failed(self, file_id: str, error: str):
        key = file_id.encode()
        # Paksa string: bila pemanggil mengirim objek exception, json.dumps bisa
        # melempar TypeError sehingga file gagal tidak pernah tertandai dan
        # terjebak loop re-pending.
        err_str = error if isinstance(error, str) else str(error)
        def _do(txn):
            txn.delete(key, db=self._pending_db)
            txn.put(key, json.dumps(err_str).encode(), db=self._failed_db)
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def re_pending(self, file_id: str):
        """Kembalikan entry ke pending agar diproses ulang."""
        key = file_id.encode()
        def _do(txn):
            if txn.get(key, db=self._files_db) is not None:
                txn.delete(key, db=self._processed_db)
                txn.delete(key, db=self._failed_db)
                txn.put(key, b"1", db=self._pending_db)
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def re_pending_all_failed(self) -> int:
        """Kembalikan semua file yang gagal di-hash ke pending agar di-retry pada
        run ini. Menjamin tidak ada file yang terjebak permanen di failed_db dan
        terlewat dari deteksi duplikat. Return jumlah file yang di-retry."""
        moved = 0
        def _do(txn):
            nonlocal moved
            keys = [k for k, _ in txn.cursor(db=self._failed_db)]
            for key in keys:
                if txn.get(key, db=self._files_db) is None:
                    # File sudah tidak ada di manifest: bersihkan saja.
                    txn.delete(key, db=self._failed_db)
                    continue
                txn.delete(key, db=self._failed_db)
                txn.delete(key, db=self._processed_db)
                txn.put(key, b"1", db=self._pending_db)
                moved += 1
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)
        return moved

    def reconcile(self) -> Dict:
        """Rekonsiliasi keseimbangan scan sebagai bukti tidak ada file lolos.

        Setiap file yang ter-indeks (ada di files_db) harus berada di salah satu
        status: processed (selesai), failed (gagal, akan di-retry), atau pending
        (belum diproses). Bila indexed != processed + failed + pending, ada
        anomali (file ter-indeks tapi tak terhitung di status mana pun) ->
        balanced=False agar ditampilkan sebagai peringatan.
        """
        with self.lock, self.env.begin() as txn:
            indexed   = txn.stat(db=self._files_db)['entries']
            processed = txn.stat(db=self._processed_db)['entries']
            failed    = txn.stat(db=self._failed_db)['entries']
            pending   = txn.stat(db=self._pending_db)['entries']
            scan_complete = bool(self._meta_get('scan_complete', False))
            failed_folders = self._meta_get('failed_folders', {}) or {}
        with self.lock, self.env.begin() as txn:
            drift = self._meta_get('scan_drift', {}) or {}
        # drift_known=False bila API gagal -> jaminan ditahan (fail-safe).
        drift_known    = bool(drift.get('known', False))
        drift_count    = int(drift.get('count', 0) or 0)
        drift_detected = drift_known and drift_count > 0
        with self.lock, self.env.begin() as txn:
            lv = self._meta_get('listing_verify', {}) or {}
        # ran=False bila fitur off; known=False bila pass kedua gagal.
        lv_ran         = bool(lv.get('ran', False))
        lv_known       = bool(lv.get('known', False))
        lv_discrepancy = int(lv.get('discrepancy', 0) or 0)
        balanced = (indexed == processed + failed + pending) and not failed_folders
        return {
            'indexed': indexed, 'processed': processed,
            'failed': failed, 'pending': pending,
            'scan_complete': scan_complete,
            'failed_folder_count': len(failed_folders),
            'balanced': balanced,
            'drift_known': drift_known,
            'drift_count': drift_count,
            'drift_detected': drift_detected,
            'listing_verify_ran': lv_ran,
            'listing_verify_known': lv_known,
            'listing_verify_discrepancy': lv_discrepancy,
        }

    def set_scan_drift(self, count: int, known: bool):
        """Simpan hasil deteksi drift (jumlah perubahan Drive selama scan).
        known=False bila tak terhitung (API gagal) -> reconcile menahan jaminan."""
        def _do(txn):
            self._meta_put_txn(txn, 'scan_drift', {'count': int(count), 'known': bool(known)})
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def set_listing_verify(self, ran: bool, known: bool, discrepancy: int):
        """Simpan hasil verifikasi double-listing (pass kedua).
        ran=False bila fitur off; known=False bila pass kedua tak lengkap."""
        def _do(txn):
            self._meta_put_txn(txn, 'listing_verify',
                               {'ran': bool(ran), 'known': bool(known),
                                'discrepancy': int(discrepancy)})
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def get_stats(self) -> Dict:
        with self.lock, self.env.begin() as txn:
            return {
                'total': txn.stat(db=self._files_db)['entries'],
                'pending': txn.stat(db=self._pending_db)['entries'],
                'processed': txn.stat(db=self._processed_db)['entries'],
                'failed': txn.stat(db=self._failed_db)['entries'],
                'scan_complete': bool(self._meta_get('scan_complete', False)),
                'failed_folders': self._meta_get('failed_folders', {}) or {},
            }

    def flush(self):
        # sync(force=True) memaksa flush ke storage, penting di Drive FUSE.
        with self.lock:
            try: self.env.sync(force=True)
            except Exception as e: _log('debug', f"manifest sync: {e}")

    def remove_deleted_files(self, current_ids: Set[str]):
        def _do(txn):
            to_remove = []
            for k, _ in txn.cursor(db=self._files_db):
                if k.decode() not in current_ids:
                    to_remove.append(k)
            for key in to_remove:
                txn.delete(key, db=self._files_db)
                txn.delete(key, db=self._pending_db)
                txn.delete(key, db=self._processed_db)
                txn.delete(key, db=self._failed_db)
            if to_remove:
                _log('info', f"Removed {len(to_remove)} deleted files from manifest")
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

# ═════════════════════════════════════════════════════════════════
# Penyimpanan utama berbasis LMDB (mendukung jutaan file tanpa OOM).
# ═════════════════════════════════════════════════════════════════
class LMDBStorage:
    def __init__(self, folder_id: str, env_path: Optional[str] = None):
        self.folder_id = folder_id
        self.env_path = env_path or folder_paths(folder_id)['hash_env']
        self.env = _open_lmdb_env(self.env_path)
        self.lock = threading.RLock()
        self._meta_db   = self.env.open_db(b"meta")
        # blake3_idx dibuka untuk kompatibilitas env lama; tidak lagi dibaca/ditulis.
        self._blake3_idx= self.env.open_db(b"blake3_idx")
        self._ident_db  = self.env.open_db(b"ident")
        _stamp_or_verify_identity(self.env, self._ident_db, 'hash', folder_id, self.lock)

    def put(self, file_id: str, record: Dict):
        # blake3_idx tidak lagi dipelihara; record cukup ditulis ke meta_db.
        def _do(txn):
            txn.put(file_id.encode(), _record_dumps(record), db=self._meta_db)
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def get(self, file_id: str) -> Optional[Dict]:
        with self.lock:
            with self.env.begin() as txn:
                data = txn.get(file_id.encode(), db=self._meta_db)
                return _record_loads(data) if data else None

    def delete(self, file_id: str):
        with self.lock:
            record = self.get(file_id)
            if not record:
                return
            # blake3_idx tidak lagi dipelihara; cukup hapus dari meta_db.
            def _do(txn):
                txn.delete(file_id.encode(), db=self._meta_db)
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)

    def exists(self, file_id: str) -> bool:
        with self.lock:
            with self.env.begin() as txn:
                return txn.get(file_id.encode(), db=self._meta_db) is not None

    def count(self) -> int:
        with self.lock:
            with self.env.begin() as txn:
                return txn.stat(db=self._meta_db)['entries']

    def iterate(self):
        """Generator (fid, record) streaming tanpa memuat semua ke RAM.
        Lock dipegang selama iterasi agar tidak bentrok dengan set_mapsize()."""
        with self.lock:
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self._meta_db)
                if cursor.first():
                    while True:
                        yield cursor.key().decode(), _record_loads(cursor.value())
                        if not cursor.next():
                            break

    def prune_orphans(self, valid_ids: Set[str]) -> int:
        """Hapus record hash yang file_id-nya TIDAK ada di `valid_ids` (himpunan
        file yang masih ada menurut manifest terbaru). Dipakai untuk membuang
        hash orphan (file yang sudah dihapus/keluar) agar hash.mdb tidak
        membengkak. PEMANGGIL WAJIB memastikan valid_ids lengkap & tepercaya
        (scan tuntas, tak ada folder gagal, API sehat) sebelum memanggil ini;
        bila valid_ids tidak lengkap, record valid bisa terhapus keliru.
        Return jumlah record yang dihapus.
        """
        # Kumpulkan orphan lewat cursor read dulu (jangan menghapus saat cursor
        # read aktif -> perilaku tak terdefinisi di LMDB).
        orphan_keys: List[bytes] = []
        with self.lock:
            with self.env.begin() as txn:
                cur = txn.cursor(db=self._meta_db)
                if cur.first():
                    while True:
                        k = cur.key()
                        if k.decode() not in valid_ids:
                            orphan_keys.append(bytes(k))
                        if not cur.next():
                            break
        if not orphan_keys:
            return 0
        def _do(txn):
            for key in orphan_keys:
                txn.delete(key, db=self._meta_db)
        with self.lock:
            _lmdb_write_with_grow(self.env, _do, lock=self.lock)
        _log('info', f"prune hash orphan: {len(orphan_keys)} record dibuang (file sudah tidak ada di manifest)")
        return len(orphan_keys)

    def sync(self):
        # force=True: paksa flush ke storage, kurangi jendela data-loss di Drive FUSE.
        try:
            self.env.sync(force=True)
        except Exception as e:
            _log('debug', f"storage sync: {e}")

    def close(self):
        self.env.close()

# ───────────────────── DEEP FOLDER JOURNAL (persistent BFS queue) ─────────────────────
class FolderJournal:
    """Journal scan folder berbasis LMDB. Queue pending, visited, dan page token
    dipersist agar scan bisa resume setelah crash. Menyimpan juga Drive Changes
    startPageToken untuk incremental scan."""
    def __init__(self, folder_id: str, env_path: Optional[str] = None):
        self.folder_id = folder_id
        self._lock = threading.RLock()
        self.env_path = env_path or folder_paths(folder_id)['journal_env']
        self.env = _open_lmdb_env(self.env_path, max_dbs=8)
        self._visited_db = self.env.open_db(b"visited")
        self._ptok_db    = self.env.open_db(b"page_tokens")
        self._meta_db    = self.env.open_db(b"meta")  # pending list, changes_token
        self._ident_db   = self.env.open_db(b"ident")
        _stamp_or_verify_identity(self.env, self._ident_db, 'journal', folder_id, self._lock)

    def set_pending(self, folder_ids: List[str]):
        def _do(txn):
            txn.put(b"pending", json.dumps(list(folder_ids)).encode(), db=self._meta_db)
        with self._lock:
            _lmdb_write_with_grow(self.env, _do, lock=self._lock)

    def get_pending(self) -> List[str]:
        with self._lock, self.env.begin() as txn:
            v = txn.get(b"pending", db=self._meta_db)
            return json.loads(v) if v else []

    def visited_set(self) -> Set[str]:
        with self._lock, self.env.begin() as txn:
            return {k.decode() for k, _ in txn.cursor(db=self._visited_db)}

    def set_page_token(self, folder_id: str, token: str):
        def _do(txn):
            txn.put(folder_id.encode(), token.encode(), db=self._ptok_db)
        with self._lock:
            _lmdb_write_with_grow(self.env, _do, lock=self._lock)

    def get_page_token(self, folder_id: str) -> Optional[str]:
        with self._lock, self.env.begin() as txn:
            v = txn.get(folder_id.encode(), db=self._ptok_db)
            return v.decode() if v else None

    def clear_page_token(self, folder_id: str):
        def _do(txn):
            txn.delete(folder_id.encode(), db=self._ptok_db)
        with self._lock:
            _lmdb_write_with_grow(self.env, _do, lock=self._lock)

    # ---- Drive Changes API (incremental scan) ----
    def set_changes_token(self, token: str):
        def _do(txn):
            txn.put(b"changes_token", token.encode(), db=self._meta_db)
        with self._lock:
            _lmdb_write_with_grow(self.env, _do, lock=self._lock)

    def get_changes_token(self) -> Optional[str]:
        with self._lock, self.env.begin() as txn:
            v = txn.get(b"changes_token", db=self._meta_db)
            return v.decode() if v else None

    def add_visited(self, folder_ids: List[str]):
        """Persist folder baru agar subtree run berikutnya tetap mengenalnya."""
        def _do(txn):
            for fid in folder_ids:
                txn.put(fid.encode(), b"1", db=self._visited_db)
        with self._lock:
            _lmdb_write_with_grow(self.env, _do, lock=self._lock)

    def add_pending(self, folder_ids: List[str]):
        """Append folder baru ke pending list secara atomik tanpa menimpa list
        yang ada. Idempoten via dedup. Aman dipanggil dari worker scan paralel."""
        if not folder_ids:
            return
        def _do(txn):
            v = txn.get(b"pending", db=self._meta_db)
            cur = json.loads(v) if v else []
            seen = set(cur)
            changed = False
            for fid in folder_ids:
                if fid not in seen:
                    cur.append(fid); seen.add(fid); changed = True
            if changed:
                txn.put(b"pending", json.dumps(cur).encode(), db=self._meta_db)
        with self._lock:
            _lmdb_write_with_grow(self.env, _do, lock=self._lock)

    def clear_scan_state(self):
        """Reset state scan (visited/page_tokens/pending) untuk full scan baru.
        changes_token dipertahankan."""
        def _do(txn):
            txn.drop(self._visited_db, delete=False)
            txn.drop(self._ptok_db, delete=False)
            txn.delete(b"pending", db=self._meta_db)
        with self._lock:
            _lmdb_write_with_grow(self.env, _do, lock=self._lock)

# ───────────────────── DOWNLOAD ENGINE ─────────────────────
_tls = threading.local()

def _build_drive():
    return build('drive', 'v3')

def _thread_drive():
    if not hasattr(_tls, 'svc'):
        _tls.svc = _build_drive()
    return _tls.svc

def download_bytes_stream(file_id: str, size_mb: float = 0, file_path: Optional[str] = None):
    """Download bytes (atau langsung ke file_path).
    Return bytes/True jika sukses (file 0-byte mengembalikan b''), None jika
    gagal. Request dibangun ulang tiap percobaan.
    """
    for attempt in range(MAX_RETRY):
        if _circuit_breaker.is_open():
            return None
        _rate_limiter.acquire()
        try:
            svc = _thread_drive()
            req = svc.files().get_media(fileId=file_id)
            if file_path:
                with open(file_path, 'wb') as f:
                    dl = MediaIoBaseDownload(f, req, chunksize=STREAM_CHUNK)
                    done = False
                    while not done:
                        _, done = dl.next_chunk()
                _rate_limiter.reward()
                return True
            else:
                buf = io.BytesIO()
                dl  = MediaIoBaseDownload(buf, req, chunksize=STREAM_CHUNK)
                done = False
                while not done:
                    _, done = dl.next_chunk()
                _rate_limiter.reward()
                return buf.getvalue()
        except HttpError as e:
            code = int(e.resp.status)
            kind = _classify_http_error(code, _http_error_reason(e))
            if kind == 'fatal_global':
                _circuit_breaker.open(f"http_{code}: {_http_error_reason(e)[:160]}")
                return None
            if kind == 'fatal': return None
            _rate_limiter.penalize()
            if attempt < MAX_RETRY - 1:
                _backoff_sleep(attempt); continue
            return None
        except Exception as ex:
            _log('warning', f"download exception fid={file_id} attempt={attempt}: {ex}")
            if attempt < MAX_RETRY - 1:
                _backoff_sleep(attempt); continue
            return None
    return None

def download_and_hash_stream(file_id: str) -> Tuple[Optional[str], int]:
    """Streaming hash anti-OOM: buffer dikosongkan tiap chunk."""
    for attempt in range(MAX_RETRY):
        if _circuit_breaker.is_open():
            return None, 0
        _rate_limiter.acquire()
        try:
            svc = _thread_drive()
            req = svc.files().get_media(fileId=file_id)
            h   = blake3.blake3()
            buf = io.BytesIO()
            dl  = MediaIoBaseDownload(buf, req, chunksize=STREAM_CHUNK)
            done = False
            total = 0
            while not done:
                _, done = dl.next_chunk()
                chunk = buf.getvalue()
                if chunk:
                    h.update(chunk)
                    total += len(chunk)
                    buf.seek(0); buf.truncate(0)
            _rate_limiter.reward()
            return h.hexdigest(), total
        except HttpError as e:
            code = int(e.resp.status)
            kind = _classify_http_error(code, _http_error_reason(e))
            if kind == 'fatal_global':
                _circuit_breaker.open(f"http_{code}: {_http_error_reason(e)[:160]}")
                return None, 0
            if kind == 'fatal': return None, 0
            _rate_limiter.penalize()
            if attempt < MAX_RETRY - 1:
                _backoff_sleep(attempt); continue
            return None, 0
        except Exception as ex:
            _log('warning', f"stream_hash exception fid={file_id}: {ex}")
            if attempt < MAX_RETRY - 1:
                _backoff_sleep(attempt); continue
            return None, 0
    return None, 0

def compute_blake3_hash(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()

# ───────────────────── PILLOW SAFE OPEN ─────────────────────
def _safe_open_rgb(img_data: bytes) -> Image.Image:
    buf = io.BytesIO(img_data)
    img = Image.open(buf)
    img.load()
    try:
        # Normalisasi orientasi EXIF; exif_transpose bisa mengembalikan objek baru.
        try:
            transposed = ImageOps.exif_transpose(img)
        except Exception as e:
            _log('debug', f"exif_transpose failed: {e}")
            transposed = img
        if transposed is not img:
            img.close()
            img = transposed
        if img.mode != 'RGB':
            converted = img.convert('RGB')
            if converted is not img:
                img.close()
                img = converted
        return img
    finally:
        # BytesIO sumber tidak lagi diperlukan setelah Image dimuat.
        try:
            buf.close()
        except Exception:
            pass

def _safe_resize(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    max_w, max_h = target_size
    w, h = img.size
    if w <= max_w and h <= max_h:
        return img
    ratio = min(max_w / w, max_h / h)
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.Resampling.LANCZOS)

# ───────────────────── HASHING ─────────────────────
def _dhash_vertical(img: Image.Image) -> str:
    """dHash V (gradien atas-bawah): dihitung sebagai dHash H pada citra yang
    dirotasi 90°. Melengkapi dHash H untuk foto identik yang diturunkan resolusinya."""
    rotated = img.rotate(90, expand=True)
    try:
        return str(imagehash.dhash(rotated, hash_size=DHASH_HASH_SIZE))
    finally:
        if rotated is not img:
            try: rotated.close()
            except Exception: pass

def _canonical_phash_dhash(img: Image.Image) -> Tuple[str, str, str]:
    """Hitung triplet (pHash, dHash H, dHash V) dari orientasi citra apa adanya."""
    ph = imagehash.phash(img, hash_size=PHASH_HASH_SIZE)
    dh = imagehash.dhash(img, hash_size=DHASH_HASH_SIZE)
    dv = _dhash_vertical(img)
    return str(ph), str(dh), dv

def _color_grid(img: Image.Image) -> Optional[List[List[float]]]:
    """Sidik warna per-blok: rata-rata RGB tiap blok pada grid COLOR_GRID x COLOR_GRID
    setelah resize ke COLOR_GRID_IMG_SIZE. Return list [[r,g,b],...] atau None."""
    try:
        n = COLOR_GRID
        size = COLOR_GRID_IMG_SIZE
        # Konversi dan resize menghasilkan objek Image baru; tutup setelah array diambil.
        rgb_conv = img.convert('RGB')
        try:
            rgb = rgb_conv.resize((size, size), Image.Resampling.BILINEAR)
        finally:
            if rgb_conv is not img:
                try: rgb_conv.close()
                except Exception: pass
        try:
            a = np.asarray(rgb, dtype=np.float64)  # (size, size, 3)
            h, w, _ = a.shape
            blocks: List[List[float]] = []
            for by in range(n):
                y0 = (by * h) // n
                y1 = ((by + 1) * h) // n
                for bx in range(n):
                    x0 = (bx * w) // n
                    x1 = ((bx + 1) * w) // n
                    sub = a[y0:y1, x0:x1, :]
                    if sub.size:
                        m = sub.reshape(-1, 3).mean(axis=0)
                        blocks.append([float(m[0]), float(m[1]), float(m[2])])
                    else:
                        blocks.append([0.0, 0.0, 0.0])
            return blocks
        finally:
            try: rgb.close()
            except Exception: pass
    except Exception as e:
        _log('debug', f"_color_grid failed: {e}")
        return None

def _color_histogram(img: Image.Image) -> Optional[List[List[float]]]:
    """Histogram warna global per-channel RGB, dinormalisasi (jumlah=1).
    Return [hist_r, hist_g, hist_b] atau None bila gagal."""
    try:
        # Resize ke SSIM_IMG_SIZE dulu: histogram ternormalisasi invariant terhadap
        # downscale, tapi hemat memori untuk foto resolusi sangat tinggi.
        size = SSIM_IMG_SIZE
        rgb_conv = img.convert('RGB')
        try:
            rgb = rgb_conv.resize((size, size), Image.Resampling.BILINEAR)
        finally:
            if rgb_conv is not img:
                try: rgb_conv.close()
                except Exception: pass
        try:
            a = np.asarray(rgb, dtype=np.uint8)  # (size, size, 3)
        finally:
            try: rgb.close()
            except Exception: pass
        if a.ndim != 3 or a.shape[2] < 3:
            return None
        hists: List[List[float]] = []
        for ch in range(3):
            counts, _ = np.histogram(a[:, :, ch], bins=HIST_BINS, range=(0, 256))
            total = counts.sum()
            if total > 0:
                norm = counts.astype(np.float64) / float(total)
            else:
                norm = counts.astype(np.float64)
            hists.append([float(x) for x in norm])
        return hists
    except Exception as e:
        _log('debug', f"_color_histogram failed: {e}")
        return None

def _aspect_ratio(img: Image.Image) -> Optional[float]:
    """Rasio aspek (lebar/tinggi) citra (langkah 1 diagram: Aspect Ratio).
    None bila tinggi 0/tak valid. Murah: hanya dari dimensi citra."""
    try:
        w, h = img.size
        if not h:
            return None
        return float(w) / float(h)
    except Exception as e:
        _log('debug', f"_aspect_ratio failed: {e}")
        return None

def _sharpness_blocks(img: Image.Image) -> Optional[List[float]]:
    """Peta ketajaman per-blok via varians Laplacian pada grid BLUR_GRID x BLUR_GRID
    setelah resize ke BLUR_REGION_IMG_SIZE. Return list datar atau None."""
    try:
        n = BLUR_GRID
        size = BLUR_REGION_IMG_SIZE
        # Konversi dan resize menghasilkan objek Image baru; tutup setelah array diambil.
        g_conv = img.convert('L')
        try:
            g = g_conv.resize((size, size), Image.Resampling.BILINEAR)
        finally:
            if g_conv is not img:
                try: g_conv.close()
                except Exception: pass
        try:
            a = np.asarray(g, dtype=np.float64)
            # Laplacian 4-tetangga; tepi 1px dibuang.
            lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
                   - 4.0 * a[1:-1, 1:-1])
            h, w = lap.shape
            blocks: List[float] = []
            for by in range(n):
                y0 = (by * h) // n
                y1 = ((by + 1) * h) // n
                for bx in range(n):
                    x0 = (bx * w) // n
                    x1 = ((bx + 1) * w) // n
                    sub = lap[y0:y1, x0:x1]
                    blocks.append(float(sub.var()) if sub.size else 0.0)
            return blocks
        finally:
            try: g.close()
            except Exception: pass
    except Exception as e:
        _log('debug', f"_sharpness_blocks failed: {e}")
        return None

def _edge_density_blocks(img: Image.Image) -> Optional[List[float]]:
    """Peta kepadatan tepi per-blok via gradien Sobel pada grid EDGE_GRID x EDGE_GRID
    setelah resize ke EDGE_REGION_IMG_SIZE. Return list datar atau None."""
    try:
        n = EDGE_GRID
        size = EDGE_REGION_IMG_SIZE
        # Konversi dan resize menghasilkan objek Image baru; tutup setelah array diambil.
        g_conv = img.convert('L')
        try:
            g = g_conv.resize((size, size), Image.Resampling.BILINEAR)
        finally:
            if g_conv is not img:
                try: g_conv.close()
                except Exception: pass
        try:
            a = np.asarray(g, dtype=np.float64)
            # Sobel 3x3 via geseran array (tanpa scipy/OpenCV); tepi 1px dibuang.
            tl, t, tr = a[:-2, :-2], a[:-2, 1:-1], a[:-2, 2:]
            l,  r     = a[1:-1, :-2], a[1:-1, 2:]
            bl, b, br = a[2:, :-2],  a[2:, 1:-1],  a[2:, 2:]
            gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl)
            gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr)
            mag = np.sqrt(gx * gx + gy * gy)
            h, w = mag.shape
            blocks: List[float] = []
            for by in range(n):
                y0 = (by * h) // n
                y1 = ((by + 1) * h) // n
                for bx in range(n):
                    x0 = (bx * w) // n
                    x1 = ((bx + 1) * w) // n
                    sub = mag[y0:y1, x0:x1]
                    blocks.append(float(sub.mean()) if sub.size else 0.0)
            return blocks
        finally:
            try: g.close()
            except Exception: pass
    except Exception as e:
        _log('debug', f"_edge_density_blocks failed: {e}")
        return None

def _blockiness_blocks(img: Image.Image) -> Optional[List[float]]:
    """Peta blockiness JPEG per-blok: skor = mean|beda di BATAS blok 8x8| - mean|beda
    di DALAM blok|. Grid BLOCKINESS_GRID x BLOCKINESS_GRID setelah resize ke
    BLOCKINESS_IMG_SIZE. Return list datar atau None. Dipakai hanya sebagai
    pembela (rescue) gerbang blur, tidak pernah menolak sendirian."""
    try:
        n = BLOCKINESS_GRID
        size = BLOCKINESS_IMG_SIZE
        g_conv = img.convert('L')
        try:
            g = g_conv.resize((size, size), Image.Resampling.BILINEAR)
        finally:
            if g_conv is not img:
                try: g_conv.close()
                except Exception: pass
        try:
            a = np.asarray(g, dtype=np.float64)
            h, w = a.shape
            # Beda absolut antar-piksel bersebelahan (horizontal & vertikal).
            dh = np.abs(a[:, 1:] - a[:, :-1])
            dv = np.abs(a[1:, :] - a[:-1, :])
            # Mask batas blok 8x8: pasangan piksel yang melintasi kelipatan 8.
            cols = np.arange(w - 1)
            rows = np.arange(h - 1)
            hbound_cols = ((cols + 1) % 8 == 0)
            vbound_rows = ((rows + 1) % 8 == 0)
            blocks: List[float] = []
            for by in range(n):
                y0 = (by * h) // n
                y1 = ((by + 1) * h) // n
                for bx in range(n):
                    x0 = (bx * w) // n
                    x1 = ((bx + 1) * w) // n
                    # Pisahkan pasangan di batas blok vs di dalam blok.
                    sub_dh = dh[y0:y1, x0:max(x0, x1 - 1)]
                    sub_hb = hbound_cols[x0:max(x0, x1 - 1)]
                    sub_dv = dv[y0:max(y0, y1 - 1), x0:x1]
                    sub_vb = vbound_rows[y0:max(y0, y1 - 1)]
                    bnd_vals = []
                    inr_vals = []
                    if sub_dh.size and sub_hb.size:
                        if sub_hb.any():
                            bnd_vals.append(sub_dh[:, sub_hb])
                        if (~sub_hb).any():
                            inr_vals.append(sub_dh[:, ~sub_hb])
                    if sub_dv.size and sub_vb.size:
                        if sub_vb.any():
                            bnd_vals.append(sub_dv[sub_vb, :])
                        if (~sub_vb).any():
                            inr_vals.append(sub_dv[~sub_vb, :])
                    bnd_mean = (np.concatenate([v.ravel() for v in bnd_vals]).mean()
                                if bnd_vals else 0.0)
                    inr_mean = (np.concatenate([v.ravel() for v in inr_vals]).mean()
                                if inr_vals else 0.0)
                    blocks.append(float(bnd_mean - inr_mean))
            return blocks
        finally:
            try: g.close()
            except Exception: pass
    except Exception as e:
        _log('debug', f"_blockiness_blocks failed: {e}")
        return None

def _ssim_thumb_encode(img: Image.Image) -> Optional[str]:
    """Encode citra kanonik grayscale SSIM_IMG_SIZE x SSIM_IMG_SIZE ke PNG
    base64 untuk disimpan di record LMDB (field 'ssim_thumb').

    Citra dikecilkan ke SSIM_IMG_SIZE lalu dikonversi ke grayscale sebelum
    di-encode PNG. Ukuran kecil (256x256 grayscale) menghasilkan PNG ~3-8 KB
    (base64 ~4-11 KB per foto). Disimpan di LMDB agar tersedia tanpa download
    ulang saat analisis duplikat (thumbnail disk lokal bersifat ephemeral).
    Return None bila encode gagal (defensif: record tetap valid tanpa field ini).
    """
    try:
        size = SSIM_IMG_SIZE
        # Konversi dan resize menghasilkan objek Image baru; tutup keduanya
        # setelah PNG di-encode agar buffer piksel tidak bocor.
        g_conv = img.convert('L')
        try:
            g = g_conv.resize((size, size), Image.Resampling.BILINEAR)
        finally:
            if g_conv is not img:
                try: g_conv.close()
                except Exception: pass
        try:
            buf = io.BytesIO()
            g.save(buf, 'PNG', optimize=True)
            return base64.b64encode(buf.getvalue()).decode('ascii')
        finally:
            try: g.close()
            except Exception: pass
    except Exception as e:
        _log('debug', f"_ssim_thumb_encode gagal: {e}")
        return None

def _ssim_match(rec_a: Dict, rec_b: Dict) -> Optional[float]:
    """Hitung skor SSIM antara dua foto dari citra kanonik di record LMDB.

    Decode field 'ssim_thumb' (PNG base64 grayscale SSIM_IMG_SIZE x SSIM_IMG_SIZE)
    dari kedua record, lalu hitung SSIM grayscale via scikit-image (CPU, tanpa
    GPU/AI). Return skor float [0..1] atau None bila:
      - scikit-image tidak tersedia (SSIM_AVAILABLE=False)
      - salah satu record tidak punya 'ssim_thumb' (record lama, kompatibel mundur)
      - decode gagal (defensif)
    Bila None dikembalikan, gerbang SSIM dilewati (fail-open, seperti pola
    gerbang existing yang melewati gerbang bila field tak tersedia).
    """
    if not _SSIM_AVAILABLE or _skimage_ssim is None:
        return None
    ta = rec_a.get('ssim_thumb')
    tb = rec_b.get('ssim_thumb')
    if not ta or not tb:
        # Record lama tanpa ssim_thumb: lewati gerbang (kompatibel mundur).
        return None
    try:
        # Decode PNG base64 -> Image -> array. Setiap Image.open().convert()
        # menghasilkan dua objek Image baru yang harus ditutup secara eksplisit
        # agar buffer piksel tidak bocor (dipanggil untuk setiap pasangan
        # kandidat foto yang lolos gerbang sebelumnya).
        def _decode_thumb(b64_data: str) -> np.ndarray:
            raw = Image.open(io.BytesIO(base64.b64decode(b64_data)))
            gray = None
            try:
                gray = raw.convert('L')
                return np.asarray(gray, dtype=np.float64) / 255.0
            finally:
                # Tutup kedua objek Image agar buffer piksel tidak bocor.
                if gray is not None and gray is not raw:
                    try: gray.close()
                    except Exception: pass
                try: raw.close()
                except Exception: pass

        arr_a = _decode_thumb(ta)
        arr_b = _decode_thumb(tb)
        # Pastikan ukuran sama (seharusnya selalu SSIM_IMG_SIZE x SSIM_IMG_SIZE).
        if arr_a.shape != arr_b.shape:
            return None
        score = float(_skimage_ssim(arr_a, arr_b, data_range=1.0))
        return score
    except Exception as e:
        _log('debug', f"_ssim_match gagal: {e}")
        return None

def _hash_image_full(img: Image.Image):
    """Hitung semua fitur foto dari satu objek Image: pHash, dHash H/V, peta
    ketajaman, warna, tepi, blockiness, ssim_thumb, histogram, rasio aspek,
    dan dimensi piksel. Khusus foto; video memakai _canonical_phash_dhash."""
    ph, dh, dv = _canonical_phash_dhash(img)
    sharp_blocks = _sharpness_blocks(img) if BLUR_REGION_GATE else None
    color_grid = _color_grid(img) if COLOR_GRID_GATE else None
    edge_blocks = _edge_density_blocks(img) if EDGE_REGION_GATE else None
    # Pembela gerbang blur: membedakan re-kompresi dari blur editan.
    blockiness_blocks = _blockiness_blocks(img) if BLOCKINESS_RESCUE_GATE else None
    color_hist = _color_histogram(img) if HIST_CORR_GATE else None
    aspect_ratio = _aspect_ratio(img) if ASPECT_RATIO_GATE else None
    # ssim_thumb selalu di-encode (tidak bergantung _SSIM_AVAILABLE) agar
    # gerbang SSIM langsung aktif bila scikit-image dipasang belakangan.
    ssim_thumb = _ssim_thumb_encode(img)
    # Dimensi asli untuk laporan PDF (menandai duplikat HD vs SD).
    try:
        _w, _h = img.size
        width, height = int(_w), int(_h)
    except Exception:
        width, height = None, None
    return (ph, dh, dv, sharp_blocks, color_grid, edge_blocks, ssim_thumb,
            color_hist, aspect_ratio, blockiness_blocks, width, height)

def _hash_image(img_data: bytes):
    """Decode bytes lalu hitung semua fitur foto via _hash_image_full."""
    try:
        img = _safe_open_rgb(img_data)
        return _hash_image_full(img)
    except Exception as e:
        _log('debug', f"_hash_image failed: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None

def _safe_open_rgb_path(path: str) -> Image.Image:
    """Buka foto dari path (anti-OOM untuk file besar). Normalisasi EXIF +
    konversi RGB identik dengan _safe_open_rgb sehingga nilai hash konsisten."""
    img = Image.open(path)
    img.load()
    try:
        try:
            transposed = ImageOps.exif_transpose(img)
        except Exception as e:
            _log('debug', f"exif_transpose (path) failed: {e}")
            transposed = img
        if transposed is not img:
            img.close()
            img = transposed
        if img.mode != 'RGB':
            converted = img.convert('RGB')
            if converted is not img:
                img.close()
                img = converted
        return img
    except Exception:
        try: img.close()
        except Exception: pass
        raise

def _hash_image_from_path(path: str):
    """Hash foto dari path (anti-OOM). Hasil identik dengan _hash_image(bytes)."""
    img = None
    try:
        img = _safe_open_rgb_path(path)
        return _hash_image_full(img)
    except Exception as e:
        _log('debug', f"_hash_image_from_path failed: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None
    finally:
        if img is not None:
            try: img.close()
            except Exception: pass

# Path ffprobe di-resolve sekali. Bila tidak ada, durasi fallback ke metadata imageio.
_FFPROBE_EXE: Optional[str] = None
_FFPROBE_RESOLVED = False
_FFPROBE_LOCK = threading.Lock()

def _ffprobe_exe() -> Optional[str]:
    global _FFPROBE_EXE, _FFPROBE_RESOLVED
    with _FFPROBE_LOCK:
        if _FFPROBE_RESOLVED:
            return _FFPROBE_EXE
        _FFPROBE_RESOLVED = True
        # Cari ffprobe di PATH, lalu di direktori ffmpeg imageio-ffmpeg.
        cand = shutil.which('ffprobe')
        if not cand:
            try:
                import imageio_ffmpeg
                ff = imageio_ffmpeg.get_ffmpeg_exe()
                guess = os.path.join(os.path.dirname(ff), 'ffprobe')
                if os.path.exists(guess):
                    cand = guess
            except Exception:
                cand = None
        _FFPROBE_EXE = cand
        return _FFPROBE_EXE

def _ffprobe_duration(video_path: str) -> Optional[float]:
    """Durasi video (detik) via ffprobe tanpa decode frame. None bila tidak
    tersedia; pemanggil fallback ke metadata imageio."""
    exe = _ffprobe_exe()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=30)
        val = (out.stdout or '').strip()
        if val and val.lower() != 'n/a':
            d = float(val)
            return d if d > 0 else None
    except Exception as e:
        _log('debug', f"ffprobe duration gagal {video_path}: {e}")
    return None

def _ffprobe_resolution(video_path: str) -> Tuple[Optional[int], Optional[int]]:
    """Resolusi video (lebar, tinggi) via ffprobe tanpa decode frame.
    Return (None, None) bila ffprobe tidak tersedia atau gagal; pemanggil
    fallback ke metadata imageio (meta['size'])."""
    exe = _ffprobe_exe()
    if not exe:
        return None, None
    try:
        out = subprocess.run(
            [exe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=30)
        lines = [ln.strip() for ln in (out.stdout or '').splitlines() if ln.strip()]
        if len(lines) >= 2:
            w = int(float(lines[0]))
            h = int(float(lines[1]))
            if w > 0 and h > 0:
                return w, h
    except Exception as e:
        _log('debug', f"ffprobe resolution gagal {video_path}: {e}")
    return None, None

def _ffprobe_fps(video_path: str) -> Optional[float]:
    """Frame rate video via ffprobe (avg_frame_rate, tanpa decode). None bila
    tidak tersedia; pemanggil fallback ke fps metadata imageio."""
    exe = _ffprobe_exe()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=avg_frame_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=30)
        val = (out.stdout or '').strip()
        if not val or val.lower() == 'n/a':
            return None
        if '/' in val:
            num, den = val.split('/', 1)
            num = float(num); den = float(den)
            if den == 0:
                return None
            f = num / den
        else:
            f = float(val)
        return f if f > 0 else None
    except Exception as e:
        _log('debug', f"ffprobe fps gagal {video_path}: {e}")
        return None

def _hash_one_frame_array(frame) -> Optional[Tuple[str, str, str]]:
    """Hitung triplet (pHash, dHash H, dHash V) dari satu frame (array RGB).
    Return None bila frame terlalu gelap/terang atau gagal. Logika identik
    untuk jalur FFmpeg maupun imageio agar nilai hash konsisten."""
    img = None
    try:
        img = Image.fromarray(frame)
        if img.mode != 'RGB':
            converted = img.convert('RGB')
            if converted is not img:
                img.close()
                img = converted
        # resize menghasilkan objek baru; tutup setelah array diambil.
        _thumb = img.resize((32, 32), Image.Resampling.BILINEAR)
        try:
            arr = np.array(_thumb)
        finally:
            try: _thumb.close()
            except Exception: pass
        if arr.mean() < 10 or arr.mean() > 248:
            return None
        ph, dh, dv = _canonical_phash_dhash(img)
        return ph, dh, dv
    except Exception as e:
        _log('debug', f"hash frame gagal: {e}")
        return None
    finally:
        if img is not None:
            try: img.close()
            except Exception: pass

def _grid_targets(duration: float) -> List[float]:
    """Titik waktu sampel (detik) pada GRID WAKTU ABSOLUT TETAP:
    t_k = (k+0.5)*VIDEO_SAMPLE_SEC -> 0.5s, 1.5s, 2.5s, ...

    KENAPA absolut, bukan fraksi durasi: titik fraksi (i+0.5)/N*durasi membuat
    N (=jumlah sampel) menentukan posisi, dan N=int(durasi/SAMPLE_SEC) diskontinu
    di batas integer -> dua video durasi nyaris sama bisa beda N -> grid bergeser
    -> frame ke-i tak selaras. Dengan grid absolut, titik ke-k SELALU di detik
    yang sama untuk video apa pun; beda durasi kecil hanya mengubah JUMLAH titik
    di ekor (ditangani perbandingan posisional). Inilah kunci keselarasan.

    Jumlah titik di-clamp ke [VIDEO_MIN_SAMPLES, VIDEO_MAX_SAMPLES]. Bila grid
    absolut menghasilkan lebih dari cap (video panjang), spacing diregangkan
    seragam berbasis durasi; karena dua video yang dibandingkan punya durasi
    ~sama (gerbang durasi), spacing-nya ~sama sehingga tetap selaras.
    Dipakai SAMA persis oleh jalur FFmpeg & fallback imageio."""
    if duration is None or duration <= 0:
        return []
    # Guard durasi tidak-hingga atau NaN: metadata imageio yang rusak bisa
    # mengembalikan float('inf') sebagai durasi. Titik grid inf akan dikirim
    # ke FFmpeg sebagai '-ss inf' yang menyebabkan error/hang per-frame.
    # Durasi NaN juga tidak valid untuk perhitungan grid.
    if duration != duration or duration == float('inf'):
        return []
    # Guard VIDEO_SAMPLE_SEC non-positif: hindari ZeroDivisionError.
    if VIDEO_SAMPLE_SEC <= 0:
        _log('warning', f"VIDEO_SAMPLE_SEC={VIDEO_SAMPLE_SEC} tidak valid (harus > 0); _grid_targets mengembalikan []")
        return []
    # Kuantisasi durasi untuk menstabilkan jumlah titik grid agar jitter durasi
    # re-encode tidak menggeser keselarasan posisional antar video.
    q = VIDEO_GRID_DURATION_QUANTUM if VIDEO_GRID_DURATION_QUANTUM > 0 else VIDEO_SAMPLE_SEC
    dur_q = round(duration / q) * q
    nat = int(dur_q / VIDEO_SAMPLE_SEC)
    n = max(VIDEO_MIN_SAMPLES, min(VIDEO_MAX_SAMPLES, nat))
    if n < 1:
        return []
    if nat <= VIDEO_MAX_SAMPLES:
        # Grid absolut: t_k = (k+0.5)*SAMPLE_SEC. Bila durasi terlalu pendek,
        # regangkan ke dalam durasi agar tetap dapat n titik.
        if nat >= VIDEO_MIN_SAMPLES:
            return [(k + 0.5) * VIDEO_SAMPLE_SEC for k in range(n)]
        return [(k + 0.5) * duration / float(n) for k in range(n)]
    # Video panjang: regangkan seragam ke VIDEO_MAX_SAMPLES titik.
    return [(k + 0.5) * duration / float(n) for k in range(n)]

def _extract_frames_ffmpeg(video_path: str, targets: List[float]) -> Optional[List]:
    """Ekstrak satu frame per titik waktu di `targets` (detik) via FFmpeg.
    Return list sepanjang len(targets) berisi array RGB atau None per slot.
    Posisi slot dipertahankan untuk keselarasan posisional antar video.
    Return None bila ffmpeg tidak tersedia atau tidak ada frame yang berhasil.
    """
    if not targets:
        return None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return None

    num_samples = len(targets)
    tmpdir = tempfile.mkdtemp(prefix='dgv_frames_')
    _t_start = time.monotonic()
    try:
        # Seek akurat: -ss setelah -i agar FFmpeg decode sampai timestamp tepat,
        # bukan snap ke keyframe terdekat. Lebih lambat tapi selaras dengan
        # jalur imageio (idx=round(t*fps)) dan konsisten lintas resolusi.
        frames: List = [None] * num_samples
        for i, t in enumerate(targets):
            if (time.monotonic() - _t_start) > VIDEO_FFMPEG_TOTAL_TIMEOUT:
                _log('debug', f"ffmpeg total timeout terlampaui {video_path} di slot={i}")
                break
            out_png = os.path.join(tmpdir, f"f_{i:08d}.png")
            cmd = [ffmpeg, '-v', 'error',
                   '-i', video_path, '-ss', f"{max(0.0, t):.6f}",
                   '-frames:v', '1', '-q:v', '2', out_png]
            try:
                # Timeout per-frame: sisa waktu total, dijepit ke [5, 120] detik.
                _elapsed = time.monotonic() - _t_start
                _per_frame_timeout = max(5.0, min(120.0, VIDEO_FFMPEG_TOTAL_TIMEOUT - _elapsed))
                subprocess.run(cmd, capture_output=True, timeout=_per_frame_timeout, check=False)
            except Exception as e:
                _log('debug', f"ffmpeg seek t={t:.3f} gagal {video_path}: {e}")
                continue
            if not os.path.exists(out_png):
                continue
            im = None
            try:
                im = Image.open(out_png)
                im.load()
                frames[i] = np.array(im)
            except Exception as e:
                _log('debug', f"baca frame ffmpeg gagal slot={i}: {e}")
                continue
            finally:
                if im is not None:
                    try: im.close()
                    except Exception: pass
        if not any(f is not None for f in frames):
            return None
        return frames
    except Exception as e:
        _log('debug', f"_extract_frames_ffmpeg gagal {video_path}: {e}")
        return None
    finally:
        try: shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception: pass

def _hash_video(video_path: str):
    """Baca video, return (phash_median, dhash_h_median, dhash_v_median,
    phash_list, dhash_h_list, dhash_v_list, duration_sec, width, height). List
    frame sejajar posisi untuk match per-frame. width/height dipakai untuk label
    kualitas video (resolusi) di laporan. Utamakan FFmpeg; fallback ke imageio."""
    reader = None
    duration = None
    vid_w: Optional[int] = None
    vid_h: Optional[int] = None
    try:
        # Durasi: utamakan ffprobe (akurat, tanpa decode). Fallback ke imageio.
        duration = _ffprobe_duration(video_path)
        # fps: utamakan ffprobe agar indeks fallback imageio selaras grid FFmpeg.
        probe_fps = _ffprobe_fps(video_path)
        # Resolusi: utamakan ffprobe; fallback ke meta['size'] imageio di bawah.
        vid_w, vid_h = _ffprobe_resolution(video_path)
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        # Fallback resolusi dari metadata imageio bila ffprobe tak tersedia.
        if not (vid_w and vid_h):
            _sz = meta.get('size')
            try:
                if _sz and len(_sz) >= 2:
                    _w, _h = int(_sz[0]), int(_sz[1])
                    if _w > 0 and _h > 0:
                        vid_w, vid_h = _w, _h
            except (TypeError, ValueError):
                pass
        fps  = probe_fps if probe_fps else (meta.get('fps', 25) or 25)
        # Guard fps garbage/negatif dari codec tertentu.
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 25.0
        if fps <= 0:
            fps = 25.0
        n    = meta.get('nframes', None)
        # Durasi efektif: pakai metadata bila ada, selain itu turunkan dari nframes/fps.
        if duration is None:
            _meta_dur = meta.get('duration', None)
            if _meta_dur and _meta_dur != float('inf'):
                duration = float(_meta_dur)
        # nframes sering tidak akurat/inf; estimasi dari durasi sebagai cadangan.
        if not n or n == float('inf'):
            n = max(1, int((meta.get('duration', 10) or 10) * fps))
        n = max(1, int(n))
        if duration is None:
            duration = n / float(fps) if fps else None
        max_idx = n - 1
        # Grid waktu absolut (lihat _grid_targets): sumber tunggal posisi sampel.
        eff_duration = duration if (duration and duration > 0) else (n / float(fps) if fps else None)
        targets = _grid_targets(eff_duration)
        if not targets:
            return None, None, None, [], [], [], duration
        num_samples = len(targets)
        # List sejajar-posisi: slot gagal diisi None agar tidak menggeser posisi.
        good_p: List[Optional[str]] = [None] * num_samples
        good_d: List[Optional[str]] = [None] * num_samples
        good_v: List[Optional[str]] = [None] * num_samples

        # Jalur cepat: FFmpeg.
        ff_frames = _extract_frames_ffmpeg(video_path, targets)
        if ff_frames:
            for i, frame in enumerate(ff_frames):
                if i >= num_samples or frame is None:
                    continue
                trip = _hash_one_frame_array(frame)
                if trip is None:
                    continue
                good_p[i] = trip[0]; good_d[i] = trip[1]; good_v[i] = trip[2]

        # Fallback: imageio seek per-frame, grid waktu identik dengan FFmpeg.
        if not any(p is not None for p in good_p):
            for i, t in enumerate(targets):
                idx = min(max(0, int(round(t * fps))), max_idx)
                try:
                    frame = reader.get_data(idx)
                except (IndexError, StopIteration):
                    _log('debug', f"video frame idx={idx} di luar jangkauan")
                    continue
                except Exception as e:
                    _log('debug', f"video frame hash idx={idx}: {e}")
                    continue
                trip = _hash_one_frame_array(frame)
                if trip is None:
                    continue
                good_p[i] = trip[0]; good_d[i] = trip[1]; good_v[i] = trip[2]
        # Median dari slot non-None (untuk field ringkas; pencocokan nyata
        # memakai list sejajar-posisi).
        _vp = [x for x in good_p if x]
        _vd = [x for x in good_d if x]
        _vv = [x for x in good_v if x]
        ph = _vp[len(_vp)//2] if _vp else None
        dh = _vd[len(_vd)//2] if _vd else None
        dv = _vv[len(_vv)//2] if _vv else None
        # Bila tidak ada slot pHash valid, kembalikan list kosong (bukan [None,...])
        # agar record diperlakukan sebagai exact-match murni.
        if not _vp:
            return ph, dh, dv, [], [], [], duration, vid_w, vid_h
        return ph, dh, dv, good_p, good_d, good_v, duration, vid_w, vid_h
    except Exception as e:
        _log('debug', f"_hash_video failed: {e}")
        return None, None, None, [], [], [], None, vid_w, vid_h
    finally:
        if reader is not None:
            try: reader.close()
            except Exception: pass

def extract_features(file_id: str, name: str, mime_type: str, size_mb: float,
                     expected_md5: Optional[str] = None,
                     expected_sha256: Optional[str] = None,
                     expected_size: Optional[int] = None) -> Tuple[Optional[Dict], Optional[str]]:
    # Kondisi deterministik ditandai prefix 'skip:' agar pemanggil menandainya
    # processed (di-skip permanen), bukan failed (yang akan di-retry tiap run).
    if size_mb * 1024 * 1024 < MIN_FILE_BYTES:
        return None, "skip:terlalu kecil"
    ftype = get_file_type(name, mime_type)
    if not ftype:
        return None, "skip:bukan media file"

    # Circuit breaker terbuka: kembalikan 'circuit_open' agar file tetap pending.
    if _circuit_breaker.is_open():
        return None, "circuit_open"

    temp_path = None
    try:
        md5_actual = None
        sha256_actual = None
        actual_size = None
        phs: List[str] = []
        dhs: List[str] = []
        dvs: List[str] = []
        dv: Optional[str] = None
        sharp_blocks: Optional[List[float]] = None
        color_grid: Optional[List[List[float]]] = None
        edge_blocks: Optional[List[float]] = None
        ssim_thumb: Optional[str] = None
        color_hist: Optional[List[List[float]]] = None
        aspect_ratio: Optional[float] = None
        blockiness_blocks: Optional[List[float]] = None
        img_width: Optional[int] = None
        img_height: Optional[int] = None
        vid_duration: Optional[float] = None
        if size_mb > MAX_EMBED_MB or ftype == 'video':
            suffix = '.mp4' if ftype == 'video' else '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name

            ok = download_bytes_stream(file_id, size_mb, file_path=temp_path)
            if ok is not True:
                return None, "download gagal"

            # Satu kali baca: hitung BLAKE3 + md5/sha256 bila checksum tersedia.
            h = blake3.blake3()
            hm = hashlib.md5() if expected_md5 else None
            hs = hashlib.sha256() if expected_sha256 else None
            with open(temp_path, 'rb') as f:
                while True:
                    chunk = f.read(STREAM_CHUNK)
                    if not chunk: break
                    h.update(chunk)
                    if hm is not None: hm.update(chunk)
                    if hs is not None: hs.update(chunk)
            b3 = h.hexdigest()
            actual_size = os.path.getsize(temp_path)
            if hm is not None: md5_actual = hm.hexdigest()
            if hs is not None: sha256_actual = hs.hexdigest()

            if ftype == 'image':
                # Foto besar: hash dari path (anti-OOM).
                ph, dh, dv, sharp_blocks, color_grid, edge_blocks, ssim_thumb, color_hist, aspect_ratio, blockiness_blocks, img_width, img_height = _hash_image_from_path(temp_path)
            else:
                ph, dh, dv, phs, dhs, dvs, vid_duration, img_width, img_height = _hash_video(temp_path)
        else:
            data = download_bytes_stream(file_id, size_mb)
            # Cek None, bukan falsy: file 0-byte (b'') bukan kegagalan.
            if data is None:
                return None, "download gagal"
            b3 = compute_blake3_hash(data)
            actual_size = len(data)
            if expected_md5:
                md5_actual = hashlib.md5(data).hexdigest()
            if expected_sha256:
                sha256_actual = hashlib.sha256(data).hexdigest()
            if ftype == 'image':
                ph, dh, dv, sharp_blocks, color_grid, edge_blocks, ssim_thumb, color_hist, aspect_ratio, blockiness_blocks, img_width, img_height = _hash_image(data)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    temp_path = tmp.name
                    tmp.write(data)
                ph, dh, dv, phs, dhs, dvs, vid_duration, img_width, img_height = _hash_video(temp_path)

        # Verifikasi ukuran: satu-satunya proteksi integritas untuk file tanpa
        # md5Checksum (video/Google-native). expected_size=0 dilewati by-design.
        if expected_size and actual_size is not None and actual_size != expected_size:
            _log('error', f"size mismatch fid={file_id} expected={expected_size} actual={actual_size}")
            return None, "size mismatch (download tidak lengkap)"

        if expected_md5 and md5_actual and md5_actual != expected_md5:
            _log('error', f"md5 mismatch fid={file_id} expected={expected_md5} actual={md5_actual}")
            return None, "md5 mismatch (download korup)"

        # sha256 melengkapi md5 (Shared Drive kadang punya sha256 tanpa md5).
        if expected_sha256 and sha256_actual and sha256_actual != expected_sha256:
            _log('error', f"sha256 mismatch fid={file_id} expected={expected_sha256} actual={sha256_actual}")
            return None, "sha256 mismatch (download korup)"

        # Tolak hanya bila BLAKE3 tidak ada. Bila perceptual-hash gagal tapi
        # BLAKE3 valid, record tetap disimpan untuk exact-match.
        if not b3:
            return None, "hashing gagal (BLAKE3 tidak terhitung)"
        if ph is None and not phs:
            _log('debug', f"perceptual-hash gagal, simpan BLAKE3 saja fid={file_id} (exact-match tetap jalan)")

        rec = {'blake3': b3, 'phash': ph, 'dhash': dh, 'dvhash': dv,
               'sharpness_blocks': sharp_blocks,   # peta ketajaman per-blok (anti-blur lokal)
               'color_grid': color_grid,            # sidik warna per-blok (anti-filter warna)
               'edge_blocks': edge_blocks,           # kepadatan tepi per-blok (anti-emoji/stiker)
               'ssim_thumb': ssim_thumb,             # citra kanonik grayscale untuk SSIM
               'color_hist': color_hist,             # histogram warna global
               'aspect_ratio': aspect_ratio,         # rasio aspek (pra-filter)
               'width': img_width,
               'height': img_height,
               'blockiness_blocks': blockiness_blocks,  # pembela gerbang blur
               'file_type': ftype,
               'md5': expected_md5 or md5_actual,
               'sha256': expected_sha256 or sha256_actual,
               'size_bytes': actual_size}            # ukuran byte asli
        if ftype == 'video':
            rec['phashes'] = phs
            rec['dhashes'] = dhs
            rec['dvhashes'] = dvs
            rec['duration'] = vid_duration
        return rec, None

    except Exception as ex:
        _log('error', f"extract_features exception fid={file_id}: {traceback.format_exc()}")
        return None, str(ex)
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception as e:
                _log('warning', f"temp cleanup failed {temp_path}: {e}")

# ───────────────────── THUMBNAIL ─────────────────────
# Di-cache di disk; tidak pernah mengembalikan string kosong.
_thumb_mem   = {}
_thumb_lock  = threading.Lock()
# Direktori thumb_cache bundle yang sedang diproses, di-set oleh analyze_folder.
_thumb_dir   = None

def set_thumb_dir(path: Optional[str]):
    global _thumb_dir
    with _thumb_lock:
        if path != _thumb_dir:
            # Ganti bundle: kosongkan mem-cache agar tidak menahan thumbnail lama.
            _thumb_mem.clear()
        _thumb_dir = path
    if path:
        try: os.makedirs(path, exist_ok=True)
        except Exception as e: _log('debug', f"makedirs thumb dir gagal: {e}")

def _safe_cache_key(file_id: str) -> str:
    """Sanitasi id menjadi nama file aman (cegah path traversal)."""
    return ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in (file_id or ''))

def _thumb_disk_path(file_id: str) -> Optional[str]:
    with _thumb_lock:
        d = _thumb_dir
    if not d:
        return None
    return os.path.join(d, _safe_cache_key(file_id) + ".jpg.b64")

def _thumb_jpg_path(file_id: str) -> Optional[str]:
    """Path file JPEG biner thumbnail untuk dirujuk WeasyPrint via file://.
    Menghindari embed base64 inline di HTML yang bisa OOM pada banyak duplikat."""
    with _thumb_lock:
        d = _thumb_dir
    if not d:
        return None
    return os.path.join(d, _safe_cache_key(file_id) + ".thumb.jpg")

def _b64_to_jpg_bytes(b64_data_uri: str) -> Optional[bytes]:
    """Ekstrak byte JPEG dari data URI base64. None bila bukan data URI valid."""
    if not b64_data_uri or ',' not in b64_data_uri:
        return None
    try:
        return base64.b64decode(b64_data_uri.split(',', 1)[1])
    except Exception as e:
        _log('debug', f"decode base64 thumbnail gagal: {e}")
        return None

def _file_uri(path: str) -> str:
    """Bentuk URI file:// yang ter-escape dari path absolut (untuk WeasyPrint)."""
    return "file://" + urllib.request.pathname2url(path)

def get_image_file_uri(file_id: str, file_name: str, file_type: str = 'image', size_mb: float = 0) -> Optional[str]:
    """Return URI file:// menuju thumbnail JPEG biner di disk untuk WeasyPrint,
    atau None bila tidak tersedia (pemanggil pakai placeholder base64)."""
    jpg_path = _thumb_jpg_path(file_id)
    if jpg_path and os.path.exists(jpg_path) and os.path.getsize(jpg_path) > 0:
        return _file_uri(jpg_path)
    b64 = get_image_base64(file_id, file_name, file_type, size_mb)
    if not b64 or b64 == _PLACEHOLDER_B64:
        return None
    raw = _b64_to_jpg_bytes(b64)
    if raw is None or not jpg_path:
        return None
    # Tulis atomik (.tmp lalu os.replace) agar file final selalu utuh.
    tmp = jpg_path + ".tmp"
    try:
        with open(tmp, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, jpg_path)
        return _file_uri(jpg_path)
    except Exception as e:
        _log('debug', f"tulis thumbnail jpg gagal {file_id}: {e}")
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return None

def prune_thumb_cache(thumb_dir: str, current_ids: Set[str]):
    """Buang thumbnail yang file_id-nya sudah tidak ada di manifest."""
    if not thumb_dir or not os.path.isdir(thumb_dir):
        return
    removed = 0
    safe_current = {_safe_cache_key(i) for i in current_ids}
    # Pangkas semua artefak cache: .jpg.b64, .thumb.jpg, .thumb.fail.
    # Urut dari suffix terpanjang agar tidak salah potong.
    suffixes = (".thumb.jpg", ".thumb.fail", ".jpg.b64")
    try:
        for entry in os.listdir(thumb_dir):
            sfx = next((s for s in suffixes if entry.endswith(s)), None)
            if sfx is None:
                continue
            fid = entry[:-len(sfx)]
            # Guard: fid kosong berarti nama file sama persis dengan suffix.
            if not fid:
                continue
            if fid not in safe_current:
                try:
                    os.unlink(os.path.join(thumb_dir, entry))
                    removed += 1
                except Exception:
                    pass
    except Exception as e:
        _log('debug', f"prune_thumb_cache error: {e}")
    if removed:
        _log('info', f"prune thumb_cache: {removed} thumbnail usang dibuang")

# TTL penanda kegagalan thumbnail (negative-cache).
# Decode gagal = deterministik (file korup) -> TTL panjang.
# Download gagal = transien (rate-limit/koneksi) -> TTL pendek.
_THUMB_FAIL_TTL_DECODE   = 7 * 24 * 3600.0   # 7 hari
_THUMB_FAIL_TTL_DOWNLOAD = 6 * 3600.0        # 6 jam

def _thumb_fail_path(file_id: str) -> Optional[str]:
    with _thumb_lock:
        d = _thumb_dir
    if not d:
        return None
    return os.path.join(d, _safe_cache_key(file_id) + ".thumb.fail")

def _thumb_fail_active(file_id: str) -> bool:
    """True bila penanda kegagalan thumbnail masih berlaku (belum melewati TTL).
    Penanda kedaluwarsa dihapus agar generate dicoba ulang."""
    p = _thumb_fail_path(file_id)
    if not p or not os.path.exists(p):
        return False
    try:
        kind = (_read_text_file(p) or 'download').strip()
        ttl = _THUMB_FAIL_TTL_DECODE if kind == 'decode' else _THUMB_FAIL_TTL_DOWNLOAD
        age = time.time() - os.path.getmtime(p)
        if age < ttl:
            return True
        try: os.unlink(p)
        except Exception: pass
        return False
    except Exception:
        return False

def _thumb_mark_fail(file_id: str, kind: str):
    """Tulis penanda kegagalan thumbnail secara atomik.
    kind='decode' (deterministik) atau 'download' (transien)."""
    p = _thumb_fail_path(file_id)
    if not p:
        return
    try:
        _write_text_atomic(p, 'decode' if kind == 'decode' else 'download')
    except Exception as e:
        _log('debug', f"tulis penanda gagal thumbnail {file_id}: {e}")

def _thumb_load_disk(file_id: str) -> Optional[str]:
    p = _thumb_disk_path(file_id)
    if p and os.path.exists(p):
        try:
            with open(p, 'r') as f:
                val = f.read().strip()
            if not val or val == _PLACEHOLDER_B64:
                return None
            return val
        except Exception as e:
            _log('debug', f"thumb disk load failed {file_id}: {e}")
    return None

def _thumb_save_disk(file_id: str, b64: str):
    p = _thumb_disk_path(file_id)
    if not p:
        return
    # Tulis atomik (.tmp lalu os.replace) agar file final selalu utuh.
    tmp = p + ".tmp"
    try:
        with open(tmp, 'w') as f:
            f.write(b64)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception as e:
        _log('debug', f"thumb disk save failed {file_id}: {e}")
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass

def get_image_base64(file_id: str, file_name: str, file_type: str = 'image', size_mb: float = 0) -> str:
    with _thumb_lock:
        if file_id in _thumb_mem:
            # Move-to-end: tandai sebagai paling baru dipakai (LRU).
            val = _thumb_mem.pop(file_id)
            _thumb_mem[file_id] = val
            return val

    cached = _thumb_load_disk(file_id)
    if cached:
        with _thumb_lock:
            _thumb_mem[file_id] = cached
            # Eviction LRU agar disk cache hit tidak melampaui THUMB_MEM_MAX.
            while len(_thumb_mem) > THUMB_MEM_MAX:
                _thumb_mem.pop(next(iter(_thumb_mem)))
        return cached

    # Negative-cache: bila file ini sudah pernah gagal dan penandanya masih
    # berlaku (TTL), langsung kembalikan placeholder tanpa download ulang.
    if _thumb_fail_active(file_id):
        with _thumb_lock:
            _thumb_mem[file_id] = _PLACEHOLDER_B64
            while len(_thumb_mem) > THUMB_MEM_MAX:
                _thumb_mem.pop(next(iter(_thumb_mem)))
        return _PLACEHOLDER_B64

    result   = _PLACEHOLDER_B64
    is_real  = False
    fail_kind: Optional[str] = None  # 'decode' (deterministik) / 'download' (transien)
    # File besar di-stream ke temp agar tidak menarik seluruh byte ke RAM.
    if size_mb > MAX_EMBED_MB:
        tmp_path = None
        try:
            suffix = '.mp4' if file_type == 'video' else '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as _tf:
                tmp_path = _tf.name
            ok = download_bytes_stream(file_id, size_mb, file_path=tmp_path)
            if ok is True:
                generated = _make_thumbnail_from_path(tmp_path, file_type)
                if generated:
                    result  = generated
                    is_real = True
                else:
                    fail_kind = 'decode'  # download sukses, decode gagal -> deterministik
                    _log('warning', f"thumbnail generate gagal fid={file_id} name={file_name} type={file_type}")
            else:
                fail_kind = 'download'  # transien (rate-limit/koneksi)
                _log('warning', f"thumbnail download gagal fid={file_id} name={file_name} size_mb={size_mb:.2f}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except Exception: pass
    else:
        data = download_bytes_stream(file_id, size_mb)
        if data is not None:
            generated = _make_thumbnail(data, file_type)
            if generated:
                result  = generated
                is_real = True
            else:
                fail_kind = 'decode'
                _log('warning', f"thumbnail generate gagal fid={file_id} name={file_name} type={file_type} bytes={len(data)}")
        else:
            fail_kind = 'download'
            _log('warning', f"thumbnail download gagal fid={file_id} name={file_name} size_mb={size_mb:.2f}")

    # Catat penanda kegagalan hanya saat sirkuit API tidak terbuka.
    # Bila sirkuit terbuka, kegagalan bersifat global sesaat; jangan tandai
    # agar file valid tidak terblokir saat kuota pulih.
    if (not is_real) and fail_kind and not _circuit_breaker.is_open():
        _thumb_mark_fail(file_id, fail_kind)

    with _thumb_lock:
        _thumb_mem[file_id] = result
        # Batasi mem-cache (LRU): buang entri terlama bila melebihi kapasitas.
        while len(_thumb_mem) > THUMB_MEM_MAX:
            _thumb_mem.pop(next(iter(_thumb_mem)))
    if is_real:
        _thumb_save_disk(file_id, result)
    return result

def _encode_thumb_image(img: Image.Image) -> str:
    """Encode img ke JPEG base64 data URI setelah resize ke THUMBNAIL_SIZE.
    Objek resize ditutup bila berbeda dari img asli; img asli tidak ditutup di sini."""
    resized = _safe_resize(img, THUMBNAIL_SIZE)
    try:
        buf = io.BytesIO()
        resized.save(buf, "JPEG", quality=THUMBNAIL_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    finally:
        if resized is not img:
            try: resized.close()
            except Exception: pass

def _frame_to_thumb(frame) -> Optional[str]:
    """Encode satu frame (array RGB) ke thumbnail JPEG base64. Lewati frame
    nyaris hitam/putih (mean<10 atau >248). Return None bila tidak layak/gagal."""
    img = None
    try:
        img = Image.fromarray(frame)
        if img.mode != 'RGB':
            converted = img.convert('RGB')
            if converted is not img:
                img.close()
                img = converted
        # Lewati frame nyaris hitam/putih (intro gelap, fade, frame rusak).
        try:
            _probe = img.resize((32, 32), Image.Resampling.BILINEAR)
            try:
                _m = float(np.asarray(_probe).mean())
            finally:
                try: _probe.close()
                except Exception: pass
            if _m < 10.0 or _m > 248.0:
                return None
        except Exception:
            pass  # bila probe gagal, tetap coba encode
        return _encode_thumb_image(img)
    except Exception as e:
        _log('debug', f"_frame_to_thumb gagal: {e}")
        return None
    finally:
        if img is not None:
            try: img.close()
            except Exception: pass

def _make_video_thumb_from_path(video_path: str) -> Optional[str]:
    """Buat thumbnail video: coba beberapa titik waktu (30/50/70/15/85/5%
    durasi), lewati frame hitam/rusak. Fallback ke FFmpeg bila imageio gagal."""
    reader = None
    try:
        reader = imageio.get_reader(video_path)
        meta   = reader.get_meta_data()
        probe_dur = _ffprobe_duration(video_path)
        probe_fps = _ffprobe_fps(video_path)
        fps    = probe_fps if probe_fps else (meta.get('fps', 25) or 25)
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 25.0
        if fps <= 0:
            fps = 25.0
        n      = meta.get('nframes', 30)
        if not n or n == float('inf'):
            _dur = probe_dur if probe_dur else (meta.get('duration', 5) or 5)
            n = max(1, int(_dur * fps))
        n = max(1, int(n))
        _dur = probe_dur if probe_dur else (meta.get('duration', None))
        # Beberapa titik kandidat (fraksi durasi). Dicoba berurutan sampai
        # satu frame layak berhasil di-encode.
        fractions = [0.30, 0.50, 0.70, 0.15, 0.85, 0.05]
        for fr in fractions:
            if _dur and _dur > 0:
                target_idx = int(round(_dur * fr * fps))
            else:
                target_idx = int(n * fr)
            idx = max(0, min(target_idx, n - 1))
            try:
                frame = reader.get_data(idx)
            except (IndexError, StopIteration):
                continue
            except Exception as e:
                _log('debug', f"thumb video get_data idx={idx} gagal: {e}")
                continue
            thumb = _frame_to_thumb(frame)
            if thumb:
                return thumb
    except Exception as e:
        _log('debug', f"_make_video_thumb_from_path imageio gagal: {e}")
    finally:
        if reader is not None:
            try: reader.close()
            except Exception: pass

    # Fallback: imageio gagal -> ekstrak via FFmpeg.
    try:
        dur = _ffprobe_duration(video_path)
        if dur and dur > 0:
            targets = [dur * fr for fr in (0.30, 0.50, 0.70, 0.15, 0.85)]
        else:
            targets = [1.0, 2.0, 5.0, 10.0]
        ff_frames = _extract_frames_ffmpeg(video_path, targets)
        if ff_frames:
            for frame in ff_frames:
                if frame is None:
                    continue
                thumb = _frame_to_thumb(frame)
                if thumb:
                    return thumb
    except Exception as e:
        _log('debug', f"_make_video_thumb_from_path ffmpeg fallback gagal: {e}")
    return None

def _make_thumbnail_from_path(path: str, ftype: str) -> Optional[str]:
    """Buat thumbnail dari file di disk (anti-OOM, tanpa menahan seluruh byte
    di RAM). Fallback ke bytes bila buka dari path gagal."""
    if ftype == 'image':
        img = None
        try:
            img = _safe_open_rgb_path(path)
            return _encode_thumb_image(img)
        except Exception as e:
            _log('debug', f"_make_thumbnail_from_path image (path) gagal, coba fallback bytes: {e}")
        finally:
            if img is not None:
                try: img.close()
                except Exception: pass
        # Fallback bytes: sebagian file dengan EXIF/metadata rusak parsial bisa
        # gagal dari path tapi berhasil dari buffer bytes.
        try:
            with open(path, 'rb') as _f:
                _data = _f.read()
            return _make_thumbnail(_data, 'image')
        except Exception as e:
            _log('debug', f"_make_thumbnail_from_path image (bytes fallback) gagal: {e}")
            return None
    return _make_video_thumb_from_path(path)

def _make_thumbnail(data: bytes, ftype: str) -> Optional[str]:
    if ftype == 'image':
        img = None
        try:
            img = _safe_open_rgb(data)
            return _encode_thumb_image(img)
        except Exception as e:
            _log('debug', f"_make_thumbnail image failed: {e}")
            return None
        finally:
            if img is not None:
                try: img.close()
                except Exception: pass
    # Video: tulis ke temp lalu pakai jalur path (imageio butuh file).
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(data); tmp = f.name
        return _make_video_thumb_from_path(tmp)
    except Exception as e:
        _log('debug', f"_make_thumbnail video failed: {e}")
        return None
    finally:
        if tmp and os.path.exists(tmp):
            try: os.unlink(tmp)
            except Exception: pass

def clear_thumb_mem_cache():
    with _thumb_lock:
        _thumb_mem.clear()

# ───────────────────── PROOF OF SCAN SYSTEM ─────────────────────
class ScanManifest:
    def __init__(self):
        self.expected_file_count   = 0
        self.expected_folder_count = 0
        self.expected_total_size   = 0.0
        self._lock = threading.Lock()

    def record_file(self, size_mb: float):
        with self._lock:
            self.expected_file_count += 1
            self.expected_total_size += size_mb

    def record_folder(self):
        with self._lock:
            self.expected_folder_count += 1

    def validate(self, actual_file_count: int, actual_size_mb: float, tolerance: float = 0.01) -> Tuple[bool, str]:
        if actual_file_count != self.expected_file_count:
            return False, (f"SCAN_INVALID: file count mismatch "
                           f"expected={self.expected_file_count} actual={actual_file_count}")
        if self.expected_total_size > 0:
            size_diff = abs(actual_size_mb - self.expected_total_size)
            if size_diff / self.expected_total_size > tolerance:
                return False, (f"SCAN_INVALID: size mismatch "
                               f"expected={self.expected_total_size:.2f}MB actual={actual_size_mb:.2f}MB")
        return True, "SCAN_VALID"

# ═════════════════════════════════════════════════════════════════
# Scan engine
# - Queue folder pending dipersist agar dapat resume setelah crash.
# - Folder gagal masuk dead-letter dan di-retry sampai SCAN_DEFER_PASSES.
# - Shortcut ke file diresolusi ke metadata target.
# ═════════════════════════════════════════════════════════════════
def _resolve_shortcut_target(target_id: str) -> Optional[Dict]:
    # Dipanggil dari dalam worker _list_children (paralel) -> service per-thread.
    _svc = _thread_drive()
    res, err = _drive_execute(lambda _tid=target_id: _svc.files().get(
        fileId=_tid,
        fields="id,name,createdTime,owners(emailAddress),mimeType,size,md5Checksum,sha256Checksum",
        supportsAllDrives=True))
    if res is None:
        _log('warning', f"shortcut target resolve gagal tid={target_id}: {err}")
    return res

def _post_scan_integrity(folder_id: str, journal: 'FolderJournal',
                         file_manifest: 'PersistentFileManifest',
                         drive_id: Optional[str], start_tok: Optional[str]):
    """Integritas pasca-scan: (1) drift detector via Changes API, (2) verifikasi
    double-listing bila VERIFY_DOUBLE_LISTING aktif. Keduanya fail-safe."""
    # (1) Drift detector, dibatasi ke subtree folder yang di-scan.
    try:
        _subtree = journal.visited_set() | {folder_id}
        _dk, _dc, _ = _count_changes_since(start_tok, drive_id, subtree=_subtree)
        file_manifest.set_scan_drift(_dc, _dk)
    except Exception as _e_drift:
        _log('warning', f"drift detector gagal: {_e_drift}")
        file_manifest.set_scan_drift(0, False)

    # (2) Double-listing (opsional).
    if not VERIFY_DOUBLE_LISTING:
        file_manifest.set_listing_verify(ran=False, known=False, discrepancy=0)
        return
    try:
        folder_ids = journal.visited_set() | {folder_id}
        complete, found = _list_media_ids_once(folder_ids, drive_id)
        if not complete:
            # Pass kedua tidak lengkap -> tidak dapat dipercaya.
            file_manifest.set_listing_verify(ran=True, known=False, discrepancy=0)
            _log('warning', "verifikasi double-listing tidak lengkap -> known=False")
            return
        known_ids = file_manifest.get_all_file_ids()
        missing = [found[i] for i in found.keys() if i not in known_ids]
        if missing:
            file_manifest.add_files(missing)
            file_manifest.flush()
            _log('error', f"double-listing: {len(missing)} file bocor dari listing pertama, ditambahkan")
        file_manifest.set_listing_verify(ran=True, known=True, discrepancy=len(missing))
        # Simpan pesan agar dicetak SETELAH garis batas (bukan sebelumnya).
        file_manifest._deferred_status_msg = (
            f"  {Colors.DIM}↳ Verifikasi double-listing: selisih {len(missing)} file{Colors.RESET}")
    except Exception as _e_lv:
        _log('warning', f"verifikasi double-listing gagal: {_e_lv}")
        file_manifest.set_listing_verify(ran=True, known=False, discrepancy=0)

def scan_folder_recursively(folder_id: str, journal: Optional[FolderJournal] = None,
                            file_manifest: Optional[PersistentFileManifest] = None
                            ) -> Tuple[List[Dict], ScanManifest, Dict[str, str]]:
    visited  = journal.visited_set() if journal else set()
    resume_q = journal.get_pending() if journal else []
    queue    = deque(resume_q) if resume_q else deque([folder_id])
    files    = []
    manifest = ScanManifest()
    failed_folders: Dict[str, str] = {}
    attempts: Dict[str, int] = defaultdict(int)
    # Dedup file dalam satu run (file bisa muncul dari beberapa parent/shortcut).
    run_seen: Set[str] = set()
    seen_file_ids = run_seen
    # Listing folder diparalelkan; lock melindungi struktur bersama.
    state_lock = threading.Lock()

    def _list_children(fid: str) -> Tuple[bool, Optional[str]]:
        """List semua halaman satu folder (thread-safe). Return (sukses, error).
        Memakai service per-thread agar tidak bentrok antar worker paralel."""
        _svc = _thread_drive()
        pt = journal.get_page_token(fid) if journal else None
        while True:
            res, err = _drive_execute(
                lambda _fid=fid, _pt=pt: _svc.files().list(
                    q=f"'{_fid}' in parents and trashed=false",
                    fields="nextPageToken, files(id,name,createdTime,owners(emailAddress),mimeType,size,md5Checksum,sha256Checksum,parents,shortcutDetails)",
                    pageSize=1000, pageToken=_pt,
                    supportsAllDrives=True, includeItemsFromAllDrives=True),
                max_retry=MAX_LIST_ATTEMPTS)
            if res is None:
                return False, err
            batch = []
            # Folder anak yang ditemukan di page ini. Dipersist ke journal
            # sebelum page token maju agar tidak hilang saat resume bila crash.
            child_folders: List[str] = []
            shortcut_targets: List[Tuple[str, str]] = []  # (shortcut_id, target_id)
            with state_lock:
                for item in res.get('files', []):
                    mime = item.get('mimeType', '')
                    if mime == 'application/vnd.google-apps.folder':
                        tid = item['id']
                        if tid not in visited:
                            queue.append(tid); child_folders.append(tid)
                    elif mime == 'application/vnd.google-apps.shortcut':
                        sc  = item.get('shortcutDetails', {})
                        tid = sc.get('targetId')
                        if tid and sc.get('targetMimeType') == 'application/vnd.google-apps.folder':
                            if tid not in visited:
                                queue.append(tid); child_folders.append(tid)
                        elif tid:
                            # Resolusi shortcut di luar lock (network call).
                            shortcut_targets.append((item['id'], tid))
                    else:
                        iid = item.get('id')
                        if iid and iid not in seen_file_ids:
                            seen_file_ids.add(iid)
                            files.append(item)
                            batch.append(item)
                            manifest.record_file(_safe_int_size(item.get('size')) / (1024 * 1024))
            for sid, tid in shortcut_targets:
                real = _resolve_shortcut_target(tid)
                with state_lock:
                    if not real:
                        failed_folders[f"shortcut:{sid}"] = f"target {tid} unresolvable"
                        continue
                    rmime = real.get('mimeType', '')
                    if rmime == 'application/vnd.google-apps.folder':
                        # Target berupa folder: telusuri sebagai subfolder.
                        rid = real.get('id')
                        if rid and rid not in visited:
                            queue.append(rid)
                            child_folders.append(rid)
                        continue
                    if rmime == 'application/vnd.google-apps.shortcut':
                        # Shortcut berantai: tolak.
                        failed_folders[f"shortcut:{sid}"] = f"target {tid} shortcut berantai"
                        continue
                    rid = real.get('id')
                    if rid and rid not in seen_file_ids:
                        seen_file_ids.add(rid)
                        files.append(real)
                        batch.append(real)
                        manifest.record_file(_safe_int_size(real.get('size')) / (1024 * 1024))
            # File dari page ini harus persisten sebelum page token maju agar
            # tidak hilang saat resume.
            if file_manifest and batch:
                file_manifest.add_files(batch)
            # Persist subfolder yang ditemukan SEBELUM page token maju agar tidak
            # hilang saat resume. add_pending hanya menambah (atomik + dedup).
            if journal and child_folders:
                journal.add_pending(child_folders)
            pt = res.get('nextPageToken')
            # Invariant lintas-env: paksa file batch durable di Drive (flush)
            # SEBELUM page token maju. Bila page token ter-mirror lebih dulu lalu
            # runtime mati, resume melewati page yang file-nya belum tersimpan.
            if pt and file_manifest and batch:
                try:
                    file_manifest.flush()
                except Exception as _e_fl:
                    _log('warning', f"flush manifest sebelum page token gagal fid={fid}: {_e_fl}")
            if journal and pt: journal.set_page_token(fid, pt)
            if not pt:
                if journal: journal.clear_page_token(fid)
                return True, None

    # BFS paralel per level: semua folder satu level di-list serentak,
    # anak-anaknya menjadi level berikutnya. Urutan persist (file lalu visited)
    # tetap dijaga.
    scan_workers = min(4, CONCURRENT_WORKERS)
    while queue:
        with state_lock:
            level = []
            level_seen = set()
            while queue:
                fid = queue.popleft()
                if fid in visited or fid in level_seen: continue
                level_seen.add(fid)
                level.append(fid)
        if not level:
            break
        with ThreadPoolExecutor(max_workers=scan_workers) as exe:
            results = {exe.submit(_list_children, fid): fid for fid in level}
            for fut in as_completed(results):
                fid = results[fut]
                try:
                    ok, err = fut.result()
                except Exception as ex:
                    ok, err = False, str(ex)
                with state_lock:
                    if ok:
                        visited.add(fid)
                        manifest.record_folder()
                        failed_folders.pop(fid, None)
                        # Persist visited agar subtree incremental run berikutnya lengkap.
                        if journal: journal.add_visited([fid])
                    else:
                        attempts[fid] += 1
                        if attempts[fid] <= SCAN_DEFER_PASSES:
                            # Dead-letter: re-queue untuk di-retry.
                            queue.append(fid)
                            _log('warning', f"scan defer fid={fid} pass={attempts[fid]} err={err}")
                        else:
                            failed_folders[fid] = err or 'unknown'
                            _log('error', f"scan list PERMANENT FAIL fid={fid}: {err}")
        # Persist progress level agar dapat resume di posisi terakhir.
        if journal:
            with state_lock:
                journal.set_pending(list(queue))

    if journal:
        journal.set_pending([])

    return files, manifest, failed_folders

# ═════════════════════════════════════════════════════════════════
# Incremental scan (Drive Changes API)
# Hanya delta sejak token terakhir yang diambil; fallback ke full scan bila
# token invalid/expired (HTTP 410) atau belum ada.
# ═════════════════════════════════════════════════════════════════
def _folder_drive_id(folder_id: str) -> Optional[str]:
    """driveId folder target bila berada di Shared Drive (None untuk My Drive)."""
    res, _err = _drive_execute(lambda _fid=folder_id: drive_service.files().get(
        fileId=_fid, fields="driveId", supportsAllDrives=True))
    return res.get('driveId') if res else None

def _list_media_ids_once(folder_ids: Set[str], drive_id: Optional[str] = None
                         ) -> Tuple[bool, Dict[str, Dict]]:
    """Listing independen pass kedua untuk verifikasi double-listing.
    Return (complete, {file_id: metadata}). complete=False bila ada folder
    yang gagal di-list. Shortcut dilewati (by-design; lihat komentar kode).
    """
    found: Dict[str, Dict] = {}
    complete = True
    for fid in folder_ids:
        page_token = None
        while True:
            def _req(_fid=fid, _pt=page_token):
                return drive_service.files().list(
                    q=f"'{_fid}' in parents and trashed=false",
                    fields="nextPageToken, files(id,name,mimeType,size,md5Checksum,"
                           "sha256Checksum,createdTime,owners(emailAddress),shortcutDetails)",
                    pageSize=1000, pageToken=_pt,
                    supportsAllDrives=True, includeItemsFromAllDrives=True)
            res, err = _drive_execute(_req, max_retry=MAX_LIST_ATTEMPTS)
            if res is None:
                _log('warning', f"verify list gagal fid={fid}: {err}")
                complete = False
                break
            for item in res.get('files', []):
                mime = item.get('mimeType', '')
                if mime in ('application/vnd.google-apps.folder',
                            'application/vnd.google-apps.shortcut'):
                    continue
                if not is_media_file(item.get('name'), mime):
                    continue
                iid = item.get('id')
                if iid:
                    found[iid] = item
            page_token = res.get('nextPageToken')
            if not page_token:
                break
    return complete, found

def _count_changes_since(token: Optional[str], drive_id: Optional[str] = None,
                         cap: int = 100000,
                         subtree: Optional[Set[str]] = None) -> Tuple[bool, int, Optional[str]]:
    """Hitung change record Drive sejak `token` (drift detector). Murni baca.
    Return (known, count, new_token). known=False bila token tidak ada atau API
    gagal. count di-cap agar tidak menelusuri tak terbatas."""
    if not token:
        return False, 0, None
    count = 0
    page_token = token
    new_token = None
    while True:
        def _req():
            # Bila filter subtree aktif, butuh parents tiap file change.
            _fields = ("nextPageToken, newStartPageToken, changes(fileId)"
                       if subtree is None else
                       "nextPageToken, newStartPageToken, changes(fileId,"
                       "file(parents,trashed))")
            kw = dict(pageToken=page_token, spaces='drive',
                      fields=_fields,
                      pageSize=1000, includeRemoved=True,
                      supportsAllDrives=True, includeItemsFromAllDrives=True)
            if drive_id: kw['driveId'] = drive_id
            return drive_service.changes().list(**kw)
        res, err = _drive_execute(_req)
        if res is None:
            _log('warning', f"drift count gagal: {err}")
            return False, count, None
        changes = res.get('changes', [])
        if subtree is None:
            count += len(changes)
        else:
            for ch in changes:
                f = ch.get('file') or {}
                parents = set(f.get('parents') or [])
                # File tanpa parents (dihapus/di luar akses) tetap dihitung
                # agar tidak under-count perubahan nyata (fail-safe).
                if parents & subtree or not f.get('parents'):
                    count += 1
        if count >= cap:
            _log('info', f"drift count mencapai cap {cap}")
            return True, count, res.get('newStartPageToken')
        page_token = res.get('nextPageToken')
        if not page_token:
            new_token = res.get('newStartPageToken')
            break
    return True, count, new_token

def _drive_start_page_token(drive_id: Optional[str] = None) -> Optional[str]:
    def _req():
        kw = {'supportsAllDrives': True}
        if drive_id: kw['driveId'] = drive_id
        return drive_service.changes().getStartPageToken(**kw)
    res, err = _drive_execute(_req)
    if res is None:
        _log('warning', f"getStartPageToken gagal: {err}")
        return None
    return res.get('startPageToken')

def _build_subtree(folder_id: str, journal: Optional[FolderJournal]) -> Set[str]:
    """Set folder yang termasuk subtree target (dari journal visited bila ada)."""
    if journal:
        vs = journal.visited_set()
        if vs:
            return vs | {folder_id}
    return {folder_id}

def incremental_scan_via_changes(folder_id: str, journal: FolderJournal,
                                 file_manifest: PersistentFileManifest,
                                 drive_id: Optional[str] = None) -> Tuple[bool, int, int]:
    """Terapkan delta Changes API ke manifest. Return (berhasil, n_added,
    n_removed). Bila berhasil=False, pemanggil wajib fallback ke full scan.
    drive_id wajib untuk folder di Shared Drive. Subfolder baru diekspansi ke
    subtree sampai fixpoint agar file di dalamnya tidak terlewat.
    """
    token = journal.get_changes_token()
    if not token:
        return False, 0, 0
    subtree = _build_subtree(folder_id, journal)
    all_changes: List[Dict] = []
    page_token = token
    new_start = None
    while True:
        def _req():
            kw = dict(
                pageToken=page_token,
                spaces='drive',
                fields="nextPageToken, newStartPageToken, changes(removed,fileId,"
                       "file(id,name,createdTime,owners(emailAddress),mimeType,size,"
                       "md5Checksum,sha256Checksum,trashed,parents,shortcutDetails))",
                pageSize=1000,
                includeRemoved=True,
                supportsAllDrives=True, includeItemsFromAllDrives=True)
            if drive_id: kw['driveId'] = drive_id
            return drive_service.changes().list(**kw)
        res, err = _drive_execute(_req)
        if res is None:
            if err and 'fatal:410' in str(err):
                _log('info', "changes token expired (410) -> fallback full scan")
            else:
                _log('warning', f"changes.list gagal: {err} -> fallback full scan")
            return False, 0, 0
        all_changes.extend(res.get('changes', []))
        page_token = res.get('nextPageToken')
        if not page_token:
            new_start = res.get('newStartPageToken')
            break

    # Pass 1: ekspansi subtree dengan folder baru sampai fixpoint.
    folder_changes = [ch.get('file') or {} for ch in all_changes
                      if not ch.get('removed')
                      and (ch.get('file') or {}).get('mimeType') == 'application/vnd.google-apps.folder'
                      and not (ch.get('file') or {}).get('trashed')]
    new_subtree_folders: List[str] = []
    while True:
        grew = False
        for f in folder_changes:
            fid = f.get('id')
            # Lewati bila fid None (change tanpa metadata id).
            if not fid or fid in subtree: continue
            if set(f.get('parents') or []) & subtree:
                subtree.add(fid); new_subtree_folders.append(fid); grew = True
        if not grew:
            break

    # Bila ada folder baru masuk subtree, file lama di dalamnya tidak terbawa
    # delta -> fallback ke full scan. Persist folder baru sebagai visited;
    # changes token tidak diperbarui agar delta dievaluasi ulang saat full scan.
    if new_subtree_folders:
        journal.add_visited(new_subtree_folders)
        _log('info', f"incremental: {len(new_subtree_folders)} folder baru masuk subtree "
                     f"-> fallback full scan (isi folder tidak terbawa delta)")
        return False, 0, 0

    # Pass 2: evaluasi file changes terhadap subtree final.
    # Semantik LAST-WINS: hanya change terakhir per-fid yang menentukan nasib file.
    decision: Dict[str, Tuple[str, Optional[Dict]]] = {}
    for ch in all_changes:
        fid = ch.get('fileId')
        f   = ch.get('file') or {}
        if not fid:
            continue
        if ch.get('removed') or f.get('trashed'):
            decision[fid] = ('remove', None); continue
        if f.get('mimeType') == 'application/vnd.google-apps.folder':
            continue
        # Shortcut tidak bisa di-download; tolak seperti jalur full-scan.
        if f.get('mimeType') == 'application/vnd.google-apps.shortcut':
            continue
        # Change tanpa 'id' di objek file: tidak bisa diproses, lewati.
        has_record = file_manifest.get_file(fid) is not None
        if not f.get('id'):
            continue
        # Bila parents tidak tersedia, jangan simpulkan file pindah keluar
        # subtree (bisa menghapus file valid secara keliru).
        raw_parents = f.get('parents')
        parents = set(raw_parents or [])
        in_subtree = bool(parents & subtree) or fid in subtree
        if in_subtree:
            decision[fid] = ('add', f)
        elif raw_parents:
            # Parents diketahui & seluruhnya di luar subtree -> pindah keluar.
            decision[fid] = ('remove', None)
        else:
            # Parents tidak diketahui: pertahankan file yang sudah dikenal;
            # tambahkan file baru bermetadata lengkap (Drive kadang menghilangkan
            # 'parents' pada change file baru di subfolder yang sudah ada).
            if has_record:
                decision[fid] = ('add', f)
            elif is_media_file(f.get('name'), f.get('mimeType')) and (
                    f.get('md5Checksum') or f.get('size')):
                decision[fid] = ('add', f)

    # Materialisasi keputusan akhir per-fid menjadi add/remove.
    added: List[Dict] = [meta for (act, meta) in decision.values()
                         if act == 'add' and meta is not None]
    removed: Set[str] = {fid for fid, (act, _) in decision.items() if act == 'remove'}

    if added:
        file_manifest.add_files(added)
        file_manifest.flush()
    if removed:
        current = file_manifest.get_all_file_ids() - removed
        file_manifest.remove_deleted_files(current)
        file_manifest.flush()
    # new_subtree_folders selalu kosong di titik ini (sudah ditangani di atas).
    if new_start:
        journal.set_changes_token(new_start)
    _log('info', f"incremental scan: +{len(added)} / -{len(removed)} (token diperbarui)")
    return True, len(added), len(removed)

# ───────────────────── AUDIT MODE ─────────────────────
def run_audit_mode(storage: LMDBStorage, file_manifest: Optional[PersistentFileManifest] = None
                   ) -> Tuple[bool, List[Dict]]:
    """Audit file: bandingkan md5/size record terhadap manifest. Tier 1 (ada
    md5): bandingkan md5. Tier 3 (tanpa md5): bandingkan size_bytes; bila beda,
    re-download + rehash BLAKE3. Entry mismatch dihapus dan di-re-pending."""
    mismatches = []
    n_checked = 0

    # Kumpulkan snapshot ringan dulu, lalu verifikasi di luar cursor.
    # Menghapus (write txn) sambil cursor read aktif -> perilaku tak terdefinisi.
    audit_rows: List[Tuple[str, str, Optional[str], Optional[int], float]] = []
    for fid, rec in storage.iterate():
        if not rec or not rec.get('blake3'):
            continue
        audit_rows.append((fid, rec.get('blake3', ''), rec.get('md5'),
                           rec.get('size_bytes'), rec.get('size_mb', 0)))

    for fid, stored_b3, stored_md5, stored_size_bytes, size_mb in audit_rows:
        n_checked += 1
        try:
            # Metadata manifest (md5 + size) — sumber kebenaran lokal, 0 API call.
            remote_md5 = None
            manifest_size = None
            if file_manifest:
                info = file_manifest.get_file(fid)
                if info:
                    remote_md5 = info.get('md5Checksum')
                    manifest_size = _safe_int_size(info.get('size'))

            bad = False
            if stored_md5 and remote_md5:
                # Tier 1: deteksi via md5 (0 API).
                bad = (stored_md5 != remote_md5)
            else:
                # Tier 3: tanpa md5, bandingkan size_bytes (0 API).
                # Pakai size_bytes record bila ada; fallback ke perhitungan MB
                # hanya untuk record lama (round-trip MB kehilangan presisi).
                stored_size = stored_size_bytes
                if stored_size is None:
                    stored_size = int(round((size_mb or 0) * 1024 * 1024))
                if manifest_size is not None and manifest_size > 0 and manifest_size == stored_size:
                    continue
                # Size berbeda/tak diketahui: rehash file ini saja.
                if size_mb > MAX_EMBED_MB:
                    b3_actual, _ = download_and_hash_stream(fid)
                else:
                    data      = download_bytes_stream(fid, size_mb)
                    b3_actual = compute_blake3_hash(data) if data is not None else None
                # Download gagal: lewati (jangan hapus record) agar bisa
                # diverifikasi ulang di run berikutnya.
                if b3_actual is None:
                    _log('warning', f"audit skip (re-download gagal) fid={fid}")
                    continue
                bad = (b3_actual != stored_b3)
                # Backfill size_bytes bila file valid tapi record lama belum
                # punya size_bytes, agar tidak di-download ulang setiap audit.
                if (not bad and stored_size_bytes is None
                        and manifest_size is not None and manifest_size > 0):
                    try:
                        _r = storage.get(fid)
                        if _r is not None and _r.get('size_bytes') is None:
                            _r['size_bytes'] = manifest_size
                            storage.put(fid, _r)
                    except Exception as _e_bf:
                        _log('debug', f"backfill size_bytes gagal fid={fid}: {_e_bf}")

            if bad:
                mismatches.append({'file_id': fid, 'name': None,
                                   'stored': stored_md5 or stored_b3, 'actual': remote_md5 or 'rehash'})
                _log('error', f"AUDIT MISMATCH fid={fid} stored_md5={stored_md5} remote_md5={remote_md5}")
                # Entry korup/berubah: hapus dan kembalikan ke pending.
                storage.delete(fid)
                if file_manifest:
                    file_manifest.re_pending(fid)
        except Exception as e:
            _log('warning', f"audit verify failed fid={fid}: {e}")

    _log('info', f"Audit mode: verifikasi {n_checked} file (metadata manifest, ~0 API call)")
    return len(mismatches) == 0, mismatches

# ═════════════════════════════════════════════════════════════════
# Deteksi duplikat dengan Multi-Index Hashing (MIH).
# pHash dibagi MIH_NUM_BANDS band. Prinsip pigeonhole: dua hash berjarak
# Hamming <= (MIH_NUM_BANDS-1) pasti identik di minimal satu band.
# ═════════════════════════════════════════════════════════════════
def _mih_bands(phash_hex: str) -> List[str]:
    """Bagi phash_hex menjadi MIH_NUM_BANDS band dengan distribusi merata.
    Jaminan pigeonhole: dua hash berjarak Hamming <= (MIH_NUM_BANDS-1) pasti
    identik di minimal satu band."""
    if not phash_hex or len(phash_hex) < MIH_NUM_BANDS:
        return [phash_hex] if phash_hex else []
    n = len(phash_hex)
    # Distribusi merata: r band pertama mendapat (q+1) char, sisanya q char.
    q, r = divmod(n, MIH_NUM_BANDS)
    bands: List[str] = []
    pos = 0
    for i in range(MIH_NUM_BANDS):
        width = q + 1 if i < r else q
        bands.append(phash_hex[pos:pos + width])
        pos += width
    return bands

def analyze_duplicates(storage: LMDBStorage,
                       file_manifest: Optional['PersistentFileManifest'] = None) -> Dict:
    # Kumpulkan record kompak (hash di-precompute ke int) lewat streaming cursor.
    recs: Dict[str, Dict] = {}
    # Set fid yang sudah punya record di storage (termasuk record skip permanen).
    # Dipakai oleh hitungan unscanned_media agar file skip-permanen tidak
    # dihitung sebagai 'belum ter-scan': mereka sudah dievaluasi dan sengaja
    # dilewati secara deterministik (terlalu kecil / bukan media menurut
    # extract_features), bukan file yang menunggu diproses ulang.
    storage_fids: Set[str] = set()
    total_images = total_videos = 0
    total_size   = 0.0
    for fid, rec in storage.iterate():
        if not rec:
            continue
        storage_fids.add(fid)
        # Record skip permanen (terlalu kecil/non-media): tidak ikut pencocokan
        # dan tidak dihitung sebagai 'belum ter-scan'.
        if rec.get('skipped'):
            continue
        ft = rec.get('file_type')
        if ft == 'image': total_images += 1
        elif ft == 'video': total_videos += 1
        total_size += rec.get('size_mb', 0)
        ph = rec.get('phash')
        dh = rec.get('dhash')
        dv = rec.get('dvhash')
        sharp_blocks = rec.get('sharpness_blocks')
        color_grid = rec.get('color_grid')
        edge_blocks = rec.get('edge_blocks')
        ssim_thumb = rec.get('ssim_thumb')
        color_hist = rec.get('color_hist')
        aspect_ratio = rec.get('aspect_ratio')
        blockiness_blocks = rec.get('blockiness_blocks')
        # Record lama hanya punya phash/dhash tunggal -> diperlakukan sebagai
        # list satu elemen. dvhash kosong -> verifikasi dvhash dilewati.
        ph_list = rec.get('phashes') or ([ph] if ph else [])
        dh_list = rec.get('dhashes') or ([dh] if dh else [])
        dv_list = rec.get('dvhashes') or ([dv] if dv else [])
        # recs hanya menyimpan data yang dipakai fase pencocokan (hash). Metadata
        # (name/createdTime/ownerEmail) diambil dari storage via _meta() hanya
        # saat sebuah file terbukti duplikat dan dimasukkan ke output, sehingga
        # tidak ditahan di RAM untuk seluruh file (anti-OOM pada jutaan file).
        # dhash hex tidak disimpan: jarak cukup dari d_int; band MIH hanya pakai
        # phash hex. size_mb tetap disimpan karena ikut di tiap entri output.
        recs[fid] = {
            'size_mb': rec.get('size_mb', 0),
            'createdTime': rec.get('createdTime', ''),  # untuk memilih anchor tertua
            'file_type': ft, 'blake3': rec.get('blake3'),
            'width': rec.get('width'),
            'height': rec.get('height'),
            'phash': ph,
            'p_int': _hex_to_int(ph),
            'd_int': _hex_to_int(dh),
            'dv_int': _hex_to_int(dv),
            'sharpness_blocks': sharp_blocks,
            'color_grid': color_grid,
            'edge_blocks': edge_blocks,
            'ssim_thumb': ssim_thumb,
            'color_hist': color_hist,
            'aspect_ratio': aspect_ratio,
            'blockiness_blocks': blockiness_blocks,
            # Sejajar-posisi: None per slot dipertahankan agar perbandingan
            # frame-i vs frame-i tidak bergeser.
            'p_ints': [_hex_to_int(x) for x in ph_list],
            'd_ints': [_hex_to_int(x) for x in dh_list],
            'dv_ints': [_hex_to_int(x) for x in dv_list],
            # p_list untuk bucketing MIH: dipadatkan ke hash valid saja.
            'p_list': [x for x in ph_list if x and _hex_to_int(x) is not None],
            'duration': rec.get('duration'),
        }

    def _meta(_fid: str) -> Dict:
        """Metadata file dari storage untuk satu entri output (hanya dipanggil
        untuk file yang terbukti duplikat). Membawa id/parent/blake3/size_bytes
        agar program remover dapat memverifikasi ulang."""
        m = storage.get(_fid) or {}
        return {
            'id': _fid,
            'name': m.get('name', ''),
            'createdTime': m.get('createdTime', ''),
            'ownerEmail': m.get('ownerEmail', 'N/A'),
            'parent': m.get('parent'),
            'blake3': m.get('blake3'),
            'size_bytes': m.get('size_bytes'),
            'width': m.get('width'),
            'height': m.get('height'),
        }

    # Total file/ukuran akurat: dari manifest (termasuk yang gagal di-hash).
    unscanned_media = 0
    if file_manifest is not None:
        m_images = m_videos = 0
        m_size = 0.0
        for _fid, info in file_manifest.iter_files():
            info = info or {}
            ftm = get_file_type(info.get('name'), info.get('mimeType'))
            if ftm == 'image':   m_images += 1
            elif ftm == 'video': m_videos += 1
            else:                continue
            m_size += _safe_int_size(info.get('size')) / (1024 * 1024)
            # Media tanpa record di storage = belum/gagal ter-scan. Cek terhadap
            # storage_fids (bukan recs) agar file skip-permanen tidak terhitung.
            if _fid not in storage_fids:
                unscanned_media += 1
        if (m_images + m_videos) > 0:
            total_images = m_images
            total_videos = m_videos
            total_size   = m_size
    total_files_stat = total_images + total_videos if file_manifest is not None else len(recs)

    if len(recs) < 2:
        return {
            'image': {}, 'video': {},
            'stats': {
                'total_files': total_files_stat, 'total_size_mb': total_size,
                'total_images': total_images, 'total_videos': total_videos,
                'total_duplicates': 0, 'image_duplicates': 0,
                'video_duplicates': 0, 'wasted_size_mb': 0,
                'unscanned_media': unscanned_media
            },
            'process_trace': {'image': {}}
        }

    # Urut berdasarkan createdTime (tertua = anchor 'asli'), file_id sebagai
    # tie-breaker agar urutan deterministik saat createdTime kosong/sama.
    sorted_ids = sorted(recs, key=lambda k: (recs[k].get('createdTime') or '', k))
    dups       = {'image': {}, 'video': {}}
    marked     = set()

    # Jejak PROSES (diagnostik): hanya dikumpulkan bila PROCESS_REPORT aktif.
    # Hanya foto yang punya gerbang berlapis; video dilewati (jalur berbeda).
    process_trace: Dict[str, Dict[str, Dict]] = {'image': {}}
    def _pt_anchor(fid: str) -> Optional[Dict]:
        """Ambil/buat entri jejak untuk satu anchor foto. None bila nonaktif.
        Node hanya dibuat saat benar-benar dipanggil (ada kandidat/near-miss
        yang akan dicatat), agar tidak ada dict kosong menumpuk di RAM."""
        if not PROCESS_REPORT:
            return None
        node = process_trace['image'].get(fid)
        if node is None:
            node = {'candidates': [], 'near_miss': [],
                    'gates': [], 'lolos': []}
            process_trace['image'][fid] = node
        return node

    # Exact match: pengelompokan berdasarkan BLAKE3.
    b3_groups = defaultdict(list)
    for fid in sorted_ids:
        b3 = recs[fid]['blake3']
        if b3: b3_groups[b3].append(fid)

    # Anchor grup exact (ASLI sebuah grup BLAKE3). Dikumpulkan agar tidak
    # didemosi jadi 'duplikat' di pass visual (lihat 'candidates -= exact_anchors'
    # di bawah), sehingga tidak tampil ganda sebagai ASLI sekaligus DUP.
    exact_anchors: Set[str] = set()
    for b3, ids in b3_groups.items():
        if len(ids) < 2: continue
        orig = ids[0]
        ft   = recs[orig]['file_type'] or 'image'
        exact_anchors.add(orig)
        for dup in ids[1:]:
            marked.add(dup)
            _md = _meta(dup)
            dups[ft].setdefault(orig, []).append({
                'id': dup, 'name': _md['name'],
                'createdTime': _md['createdTime'],
                'ownerEmail': _md['ownerEmail'],
                'parent': _md['parent'],
                'blake3': _md['blake3'],
                'size_bytes': _md['size_bytes'],
                'match_type': 'exact',
                'size_mb': recs[dup]['size_mb']
            })

    # Visual match: bucket MIH lalu verifikasi jarak Hamming.
    remain = [f for f in sorted_ids if f not in marked]

    def _region_blurred_apart(rb_a, rb_b) -> bool:
        """True bila dua foto VERSI BEDA karena blur LOKAL: ada >= BLUR_REGION_MIN_BLOCKS
        blok di mana sisi tertajam bertekstur (hi >= BLUR_REGION_MIN_SHARP) tapi
        rasio ketajaman (lo/hi) < BLUR_REGION_RATIO_MIN (satu sisi jauh lebih
        blur, mis. wajah disensor). Blok rata diabaikan. False bila peta blok tak
        tersedia/tak sebanding (kompatibel mundur -> gerbang dilewati)."""
        if not BLUR_REGION_GATE:
            return False
        if not rb_a or not rb_b or len(rb_a) != len(rb_b):
            return False
        blurred = 0
        for sa, sb in zip(rb_a, rb_b):
            if sa is None or sb is None:
                continue
            lo, hi = (sa, sb) if sa <= sb else (sb, sa)
            # Hanya sisi tertajam (hi) yang wajib bertekstur; sisi blur (lo) boleh rendah.
            if hi < BLUR_REGION_MIN_SHARP:
                continue
            if (lo / hi) < BLUR_REGION_RATIO_MIN:
                blurred += 1
                if blurred >= BLUR_REGION_MIN_BLOCKS:
                    return True
        return False

    def _color_changed_apart(cg_a, cg_b) -> bool:
        """True bila dua foto VERSI BEDA karena perubahan WARNA: ada minimal
        COLOR_GRID_MIN_BLOCKS blok yang selisih warna rata-ratanya (jarak Euclid
        RGB) > COLOR_GRID_MAX_DIST. Menangkap B&W, sepia, tint, dan color grading
        lokal. False bila peta warna tak tersedia/tak sebanding (kompatibel
        mundur -> gerbang dilewati sampai migrasi me-rehash)."""
        if not COLOR_GRID_GATE:
            return False
        if not cg_a or not cg_b or len(cg_a) != len(cg_b):
            return False
        changed = 0
        for ba, bb in zip(cg_a, cg_b):
            if not ba or not bb or len(ba) < 3 or len(bb) < 3:
                continue
            dr = ba[0] - bb[0]
            dg = ba[1] - bb[1]
            db = ba[2] - bb[2]
            dist = (dr * dr + dg * dg + db * db) ** 0.5
            if dist > COLOR_GRID_MAX_DIST:
                changed += 1
                if changed >= COLOR_GRID_MIN_BLOCKS:
                    return True
        return False

    def _edge_added_apart(ea, eb) -> bool:
        """True bila dua foto VERSI BEDA karena PENAMBAHAN tepi: ada >=
        EDGE_REGION_MIN_BLOCKS blok di mana satu sisi punya tepi jauh lebih
        banyak (emoji/stiker/teks/watermark). Syarat ganda agar duplikat sah
        lolos: hi >= EDGE_BLOCK_MIN_DENSITY, selisih hi-lo > EDGE_REGION_DELTA_MIN,
        DAN rasio lo/hi < EDGE_REGION_RATIO_MAX. False bila peta tak tersedia."""
        if not EDGE_REGION_GATE:
            return False
        if not ea or not eb or len(ea) != len(eb):
            return False
        added = 0
        for da, db in zip(ea, eb):
            if da is None or db is None:
                continue
            lo, hi = (da, db) if da <= db else (db, da)
            # Sisi ber-tepi-banyak harus punya tepi nyata (bukan dua blok polos).
            if hi < EDGE_BLOCK_MIN_DENSITY:
                continue
            if (hi - lo) <= EDGE_REGION_DELTA_MIN:
                continue
            if (lo / hi) < EDGE_REGION_RATIO_MAX:
                added += 1
                if added >= EDGE_REGION_MIN_BLOCKS:
                    return True
        return False

    # Helper diagnostik: jumlah blok pemicu tiap gerbang. Hanya untuk logging
    # kalibrasi (GATE-REJECT); tidak memengaruhi keputusan match.
    def _color_changed_count(cg_a, cg_b) -> int:
        if not cg_a or not cg_b or len(cg_a) != len(cg_b):
            return -1
        changed = 0
        for ba, bb in zip(cg_a, cg_b):
            if not ba or not bb or len(ba) < 3 or len(bb) < 3:
                continue
            dr = ba[0] - bb[0]; dg = ba[1] - bb[1]; db = ba[2] - bb[2]
            if (dr * dr + dg * dg + db * db) ** 0.5 > COLOR_GRID_MAX_DIST:
                changed += 1
        return changed

    def _edge_added_count(ea, eb) -> int:
        if not ea or not eb or len(ea) != len(eb):
            return -1
        added = 0
        for da, db in zip(ea, eb):
            if da is None or db is None:
                continue
            lo, hi = (da, db) if da <= db else (db, da)
            if hi < EDGE_BLOCK_MIN_DENSITY:
                continue
            if (hi - lo) <= EDGE_REGION_DELTA_MIN:
                continue
            if (lo / hi) < EDGE_REGION_RATIO_MAX:
                added += 1
        return added

    def _blur_apart_count(rb_a, rb_b) -> int:
        if not rb_a or not rb_b or len(rb_a) != len(rb_b):
            return -1
        blurred = 0
        for sa, sb in zip(rb_a, rb_b):
            if sa is None or sb is None:
                continue
            lo, hi = (sa, sb) if sa <= sb else (sb, sa)
            if hi < BLUR_REGION_MIN_SHARP:
                continue
            if (lo / hi) < BLUR_REGION_RATIO_MIN:
                blurred += 1
        return blurred

    def _blur_diag(rb_a, rb_b):
        """Diagnostik sebaran blur untuk kalibrasi (TIDAK memengaruhi keputusan).
        Membedakan dua pola blur yang jumlah bloknya bisa sama tapi sifatnya
        beda:
          - REKOMPRES GLOBAL : blur DALAM tersebar MERATA ke hampir semua blok
            bertekstur (fraksi blur tinggi).
          - EDITAN LOKAL     : blur DALAM terkumpul di SEBAGIAN blok (wajah/
            objek/bokeh), sisanya tetap tajam (fraksi blur rendah-menengah).

        Mengembalikan dict:
          tekstur   : jumlah blok yang sisi tertajamnya bertekstur
                      (hi >= BLUR_REGION_MIN_SHARP) -> kandidat penilaian.
          blur_lok  : jumlah blok bertekstur yang rasio lo/hi < BLUR_REGION_RATIO_MIN
                      (sama dengan _blur_apart_count; blok 'blur sepihak').
          fraksi    : blur_lok / tekstur (0..1). Tinggi = blur MERATA (indikasi
                      rekompres global); rendah-menengah = blur LOKAL (editan).
          rasio_min : rasio ketajaman (lo/hi) TERDALAM di antara blok bertekstur
                      (mendekati 0 = ada area yang benar-benar hancur/disensor).
        Mengembalikan None bila peta blok tak tersedia/tak sebanding
        (kompatibel mundur)."""
        if not rb_a or not rb_b or len(rb_a) != len(rb_b):
            return None
        tekstur = 0
        blur_lok = 0
        rasio_min = None
        for sa, sb in zip(rb_a, rb_b):
            if sa is None or sb is None:
                continue
            lo, hi = (sa, sb) if sa <= sb else (sb, sa)
            if hi < BLUR_REGION_MIN_SHARP:
                continue
            tekstur += 1
            rasio = lo / hi  # hi >= BLUR_REGION_MIN_SHARP > 0 -> aman
            if rasio_min is None or rasio < rasio_min:
                rasio_min = rasio
            if rasio < BLUR_REGION_RATIO_MIN:
                blur_lok += 1
        if tekstur <= 0:
            return {'tekstur': 0, 'blur_lok': 0, 'fraksi': 0.0, 'rasio_min': None}
        return {'tekstur': tekstur, 'blur_lok': blur_lok,
                'fraksi': blur_lok / float(tekstur), 'rasio_min': rasio_min}

    def _blockiness_global_like(sb_a, sb_b, bk_a, bk_b, diag) -> bool:
        """True bila penurunan ketajaman antara dua foto beraroma RE-KOMPRESI
        GLOBAL (drop kualitas) -- sehingga vonis tolak gerbang blur sebaiknya
        DIBATALKAN dan pasangan diloloskan ke SSIM. False bila lebih mirip blur
        EDITAN/sensor LOKAL (vonis tolak gerbang blur DIPERTAHANKAN).

        Bukti yang dipakai (semua harus terpenuhi):
          1) Penurunan ketajaman MERATA: fraksi blok blur (dari _blur_diag) >=
             BLOCKINESS_GLOBAL_FRAC. Re-kompresi melembutkan SELURUH foto; sensor
             hanya sebagian (fraksi rendah).
          2) Tidak ada blok yang nyaris HANCUR total: rasio ketajaman terdalam
             >= BLOCKINESS_DEEPEST_MIN. Sensor berat menghancurkan satu wilayah
             ekstrem; re-kompresi global tidak.
          3) Sisi yang lebih BURAM menunjukkan ARTEFAK BLOK JPEG yang jauh lebih
             kuat & menyeluruh dari sisi tajam (blockiness rata-rata sisi buram
             >= BLOCKINESS_BLOCK_MIN dan >= BLOCKINESS_RATIO_MIN x sisi tajam).
             Re-kompresi MENAMBAH artefak blok 8x8; blur halus editan TIDAK.

        Fail-safe: bila peta blockiness/diag tak tersedia -> False (jangan rescue)."""
        if not BLOCKINESS_RESCUE_GATE:
            return False
        if not diag:
            return False
        # (1) Lantai fraksi longgar (buang noise). Fraksi bukan pembeda andal;
        # pembeda nyata ada di syarat (3).
        if diag.get('fraksi', 0.0) < BLOCKINESS_GLOBAL_FRAC:
            return False
        # (2) Kedalaman blur: hanya membatalkan rescue bila BLOCKINESS_DEEPEST_MIN
        # > 0 (default 0.0 = nonaktif).
        if BLOCKINESS_DEEPEST_MIN > 0:
            _rm = diag.get('rasio_min')
            if _rm is not None and _rm < BLOCKINESS_DEEPEST_MIN:
                return False
        # (3) PEMUTUS UTAMA: sisi buram harus ber-artefak-blok JPEG jauh lebih
        # kuat & menyebar (ciri re-kompresi; blur editan justru menghapus artefak).
        if (not bk_a or not bk_b or not sb_a or not sb_b
                or len(bk_a) != len(bk_b) or len(sb_a) != len(sb_b)):
            return False
        # Tentukan, per peta ketajaman, sisi mana yang lebih BURAM secara global
        # (jumlah ketajaman lebih kecil), lalu bandingkan blockiness kedua sisi.
        try:
            sum_a = sum(x for x in sb_a if x is not None)
            sum_b = sum(x for x in sb_b if x is not None)
        except TypeError:
            return False
        blur_bk, sharp_bk = (bk_a, bk_b) if sum_a <= sum_b else (bk_b, bk_a)
        vb = [x for x in blur_bk if x is not None]
        vs = [x for x in sharp_bk if x is not None]
        if not vb or not vs:
            return False
        mean_blur = sum(vb) / len(vb)
        mean_sharp = sum(vs) / len(vs)
        if mean_blur < BLOCKINESS_BLOCK_MIN:
            return False
        # Sisi buram harus punya artefak blok minimal RATIO_MIN kali sisi tajam.
        # Bila sisi tajam ~0 artefak (mean_sharp kecil), rasio besar otomatis.
        if mean_blur < BLOCKINESS_RATIO_MIN * max(mean_sharp, 1e-6):
            return False
        # Cakupan: artefak kompresi harus menyebar di sisi buram (re-kompresi),
        # bukan terkumpul di sedikit blok (editan lokal -> cakupan rendah -> tolak).
        coverage = sum(1 for x in vb if x >= BLOCKINESS_BLOCK_MIN) / len(vb)
        if coverage < BLOCKINESS_MIN_COVERAGE:
            return False
        return True

    def _blockiness_diag(sb_a, sb_b, bk_a, bk_b):
        """Diagnostik MENTAH blockiness untuk kalibrasi (TIDAK memengaruhi
        keputusan). Mengembalikan dict {mean_buram, mean_tajam, rasio, cakupan}
        atau None bila peta tak tersedia/tak sebanding. Angka ini memperlihatkan
        seberapa dekat sebuah pasangan ke ambang rescue blockiness (mean buram
        >= BLOCKINESS_BLOCK_MIN, rasio >= BLOCKINESS_RATIO_MIN, cakupan >=
        BLOCKINESS_MIN_COVERAGE) agar kalibrasi tidak menebak."""
        if (not bk_a or not bk_b or not sb_a or not sb_b
                or len(bk_a) != len(bk_b) or len(sb_a) != len(sb_b)):
            return None
        try:
            sum_a = sum(x for x in sb_a if x is not None)
            sum_b = sum(x for x in sb_b if x is not None)
        except TypeError:
            return None
        blur_bk, sharp_bk = (bk_a, bk_b) if sum_a <= sum_b else (bk_b, bk_a)
        vb = [x for x in blur_bk if x is not None]
        vs = [x for x in sharp_bk if x is not None]
        if not vb or not vs:
            return None
        mean_blur = sum(vb) / len(vb)
        mean_sharp = sum(vs) / len(vs)
        rasio = mean_blur / max(mean_sharp, 1e-6)
        cakupan = sum(1 for x in vb if x >= BLOCKINESS_BLOCK_MIN) / len(vb)
        return {'mean_buram': mean_blur, 'mean_tajam': mean_sharp,
                'rasio': rasio, 'cakupan': cakupan}

    def _bk_diag_str(bkd) -> str:
        """Format diagnostik blockiness untuk laporan (string ringkas)."""
        if not bkd:
            return ''
        return (f"blockiness buram {bkd['mean_buram']:.2f} (ambang >= {BLOCKINESS_BLOCK_MIN}) "
                f"| rasio vs tajam {bkd['rasio']:.2f}x (ambang >= {BLOCKINESS_RATIO_MIN}x) "
                f"| cakupan {bkd['cakupan']:.2f} (ambang >= {BLOCKINESS_MIN_COVERAGE})")

    def _aspect_ratio_apart(ar_a, ar_b) -> bool:
        """True bila dua foto VERSI BEDA karena rasio aspek berbeda (Pre-Filter,
        langkah 1 diagram): |AR_a - AR_b| > ASPECT_RATIO_MAX_DELTA. Crop/rotasi/
        ubah kanvas menggeser rasio; resize/kompresi murni mempertahankannya.
        False bila salah satu tak tersedia (kompatibel mundur -> dilewati)."""
        if not ASPECT_RATIO_GATE:
            return False
        if ar_a is None or ar_b is None:
            return False
        try:
            return abs(float(ar_a) - float(ar_b)) > ASPECT_RATIO_MAX_DELTA
        except (TypeError, ValueError):
            return False

    def _hist_corr(ch_a, ch_b) -> Optional[float]:
        """Korelasi histogram warna global antara dua foto: rata-rata koefisien
        korelasi Pearson per-channel RGB. Return skor [-1..1] atau None bila
        histogram tak tersedia/tak sebanding (kompatibel mundur)."""
        if not ch_a or not ch_b or len(ch_a) != len(ch_b):
            return None
        try:
            scores = []
            for ha, hb in zip(ch_a, ch_b):
                if not ha or not hb or len(ha) != len(hb):
                    return None
                va = np.asarray(ha, dtype=np.float64)
                vb = np.asarray(hb, dtype=np.float64)
                sa = va.std(); sb = vb.std()
                if sa == 0 or sb == 0:
                    # Channel datar (mis. warna seragam): cocok bila keduanya
                    # datar & dekat, selain itu tak berkorelasi.
                    scores.append(1.0 if np.allclose(va, vb) else 0.0)
                    continue
                corr = float(np.corrcoef(va, vb)[0, 1])
                if corr != corr:  # NaN guard
                    corr = 0.0
                scores.append(corr)
            if not scores:
                return None
            return sum(scores) / len(scores)
        except Exception as e:
            _log('debug', f"_hist_corr gagal: {e}")
            return None

    def _hist_corr_below(ch_a, ch_b) -> bool:
        """True bila dua foto VERSI BEDA karena korelasi histogram global di
        bawah ambang (Global Warna, langkah 2 diagram: Score >= 0.93). False
        bila tak tersedia (kompatibel mundur -> gerbang dilewati)."""
        if not HIST_CORR_GATE:
            return False
        score = _hist_corr(ch_a, ch_b)
        if score is None:
            return False
        return score < HIST_CORR_THRESHOLD

    def _frame_close(pi, di, vi, pj, dj, vj, use_dv: bool) -> Tuple[bool, int, int]:
        """Apakah frame pada posisi yang sama di sisi A & B cukup dekat menurut
        ketiga hash? Return (dekat, p_dist, d_dist). Ambang tidak diubah.
        Slot kosong (None pada pHash/dHash salah satu sisi) -> tidak cocok.

        Memakai VIDEO_DHASH_FRAME_THRESHOLD / VIDEO_DVHASH_FRAME_THRESHOLD
        (bukan DHASH_THRESHOLD / DVHASH_THRESHOLD yang kini dilonggarkan untuk
        foto). Pemisahan ini menjamin jalur video tetap ketat persis seperti
        semula, tidak terpengaruh pelonggaran ambang foto."""
        if pi is None or pj is None or di is None or dj is None:
            return False, 99, 99
        pd = _popcount(pi ^ pj)
        if pd > VIDEO_PHASH_FRAME_THRESHOLD: return False, 99, 99
        dd = _popcount(di ^ dj)
        if dd > VIDEO_DHASH_FRAME_THRESHOLD: return False, 99, 99
        if use_dv and vi is not None and vj is not None:
            if _popcount(vi ^ vj) > VIDEO_DVHASH_FRAME_THRESHOLD: return False, 99, 99
        return True, pd, dd

    def _frames_match(a: Dict, b: Dict) -> Tuple[bool, int, int]:
        """Cocok HANYA bila dua video identik (boleh beda resolusi/kualitas):
        (1) durasi ~sama (gerbang durasi WAJIB) DAN (2) frame ke-i video A cocok
        ketat dengan frame ke-i video B pada hampir semua posisi.

        Perbandingan POSISIONAL (bukan LCS): sampling sudah selaras-waktu
        deterministik (lihat _extract_frames_ffmpeg), jadi frame ke-i kedua
        video mewakili momen waktu yang sama. 'Turun resolusi' tidak menggeser
        konten tiap posisi -> seluruh posisi cocok. Editan apa pun (trim, sisip,
        ubah konten, susun-ulang) menggeser isi pada posisi -> banyak posisi
        gagal -> rasio turun -> ditolak. LCS lama terlalu longgar: video berbeda
        yang banyak frame statis/mirip-tersebar bisa mencapai rasio tinggi.
        Return (cocok, best_p, best_d)."""
        ap, ad = a['p_ints'], a['d_ints']
        bp, bd = b['p_ints'], b['d_ints']
        av, bv = a.get('dv_ints') or [], b.get('dv_ints') or []
        # Defensif: d_ints/dv_ints bisa lebih pendek dari p_ints pada record lama.
        # Pad dengan None agar akses indeks di loop tidak IndexError.
        if len(ad) < len(ap):
            ad = list(ad) + [None] * (len(ap) - len(ad))
        if len(bd) < len(bp):
            bd = list(bd) + [None] * (len(bp) - len(bd))
        if len(av) < len(ap):
            av = list(av) + [None] * (len(ap) - len(av))
        if len(bv) < len(bp):
            bv = list(bv) + [None] * (len(bp) - len(bv))
        # Butuh minimal satu slot valid di kedua sisi (list kini berisi None).
        if not any(x is not None for x in ap) or not any(x is not None for x in bp):
            return False, 99, 99
        # Gerbang durasi NYATA (detik) WAJIB. 'Resolusi turun' tidak mengubah
        # durasi; editan (trim/sisip) mengubahnya. Perbandingan posisional
        # bergantung pada durasi yang sebanding agar frame ke-i kedua sisi
        # benar-benar momen yang sama, jadi durasi yang tak diketahui pada salah
        # satu sisi -> TOLAK (jangan diloloskan diam-diam).
        da, db = a.get('duration'), b.get('duration')
        if not (da and db and da > 0 and db > 0):
            return False, 99, 99
        longer = max(da, db)
        # Toleransi relatif, dijepit antara lantai mutlak (video pendek) dan
        # plafon mutlak (video panjang: cegah toleransi meloloskan editan).
        allowed = min(VIDEO_DURATION_MAX_DELTA,
                      max(VIDEO_DURATION_MIN_DELTA, longer * VIDEO_DURATION_TOLERANCE))
        if abs(da - db) > allowed:
            return False, 99, 99
        # dHash vertikal hanya diverifikasi bila kedua sisi memilikinya.
        use_dv = any(x is not None for x in av) and any(x is not None for x in bv)
        # Panjang = jumlah titik grid pHash PENUH (bukan min dengan dHash), agar
        # slot dHash yang hilang tetap menurunkan rasio (dihitung tidak-cocok
        # oleh _frame_close), bukan menaikkannya.
        la = len(ap)
        lb = len(bp)
        if la == 0 or lb == 0:
            return False, 99, 99
        # Perbandingan POSISIONAL slot-ke-slot. Denominator = jumlah titik grid
        # TERPANJANG: posisi yang hanya ada di satu sisi (beda num_samples akibat
        # beda durasi, atau trim/sisip) otomatis dihitung tidak-cocok sehingga
        # menurunkan rasio. Posisi dengan slot None di salah satu sisi juga
        # tidak-cocok (frame hilang sepihak -> bukan re-encode identik).
        k = min(la, lb)
        longer_k = max(la, lb)
        matched = 0
        valid_pos = 0
        # Denominator efektif = posisi RELEVAN. Slot yang kedua sisinya tak
        # terukur (None, mis. fade/layar hitam) dikecualikan; slot ekor dan slot
        # yang hanya satu sisi None tetap dihitung tidak-cocok (ketahanan editan).
        relevant = longer_k - k  # slot ekor selalu relevan (tidak-cocok)
        best_p = best_d = 99
        for i in range(k):
            ai_valid = ap[i] is not None and ad[i] is not None
            bi_valid = bp[i] is not None and bd[i] is not None
            if ai_valid and bi_valid:
                valid_pos += 1
            # Slot netral: kedua sisi sama-sama tak terukur -> keluarkan dari
            # denominator (jangan hukum video dengan banyak frame gelap sah).
            a_none = ap[i] is None
            b_none = bp[i] is None
            if a_none and b_none:
                continue
            relevant += 1
            vi = av[i] if use_dv else None
            vj = bv[i] if use_dv else None
            close, pd, dd = _frame_close(ap[i], ad[i], vi, bp[i], bd[i], vj, use_dv)
            if close:
                matched += 1
                if pd < best_p: best_p = pd
                if dd < best_d: best_d = dd
        # Gate ADAPTIF: minimal dua-pertiga dari min(k, VIDEO_MIN_SAMPLES) posisi
        # harus valid (lantai mutlak 2), agar keputusan tidak diambil dari
        # segelintir frame. Lantai naik ke VIDEO_MIN_VALID_POS bila grid cukup
        # panjang (k >= VIDEO_MIN_SAMPLES).
        min_valid = max(2, (min(k, VIDEO_MIN_SAMPLES) * 2) // 3)
        if k >= VIDEO_MIN_SAMPLES:
            min_valid = max(min_valid, VIDEO_MIN_VALID_POS)
        if valid_pos < min_valid:
            return False, 99, 99
        # Denominator relevan tak boleh 0 (semua slot netral): tolak agar tidak
        # membagi nol / meloloskan dari ketiadaan bukti.
        if relevant <= 0:
            return False, 99, 99
        if (matched / relevant) < VIDEO_FRAME_MATCH_RATIO:
            return False, 99, 99
        return True, best_p, best_d

    for ft in ('image', 'video'):
        # Image valid bila p_int ada. Video disaring dengan p_ints (daftar
        # frame-hash), bukan p_int (median yang bisa None walau frame valid).
        if ft == 'video':
            type_files = [fid for fid in remain
                          if recs[fid]['file_type'] == ft
                          and any(x is not None for x in recs[fid]['p_ints'])]
        else:
            type_files = [fid for fid in remain
                          if recs[fid]['file_type'] == ft and recs[fid]['p_int'] is not None]

        # Bucket MIH: image memakai pHash tunggal, video mengindeks semua frame-hash.
        mih_buckets = defaultdict(list)
        for fid in type_files:
            if ft == 'video':
                # Dedup (band_index, band) per file: banyak frame video memberi
                # band identik; tanpa dedup _frames_match dipanggil berulang.
                seen_bands = set()
                for ph_hex in recs[fid]['p_list']:
                    for bi, band in enumerate(_mih_bands(ph_hex)):
                        if (bi, band) in seen_bands:
                            continue
                        seen_bands.add((bi, band))
                        mih_buckets[(bi, band)].append(fid)
            else:
                for bi, band in enumerate(_mih_bands(recs[fid]['phash'])):
                    mih_buckets[(bi, band)].append(fid)

        seen = set()
        for fid in type_files:
            if fid in seen: continue
            r = recs[fid]
            # Tandai anchor (file tertua) sebagai sudah-terlihat agar tidak
            # muncul sebagai kandidat/duplikat di grup lain (tampil ganda).
            seen.add(fid)
            marked.add(fid)
            candidates = set()
            if ft == 'video':
                for ph_hex in r['p_list']:
                    for bi, band in enumerate(_mih_bands(ph_hex)):
                        bucket = mih_buckets.get((bi, band), ())
                        # Lewati bucket jenuh (band tak selektif). Pasangan sah
                        # tetap bertemu lewat band lain (pigeonhole); cegah O(n^2).
                        if len(bucket) > MIH_VIDEO_BUCKET_CAP:
                            continue
                        candidates.update(bucket)
            else:
                for bi, band in enumerate(_mih_bands(r['phash'])):
                    candidates.update(mih_buckets.get((bi, band), ()))
            candidates.discard(fid)
            candidates -= seen
            candidates -= marked
            # Jangan demosi anchor grup EXACT jadi 'duplikat' visual (sudah ASLI
            # di grup BLAKE3-nya). Boleh jadi anchor visual, tak boleh kandidat/dup.
            candidates -= exact_anchors

            for cand in candidates:
                c = recs[cand]
                if ft == 'video':
                    # Kelayakan video memakai daftar frame-hash, bukan median
                    # p_int (yang bisa None walau frame-hash valid).
                    if not c['p_ints']: continue
                    ok, p_dist, d_dist = _frames_match(r, c)
                    if not ok: continue
                else:
                    # Jejak LAZY: jangan buat node anchor sebelum ada yang dicatat.
                    _ptn = None
                    if c['p_int'] is None: continue
                    p_dist = _popcount(r['p_int'] ^ c['p_int'])
                    if p_dist > PHASH_THRESHOLD:
                        # Gugur di pintu pHash. Catat sebagai near-miss bila masih
                        # cukup dekat (terlihat gugur di pHash atau gerbang dalam).
                        if PROCESS_REPORT and p_dist < PROCESS_REPORT_NEAR_PHASH:
                            _ptn = _pt_anchor(fid)
                            if _ptn is not None:
                                _ptn['near_miss'].append((cand, p_dist))
                        continue
                    # Lolos pintu pHash: resmi jadi calon duplikat visual.
                    if PROCESS_REPORT:
                        _ptn = _pt_anchor(fid)
                    if _ptn is not None:
                        _ptn['candidates'].append(cand)
                    # pHash dan dHash horizontal harus sama-sama cocok untuk
                    # menekan false-positive.
                    if r['d_int'] is None or c['d_int'] is None:
                        if _ptn is not None:
                            _ptn['gates'].append({'nama': 'pHash+piksel (dHash)',
                                'keluar': [(cand, 'dHash tidak tersedia', '—')]})
                        continue
                    d_dist = _popcount(r['d_int'] ^ c['d_int'])
                    if d_dist > DHASH_THRESHOLD:
                        if _ptn is not None:
                            _ptn['gates'].append({'nama': 'pHash+piksel (dHash)',
                                'keluar': [(cand, 'dHash H jauh',
                                            f'd_dist {d_dist} | ambang tolak > {DHASH_THRESHOLD}')]})
                        continue
                    # dHash V memperkuat deteksi foto beresolusi turun. WAJIB
                    # bila kedua sisi punya dv_int; record lama dilewati.
                    if r['dv_int'] is not None and c['dv_int'] is not None:
                        _dv_dist = _popcount(r['dv_int'] ^ c['dv_int'])
                        if _dv_dist > DVHASH_THRESHOLD:
                            if _ptn is not None:
                                _ptn['gates'].append({'nama': 'pHash+piksel (dHash V)',
                                    'keluar': [(cand, 'dHash V jauh',
                                                f'dv_dist {_dv_dist} | ambang tolak > {DVHASH_THRESHOLD}')]})
                            continue
                    # [1] Gerbang ASPECT RATIO (pra-filter). Rasio aspek foto
                    # identik pasti sama; crop/rotasi menggesernya. WAJIB bila
                    # kedua sisi punya aspect_ratio; record lama dilewati.
                    if _aspect_ratio_apart(r.get('aspect_ratio'),
                                           c.get('aspect_ratio')):
                        _log('debug',
                             f"GATE-REJECT aspect fid={fid} cand={cand} "
                             f"p_dist={p_dist} d_dist={d_dist} "
                             f"ar_a={r.get('aspect_ratio')} ar_b={c.get('aspect_ratio')}")
                        if _ptn is not None:
                            try:
                                _delta = abs(float(r.get('aspect_ratio')) - float(c.get('aspect_ratio')))
                                _delta_str = f'delta {_delta:.2f}'
                            except (TypeError, ValueError):
                                _delta_str = 'delta ?'
                            _ptn['gates'].append({'nama': 'aspect ratio',
                                'keluar': [(cand, 'rasio aspek berbeda',
                                            f'{_delta_str} | ambang tolak > {ASPECT_RATIO_MAX_DELTA}')]})
                        continue
                    # Urutan gerbang foto: murah -> mahal (short-circuit). SSIM
                    # paling belakang (paling mahal). Setiap penolakan dicatat ke
                    # audit log (file saja) untuk kalibrasi ambang.
                    if STRICT_VISUAL:
                        # [2] Gerbang HISTOGRAM (Global Warna). Foto identik punya
                        # korelasi histogram ~1.0; korelasi < ambang -> tolak.
                        # WAJIB bila kedua sisi punya color_hist; record lama dilewati.
                        if _hist_corr_below(r.get('color_hist'),
                                            c.get('color_hist')):
                            _hc = _hist_corr(r.get('color_hist'), c.get('color_hist'))
                            _log('debug',
                                 f"GATE-REJECT hist fid={fid} cand={cand} "
                                 f"p_dist={p_dist} d_dist={d_dist} "
                                 f"hist_corr={_hc}")
                            if _ptn is not None:
                                _hc_str = 'korelasi ?' if _hc is None else f'korelasi {_hc:.2f}'
                                _ptn['gates'].append({'nama': 'histogram',
                                    'keluar': [(cand, 'distribusi warna global beda jauh',
                                                f'{_hc_str} | ambang lolos >= {HIST_CORR_THRESHOLD}')]})
                            continue

                        # [3] Gerbang WARNA PER-REGION (color_grid). Menangkap
                        # filter warna (B&W/sepia/tint) yang buta bagi hash
                        # grayscale & SSIM. WAJIB bila kedua sisi punya color_grid.
                        if _color_changed_apart(r.get('color_grid'),
                                                c.get('color_grid')):
                            _cc = _color_changed_count(r.get('color_grid'), c.get('color_grid'))
                            _log('debug',
                                 f"GATE-REJECT color fid={fid} cand={cand} "
                                 f"p_dist={p_dist} d_dist={d_dist} "
                                 f"color_blocks_changed={_cc}")
                            if _ptn is not None:
                                _ptn['gates'].append({'nama': 'color_grid',
                                    'keluar': [(cand, 'warna lokal berubah di banyak blok',
                                                f'{_cc} blok berubah | ambang tolak >= {COLOR_GRID_MIN_BLOCKS}')]})
                            continue

                    # [4] Gerbang EDGE PER-REGION (anti emoji/stiker/teks/watermark).
                    # Menangkap tepi tajam baru yang lolos color_grid. Aktif bila
                    # STRICT_EDGE; field tetap tersimpan walau mati (tanpa scan ulang).
                    if STRICT_EDGE:
                        if _edge_added_apart(r.get('edge_blocks'),
                                             c.get('edge_blocks')):
                            _ec = _edge_added_count(r.get('edge_blocks'), c.get('edge_blocks'))
                            _log('debug',
                                 f"GATE-REJECT edge fid={fid} cand={cand} "
                                 f"p_dist={p_dist} d_dist={d_dist} "
                                 f"edge_blocks_added={_ec}")
                            if _ptn is not None:
                                _ptn['gates'].append({'nama': 'edge',
                                    'keluar': [(cand, 'muncul tepi baru di beberapa blok',
                                                f'{_ec} blok tepi baru | ambang tolak >= {EDGE_REGION_MIN_BLOCKS}')]})
                            continue

                    # [5] Gerbang anti-blur PER-REGION (wajah/plat disensor).
                    # Menangkap blur lokal maupun seluruh foto. WAJIB bila kedua
                    # sisi punya peta blok; record lama dilewati. Aktif bila STRICT_BLUR.
                    if STRICT_BLUR:
                        # Diagnostik sebaran blur (global vs lokal) untuk
                        # kalibrasi. Dihitung sekali, dipakai baik saat blur
                        # GAGAL maupun LOLOS. Tidak memengaruhi keputusan.
                        _bd = _blur_diag(r.get('sharpness_blocks'),
                                         c.get('sharpness_blocks')) if PROCESS_REPORT else None
                        def _bd_str(bd):
                            if not bd:
                                return ''
                            _rm = bd.get('rasio_min')
                            _rm_s = '?' if _rm is None else f"{_rm:.2f}"
                            return (f"fraksi {bd['fraksi']:.2f} "
                                    f"(blur {bd['blur_lok']}/{bd['tekstur']} blok bertekstur) "
                                    f"| rasio terdalam {_rm_s}")
                        if _region_blurred_apart(r.get('sharpness_blocks'),
                                                 c.get('sharpness_blocks')):
                            _bc = _blur_apart_count(r.get('sharpness_blocks'), c.get('sharpness_blocks'))
                            # PEMBELA blockiness: bila buramnya beraroma re-kompresi
                            # global (artefak blok JPEG merata & kuat), batalkan
                            # vonis -> lanjut ke SSIM. Blur lokal/sensor tetap ditolak.
                            _bd_full = _bd if _bd is not None else _blur_diag(
                                r.get('sharpness_blocks'), c.get('sharpness_blocks'))
                            _rescued = _blockiness_global_like(
                                r.get('sharpness_blocks'), c.get('sharpness_blocks'),
                                r.get('blockiness_blocks'), c.get('blockiness_blocks'),
                                _bd_full)
                            if _rescued:
                                _log('debug',
                                     f"BLUR-RESCUE (re-kompresi global) fid={fid} cand={cand} "
                                     f"p_dist={p_dist} d_dist={d_dist} "
                                     f"blur_blocks_apart={_bc} "
                                     f"blur_fraksi={(_bd_full or {}).get('fraksi')} "
                                     f"rasio_min={(_bd_full or {}).get('rasio_min')}")
                                if _ptn is not None:
                                    _bds = _bd_str(_bd_full)
                                    _info = (f"{_bc} blok blur tapi DISELAMATKAN: "
                                             f"blockiness merata (ciri re-kompresi global)")
                                    if _bds:
                                        _info += f' || {_bds}'
                                    _ptn['gates'].append({'nama': 'blur_diag',
                                        'info': (cand, _info)})
                                # Tidak 'continue': lanjut ke gerbang SSIM.
                            else:
                                _log('debug',
                                     f"GATE-REJECT blur fid={fid} cand={cand} "
                                     f"p_dist={p_dist} d_dist={d_dist} "
                                     f"blur_blocks_apart={_bc} "
                                     f"blur_fraksi={(_bd or {}).get('fraksi')} "
                                     f"rasio_min={(_bd or {}).get('rasio_min')}")
                                if _ptn is not None:
                                    _diag_suffix = ''
                                    _bds = _bd_str(_bd)
                                    if _bds:
                                        _diag_suffix = f' || {_bds}'
                                    # Diagnostik blockiness: kenapa rescue gagal
                                    # (angka mentah vs ambang) untuk kalibrasi.
                                    _bkd = _blockiness_diag(
                                        r.get('sharpness_blocks'), c.get('sharpness_blocks'),
                                        r.get('blockiness_blocks'), c.get('blockiness_blocks'))
                                    _bkds = _bk_diag_str(_bkd)
                                    if _bkds:
                                        _diag_suffix += f' || {_bkds}'
                                    _ptn['gates'].append({'nama': 'blur',
                                        'keluar': [(cand, 'satu sisi jauh lebih blur di beberapa blok',
                                                    f'{_bc} blok blur | ambang tolak >= {BLUR_REGION_MIN_BLOCKS}{_diag_suffix}')]})
                                continue
                        # Blur LOLOS: catat diagnostik sebaran untuk kalibrasi.
                        if _ptn is not None and _bd is not None:
                            _ptn['gates'].append({'nama': 'blur_diag',
                                'info': (cand, _bd_str(_bd))})

                    # [6] Gerbang SSIM — pemutus akhir foto (MAHAL, paling
                    # belakang). Foto identik -> SSIM >= ambang -> DUPLIKAT; edit
                    # kecil menurunkannya -> BEDA. ssim_thumb None (record lama)
                    # -> gerbang dilewati (fail-open).
                    if STRICT_SSIM:
                        _ssim_score = _ssim_match(r, c)
                        if _ssim_score is not None and _ssim_score < SSIM_THRESHOLD:
                            _log('debug',
                                 f"GATE-REJECT ssim fid={fid} cand={cand} "
                                 f"p_dist={p_dist} d_dist={d_dist} "
                                 f"ssim={_ssim_score:.4f} threshold={SSIM_THRESHOLD}")
                            if _ptn is not None:
                                _kurang = SSIM_THRESHOLD - _ssim_score
                                _ptn['gates'].append({'nama': 'ssim',
                                    'keluar': [(cand, 'kemiripan struktur di bawah ambang',
                                                f'ssim {_ssim_score:.3f} | ambang lolos >= {SSIM_THRESHOLD} (kurang {_kurang:.3f})')]})
                            continue
                        # Pasangan LOLOS semua gerbang: catat skor SSIM (kalibrasi).
                        _log('debug',
                             f"GATE-PASS fid={fid} cand={cand} "
                             f"p_dist={p_dist} d_dist={d_dist} "
                             f"ssim={'NA' if _ssim_score is None else f'{_ssim_score:.4f}'}")
                        if _ptn is not None:
                            _ssim_str = 'NA' if _ssim_score is None else f'{_ssim_score:.3f}'
                            _ptn['gates'].append({'nama': 'ssim',
                                'lolos_nilai': (cand, f'ssim {_ssim_str}')})
                    if _ptn is not None:
                        _ptn['lolos'].append(cand)

                seen.add(cand); marked.add(cand)
                _md = _meta(cand)
                dups[ft].setdefault(fid, []).append({
                    'id': cand, 'name': _md['name'],
                    'createdTime': _md['createdTime'],
                    'ownerEmail': _md['ownerEmail'],
                    'parent': _md['parent'],
                    'blake3': _md['blake3'],
                    'size_bytes': _md['size_bytes'],
                    'match_type': 'visual_pd',
                    'phash_dist': p_dist,
                    'dhash_dist': d_dist,
                    'size_mb': c['size_mb'],
                    'width': _md.get('width'),
                    'height': _md.get('height'),
                })

    wasted = sum(d.get('size_mb', 0) for bucket in dups.values() for g in bucket.values() for d in g)
    stats  = {
        'total_files': total_files_stat, 'total_size_mb': total_size,
        'total_images': total_images, 'total_videos': total_videos,
        'total_duplicates': sum(len(g) for b in dups.values() for g in b.values()),
        'image_duplicates': sum(len(g) for g in dups['image'].values()),
        'video_duplicates': sum(len(g) for g in dups['video'].values()),
        'wasted_size_mb': wasted,
        'unscanned_media': unscanned_media
    }
    return {'image': dups['image'], 'video': dups['video'], 'stats': stats,
            # Jejak diagnostik per-anchor foto (kosong bila PROCESS_REPORT off).
            # Dipakai _build_process_report_html (halaman lanjutan PDF); tidak
            # memengaruhi laporan TXT/PDF utama maupun keputusan duplikat.
            'process_trace': process_trace}

# ════════════════════════════════════════════════════════════
# HALAMAN LANJUTAN PDF: LAPORAN PROSES / KEPUTUSAN DI BALIK LAYAR
# ════════════════════════════════════════════════════════════
# Di-inject sebagai halaman lanjutan di PDF (sebelum footer), membaca jejak
# gerbang (process_trace) dari analyze_duplicates. Tabel urut: pHash -> dHash H
# -> dHash V -> Aspect -> Histogram -> Color G -> Edge -> Blur -> SSIM; tiap
# kandidat ditandai ✓ (lolos), ✗ (gugur di gerbang itu), atau — (tak diuji).
# Bila PROCESS_REPORT off / tidak ada jejak, kembalikan '' (PDF tetap seperti biasa).
def _build_process_report_html(folder_name: str, storage: 'LMDBStorage',
                               duplicate_result: Dict) -> str:
    if not PROCESS_REPORT:
        return ''
    trace = (duplicate_result or {}).get('process_trace') or {}
    img_trace: Dict[str, Dict] = trace.get('image') or {}
    image_dups = (duplicate_result or {}).get('image', {}) or {}
    video_dups = (duplicate_result or {}).get('video', {}) or {}

    def _orig(oid):
        return storage.get(oid) or {}

    def _name(fid: str) -> str:
        m = storage.get(fid) or {}
        return m.get('name') or fid

    def _flink(name, file_id):
        """Nama file sebagai link Drive (ellipsis ditangani CSS, bukan dipotong)."""
        url = f"https://drive.google.com/file/d/{file_id}/view"
        return f'<a href="{url}" title="{html.escape(str(name))}">{html.escape(str(name))}</a>'

    # Urutan & label gerbang (sinkron dengan analyze_duplicates). Pra-filter
    # 'pHash+piksel...' dipecah jadi 3 kolom (pHash/dHash H/dHash V) yang
    # lolos/gugur bersama. blur_diag bukan gerbang (diabaikan di tabel).
    GATE_FLOW = ['__prefilter__', 'aspect ratio', 'histogram',
                 'color_grid', 'edge', 'blur', 'ssim']
    GAGAL_LABEL = {
        '__prefilter__': 'pHash + dHash H/V',
        'aspect ratio':  'Aspect Ratio', 'histogram': 'Histogram',
        'color_grid':    'Color Grid',   'edge':      'Edge',
        'blur':          'Blur',         'ssim':      'SSIM',
    }
    LOLOS_NAMA = {
        '__prefilter__': 'pHash, dHash H, dHash V', 'aspect ratio': 'Aspect',
        'histogram': 'Histogram', 'color_grid': 'Color Grid', 'edge': 'Edge',
        'blur': 'Blur', 'ssim': 'SSIM',
    }
    DETAIL_BATAS = {
        'color_grid': f"blok dihitung berubah bila selisih warna > {COLOR_GRID_MAX_DIST}",
        'edge':       (f"blok dihitung bila tepi > {EDGE_BLOCK_MIN_DENSITY}, "
                       f"selisih > {EDGE_REGION_DELTA_MIN}, rasio < {EDGE_REGION_RATIO_MAX}"),
        'blur':       (f"blok dihitung bila rasio ketajaman < {BLUR_REGION_RATIO_MIN} "
                       f"dan area tajam > {BLUR_REGION_MIN_SHARP}"),
    }
    def _lolos_hingga(gate_name):
        try:
            idx = GATE_FLOW.index(gate_name)
        except ValueError:
            return '-'
        passed = [LOLOS_NAMA[g] for g in GATE_FLOW[:idx]]
        return ', '.join(passed) if passed else '-'

    # 9 kolom gerbang di tabel (pra-filter dipecah jadi pHash/dHash H/dHash V).
    GATE_COLS = [
        ('__prefilter__', 'pHash'), ('__prefilter__', 'dHash H'),
        ('__prefilter__', 'dHash V'), ('aspect ratio', 'Aspect'),
        ('histogram', 'Hist'), ('color_grid', 'Color G'),
        ('edge', 'Edge'), ('blur', 'Blur'), ('ssim', 'SSIM'),
    ]

    # Status saklar gerbang (untuk tabel "GERBANG AKTIF").
    _hist_on  = STRICT_VISUAL and HIST_CORR_GATE
    _color_on = STRICT_VISUAL and COLOR_GRID_GATE
    gerbang_rows = [
        ('BLAKE3 (EXACT)',  True,            "identik byte-per-byte (tanpa ambang) = pasti duplikat"),
        ('pHash + dHash H/V', True,         f"pHash ≤ {PHASH_THRESHOLD}, dHash H ≤ {DHASH_THRESHOLD}, dHash V ≤ {DVHASH_THRESHOLD}"),
        ('Aspect Ratio',   ASPECT_RATIO_GATE, f"delta ≤ {ASPECT_RATIO_MAX_DELTA}"),
        ('Histogram',      _hist_on,         f"korelasi ≥ {HIST_CORR_THRESHOLD}"),
        ('Color Grid',     _color_on,        f"tolak ≥ {COLOR_GRID_MIN_BLOCKS} blok"),
        ('Edge',           STRICT_EDGE and EDGE_REGION_GATE, f"tolak ≥ {EDGE_REGION_MIN_BLOCKS} blok"),
        ('Blur',           STRICT_BLUR and BLUR_REGION_GATE, f"tolak ≥ {BLUR_REGION_MIN_BLOCKS} blok"),
        ('SSIM',           STRICT_SSIM,      f"lolos ≥ {SSIM_THRESHOLD}"),
    ]

    # Grup EXACT (BLAKE3) dari hasil (anggota match_type == 'exact').
    exact_groups = []
    for bucket in (image_dups, video_dups):
        for oid, dlist in bucket.items():
            ex = [d for d in dlist if d.get('match_type') == 'exact']
            if ex:
                exact_groups.append((oid, ex))

    # Syarat masuk laporan: lolos pHash (punya 'candidates'); anchor near-miss
    # saja diabaikan. Grup ADA HASIL ditaruh di atas, grup TANPA HASIL di bawah.
    _all_anchors = [a for a in sorted(img_trace.keys())
                    if (img_trace[a].get('candidates') or [])]
    _anchors_hasil = [a for a in _all_anchors if (img_trace[a].get('lolos') or [])]
    _anchors_kosong = [a for a in _all_anchors if not (img_trace[a].get('lolos') or [])]
    visual_anchors = _anchors_hasil + _anchors_kosong
    _kosong_set = set(_anchors_kosong)

    # CSS section proses (gaya konsisten dgn laporan duplikat di atas).
    css = """
    <style>
        .proc-divider { page-break-before: always; border-top: 3px solid #0f172a; margin: 30px 0 20px 0; padding-top: 15px; }
        .proc-head { text-align: center; margin-bottom: 20px; }
        .proc-head h2 { font-size: 15pt; color: #0f172a; margin: 0 0 5px 0; }
        .proc-head p { font-size: 9.5pt; color: #64748b; margin: 0; }
        .proc-sec-title { font-size: 11pt; font-weight: bold; color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; margin: 18px 0 10px 0; }
        .gate-cfg { width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 9pt; }
        .gate-cfg th { background: #f1f5f9; padding: 6px 8px; border: 1px solid #cbd5e1; text-align: left; }
        .gate-cfg td { padding: 6px 8px; border: 1px solid #cbd5e1; background: #fff; }
        .gc-on { color: #059669; font-weight: bold; font-family: 'DejaVu Sans', sans-serif; } .gc-off { color: #94a3b8; font-family: 'DejaVu Sans', sans-serif; }
        .proc-info { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 5px; padding: 8px 12px; font-size: 8.5pt; color: #64748b; margin-bottom: 12px; }
        .jalur-title { background: #0f172a; color: #fff; padding: 8px 12px; font-size: 10.5pt; font-weight: bold; margin-bottom: 10px; }
        .ex-box { border: 2px solid #fecaca; border-radius: 8px; margin-bottom: 12px; page-break-inside: avoid; }
        .ex-head { background: #fef2f2; padding: 10px 12px; border-bottom: 1px solid #fecaca; }
        .ex-title { font-size: 10.5pt; font-weight: bold; color: #dc2626; margin-bottom: 6px; }
        .kv { font-size: 8.5pt; color: #475569; margin: 2px 0; }
        .kv .k { display: inline-block; width: 70px; color: #94a3b8; }
        .kv .s { display: inline-block; width: 12px; text-align: center; }
        /* Tiap grup visual mulai di halaman baru (.vi-break). Grup boleh meluber
           lintas halaman; TIDAK lagi page-break-inside:avoid agar grup panjang
           (tabel + penjelasan) tidak dipaksa muat satu halaman / mengecil. */
        .vi-box { border: 2px solid #bbf7d0; border-radius: 8px; margin-bottom: 16px; }
        .vi-break { page-break-before: always; }
        .proc-pagebreak { page-break-before: always; }
        .vi-head { background: #f0fdf4; padding: 10px 12px; border-bottom: 1px solid #bbf7d0; }
        .vi-title { font-size: 10.5pt; font-weight: bold; color: #059669; margin-bottom: 6px; }
        .gate-tbl { width: 100%; border-collapse: collapse; font-size: 7.5pt; margin: 8px 0; table-layout: fixed; }
        .gate-tbl th { background: #f1f5f9; padding: 5px 2px; border: 1px solid #cbd5e1; text-align: center; font-weight: bold; font-size: 7pt; }
        .gate-tbl td { padding: 5px 2px; border: 1px solid #cbd5e1; text-align: center; background: #fff; }
        .gate-tbl td.fn { text-align: left; padding-left: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .gate-tbl td.fn a { color: #0ea5e9; text-decoration: none; }
        /* Simbol lolos/gugur/skip menggunakan karakter Unicode dengan font
           Noto Sans yang di-install via apt-get di awal script. Font ini
           mendukung karakter ✓ (U+2713), ✗ (U+2717), dan — (U+2014). */
        .g-pass { color: #059669; font-family: 'Noto Sans', 'DejaVu Sans', sans-serif; font-size: 11pt; font-weight: bold; }
        .g-fail { color: #dc2626; font-family: 'Noto Sans', 'DejaVu Sans', sans-serif; font-size: 11pt; font-weight: bold; }
        .g-skip { color: #94a3b8; font-family: 'Noto Sans', 'DejaVu Sans', sans-serif; font-size: 10pt; }
        .h-dup { background: #059669; color: #fff; padding: 2px 4px; border-radius: 6px; font-size: 6.5pt; font-weight: bold; }
        .h-no { background: #dc2626; color: #fff; padding: 2px 4px; border-radius: 6px; font-size: 6.5pt; font-weight: bold; }
        .ringkas { background: #f8fafc; padding: 6px 12px; font-size: 8.5pt; border-top: 1px solid #e2e8f0; }
        .pj-box { background: #fef3c7; padding: 10px 12px; border-top: 2px solid #fcd34d; }
        .pj-title { font-weight: bold; color: #b45309; margin-bottom: 8px; font-size: 9.5pt; }
        .pj-item { background: #fff; border: 1px solid #fcd34d; border-radius: 5px; padding: 8px 10px; margin-bottom: 6px; }
        .pj-fn { font-weight: bold; color: #dc2626; margin-bottom: 5px; font-size: 8.5pt; }
        .pj-fn a { color: #dc2626; }
        .pj-kv { font-size: 7.5pt; margin: 3px 0; color: #475569; }
        .pj-kv .k { display: inline-block; width: 90px; color: #64748b; }
        .pj-kv .s { display: inline-block; width: 12px; text-align: center; }
        .cara { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 5px; padding: 12px; font-size: 8.5pt; color: #475569; margin-top: 16px; page-break-before: always; page-break-inside: avoid; }
        .cara h4 { color: #1e293b; margin: 0 0 8px 0; font-size: 9.5pt; }
        .cara p { margin: 5px 0; }
        .cara-tbl { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 8.5pt; page-break-inside: avoid; }
        .cara-tbl th { background: #f1f5f9; padding: 5px 8px; border: 1px solid #cbd5e1; text-align: left; color: #1e293b; }
        .cara-tbl td { padding: 5px 8px; border: 1px solid #cbd5e1; background: #fff; vertical-align: top; }
        .cara-tbl td:first-child { width: 32%; font-weight: bold; color: #0f172a; white-space: nowrap; }
        .sym-p { color: #059669; font-family: 'Noto Sans', 'DejaVu Sans', sans-serif; font-size: 11pt; font-weight: bold; } .sym-f { color: #dc2626; font-family: 'Noto Sans', 'DejaVu Sans', sans-serif; font-size: 11pt; font-weight: bold; } .sym-s { color: #94a3b8; font-family: 'Noto Sans', 'DejaVu Sans', sans-serif; font-size: 10pt; }
    </style>
    """

    h = [css]
    h.append('<div class="proc-divider"><div class="proc-head">')
    h.append('<h2>LAPORAN PROSES / KEPUTUSAN DI BALIK LAYAR</h2>')
    h.append('<p>Detail teknis evaluasi setiap file melalui gerbang deteksi</p></div>')

    # ── GERBANG AKTIF ──
    h.append('<div class="proc-sec-title">GERBANG AKTIF</div>')
    h.append('<table class="gate-cfg"><tr><th style="width:70px;">Status</th><th>Gerbang</th><th>Batas / Threshold</th></tr>')
    for nama, aktif, batas in gerbang_rows:
        cls = 'gc-on' if aktif else 'gc-off'
        _dot = '<span class="g-pass">✓</span>' if aktif else '<span class="g-fail">✗</span>'
        txt = 'Aktif' if aktif else 'Mati'
        h.append(f'<tr><td class="{cls}">{_dot} {txt}</td><td><strong>{html.escape(nama)}</strong></td><td>{html.escape(batas)}</td></tr>')
    h.append('</table>')
    h.append('<div class="proc-info">Gerbang Mati tidak dievaluasi. Penanda strip abu-abu di tabel berarti tidak diuji (sudah gugur sebelumnya), bukan lolos.</div>')

    # ── JALUR 1 — EXACT (BLAKE3) ──
    h.append('<div class="jalur-title">JALUR 1 — EXACT (BLAKE3)</div>')
    h.append('<div class="proc-info">File identik byte-per-byte. Hash BLAKE3 sama = pasti duplikat 100%, tanpa gerbang.</div>')
    if exact_groups:
        for ei, (oid, dups) in enumerate(exact_groups, 1):
            o = _orig(oid)
            # BLAKE3 penuh (tidak dipotong) agar bisa diverifikasi/disalin utuh.
            b3s = o.get('blake3') or '-'
            h.append('<div class="ex-box"><div class="ex-head">')
            h.append(f'<div class="ex-title">EXACT #{ei} — FILE ASLI</div>')
            h.append(f'<div class="kv"><span class="k">Nama</span><span class="s">:</span>{_flink(o.get("name","Unknown"), oid)}</div>')
            h.append(f'<div class="kv"><span class="k">ID File</span><span class="s">:</span>{html.escape(str(oid))}</div>')
            h.append(f'<div class="kv"><span class="k">Ukuran</span><span class="s">:</span>{format_size(o.get("size_mb",0))}</div>')
            h.append(f'<div class="kv"><span class="k">BLAKE3</span><span class="s">:</span><span style="word-break:break-all;">{html.escape(b3s)}</span></div>')
            h.append('</div><div style="padding:10px 12px;">')
            h.append(f'<div style="font-weight:bold; margin-bottom:6px; font-size:8.5pt;">Duplikat ({len(dups)} file):</div>')
            for di, d in enumerate(dups, 1):
                h.append('<div style="background:#fef2f2; padding:6px 8px; border-radius:4px; margin-bottom:4px; font-size:8.5pt;">')
                h.append(f'<strong>{di}. {_flink(d.get("name","Unknown"), d.get("id"))}</strong><br>')
                h.append(f'<span style="color:#64748b; font-size:7.5pt;">ID: {html.escape(str(d.get("id")))} | {format_size(d.get("size_mb",0))}</span>')
                h.append('</div>')
            h.append('</div></div>')
    else:
        h.append('<div class="proc-info">Tidak ada duplikat EXACT (tidak ada file identik byte-per-byte).</div>')

    # ── JALUR 2 — VISUAL ──
    h.append('<div class="jalur-title">JALUR 2 — VISUAL</div>')
    h.append('<div class="proc-info">Alur gerbang: pHash → dHash H → dHash V → Aspect → Histogram → Color Grid → Edge → Blur → SSIM.</div>')
    if not visual_anchors:
        h.append('<div class="proc-info">Tidak ada kandidat visual yang lolos pintu pHash + dHash H/V.</div>')
    _kosong_marker_written = False
    # Blok visual pertama tanpa page-break (cegah halaman kosong di awal);
    # blok berikutnya selalu mulai di halaman baru.
    _first_visual_block = True
    for gi, anchor in enumerate(visual_anchors, 1):
        _marker_emitted = False
        if (not _kosong_marker_written) and anchor in _kosong_set:
            _kosong_marker_written = True
            _marker_emitted = True
            # Penanda transisi mulai halaman baru (kecuali blok pertama); grup
            # setelahnya tidak menambah break agar tidak ada halaman kosong.
            _mk_cls = 'jalur-title' if _first_visual_block else 'jalur-title proc-pagebreak'
            h.append(f'<div class="{_mk_cls}" style="background:#64748b;">'
                     'GRUP TANPA HASIL DUPLIKAT (bukan duplikat)</div>')
            h.append('<div class="proc-info">Kandidat di bawah ini dekat secara hash '
                     'tetapi dijegal gerbang verifikasi. Data analisis, bukan duplikat.</div>')
            _first_visual_block = False
        # Grup mulai di halaman baru, kecuali grup pertama atau grup tepat
        # setelah penanda 'tanpa hasil' (penanda sudah memicu halaman baru).
        _vi_break = '' if (_first_visual_block or _marker_emitted) else ' vi-break'
        _first_visual_block = False
        node = img_trace[anchor]
        cands = node.get('candidates') or []
        o = _orig(anchor)
        # Kumpulkan keluar per gerbang + skor lolos (selaras laporan PROSES TXT).
        keluar_by_gate: Dict[str, list] = defaultdict(list)
        lolos_ssim: Dict[str, str] = {}
        for g in node.get('gates') or []:
            nm = g.get('nama')
            if nm and nm.startswith('pHash+piksel'):
                nm = '__prefilter__'
            if nm == 'blur_diag':
                continue
            for item in g.get('keluar', []) or []:
                keluar_by_gate[nm].append(item)
            lv = g.get('lolos_nilai')
            if lv:
                lolos_ssim[lv[0]] = lv[1]
        lolos = set(node.get('lolos') or [])
        # Gerbang tempat tiap kandidat gugur (None = lolos semua).
        gugur_di: Dict[str, str] = {}
        gugur_detail: Dict[str, tuple] = {}
        for _gn in GATE_FLOW:
            for (cfid, alasan, nilai) in (keluar_by_gate.get(_gn) or []):
                if cfid not in gugur_di:
                    gugur_di[cfid] = _gn
                    gugur_detail[cfid] = (_gn, alasan, nilai)

        _lolos_n = sum(1 for c in cands if c in lolos)
        _gugur_n = len(cands) - _lolos_n

        h.append(f'<div class="vi-box{_vi_break}"><div class="vi-head">')
        h.append(f'<div class="vi-title">GRUP #{gi} — FILE ASLI</div>')
        h.append(f'<div class="kv"><span class="k">Nama</span><span class="s">:</span>{_flink(o.get("name", anchor), anchor)}</div>')
        h.append(f'<div class="kv"><span class="k">ID File</span><span class="s">:</span>{html.escape(str(anchor))}</div>')
        h.append(f'<div class="kv"><span class="k">Ukuran</span><span class="s">:</span>{format_size(o.get("size_mb",0))}</div>')
        h.append('</div><div style="padding:8px 10px;">')
        if cands:
            h.append('<table class="gate-tbl"><colgroup>'
                     '<col style="width:4%;"><col style="width:16%;">'
                     '<col style="width:7%;"><col style="width:8%;"><col style="width:8%;">'
                     '<col style="width:7%;"><col style="width:6%;"><col style="width:8%;">'
                     '<col style="width:6%;"><col style="width:6%;"><col style="width:6%;">'
                     '<col style="width:18%;"></colgroup><tr>')
            h.append('<th>No</th><th>Filename</th>')
            for _gn, lbl in GATE_COLS:
                h.append(f'<th>{lbl}</th>')
            h.append('<th>Hasil</th></tr>')
            for ci, cfid in enumerate(cands, 1):
                is_dup = cfid in lolos
                g_out = gugur_di.get(cfid)  # gerbang tempat gugur (None=lolos)
                # Indeks gerbang tempat gugur dalam GATE_FLOW (untuk — setelahnya).
                fail_idx = GATE_FLOW.index(g_out) if (g_out in GATE_FLOW) else None
                h.append(f'<tr><td>{ci}</td><td class="fn">{_flink(_name(cfid), cfid)}</td>')
                for _gn, _lbl in GATE_COLS:
                    gidx = GATE_FLOW.index(_gn)
                    if is_dup:
                        cell = '<span class="g-pass">✓</span>'
                    elif fail_idx is None:
                        cell = '<span class="g-pass">✓</span>'
                    elif gidx < fail_idx:
                        cell = '<span class="g-pass">✓</span>'
                    elif gidx == fail_idx:
                        cell = '<span class="g-fail">✗</span>'
                    else:
                        cell = '<span class="g-skip">—</span>'
                    h.append(f'<td>{cell}</td>')
                hcls = 'h-dup' if is_dup else 'h-no'
                htxt = 'DUPLIKAT' if is_dup else 'TIDAK'
                h.append(f'<td><span class="{hcls}">{htxt}</span></td></tr>')
            h.append('</table>')
        else:
            h.append('<div class="proc-info">Tidak ada kandidat yang lolos pintu pHash.</div>')
        h.append('</div>')
        h.append(f'<div class="ringkas"><strong>Ringkasan:</strong> Lolos sebagai duplikat: <strong>{_lolos_n}</strong> file | Tidak lolos: <strong>{_gugur_n}</strong> file</div>')

        # Penjelasan file yang gugur.
        gagal_list = [(gugur_detail[c][0], c, gugur_detail[c][1], gugur_detail[c][2])
                      for c in cands if (c not in lolos) and c in gugur_detail]
        if gagal_list:
            h.append('<div class="pj-box"><div class="pj-title">PENJELASAN FILE YANG TIDAK LOLOS</div>')
            for (gate_name, cfid, alasan, nilai) in gagal_list:
                tahap = GAGAL_LABEL.get(gate_name, gate_name or '?')
                h.append('<div class="pj-item">')
                h.append(f'<div class="pj-fn">{_flink(_name(cfid), cfid)}</div>')
                h.append(f'<div class="pj-kv"><span class="k">Gugur di</span><span class="s">:</span><strong>{html.escape(str(tahap))}</strong></div>')
                if nilai:
                    h.append(f'<div class="pj-kv"><span class="k">Nilai Detail</span><span class="s">:</span>{html.escape(str(nilai))}</div>')
                _db = DETAIL_BATAS.get(gate_name)
                if _db:
                    h.append(f'<div class="pj-kv"><span class="k">Detail Batas</span><span class="s">:</span>{html.escape(str(_db))}</div>')
                h.append(f'<div class="pj-kv"><span class="k">Lolos hingga</span><span class="s">:</span>{html.escape(_lolos_hingga(gate_name))}</div>')
                if alasan:
                    h.append(f'<div class="pj-kv"><span class="k">Alasan</span><span class="s">:</span>{html.escape(str(alasan))}</div>')
                h.append('</div>')
            h.append('</div>')
        h.append('</div>')

    # ── Cara baca ──
    h.append('<div class="cara"><h4>CARA MEMBACA TABEL</h4>')
    h.append('<p><strong>Keterangan simbol</strong><br>'
             '<span class="sym-p">✓</span> &nbsp; Lolos gerbang<br>'
             '<span class="sym-f">✗</span> &nbsp; Gugur pada gerbang ini<br>'
             '<span class="sym-s">—</span> &nbsp; Tidak diuji (telah gugur pada gerbang sebelumnya)</p>')
    h.append('<p>Klik nama berkas untuk membukanya di Google Drive.</p>')
    h.append('<p>Pemeriksaan dilakukan melalui dua jalur deteksi yang berbeda.</p>')
    h.append('<p><strong>JALUR 1 — EXACT (BLAKE3)</strong><br>'
             'Mendeteksi berkas yang identik byte demi byte. Nilai hash BLAKE3 yang '
             'sama menandakan duplikat mutlak (100%), sehingga tidak memerlukan '
             'pemeriksaan gerbang maupun ambang batas. Jalur ini tidak ditampilkan '
             'sebagai kolom pada tabel kandidat; hasilnya disajikan terpisah pada '
             'bagian “JALUR 1 — EXACT (BLAKE3)”.</p>')
    h.append('<p><strong>JALUR 2 — VISUAL</strong><br>'
             'Setiap gerbang berikut mewakili satu kolom pada tabel, disusun dari '
             'pemeriksaan paling ringan hingga paling berat.</p>')
    h.append('<table class="cara-tbl">'
             '<tr><th>Gerbang</th><th>Keterangan</th></tr>'
             '<tr><td>pHash, dHash H, dHash V</td><td>Pra-filter kemiripan struktur secara kasar dan cepat.</td></tr>'
             '<tr><td>Aspect</td><td>Rasio lebar terhadap tinggi; mendeteksi pemotongan, rotasi, dan perubahan kanvas.</td></tr>'
             '<tr><td>Hist</td><td>Distribusi warna global; mendeteksi perubahan warna menyeluruh.</td></tr>'
             '<tr><td>Color G</td><td>Komposisi warna lokal per blok; mendeteksi konversi hitam-putih, sepia, dan tint.</td></tr>'
             '<tr><td>Edge</td><td>Penambahan tepi baru; mendeteksi stiker, emoji, teks, dan watermark.</td></tr>'
             '<tr><td>Blur</td><td>Area yang dikaburkan; mendeteksi pengaburan pada sebagian atau seluruh gambar.</td></tr>'
             '<tr><td>SSIM</td><td>Kemiripan struktur piksel; berperan sebagai penentu akhir dan paling berat.</td></tr>'
             '</table>')
    h.append('<p>Pada Jalur Visual, sebuah berkas dinyatakan duplikat hanya apabila '
             'lolos seluruh gerbang. Kegagalan pada satu gerbang saja sudah cukup '
             'untuk menyatakannya bukan duplikat. Jalur Exact tidak melewati '
             'gerbang-gerbang ini.</p>')
    h.append('<p>Nilai terukur beserta batas tiap gerbang dapat dilihat pada bagian '
             '“PENJELASAN BERKAS YANG TIDAK LOLOS”.</p>')
    h.append('</div></div>')

    return ''.join(h)

# ════════════════════════════════════════════════════════════
# PEMENANG "RESOLUSI LEBIH TINGGI DARI ASLI" (satu per grup).
# ════════════════════════════════════════════════════════════
# Dalam satu grup duplikat, PALING BANYAK SATU file boleh diberi label
# "RESOLUSI LEBIH TINGGI DARI ASLI": yaitu duplikat yang metriknya PALING
# TINGGI dan MELAMPAUI file asli (anchor). Metrik komposit:
#   1) resolusi (width*height) -> jumlah piksel absolut, pembanding utama.
#   2) ukuran file (bytes)     -> tie-breaker saat resolusi SAMA (mis. dua foto
#                                 2048x2048: yang ukurannya lebih besar menang,
#                                 karena kompresi lebih ringan = kualitas lebih baik).
# Bila resolusi/ukuran duplikat tidak melampaui asli, TIDAK ada label.
# Fungsi ini HANYA memberi label informasi di laporan; DGV23 tetap read-only
# (tidak menghapus/memindahkan file).

def _res_metric(e: Dict) -> Tuple[int, int]:
    """Metrik komposit satu file: (jumlah_piksel, ukuran_bytes). Dipakai untuk
    membandingkan kualitas antar file duplikat. Nilai 0 bila tak tersedia."""
    e = e or {}
    try:
        w = int(e.get('width') or 0); h = int(e.get('height') or 0)
        pixels = w * h if (w > 0 and h > 0) else 0
    except (TypeError, ValueError):
        pixels = 0
    size_bytes = _safe_int_size(e.get('size_bytes'))
    if size_bytes <= 0:
        # Tie-breaker WAJIB terisi: bila size_bytes kosong/0, turunkan dari
        # size_mb (perkiraan). Tanpa ini, dua foto beresolusi SAMA (W×H sama)
        # yang size_bytes-nya kosong menghasilkan metrik SERI -> label
        # "RESOLUSI LEBIH TINGGI DARI ASLI" bisa bocor ke lebih dari satu file.
        try:
            smb = float(e.get('size_mb')) if e.get('size_mb') is not None else 0.0
        except (TypeError, ValueError):
            smb = 0.0
        if smb > 0:
            size_bytes = int(smb * 1024 * 1024)
    return (pixels, size_bytes)

def _pick_res_winner(orig: Dict, dlist: List[Dict]) -> Optional[str]:
    """Pilih id SATU duplikat yang berhak label "RESOLUSI LEBIH TINGGI DARI
    ASLI": metrik komposit (piksel, ukuran) TERTINGGI di antara duplikat DAN
    strictly melampaui metrik asli. Return id pemenang, atau None bila tak ada
    duplikat yang melampaui asli. Deterministik: bila metrik seri, id lebih
    kecil menang. Menjamin MAKSIMAL satu pemenang per grup."""
    orig_metric = _res_metric(orig)
    best_id = None
    best_metric = None
    for d in dlist or []:
        did = (d or {}).get('id')
        if not did:
            continue
        dm = _res_metric(d)
        # Harus MELAMPAUI asli (piksel lalu ukuran). Seri dengan asli -> tidak
        # layak label (bukan "lebih tinggi").
        if dm <= orig_metric:
            continue
        # Pilih pemenang tunggal secara deterministik. Metrik lebih tinggi
        # menang; bila metrik SERI (W×H sama DAN ukuran sama persis), id lebih
        # kecil menang. Guard best_id None di iterasi pertama agar tidak
        # membandingkan None (dijamin MAKSIMAL satu pemenang per grup).
        if best_metric is None:
            best_id, best_metric = did, dm
        elif dm > best_metric:
            best_id, best_metric = did, dm
        elif dm == best_metric and (best_id is None or did < best_id):
            best_id, best_metric = did, dm
    return best_id

def _group_match_type(dlist: List[Dict]) -> str:
    """Tipe sebuah grup duplikat: 'exact' bila anggotanya identik byte (BLAKE3),
    selain itu 'visual'. Anggota satu grup selalu seragam match_type-nya (grup
    exact dari BLAKE3 vs grup visual dari MIH), jadi cukup lihat elemen pertama."""
    return 'exact' if (dlist and dlist[0].get('match_type') == 'exact') else 'visual'

def _grouped_report_sections(image_dups: Dict, video_dups: Dict) -> List[Tuple[str, bool, list]]:
    """Susun bagian laporan dalam urutan yang diminta pengguna:
      1) Foto 100% EXACT
      2) Foto VISUAL MATCH
      3) Video 100% EXACT
      4) Video VISUAL MATCH (bila ada)

    Mengembalikan list (judul_section, is_video, [(orig_id, dlist), ...]).
    Section tanpa grup dilewati oleh pemanggil. Urutan grup di dalam tiap
    section mengikuti urutan asli bucket (insertion order dict).
    """
    def _split(bucket: Dict) -> Tuple[list, list]:
        exact, visual = [], []
        for oid, dlist in bucket.items():
            (exact if _group_match_type(dlist) == 'exact' else visual).append((oid, dlist))
        return exact, visual

    img_exact, img_visual = _split(image_dups)
    vid_exact, vid_visual = _split(video_dups)
    return [
        ("DUPLIKAT FOTO — 100% EXACT",   False, img_exact),
        ("DUPLIKAT FOTO — VISUAL MATCH", False, img_visual),
        ("DUPLIKAT VIDEO — 100% EXACT",  True,  vid_exact),
        ("DUPLIKAT VIDEO — VISUAL MATCH", True, vid_visual),
    ]

# ───────────────────── LAPORAN TXT & PDF ─────────────────────
def save_analysis_to_drive(folder_name: str, folder_id: str, storage: LMDBStorage, duplicate_result: Dict):
    paths = folder_paths(folder_id, folder_name)
    # Thumbnail bundle aktif, terpisah per folder.
    set_thumb_dir(paths['thumb'])
    # Bundle sudah unik per folder Drive, jadi laporan ditulis langsung di
    # 2_laporan tanpa subfolder bersarang. Nama berkas tetap deskriptif.
    subfolder_name = report_subdir_name(folder_name, folder_id)
    subfolder_path = paths['report']
    os.makedirs(subfolder_path, exist_ok=True)

    stats          = duplicate_result['stats']
    image_dups     = duplicate_result['image']
    video_dups     = duplicate_result['video']
    total_size_str = format_size(stats['total_size_mb'])
    wasted_str     = format_size(stats['wasted_size_mb'])

    # Ambil hanya record asli yang dibutuhkan laporan untuk menghemat RAM.
    def _orig(oid):
        return storage.get(oid) or {}

    # ── TXT (machine-readable untuk program remover) ──
    # Tiap baris 'key=value' dipisah ' | ', 'name=' selalu paling akhir agar
    # nama yang mengandung '|'/'('/spasi tidak merusak token. Newline dibuang.
    def _clean(v) -> str:
        s = '' if v is None else str(v)
        return s.replace('\r', ' ').replace('\n', ' ')

    def _dim_str(v) -> str:
        # Dimensi piksel sebagai string; kosong bila tak tersedia/tak valid.
        if v is None:
            return ''
        try:
            iv = int(v)
            return str(iv) if iv > 0 else ''
        except (TypeError, ValueError):
            return ''

    def _file_line(prefix: str, group: int, role: str, match_type: str, e: Dict,
                   res_higher: bool = False) -> str:
        match = 'exact' if match_type == 'exact' else 'visual'
        sz = e.get('size_bytes')
        # _safe_int_size agar tipe non-int (record lama) tidak melempar error.
        sz_str = '' if sz is None else str(_safe_int_size(sz))
        # Resolusi per file (w/h). Kosong bila dimensi tak tersedia. Ditaruh
        # sebelum 'name=' agar 'name' tetap token terakhir.
        w_str = _dim_str(e.get('width'))
        h_str = _dim_str(e.get('height'))
        # res_higher=1 menandai SATU pemenang grup: duplikat dengan resolusi
        # (lalu ukuran) tertinggi yang melampaui file asli. Hanya informasi.
        rh_str = '1' if res_higher else '0'
        return (
            f"{prefix}group={group} | id={_clean(e.get('id'))}"
            f" | parent={_clean(e.get('parent'))}"
            f" | match={match}"
            f" | b3={_clean(e.get('blake3'))}"
            f" | size={sz_str}"
            f" | w={w_str}"
            f" | h={h_str}"
            f" | role={role}"
            f" | res_higher={rh_str}"
            f" | name={_clean(e.get('name'))}\n"
        )

    # String integritas dihitung sekali, dipakai bersama TXT & PDF agar isi
    # kedua laporan selalu sama.
    _rep_rec = stats.get('reconcile') or {}
    _rep_lv_ok = ((not _rep_rec.get('listing_verify_ran'))
                  or (_rep_rec.get('listing_verify_known') and _rep_rec.get('listing_verify_discrepancy', 0) == 0))
    _rep_ok = (_rep_rec.get('balanced') and _rep_rec.get('scan_complete')
               and _rep_rec.get('pending', 0) == 0 and _rep_rec.get('failed', 0) == 0
               and not _rep_rec.get('drift_detected') and _rep_rec.get('drift_known', False)
               and _rep_lv_ok)
    _rep_integrity = ('TERVERIFIKASI' if _rep_ok
                      else 'BELUM TERVERIFIKASI (jalankan ulang)')
    _rep_rec_str = (
        f"{_rep_rec.get('indexed', 0)} terindeks = {_rep_rec.get('processed', 0)} diproses"
        f" + {_rep_rec.get('failed', 0)} gagal + {_rep_rec.get('pending', 0)} belum"
    )
    _rep_failed_folder_count = _rep_rec.get('failed_folder_count', 0)
    _rep_failed_folder_str = (
        f"{_rep_failed_folder_count} folder (jalankan ulang)" if _rep_failed_folder_count > 0
        else "tidak ada (semua folder berhasil di-scan)"
    )
    if _rep_rec.get('drift_detected'):
        _rep_drift_str = f"{_rep_rec.get('drift_count', 0)} perubahan terdeteksi (jalankan ulang)"
    elif not _rep_rec.get('drift_known', True):
        _rep_drift_str = "tidak dapat diverifikasi (jalankan ulang)"
    else:
        _rep_drift_str = "tidak ada (Drive stabil selama scan)"
    if not _rep_rec.get('listing_verify_ran'):
        _rep_listing_str = "tidak dijalankan"
    elif not _rep_rec.get('listing_verify_known'):
        _rep_listing_str = "tidak lengkap (jalankan ulang)"
    elif _rep_rec.get('listing_verify_discrepancy', 0) > 0:
        _rep_listing_str = f"{_rep_rec.get('listing_verify_discrepancy')} file terlewat (akan diproses ulang, jalankan ulang)"
    else:
        _rep_listing_str = "cocok (dua listing sepakat)"
    # Status scan ikut logika terminal: kondisional pada unscanned_media.
    _rep_unscanned = stats.get('unscanned_media', 0)
    _rep_status_scan_label = 'File belum ter-scan' if _rep_unscanned > 0 else 'Status Scan'
    _rep_status_scan_str = (
        f"{_rep_unscanned} file (jalankan ulang untuk menuntaskan)" if _rep_unscanned > 0
        else "semua file media ter-scan"
    )

    txt_path = os.path.join(subfolder_path, subfolder_name + ".txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            # Header meniru tata letak PDF: judul + nama tools + Folder | tanggal.
            _hdr_date = datetime.now().strftime("%d %b %Y, %H:%M:%S")
            f.write("=" * 50 + "\n")
            f.write("LAPORAN DETEKSI DUPLIKAT\n")
            f.write("DupliGuard Vision\n")
            f.write(f"Folder: {folder_name} | {_hdr_date}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'Folder':<22} : {folder_name}\n")
            f.write(f"{'Folder ID':<22} : {folder_id}\n")
            f.write(f"{'Total File':<22} : {stats['total_files']} file ({total_size_str})\n")
            f.write(f"{'Total Foto':<22} : {stats['total_images']} file\n")
            f.write(f"{'Total Video':<22} : {stats['total_videos']} file\n")
            f.write(f"{_rep_status_scan_label:<22} : {_rep_status_scan_str}\n")
            if _rep_rec:
                f.write("\n")
                f.write(f"{'Rekonsiliasi':<22} : {_rep_rec_str}\n")
                f.write(f"{'Folder gagal di-scan':<22} : {_rep_failed_folder_str}\n")
                f.write(f"{'Verifikasi listing':<22} : {_rep_listing_str}\n")
                f.write(f"{'Perubahan Drive':<22} : {_rep_drift_str}\n")
                f.write(f"{'Integritas Scan':<22} : {_rep_integrity}\n")
            f.write("\n")
            f.write(f"{'Total Duplikat':<22} : {stats['total_duplicates']} file\n")
            f.write(f"{'Duplikat Foto':<22} : {stats['image_duplicates']} file\n")
            f.write(f"{'Duplikat Video':<22} : {stats['video_duplicates']} file\n")
            f.write(f"{'Ruang Terbuang (Karna Duplikat)':<22} : {wasted_str}\n\n")
            _group_no = 0
            for label, _isvid, groups in _grouped_report_sections(image_dups, video_dups):
                if not groups: continue
                f.write("=" * 50 + "\n" + label + "\n" + "=" * 50 + "\n\n")
                for oid, dlist in groups:
                    _group_no += 1
                    orig = _orig(oid)
                    orig_entry = {
                        'id': oid,
                        'parent': orig.get('parent'),
                        'blake3': orig.get('blake3'),
                        'size_bytes': orig.get('size_bytes'),
                        'width': orig.get('width'),
                        'height': orig.get('height'),
                        'name': orig.get('name', 'Unknown'),
                    }
                    # Label match baris ASLI mengikuti TIPE GRUP, bukan selalu
                    # 'exact'. Pada grup visual, anchor & duplikat TIDAK identik
                    # byte (b3 berbeda); menulis match=exact di sini menyesatkan
                    # program remover yang memverifikasi ulang lewat b3. Anggota
                    # satu grup selalu seragam match_type-nya (grup exact dari
                    # BLAKE3 vs grup visual dari MIH), jadi ambil dari elemen
                    # pertama dlist. role=ASLI tetap menjadi blacklist remover.
                    _grp_match = dlist[0]['match_type'] if dlist else 'exact'
                    # SATU pemenang "resolusi lebih tinggi dari asli" per grup.
                    # Berlaku untuk foto maupun video; video kini menyimpan
                    # width/height (resolusi) sehingga _pick_res_winner dapat
                    # membandingkannya. Bila resolusi tak tersedia, _res_metric
                    # jatuh ke ukuran file (fail-safe), tidak menghasilkan label
                    # keliru.
                    _res_winner = _pick_res_winner(orig_entry, dlist)
                    f.write(_file_line("[ASLI] ", _group_no, "ASLI", _grp_match, orig_entry))
                    for d in dlist:
                        f.write(_file_line("  [DUP] ", _group_no, "DUP", d['match_type'], d,
                                           res_higher=(_res_winner is not None and d.get('id') == _res_winner)))
                    f.write("\n")
            if not image_dups and not video_dups:
                f.write("Tidak ada duplikat ditemukan.\n")
        print(f"    {Colors.SUCCESS}✓ Laporan TXT{Colors.RESET}")
    except Exception as e:
        print(f"    {Colors.WARNING}✗ Gagal buat TXT: {e}{Colors.RESET}")

    # ── PDF ──
    clear_thumb_mem_cache()

    pdf_path = os.path.join(subfolder_path, f"{subfolder_name}.pdf")
    # Pakai string integritas yang sudah dihitung di awal (alias) agar HTML PDF
    # di bawah tidak perlu diubah dan isi TXT & PDF selalu sama.
    _pdf_rec = _rep_rec
    _pdf_integrity = _rep_integrity
    _pdf_rec_str = _rep_rec_str
    _pdf_failed_folder_str = _rep_failed_folder_str
    _pdf_drift_str = _rep_drift_str
    _pdf_listing_str = _rep_listing_str
    try:
        html_content = f"""<!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <title>Laporan Deteksi Duplikat - {html.escape(str(folder_name))}</title>
            <style>
                @page {{ size: A4; margin: 20mm; }}
                body {{ font-family: 'DejaVu Sans', 'Helvetica', 'Arial', sans-serif; font-size: 10pt; color: #1e293b; line-height: 1.5; }}
                a {{ color: #0ea5e9; text-decoration: underline; }}
                .header {{ text-align: center; margin-bottom: 20px; border-bottom: 2px solid #0f172a; padding-bottom: 10px; }}
                .header h1 {{ font-size: 18pt; margin: 0; }}
                .stats-table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
                .stats-table td {{ padding: 8px 12px; border: 1px solid #cbd5e1; background: #f8fafc; width: 25%; }}
                .stats-table td.val {{ font-weight: bold; font-size: 12pt; background: #ffffff; text-align: center; }}
                .stats-table td.val.red {{ color: #dc2626; }}
                .section-title {{ font-size: 12pt; font-weight: bold; border-bottom: 1px solid #cbd5e1; margin: 20px 0 15px 0; }}
                .group-box {{ border: 2px solid #e2e8f0; border-radius: 12px; margin-bottom: 30px; }}
                .group-header {{ background: #ecfdf5; padding: 20px; border-bottom: 3px solid #a7f3d0; text-align: center; }}
                .group-title {{ font-size: 11pt; font-weight: bold; color: #059669; }}
                .img-orig {{ width: 560px; height: 560px; object-fit: contain; border: 3px solid #10b981; border-radius: 10px; display: block; margin: 0 auto 20px auto; }}
                .img-dup {{ width: 560px; height: 560px; object-fit: contain; border: 3px solid #f97316; border-radius: 10px; display: block; margin: 0 auto 15px auto; }}
                .info-text {{ font-size: 9.5pt; color: #475569; margin: 4px 0; }}
                .dup-row {{ margin-bottom: 25px; padding: 20px; background: #ffffff; border-radius: 10px; border: 2px solid #e2e8f0; text-align: center; }}
                .badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 9pt; font-weight: bold; }}
                .badge-exact {{ background: #dc2626; color: white; }}
                .badge-visual {{ background: #f97316; color: white; }}
                .badge-res {{ background: #7c3aed; color: white; }}
                .res-banner {{ background: #f5f3ff; border: 1px solid #ddd6fe; color: #5b21b6; font-size: 9pt; font-weight: bold; padding: 8px 12px; border-radius: 8px; margin: 0 auto 12px auto; max-width: 560px; }}
                .footer {{ margin-top: 30px; text-align: center; font-size: 8pt; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LAPORAN DETEKSI DUPLIKAT</h1>
                <p><strong>DupliGuard Vision</strong></p>
                <p>Folder: <strong>{html.escape(str(folder_name))}</strong> | {datetime.now().strftime("%d %b %Y, %H:%M:%S")}</p>
            </div>
            <table class="stats-table">
                <tr><td><strong>Total File</strong></td><td class="val">{stats['total_files']} file ({total_size_str})</td></tr>
                <tr><td><strong>Total Foto</strong></td><td class="val">{stats['total_images']} file</td></tr>
                <tr><td><strong>Total Video</strong></td><td class="val">{stats['total_videos']} file</td></tr>
                <tr><td><strong>{_rep_status_scan_label}</strong></td><td class="val">{_rep_status_scan_str}</td></tr>
                <tr><td><strong>Rekonsiliasi</strong></td><td class="val">{_pdf_rec_str}</td></tr>
                <tr><td><strong>Folder gagal di-scan</strong></td><td class="val">{_pdf_failed_folder_str}</td></tr>
                <tr><td><strong>Verifikasi listing</strong></td><td class="val">{_pdf_listing_str}</td></tr>
                <tr><td><strong>Perubahan Drive</strong></td><td class="val">{_pdf_drift_str}</td></tr>
                <tr><td><strong>Integritas Scan</strong></td><td class="val">{_pdf_integrity}</td></tr>
                <tr><td><strong>Total Duplikat</strong></td><td class="val red">{stats['total_duplicates']} file</td></tr>
                <tr><td><strong>Duplikat Foto</strong></td><td class="val red">{stats['image_duplicates']} file</td></tr>
                <tr><td><strong>Duplikat Video</strong></td><td class="val red">{stats['video_duplicates']} file</td></tr>
                <tr><td><strong>Ruang Terbuang (Karna Duplikat)</strong></td><td class="val red">{wasted_str}</td></tr>
            </table>
        """

        # Helper resolusi: format WxH dan jumlah piksel untuk perbandingan.
        def _res_px(e):
            try:
                w = int(e.get('width')); h = int(e.get('height'))
                if w > 0 and h > 0:
                    return w, h, w * h
            except Exception:
                pass
            return None

        def _res_str(e):
            r = _res_px(e)
            return f"{r[0]} x {r[1]} px" if r else "tidak diketahui"

        for title, isvid, groups in _grouped_report_sections(image_dups, video_dups):
            if not groups: continue
            html_content += f'<div class="section-title">{title}</div>'
            # Tanda resolusi ditampilkan untuk grup VISUAL (foto maupun video).
            # Video kini menyimpan width/height sehingga resolusi dapat dibandingkan.
            # Grup EXACT dilewati karena anggotanya identik byte (resolusi sama).
            _show_res = 'VISUAL' in title.upper()
            for gi, (oid, dlist) in enumerate(groups, 1):
                orig  = _orig(oid)
                ftype = 'video' if isvid else 'image'
                # Banner/badge peringatan bila ada duplikat yang RESOLUSINYA
                # BENAR-BENAR LEBIH TINGGI dari asli.
                #
                # Karena kandidat sudah lolos seluruh gerbang (pHash..SSIM),
                # foto dianggap SAMA; penentuan 'lebih tinggi' murni MATEMATIS,
                # tanpa model/sensor. Metrik per file:
                #   - jumlah piksel (width*height) bila kedua dimensi valid;
                #   - selain itu jatuh ke size_bytes (ukuran file), yang SELALU
                #     tersimpan (tidak bergantung rehash foto). Ini menutup bug
                #     di mana width/height anchor kosong (mis. foto Q25 kecil
                #     yang gagal decode dimensi) membuat seluruh blok label
                #     terlewati walau duplikat jelas lebih besar.
                #
                # 'ASLI' (anchor) TETAP file tertua (fundamental tools, tidak
                # diubah). Label ini murni WARNING opsional di laporan: menandai
                # duplikat yang resolusinya lebih tinggi dari asli, agar pengguna
                # sadar ada versi lebih besar. Aturan label:
                #  - Fail-safe: bila metrik anchor tak tersedia (_orig_metric
                #    None), tidak ada label (deterministik antar run).
                #  - Labeli SEMUA duplikat yang metrik-nya > metrik anchor (boleh
                #    lebih dari satu). Salinan berpiksel/ukuran SAMA dengan anchor
                #    (metrik == anchor) tidak pernah kena karena syaratnya '>'.
                # SATU pemenang tunggal per grup untuk label "RESOLUSI LEBIH
                # TINGGI DARI ASLI". Bukan semua duplikat yang > anchor: hanya
                # yang metrik tertingginya melampaui asli. Metrik komposit:
                # resolusi (w*h) dulu, lalu ukuran file (bytes) sebagai
                # tie-breaker (dua foto 2048x2048 -> yang ukurannya lebih besar
                # menang). Fungsi _pick_res_winner menjamin maksimal 1 pemenang.
                _res_winner_id = None
                _has_higher_dup = False
                if _show_res:
                    # Pilih SATU pemenang: duplikat dengan metrik komposit
                    # (resolusi, ukuran) tertinggi yang MELAMPAUI asli. Bila seri
                    # dengan asli atau tak ada yang lebih tinggi -> tidak ada label.
                    _res_winner_id = _pick_res_winner(orig, dlist)
                    if _res_winner_id:
                        _has_higher_dup = True
                # Rujuk thumbnail via file:// (anti-OOM). Fallback ke placeholder
                # base64 bila file tak tersedia.
                ob64  = get_image_file_uri(oid, orig.get('name', ''), ftype, orig.get('size_mb', 0)) or _PLACEHOLDER_B64
                olink = f"https://drive.google.com/file/d/{oid}/view"
                html_content += f"""
                <div class="group-box">
                    <div class="group-header">
                        <img src="{ob64}" class="img-orig">
                        <div class="group-title">Grup #{gi} - {'VIDEO' if isvid else 'FOTO'} ASLI</div>
                        <div class="info-text"><strong>Nama:</strong> <a href="{olink}">{html.escape(str(orig.get('name', 'Unknown')))}</a></div>
                        <div class="info-text"><strong>ID File:</strong> <a href="{olink}">{html.escape(str(oid))}</a></div>
                        <div class="info-text"><strong>Tanggal:</strong> {format_timestamp(orig.get('createdTime', ''))}</div>
                        <div class="info-text"><strong>Owner:</strong> {html.escape(str(orig.get('ownerEmail','N/A')))}</div>
                        {f'<div class="info-text"><strong>Resolusi:</strong> {_res_str(orig)}</div>' if _show_res else ''}
                        <div class="info-text"><strong>Ukuran:</strong> {format_size(orig.get('size_mb',0))}</div>
                        {'<div class="res-banner">PERHATIAN: ada duplikat di grup ini yang RESOLUSINYA LEBIH TINGGI dari file asli. Pertimbangkan file mana yang ingin dipertahankan.</div>' if _has_higher_dup else ''}
                    </div>
                    <div style="padding:15px;"><div style="font-weight:bold; margin-bottom:10px;">Duplikat ({len(dlist)} file)</div>
                """
                for d in dlist:
                    db64  = get_image_file_uri(d['id'], d['name'], ftype, d.get('size_mb', 0)) or _PLACEHOLDER_B64
                    dlink = f"https://drive.google.com/file/d/{d['id']}/view"
                    badge = "badge-exact" if d['match_type'] == 'exact' else "badge-visual"
                    btext = "100% EXACT" if d['match_type'] == 'exact' else "VISUAL MATCH"
                    # Badge peringatan HANYA untuk SATU pemenang grup (resolusi
                    # -> ukuran tertinggi yang melampaui asli). Maksimal 1 per grup.
                    _res_higher = bool(_show_res and _res_winner_id and d.get('id') == _res_winner_id)
                    _res_info = (f'<div class="info-text"><strong>Resolusi:</strong> {_res_str(d)}</div>'
                                 if _show_res else '')
                    _res_badge = ('<span class="badge badge-res">RESOLUSI LEBIH TINGGI DARI ASLI</span>'
                                  if _res_higher else '')
                    html_content += f"""
                    <div class="dup-row">
                        <img src="{db64}" class="img-dup">
                        <div class="info-text"><strong>Nama:</strong> <a href="{dlink}">{html.escape(str(d['name']))}</a></div>
                        <div class="info-text"><strong>ID File:</strong> <a href="{dlink}">{html.escape(str(d['id']))}</a></div>
                        <div class="info-text"><strong>Tanggal:</strong> {format_timestamp(d['createdTime'])}</div>
                        <div class="info-text"><strong>Owner:</strong> {html.escape(str(d.get('ownerEmail','N/A')))}</div>
                        {_res_info}
                        <div class="info-text"><strong>Ukuran:</strong> {format_size(d.get('size_mb',0))}</div>
                        <div><span class="badge {badge}">{btext}</span> {_res_badge}</div>
                    </div>
                    """
                html_content += "</div></div>"

        if not image_dups and not video_dups:
            html_content += '<div style="text-align:center; padding:40px;"><h3>TIDAK ADA DUPLIKAT DITEMUKAN</h3></div>'
        # ── Halaman lanjutan: LAPORAN PROSES (di-inject sebelum footer) ──
        # Hanya menambah section; aman bila process_trace kosong / PROCESS_REPORT
        # off (_build_process_report_html mengembalikan '').
        try:
            html_content += _build_process_report_html(folder_name, storage, duplicate_result)
        except Exception as _e_proc:
            _log('error', f"build process report HTML gagal: {traceback.format_exc()}")
        html_content += '<div class="footer">BLAKE3 (Exact) + pHash + dHash H/V + Aspect Ratio + Histogram + color_grid + Edge + Blur + SSIM (Visual)</div></body></html>'

        WeasyHTML(string=html_content).write_pdf(pdf_path)
        print(f"    {Colors.SUCCESS}✓ Laporan PDF{Colors.RESET}")
    except Exception as e:
        print(f"    {Colors.WARNING}✗ Gagal buat PDF: {e}{Colors.RESET}")
        _log('error', f"PDF generation failed: {traceback.format_exc()}")

# ───────────────────── HELPERS ─────────────────────
# Lebar baku garis batas output terminal (dipakai semua separator).
SEPARATOR_WIDTH = 70

def print_separator(width: int = SEPARATOR_WIDTH, char: str = '═', color: str = Colors.BORDER):
    """Cetak satu garis batas penuh dengan gaya seragam (default '═')."""
    print(f"{color}{char * width}{Colors.RESET}")

def print_header(title: str, width: int = SEPARATOR_WIDTH, char: str = '═',
                 color: str = Colors.BORDER, title_color: str = Colors.HEADER):
    """Cetak header bergaya box: judul di-center diapit dua garis batas.

        ══════════════════════════════════════════════════════════════════════
                                 HASIL ANALISIS DUPLIKAT
        ══════════════════════════════════════════════════════════════════════
    """
    print_separator(width, char, color)
    print(f"{title_color}{Colors.BOLD}{title.upper().center(width)}{Colors.RESET}")
    print_separator(width, char, color)

def print_row_success(label: str, value: str, w: int = 22):
    print(f"{Colors.SUCCESS}{label:<{w}}{Colors.RESET} : {Colors.VALUE_SUCCESS}{value}{Colors.RESET}")

def _colorize_value(value: str) -> str:
    """Warnai value baris ringkasan: teks di luar kurung abu-abu (GRAY_VALUE),
    di dalam kurung kuning tua (YELLOW_DIM). Mendukung kurung bersarang/ganda;
    warna disesuaikan kembali berdasarkan depth setelah tiap kurung tutup."""
    out = []
    depth = 0
    for ch in value:
        if ch == '(':
            # Kurung buka: tetap abu-abu, lalu isi di dalamnya kuning tua.
            out.append(f"{Colors.GRAY_VALUE}({Colors.YELLOW_DIM}")
            depth += 1
        elif ch == ')' and depth > 0:
            depth -= 1
            if depth > 0:
                # Masih di dalam kurung luar: kembali ke kuning tua setelah ')'.
                out.append(f"{Colors.GRAY_VALUE}){Colors.YELLOW_DIM}")
            else:
                # Kembali ke level luar kurung: kembali ke abu-abu.
                out.append(f"{Colors.GRAY_VALUE})")
        else:
            out.append(ch)
    return f"{Colors.GRAY_VALUE}{''.join(out)}"

def print_row_themed(key_color: str, label: str, value: str, w: int = 22):
    """Cetak satu baris ringkasan dengan warna key per-kategori dan value
    yang otomatis dipisah warna luar/dalam kurung (lihat _colorize_value)."""
    print(f"{key_color}{label:<{w}}{Colors.RESET} : {_colorize_value(value)}{Colors.RESET}")

# ───────────────────── SESSION LOCK (deteksi sesi Colab ganda) ─────────────────────
# Ambang (detik) lock dianggap basi. Bila heartbeat lebih lama dari ini, sesi
# pemilik dianggap mati dan lock boleh diambil-alih. Dibuat longgar agar jeda
# I/O Drive wajar tidak salah dianggap basi.
SESSION_LOCK_STALE_SEC = 600
# Identitas sesi proses ini (unik per runtime Colab).
_SESSION_ID = f"{os.uname().nodename if hasattr(os, 'uname') else 'host'}:{os.getpid()}:{base64.b16encode(os.urandom(4)).decode()}"

def _session_lock_path(paths: Dict[str, str]) -> str:
    return os.path.join(paths['bundle'], ".session.lock")

def _read_session_lock(path: str) -> Optional[Dict]:
    txt = _read_text_file(path)
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None

def _acquire_session_lock(paths: Dict[str, str]) -> Tuple[Optional[str], bool]:
    """Ambil lock-file sesi di Drive. Return (lock_path, prompt_shown): lock_path
    None bila pengguna membatalkan karena sesi lain aktif; prompt_shown True bila
    prompt peringatan sempat tampil. Lock ringan berbasis heartbeat untuk
    mendeteksi dua sesi Colab pada folder sama yang akan saling menimpa .mdb."""
    lock_path = _session_lock_path(paths)
    existing = _read_session_lock(lock_path)
    now = time.time()
    prompt_shown = False
    if existing and existing.get('session') != _SESSION_ID:
        hb = float(existing.get('heartbeat', 0) or 0)
        # abs() agar jam sistem yang mundur (NTP/VM drift) tidak membuat selisih
        # negatif yang selalu < STALE_SEC (lock basi terjebak dianggap aktif).
        age = abs(now - hb)
        if age < SESSION_LOCK_STALE_SEC:
            prompt_shown = True
            _aktif_menit = max(1, int(age // 60))
            print(f"{Colors.WARNING}Terdeteksi jejak runtime Colab lain pada folder ini "
                  f"({_aktif_menit} menit lalu).{Colors.RESET}")
            print(f"{Colors.WARNING}Pastikan tidak ada runtime lain yang masih scan folder ini.{Colors.RESET}")
            print(f"{Colors.SUCCESS}y = sudah pasti tidak ada (lanjut){Colors.RESET}")
            print(f"{Colors.SUCCESS}n = masih ada / belum yakin (batal){Colors.RESET}")
            print()
            while True:
                print(f"{Colors.SUCCESS}{Colors.BOLD}Lanjut? (y/n){Colors.RESET} : ", end="")
                try:
                    ch = input().strip().lower()
                except EOFError:
                    # stdin ditutup (lingkungan non-interaktif / headless):
                    # asumsikan 'n' (batal) agar tidak crash dan tidak
                    # melanjutkan tanpa konfirmasi pengguna.
                    print()
                    ch = 'n'
                if ch in ('y', 'n'):
                    break
            if ch == 'n':
                return None, prompt_shown
        else:
            _log('info', f"session lock basi diambil-alih (heartbeat {int(age)}s lalu)")
    try:
        _write_text_atomic(lock_path, json.dumps({'session': _SESSION_ID, 'heartbeat': now}))
    except Exception as e:
        _log('debug', f"tulis session lock gagal: {e}")
    return lock_path, prompt_shown

def _touch_session_lock(lock_path: Optional[str]) -> bool:
    """Perbarui heartbeat lock bila masih milik sesi ini. Return True bila
    diperbarui, False bila lock sudah diambil-alih sesi lain / tidak ada.
    Tidak pernah menimpa lock milik sesi lain."""
    if not lock_path:
        return False
    try:
        cur = _read_session_lock(lock_path)
        # Bila lock hilang (mis. terhapus), tulis ulang sebagai milik kita.
        # Bila milik sesi lain, JANGAN sentuh.
        if cur and cur.get('session') != _SESSION_ID:
            return False
        _write_text_atomic(lock_path, json.dumps({'session': _SESSION_ID, 'heartbeat': time.time()}))
        return True
    except Exception as e:
        _log('debug', f"refresh session heartbeat gagal: {e}")
        return False

# Interval refresh heartbeat (detik). Dibuat jauh di bawah SESSION_LOCK_STALE_SEC
# (sepertiga) agar lock sesi yang masih hidup tidak pernah terlanjur dianggap
# basi oleh sesi lain di antara dua refresh.
SESSION_LOCK_HEARTBEAT_SEC = 200

def _start_session_heartbeat(lock_path: Optional[str]) -> Tuple[Optional[threading.Thread], Optional[threading.Event]]:
    """Mulai thread daemon yang memperbarui heartbeat lock secara berkala selama
    analisis berlangsung. Return (thread, stop_event); pemanggil wajib set
    stop_event lalu join saat selesai. Tanpa refresh berkala, analisis yang
    lebih lama dari SESSION_LOCK_STALE_SEC membuat lock dianggap basi dan dapat
    diambil-alih sesi lain -> dua sesi menimpa .mdb di Drive (data hilang)."""
    if not lock_path:
        return None, None
    stop = threading.Event()
    def _loop():
        while not stop.wait(SESSION_LOCK_HEARTBEAT_SEC):
            if not _touch_session_lock(lock_path):
                # Lock sudah bukan milik kita: berhenti memperbarui.
                break
    t = threading.Thread(target=_loop, daemon=True, name="dgv-session-heartbeat")
    t.start()
    return t, stop

def _release_session_lock(lock_path: Optional[str]):
    """Lepas lock hanya bila masih milik sesi ini (jangan hapus lock sesi lain
    yang mungkin mengambil-alih setelah kita)."""
    if not lock_path:
        return
    try:
        cur = _read_session_lock(lock_path)
        if cur and cur.get('session') == _SESSION_ID and os.path.exists(lock_path):
            os.unlink(lock_path)
    except Exception as e:
        _log('debug', f"lepas session lock gagal: {e}")

# ───────────────────── ANALISIS FOLDER ─────────────────────
def analyze_folder(folder_id: str, folder_name: str):
    # Semua data folder terkumpul di satu bundle DupliGuard Vision/<Nama> (<id>)/.
    paths = ensure_bundle(folder_id, folder_name)
    set_thumb_dir(paths['thumb'])

    print_header("ANALISIS FOLDER")

    # Bersihkan scratch sisa crash run sebelumnya sebelum env LMDB dibuka. Pada
    # titik ini belum ada tulis berjalan, jadi scratch aman dibuang tanpa syarat
    # umur. Aman: file final .mdb/.gen tidak pernah disentuh.
    try:
        _cleanup_scratch_now(paths['database'], paths['cache'])
    except Exception as _e_scratch0:
        _log('debug', f"cleanup scratch awal analisis gagal: {_e_scratch0}")

    # Reset circuit breaker: state-nya global modul dan bisa tertinggal terbuka
    # dari folder sebelumnya pada sesi yang sama. Tiap folder mulai dengan
    # sirkuit tertutup; bila kuota masih habis, breaker terbuka lagi otomatis.
    _circuit_breaker.reset()

    # Deteksi sesi Colab ganda pada folder yang sama sebelum membuka env LMDB.
    _session_lock, _prompt_shown = _acquire_session_lock(paths)
    if _session_lock is None:
        return

    storage       = LMDBStorage(folder_id, env_path=paths['hash_env'])
    file_manifest = PersistentFileManifest(folder_id, env_path=paths['manifest_env'])
    journal       = FolderJournal(folder_id, env_path=paths['journal_env'])
    # Jaga heartbeat lock tetap segar selama analisis (bisa berjam-jam untuk
    # folder besar) agar tidak terlanjur dianggap basi & diambil-alih sesi lain.
    _hb_thread, _hb_stop = _start_session_heartbeat(_session_lock)
    try:
        _analyze_folder_body(folder_id, folder_name, paths, storage, file_manifest, journal,
                             prompt_shown=_prompt_shown)
    finally:
        # Hentikan thread heartbeat sebelum melepas lock, agar tidak ada refresh
        # yang menulis ulang lock setelah dilepas.
        if _hb_stop is not None:
            _hb_stop.set()
        if _hb_thread is not None:
            try: _hb_thread.join(timeout=5)
            except Exception: pass
        # Bersihkan thumbnail lokal folder ini walau terjadi exception, agar
        # tidak menumpuk di /content antar-folder (risiko ENOSPC).
        _cleanup_local_thumb_dir(folder_id)
        # Lepas lock sesi (bila masih milik kita) agar sesi berikutnya tidak
        # salah mengira folder masih aktif.
        _release_session_lock(_session_lock)
        # Tutup seluruh env LMDB walau terjadi exception, agar file handle dan
        # .mdb-lock tidak bocor/terkunci antar-folder dalam satu sesi Colab.
        for _res in (storage, file_manifest, journal):
            _env = getattr(_res, 'env', None)
            if _env is None:
                continue
            # Sync dulu (force=True) agar meta+data durable sebelum handle ditutup;
            # di Drive FUSE tanpa flush final data terakhir bisa hilang.
            try: _env.sync(force=True)
            except Exception as _e_sync:
                _log('debug', f"env sync gagal: {_e_sync}")
            try: _env.close()
            except Exception as _e_close:
                _log('debug', f"env close gagal: {_e_close}")
        # Semua env LMDB folder kini ditutup: scratch (.tmp/.snap/.gen.tmp +
        # lock yatim) sudah pasti tidak diperlukan, hapus permanen sekarang.
        # Aman: file final .mdb/.gen tidak pernah disentuh.
        try:
            _cleanup_scratch_now(paths['database'], paths['cache'])
        except Exception as _e_scratch:
            _log('debug', f"cleanup scratch akhir analisis gagal: {_e_scratch}")

def _analyze_folder_body(folder_id: str, folder_name: str, paths: Dict[str, str],
                         storage: 'LMDBStorage', file_manifest: 'PersistentFileManifest',
                         journal: 'FolderJournal', prompt_shown: bool = False):
    manifest_stats = file_manifest.get_stats()
    # Changes API butuh driveId untuk Shared Drive (None untuk My Drive).
    drive_id = _folder_drive_id(folder_id)
    resumed = False
    incremental_done = False
    # Pesan status scan (Resume / Perubahan) disimpan dulu, dicetak SETELAH
    # blok MEMPROSES agar MEMPROSES tampil di atas baris ini.
    _scan_status_msg: Optional[str] = None
    if manifest_stats['scan_complete'] and manifest_stats['pending'] > 0:
        # Scan sudah lengkap tapi proses belum selesai: resume tanpa scan ulang.
        _scan_status_msg = f"  {Colors.DIM}↳ Resume: {manifest_stats['processed']}/{manifest_stats['total']} sudah diproses{Colors.RESET}"
        current_ids = file_manifest.get_all_file_ids()
        resumed = True
    elif manifest_stats['scan_complete'] and journal.get_changes_token():
        # Scan sebelumnya tuntas dan ada token: coba incremental (hanya delta).
        ok_inc, _added, _removed = incremental_scan_via_changes(folder_id, journal, file_manifest, drive_id)
        if ok_inc:
            _scan_status_msg = f"  {Colors.DIM}↳ Perubahan: +{_added} / -{_removed} sejak scan terakhir{Colors.RESET}"
            current_ids = file_manifest.get_all_file_ids()
            incremental_done = True
        else:
            # Token invalid/expired: fallback ke full scan.
            file_manifest.invalidate_scan()
            # Full scan baru (bukan resume): reset visited/page_tokens basi.
            if not journal.get_pending():
                journal.clear_scan_state()
            start_tok = _drive_start_page_token(drive_id)
            resume_mid_scan = bool(journal.get_pending())
            all_files, scan_manifest, failed_folders = scan_folder_recursively(folder_id, journal, file_manifest)
            current_ids = {f['id'] for f in all_files}
            actual_size = sum(_safe_int_size(f.get('size')) / (1024 * 1024) for f in all_files)
            scan_valid, scan_reason = scan_manifest.validate(len(all_files), actual_size)
            if not scan_valid:
                _log('error', scan_reason)
            # Saat resume mid-scan, all_files hanya memuat folder yang di-scan
            # pada run ini. Menghapus berdasarkan current_ids saja akan
            # membuang file dari folder yang sudah di-scan sebelumnya. Hanya
            # pangkas file terhapus pada full scan tuntas (bukan resume).
            if not resume_mid_scan:
                file_manifest.remove_deleted_files(current_ids)
            # Hanya tandai scan tuntas bila valid dan tidak ada folder gagal.
            # Bila ditandai tuntas padahal ada folder gagal, run berikutnya
            # beralih ke incremental scan dan folder gagal tidak akan pernah
            # di-scan ulang -> file di subtree itu hilang permanen. Changes
            # token juga hanya disimpan saat scan benar-benar lengkap, agar
            # incremental tidak dimulai dari basis yang tidak lengkap.
            scan_done = scan_valid and not failed_folders
            if scan_done:
                file_manifest.mark_scan_complete(failed_folders)
                if start_tok: journal.set_changes_token(start_tok)
            else:
                _log('warning', "scan belum tuntas (invalid/ada folder gagal) -> full scan diulang run berikutnya")
            # Integritas pasca-scan: drift + (opsional) double-listing.
            _post_scan_integrity(folder_id, journal, file_manifest, drive_id, start_tok)
            if failed_folders:
                print(f"  {Colors.WARNING}⚠ {len(failed_folders)} folder gagal di-scan setelah {SCAN_DEFER_PASSES} pass — daftar di log{Colors.RESET}")
    else:
        # Full scan, atau resume scan lewat journal pending queue.
        # startPageToken diambil sebelum scan agar perubahan selama scan
        # tertangkap di run berikutnya.
        file_manifest.invalidate_scan()
        # Full scan baru (bukan resume mid-scan): reset visited/page_tokens basi.
        resume_mid_scan = bool(journal.get_pending())
        if not resume_mid_scan:
            journal.clear_scan_state()
        start_tok = _drive_start_page_token(drive_id)
        all_files, scan_manifest, failed_folders = scan_folder_recursively(folder_id, journal, file_manifest)
        current_ids = {f['id'] for f in all_files}

        actual_size = sum(_safe_int_size(f.get('size')) / (1024 * 1024) for f in all_files)
        scan_valid, scan_reason = scan_manifest.validate(len(all_files), actual_size)
        if not scan_valid:
            _log('error', scan_reason)

        # Saat resume mid-scan, all_files tidak lengkap; jangan pangkas
        # berdasarkan current_ids agar file folder yang sudah di-scan aman.
        if not resume_mid_scan:
            file_manifest.remove_deleted_files(current_ids)
        # Hanya tandai scan tuntas bila valid dan tidak ada folder gagal
        # (lihat penjelasan di jalur fallback). Mencegah folder gagal terlewat
        # permanen akibat beralih ke incremental scan terlalu dini.
        scan_done = scan_valid and not failed_folders
        if scan_done:
            file_manifest.mark_scan_complete(failed_folders)
            if start_tok: journal.set_changes_token(start_tok)
        else:
            _log('warning', "scan belum tuntas (invalid/ada folder gagal) -> full scan diulang run berikutnya")

        # Integritas pasca-scan: drift detector + (opsional) double-listing.
        # Lihat _post_scan_integrity. Kegagalan ditandai tak-diketahui
        # (fail-safe), tidak menjatuhkan scan.
        _post_scan_integrity(folder_id, journal, file_manifest, drive_id, start_tok)

        if failed_folders:
            print(f"  {Colors.WARNING}⚠ {len(failed_folders)} folder gagal di-scan setelah {SCAN_DEFER_PASSES} pass — daftar di log{Colors.RESET}")

    # Bersihkan record storage untuk file yang sudah tidak ada di manifest
    # agar laporan tidak memuat file yang sudah hilang.
    _manifest_ids = file_manifest.get_all_file_ids()
    _stale = [sfid for sfid, _ in storage.iterate() if sfid not in _manifest_ids]
    for sfid in _stale:
        storage.delete(sfid)
    if _stale:
        _log('info', f"storage cleanup: {len(_stale)} record file terhapus dibuang")

    # Buang thumbnail usang. Cache file yang masih ada dipertahankan.
    prune_thumb_cache(paths['thumb'], _manifest_ids)

    # Migrasi skema video: record lama tanpa daftar hash per-frame (phashes +
    # dvhashes) + durasi hanya bisa dibandingkan via satu frame median (tidak
    # andal). Record video yang belum lengkap di-hash ulang sekali. Hanya
    # menyentuh video. Idempoten: setelah rehash field selalu ada (nilai atau
    # None eksplisit), sehingga tidak diulang lagi.
    _vid_migrated = 0
    _vid_to_migrate = []
    for vfid, vrec in storage.iterate():
        if not vrec or vrec.get('file_type') != 'video':
            continue
        # Lengkap = punya field per-frame + gerbang durasi. Bila tetap kosong
        # setelah rehash, record hanya ikut exact-match BLAKE3 (aman tanpa celah).
        if ('duration' not in vrec
                or 'phashes' not in vrec
                or 'dvhashes' not in vrec):
            _vid_to_migrate.append(vfid)
    for vfid in _vid_to_migrate:
        storage.delete(vfid)
        file_manifest.re_pending(vfid)
        _vid_migrated += 1
    if _vid_migrated:
        _log('info', f"migrasi video: {_vid_migrated} record lama di-hash ulang (frame-list + durasi)")

    # Migrasi skema foto: field gerbang berlapis (dvhash, sharpness_blocks,
    # color_grid, edge_blocks, ssim_thumb, color_hist, aspect_ratio, width)
    # hanya dicek bila kedua sisi memilikinya; record lama tanpa field melewati
    # verifikasi -> rawan false-positive. Record yang belum lengkap di-hash
    # ulang sekali. Idempoten: setelah rehash field selalu ada (nilai atau None).
    #
    # CATATAN: periksa 'sharpness_blocks' (peta per-blok), BUKAN 'sharpness'
    # (ketajaman global) yang sudah dihapus dan tak pernah ada di record manapun;
    # memeriksanya akan selalu True -> semua foto di-rehash tiap run.
    _img_migrated = 0
    _img_to_migrate = []
    for ifid, irec in storage.iterate():
        if not irec or irec.get('file_type') != 'image':
            continue
        if ('dvhash' not in irec
                or 'sharpness_blocks' not in irec
                or 'color_grid' not in irec
                or 'edge_blocks' not in irec
                or 'ssim_thumb' not in irec
                or 'color_hist' not in irec
                or 'aspect_ratio' not in irec
                or 'width' not in irec):
            _img_to_migrate.append(ifid)
    for ifid in _img_to_migrate:
        storage.delete(ifid)
        file_manifest.re_pending(ifid)
        _img_migrated += 1
    if _img_migrated:
        _log('info', f"migrasi foto: {_img_migrated} record lama di-hash ulang (dvhash/sharpness_blocks/color_grid/edge_blocks/ssim_thumb/color_hist/aspect_ratio/width/height)")

    # Jaminan tidak ada file lolos (processed-tanpa-record): storage.put (env
    # hash) dan mark_processed (env manifest) menulis ke dua env LMDB berbeda
    # yang di-mirror independen. Bila runtime mati di antaranya, file bisa
    # 'processed' di manifest tanpa record di storage -> terlewat permanen.
    # Lintasi manifest dan kembalikan ke pending setiap file MEDIA yang belum
    # punya record di storage. Materialisasi fid dulu, baru re_pending di luar
    # loop (write-txn di dalam cursor iter_files -> transaksi bersarang dilarang).
    #
    # Kriteria 'file media' WAJIB sama dengan analyze_duplicates saat menghitung
    # 'unscanned_media' (get_file_type), agar setiap file yang dihitung unscanned
    # pasti masuk antrean (angka tidak macet > 0 tanpa kemajuan). re_pending
    # idempoten dan hanya menyentuh file yang ada di manifest.
    _missing_fids = []
    for _mfid, _minfo in file_manifest.iter_files():
        _minfo = _minfo or {}
        if get_file_type(_minfo.get('name'), _minfo.get('mimeType')) is None:
            continue
        if not storage.exists(_mfid):
            _missing_fids.append(_mfid)
        else:
            # File sudah punya record di storage (termasuk penanda skip permanen).
            # storage.exists() True -> tidak masuk _missing_fids & tidak dihitung
            # 'unscanned_media'. Tidak ada tindakan yang diperlukan.
            pass
    for _mfid in _missing_fids:
        file_manifest.re_pending(_mfid)
    if _missing_fids:
        _log('info', f"re-pending {len(_missing_fids)} file media tanpa record di storage (belum ter-scan)")

    # Jaminan tidak ada file lolos: kembalikan semua file yang pernah gagal
    # di-hash ke pending agar di-retry pada run ini. File gagal akibat gangguan
    # sesaat (download terputus, rate-limit, dekode gagal sekali) tidak boleh
    # terjebak permanen dan terlewat dari deteksi.
    _retried = file_manifest.re_pending_all_failed()
    if _retried:
        _log('info', f"retry {_retried} file yang sebelumnya gagal di-hash")

    # Ambil pending terbaru dari manifest. Non-media langsung ditandai
    # processed; record dengan md5 sama ditandai processed; record dengan md5
    # berbeda dihapus lalu diproses ulang; tanpa record langsung diproses.
    #
    # PENTING: get_pending_files() menahan read-txn LMDB + lock manifest selama
    # generator hidup. Memanggil mark_processed()/write-txn di dalam loop itu
    # membuka transaksi bersarang pada env & thread yang sama -> dilarang LMDB
    # dan bisa deadlock. Materialisasi dulu seluruh pending ke list (read-txn
    # ditutup), baru lakukan mutasi (mark_processed/storage.delete) di luar.
    #
    # Aman terhadap RAM: record di files_db manifest hanya metadata Drive ringan
    # (id, name, mime, size, md5, createdTime, owners) — BUKAN record hash/
    # frame-list video yang besar (itu disimpan di LMDBStorage, bukan di sini).
    _pending_snapshot = list(file_manifest.get_pending_files())
    files_to_process = []
    for f in _pending_snapshot:
        fid = f['id']
        if not is_media_file(f.get('name'), f.get('mimeType')):
            file_manifest.mark_processed(fid)
            continue
        rec = storage.get(fid)
        if rec is not None:
            new_md5 = f.get('md5Checksum')
            old_md5 = rec.get('md5')
            if new_md5 and old_md5 and new_md5 != old_md5:
                storage.delete(fid)
                files_to_process.append(f)
                _log('info', f"isi file berubah, reprocess fid={fid}")
            else:
                # File TIDAK di-hash ulang (md5 sama), tapi lokasi/nama bisa
                # berubah karena file dipindah folder / di-rename. file_id & hash
                # tetap valid; cukup sinkronkan metadata ringan (parent, name)
                # dari manifest terbaru ke record hash agar laporan TXT/PDF &
                # program remover menunjuk lokasi/nama yang benar. Tanpa
                # download/hash ulang (hemat kuota). Hanya menulis bila berubah.
                _new_parent = (f.get('parents') or [None])[0]
                _new_name = f.get('name')
                _meta_changed = False
                if _new_parent is not None and rec.get('parent') != _new_parent:
                    rec['parent'] = _new_parent
                    _meta_changed = True
                if _new_name and rec.get('name') != _new_name:
                    rec['name'] = _new_name
                    _meta_changed = True
                if _meta_changed:
                    storage.put(fid, rec)
                    _log('info', f"sinkron metadata (parent/name) fid={fid} tanpa hash ulang")
                file_manifest.mark_processed(fid)
        else:
            files_to_process.append(f)

    # Garis batas pemisah sebelum blok MEMPROSES. Hanya dicetak bila prompt
    # session-lock sempat tampil di antara judul ANALISIS FOLDER dan blok ini;
    # tanpa prompt, garis batas bawah header sudah menjadi pemisah ke MEMPROSES
    # sehingga garis di sini akan dobel.
    if prompt_shown:
        print_separator()
    # Pesan verifikasi double-listing (disimpan oleh _post_scan_integrity)
    # dicetak SETELAH garis batas agar urutan tampilan benar: batas dulu, baru
    # pesan verifikasi, lalu blok MEMPROSES di bawahnya.
    _deferred_msg = getattr(file_manifest, '_deferred_status_msg', None)
    if _deferred_msg:
        print(_deferred_msg)
        file_manifest._deferred_status_msg = None
    if files_to_process:
        total  = len(files_to_process)
        done   = success = errors = 0
        lock   = threading.Lock()

        def process_one(f: Dict) -> Tuple[str, Optional[Dict], Optional[str]]:
            fid     = f['id']
            name    = f.get('name', '')
            mime    = f.get('mimeType', '')
            raw_size = _safe_int_size(f.get('size'))
            size_mb = raw_size / (1024 * 1024)
            feat, err = extract_features(fid, name, mime, size_mb,
                                         expected_md5=f.get('md5Checksum'),
                                         expected_sha256=f.get('sha256Checksum'),
                                         expected_size=(raw_size or None))
            if feat:
                return fid, {
                    **feat, 'name': name,
                    'createdTime': f.get('createdTime', ''),
                    'ownerEmail': (f.get('owners') or [{}])[0].get('emailAddress', 'N/A'),
                    'size_mb': size_mb,
                    # parent pertama: dipakai laporan agar remover mengeluarkan
                    # file dari folder yang tepat (bukan menebak via nama folder
                    # yang bisa kembar). None bila Drive tak menyertakan parents.
                    'parent': (f.get('parents') or [None])[0],
                }, None
            return fid, None, err

        def _handle_result(fid: str, rec: Optional[Dict], err: Optional[str]):
            nonlocal done, success, errors
            with lock:
                done += 1
                if rec:
                    storage.put(fid, rec)
                    file_manifest.mark_processed(fid)
                    success += 1
                    if success % SAVE_INTERVAL == 0:
                        storage.sync()
                else:
                    # Abort transien (circuit breaker kuota/auth): biarkan file
                    # tetap pending agar diproses ulang saat kuota pulih, jangan
                    # tandai failed (menggelembungkan angka 'gagal' di reconcile).
                    if err == "circuit_open" or _circuit_breaker.is_open():
                        _log('info', f"circuit open, fid={fid} tetap pending (tidak ditandai gagal)")
                    elif err and err.startswith("skip:"):
                        # Deterministik (terlalu kecil / non-media): skip permanen.
                        # Simpan record penanda skip ke storage SEBELUM mark_processed
                        # agar storage.exists(fid) True di run berikutnya, sehingga
                        # _missing_fids tidak me-re_pending-nya tiap run (loop tanpa
                        # kemajuan). Record skip tanpa blake3/phash -> tidak ikut
                        # pencocokan duplikat, hanya penanda terminal.
                        _skip_reason = err[len('skip:'):]
                        storage.put(fid, {'skipped': True, 'skip_reason': _skip_reason})
                        file_manifest.mark_processed(fid)
                        _log('info', f"skip permanen fid={fid}: {_skip_reason} (record skip disimpan ke storage)")
                    else:
                        errors += 1
                        file_manifest.mark_failed(fid, err or "unknown")
                        if err: _log('warning', f"process_one failed fid={fid}: {err}")
                pct = int(done / total * 100)
                bar = "█" * int(30 * done / total) + "░" * (30 - int(30 * done / total))
                print(f"\r{Colors.CYAN}MEMPROSES{Colors.RESET}:[{bar}] {pct}% {done}/{total}{'  '+Colors.WARNING+'✗'+str(errors) if errors else ''}", end="", flush=True)

        # Proses dalam batch ber-bounded agar tidak menahan jutaan Future di RAM
        # sekaligus. Paralelisme tetap CONCURRENT_WORKERS; hanya jumlah future
        # hidup serentak yang dibatasi ke PROCESS_BATCH_SIZE.
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as exe:
            batch: List[Dict] = []

            def _drain(items: List[Dict]):
                # WAJIB nonlocal: blok except di bawah menulis done/errors.
                # Tanpa ini, augmented assign jadi variabel lokal -> UnboundLocalError
                # saat future melempar exception.
                nonlocal done, errors
                futures = {exe.submit(process_one, f): f for f in items}
                for fut in as_completed(futures):
                    f = futures[fut]
                    try:
                        fid, rec, err = fut.result()
                    except Exception as ex:
                        _log('error', f"future result exception: {ex}")
                        with lock:
                            done += 1
                            # Exception saat circuit terbuka = abort transien:
                            # pertahankan pending, jangan tandai gagal.
                            if _circuit_breaker.is_open():
                                _log('info', f"circuit open, fid={f['id']} tetap pending (exception transien)")
                            else:
                                errors += 1
                                file_manifest.mark_failed(f['id'], str(ex))
                        continue
                    _handle_result(fid, rec, err)

            for f in files_to_process:
                # Circuit breaker terbuka (kuota/auth global): hentikan submit
                # batch baru. Sisa file tetap di pending (tidak ditandai failed)
                # agar run berikutnya melanjutkan begitu kuota pulih.
                if _circuit_breaker.is_open():
                    break
                batch.append(f)
                if len(batch) >= PROCESS_BATCH_SIZE:
                    _drain(batch)
                    batch = []
            if batch and not _circuit_breaker.is_open():
                _drain(batch)

        print()
        # Baris status scan (Perubahan/Resume) dicetak SETELAH blok MEMPROSES.
        if _scan_status_msg:
            print(_scan_status_msg)
        storage.sync()
        file_manifest.flush()
        if errors:
            print(f"  {Colors.WARNING}⚠ {errors} file gagal{Colors.RESET}")
        if _circuit_breaker.is_open():
            print(f"  {Colors.WARNING}⚠ API dihentikan (kuota/akses): {_circuit_breaker.reason()} — jalankan ulang nanti untuk melanjutkan{Colors.RESET}")
    else:
        # Tidak ada file baru untuk diproses: blok MEMPROSES tetap ditampilkan
        # demi konsistensi UI, tetapi LANGSUNG penuh (100% 0/0) tanpa animasi
        # loading bertahap (memang tidak ada pekerjaan yang berjalan).
        print(f"{Colors.CYAN}MEMPROSES{Colors.RESET}:[{'█' * 30}] 100% 0/0")
        # Baris status scan (Perubahan/Resume) dicetak SETELAH blok MEMPROSES.
        if _scan_status_msg:
            print(_scan_status_msg)

    if storage.count() > 0:
        run_audit_mode(storage, file_manifest)

    # Scan TUNTAS & SUKSES (semua ter-index, tidak ada pending/gagal): bersihkan
    # scratch sekarang tanpa menunggu run berikutnya. Gerbang ketat
    # (scan_complete & pending=0 & failed=0) mencegah pembersihan saat masih ada
    # pekerjaan tertunda. Pengaman data sama dengan _cleanup_scratch_now (file
    # final .mdb/.gen tidak pernah disentuh).
    try:
        _rec_cleanup = file_manifest.reconcile()
        if (_rec_cleanup.get('scan_complete')
                and _rec_cleanup.get('pending', 0) == 0
                and _rec_cleanup.get('failed', 0) == 0
                and _rec_cleanup.get('failed_folder_count', 0) == 0):
            # Flush dulu agar file final .mdb sudah ter-mirror sebelum scratch
            # dibuang (pengaman _final_exists mengandalkan keberadaan .mdb).
            try: storage.sync()
            except Exception: pass
            try: file_manifest.flush()
            except Exception: pass
            _n_clean = _cleanup_scratch_now(paths['database'], paths['cache'])
            if _n_clean:
                _log('info', f"scan selesai: {_n_clean} file scratch dibersihkan")
            # Prune hash orphan HANYA di titik aman ini: gerbang di atas sudah
            # menjamin scan TUNTAS (scan_complete), tanpa pending/failed, dan
            # tanpa folder gagal -> manifest lengkap & tepercaya. Tambahan
            # syarat: circuit breaker tidak terbuka (metadata Drive tepercaya).
            # Bila tidak, prune di-skip (hash tetap disimpan; fail-safe).
            if PRUNE_ORPHAN_HASH and not _circuit_breaker.is_open():
                try:
                    _valid_ids = file_manifest.get_all_file_ids()
                    if _valid_ids:
                        _n_pruned = storage.prune_orphans(_valid_ids)
                        if _n_pruned:
                            storage.sync()
                            _log('info', f"scan selesai: {_n_pruned} record hash orphan dibuang")
                except Exception as _e_prune:
                    _log('warning', f"prune hash orphan gagal (dilewati): {_e_prune}")
    except Exception as _e_scratch_done:
        _log('debug', f"cleanup scratch pasca-scan-selesai gagal: {_e_scratch_done}")

    result = analyze_duplicates(storage, file_manifest)
    stats  = result['stats']

    print_header("HASIL ANALISIS DUPLIKAT")
    print()
    print_row_themed(Colors.CYAN_KEY, "Folder",      folder_name)
    print_row_themed(Colors.CYAN_KEY, "Total File",  f"{stats['total_files']} file ({format_size(stats['total_size_mb'])})")
    print_row_themed(Colors.CYAN_KEY, "Total Foto",  f"{stats['total_images']} file")
    print_row_themed(Colors.CYAN_KEY, "Total Video", f"{stats['total_videos']} file")
    # Jaminan tidak ada file lolos scan: tampilkan status eksplisit.
    _unscanned = stats.get('unscanned_media', 0)
    if _unscanned > 0:
        print_row_themed(Colors.CYAN_KEY, "File belum ter-scan", f"{_unscanned} file (jalankan ulang untuk menuntaskan)")
    else:
        print_row_themed(Colors.CYAN_KEY, "Status Scan", "semua file media ter-scan")

    # ── REKONSILIASI SCAN: bukti angka tidak ada file lolos ──
    _rec = file_manifest.reconcile()
    stats['reconcile'] = _rec
    print()
    print_row_themed(Colors.GREEN_DIM_KEY, "Rekonsiliasi",
                     f"{_rec['indexed']} terindeks = {_rec['processed']} diproses"
                     f" + {_rec['failed']} gagal + {_rec['pending']} belum")
    if _rec['failed_folder_count'] > 0:
        print_row_themed(Colors.GREEN_DIM_KEY, "Folder gagal di-scan", f"{_rec['failed_folder_count']} folder (jalankan ulang)")
    else:
        print_row_themed(Colors.GREEN_DIM_KEY, "Folder gagal di-scan", "tidak ada (semua folder berhasil di-scan)")
    # Hitung status (dipakai beberapa baris di bawah) sebelum dicetak, agar
    # urutan tampilan log bisa: Integritas Scan -> Verifikasi Listing Ganda ->
    # Perubahan Drive Saat Scan, tanpa bergantung pada urutan cetak.
    _lv_ok = ((not _rec.get('listing_verify_ran'))
              or (_rec.get('listing_verify_known') and _rec.get('listing_verify_discrepancy', 0) == 0))
    _integrity_ok = (_rec['balanced'] and _rec['scan_complete'] and _rec['pending'] == 0
                     and _rec['failed'] == 0 and not _rec.get('drift_detected')
                     and _rec.get('drift_known', False) and _lv_ok)
    # 1) Verifikasi double-listing (opsional). Hanya mempengaruhi tampilan/
    # jaminan bila benar-benar dijalankan.
    if _rec.get('listing_verify_ran'):
        if not _rec.get('listing_verify_known'):
            print_row_themed(Colors.GREEN_DIM_KEY, "Verifikasi listing", "tidak lengkap (jalankan ulang)")
        elif _rec.get('listing_verify_discrepancy', 0) > 0:
            print_row_themed(Colors.GREEN_DIM_KEY, "Verifikasi listing", f"{_rec.get('listing_verify_discrepancy')} file terlewat (akan diproses ulang, jalankan ulang)")
        else:
            print_row_themed(Colors.GREEN_DIM_KEY, "Verifikasi listing", "cocok (dua listing sepakat)")
    else:
        print_row_themed(Colors.GREEN_DIM_KEY, "Verifikasi listing", "tidak dijalankan")
    # 2) Drift: perubahan Drive selama jendela scan. Bila terdeteksi/tak
    # diketahui, angka adalah snapshot mid-flight -> jaminan ditahan eksplisit.
    if _rec.get('drift_detected'):
        print_row_themed(Colors.GREEN_DIM_KEY, "Perubahan Drive", f"{_rec.get('drift_count', 0)} perubahan terdeteksi (jalankan ulang)")
    elif not _rec.get('drift_known', True):
        print_row_themed(Colors.GREEN_DIM_KEY, "Perubahan Drive", "tidak dapat diverifikasi (jalankan ulang)")
    else:
        print_row_themed(Colors.GREEN_DIM_KEY, "Perubahan Drive", "tidak ada (Drive stabil selama scan)")
    # 3) Integritas Scan (kesimpulan, ditampilkan paling akhir setelah bukti).
    if _integrity_ok:
        print_row_themed(Colors.GREEN_DIM_KEY, "Integritas Scan", "TERVERIFIKASI")
    else:
        print_row_themed(Colors.GREEN_DIM_KEY, "Integritas Scan", "BELUM TERVERIFIKASI (jalankan ulang)")

    if stats['total_duplicates'] > 0:
        print()
        print_row_themed(Colors.ORANGE_DIM_KEY, "DUPLIKAT DITEMUKAN", "")
        print_row_themed(Colors.ORANGE_DIM_KEY, "Total Duplikat",     f"{stats['total_duplicates']} file")
        print_row_themed(Colors.ORANGE_DIM_KEY, "Duplikat Foto",      f"{stats['image_duplicates']} file")
        print_row_themed(Colors.ORANGE_DIM_KEY, "Duplikat Video",     f"{stats['video_duplicates']} file")
        print_row_themed(Colors.ORANGE_DIM_KEY, "Ruang Terbuang (Karna Duplikat)", format_size(stats['wasted_size_mb']))
        print()
        while True:
            print(f"{Colors.SUCCESS}{Colors.BOLD}Simpan laporan? (y/n){Colors.RESET} : ", end="")
            _scroll_to_input()
            try:
                ch = input().strip().lower()
            except EOFError:
                # stdin ditutup (lingkungan non-interaktif / headless):
                # asumsikan 'n' (tidak simpan) agar tidak crash.
                print()
                ch = 'n'
            if ch in ('y', 'n'): break
        if ch == 'y':
            print(f"\n{Colors.CYAN}Membuat laporan...{Colors.RESET}")
            # Tulis laporan duplikat: TXT (machine-readable) + PDF. Keputusan
            # tiap gerbang visual (laporan PROSES) disertakan sebagai halaman
            # lanjutan di PDF yang sama (lihat _build_process_report_html).
            save_analysis_to_drive(folder_name, folder_id, storage, result)
    else:
        print()
        print_row_success("Tidak ada duplikat", "Semua file unik")

    print()
    print_separator()

# ───────────────────── ENTRY POINT ─────────────────────
def main():
    print()
    folders = []
    page_token = None
    while True:
        results, err = _drive_execute(
            lambda _pt=page_token: drive_service.files().list(
                q="('root' in parents) and trashed=false and (mimeType='application/vnd.google-apps.folder' or mimeType='application/vnd.google-apps.shortcut')",
                fields="nextPageToken, files(id,name,mimeType,shortcutDetails)",
                pageSize=1000, pageToken=_pt,
                supportsAllDrives=True, includeItemsFromAllDrives=True))
        if results is None:
            print(f"{Colors.ERROR}✗ Gagal mengambil daftar folder: {err}{Colors.RESET}")
            return
        for item in results.get('files', []):
            fid = item['id']
            if item.get('mimeType') == 'application/vnd.google-apps.shortcut':
                sc = item.get('shortcutDetails', {})
                if sc.get('targetMimeType') == 'application/vnd.google-apps.folder':
                    fid = sc.get('targetId', fid)
            folders.append({'name': item['name'], 'id': fid})
        page_token = results.get('nextPageToken')
        if not page_token:
            break

    # Shared Drive (Team Drive): root tiap Shared Drive adalah folder dengan id
    # sama dengan driveId. Seluruh jalur scan sudah mendukung Shared Drive
    # (supportsAllDrives/includeItemsFromAllDrives), jadi cukup menampilkannya
    # sebagai pilihan. Bila gagal, daftar My Drive tetap dipakai.
    sd_token = None
    while True:
        sd_res, sd_err = _drive_execute(
            lambda _pt=sd_token: drive_service.drives().list(
                pageSize=100, pageToken=_pt, fields="nextPageToken, drives(id,name)"))
        if sd_res is None:
            if sd_err:
                _log('warning', f"list Shared Drive gagal: {sd_err}")
            break
        for d in sd_res.get('drives', []):
            did = d.get('id')
            if did:
                folders.append({'name': f"[Shared Drive] {d.get('name', did)}", 'id': did})
        sd_token = sd_res.get('nextPageToken')
        if not sd_token:
            break

    if not folders:
        print(f"{Colors.WARNING}Tidak ada folder ditemukan.{Colors.RESET}")
        return

    # Dedup berdasarkan id target: folder asli dan shortcut yang menunjuk ke
    # folder yang sama tidak boleh muncul dua kali di daftar pilihan.
    _seen_fids = set()
    _unique = []
    for f in folders:
        if f['id'] in _seen_fids:
            continue
        _seen_fids.add(f['id'])
        _unique.append(f)
    folders = _unique

    print_header("DAFTAR FOLDER")
    for i, f in enumerate(folders, 1):
        print(f"{Colors.SUCCESS}{Colors.BOLD}{i:2d}.{Colors.RESET} {f['name']}")
    print_separator()

    while True:
        print(f"{Colors.SUCCESS}{Colors.BOLD}Pilih nomor folder {Colors.RESET}: ", end="")
        _scroll_to_input()
        try:
            raw = input().strip()
        except EOFError:
            # stdin ditutup (lingkungan non-interaktif / headless): tidak bisa
            # meminta pilihan pengguna. Hentikan dengan pesan yang jelas alih-
            # alih loop selamanya (infinite loop karena EOFError selalu terulang).
            print()
            print(f"{Colors.WARNING}stdin tidak tersedia (lingkungan non-interaktif). "
                  f"Panggil analyze_folder() langsung dengan folder_id yang diinginkan.{Colors.RESET}")
            return
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(folders): break
        except (ValueError, TypeError):
            pass
    analyze_folder(folders[idx]['id'], folders[idx]['name'])

if __name__ == "__main__":
    main()
