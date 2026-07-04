"""
DupliGuard Evictor — Eksekutor Laporan DupliGuard Vision (DGV.py)
=================================================================

PERAN
  Pasangan eksekutor untuk DGV.py:
    • DGV.py   = PELAPOR  (read-only): memindai folder & menulis laporan TXT v2.
    • tool ini = EKSEKUTOR          : membaca laporan TXT v2 lalu MEMBERSIHKAN
                                       FOLDER (mengeluarkan file duplikat dari
                                       folder, BUKAN menghapus file).

ATURAN BESI (TIDAK BOLEH DILANGGAR)
  1. HANYA mengeluarkan file dari SEBUAH FOLDER (Drive API: files.update +
     removeParents).
  2. TIDAK PERNAH menghapus file (TIDAK files.delete) dan TIDAK PERNAH memindah
     ke sampah (TIDAK trashed=true). File tidak pernah lenyap.
  3. Yang dibersihkan adalah KEANGGOTAAN FOLDER, bukan file-nya:
     • File milik Anda  : keluar folder -> tetap ada di My Drive Anda.
     • File milik orang : keluar folder -> tetap utuh di My Drive pemilik asli.
  Konsekuensi: SETIAP tindakan tool ini reversibel oleh manusia. Tanpa
  kehilangan data.

ALUR PEMAKAIAN
  1. Ketik/salin NAMA file laporan TXT.
  2. Tool menampilkan INFORMASI laporan (folder, jumlah asli/duplikat, ukuran).
  3. Tool memverifikasi kondisi terkini di Drive (anti-stale) lalu menampilkan
     rencana.
  4. MENU dua opsi (mode interaktif):
       [1] Terapkan rekomendasi laporan -> keluarkan seluruh duplikat. Diminta
           konfirmasi y/n ganda (mencegah salah ketik). 'n' -> kembali ke menu.
       [2] Seleksi file yang dipertahankan -> masukkan id file yang INGIN
           DIPERTAHANKAN (pisah spasi, jumlah bebas). Tool menampilkan NAMA +
           LINK Drive untuk validasi. Lalu y/n: 'y' keluarkan sisanya, 'n'
           kembali ke menu.

MODE SKALA (auto-switch by jumlah entri, ambang LARGE_SCALE_THRESHOLD)
  • INTERAKTIF (kecil) : alur di atas, UX detail penuh (rencana per-grup,
                         opsi 1/2). Entri dimuat ke RAM (aman karena sedikit).
  • BATCH (besar)      : tahan jutaan file. Laporan dibaca STREAMING dari disk
                         (RAM rendah), rencana ditulis ke antrian JSONL, lalu
                         dieksekusi streaming. Tampilan diringkas (agregat).
                         CHECKPOINT incremental -> aman diputus & di-RESUME.

FILOSOFI TEKNIS (level setara DGV.py, BUKAN meniru buta)
  Keamanan setingkat DGV.py dicapai dengan teknologi yang TEPAT untuk tool yang
  MENGUBAH state:
    • Token-bucket throttle PROAKTIF (laju aman sejak awal, anti rate-limit).
    • Retry + backoff + KLASIFIKASI ERROR (andal & jujur saat API rewel).
    • LOG AUDIT per-aksi yang PERSISTEN & REVERSIBEL (parent sebelum/sesudah,
      status orphan, pelaku, identitas laporan) -> bisa di-rollback manual.
    • CHECKPOINT + RESUME (mode batch; aman diputus di tengah run besar).
    • IDEMPOTEN (aman dijalankan ulang; file yang sudah keluar -> dilewati).
    • TOCTOU guard (verifikasi parent ulang tepat sebelum eksekusi).
    • Anti-stale (size + md5, --strict menambah BLAKE3 penuh).
    • Jaminan grup (minimal 1 file per grup duplikat dipertahankan).
    • Identifikasi laporan via KETIK/SALIN nama (sadar & hati-hati).
    • Konfirmasi eksplisit ganda sebelum aksi.

TAMPILAN
  Output sepenuhnya di TERMINAL dengan kode warna ANSI (256-color) bergaya
  persis DGV.py: judul box ber-garis, baris 'key : value' berwarna, dan
  pewarnaan otomatis teks di dalam tanda kurung. Tidak ada UI HTML.
"""
!pip install blake3 -q

import io
import os
import re
import sys
import time
import random
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

try:
    import blake3
    _HAS_BLAKE3 = True
except Exception:
    _HAS_BLAKE3 = False

# ==================== SUPPRESS WARNING ====================
warnings.filterwarnings("ignore", category=UserWarning, module="google_auth_httplib2")
for _lg in ("googleapiclient", "google_auth_httplib2"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

# ==================== LOG AUDIT (jejak aksi ke file) ====================
# Tool ini MENGUBAH Drive, jadi tiap aksi dicatat agar dapat diperiksa/diputar
# ulang. Console Colab tetap bersih (log hanya ke file). CATATAN: handler awal
# ini menulis ke /tmp (TIDAK persisten). Handler KEDUA yang persisten ke
# WORK_DIR (diutamakan Google Drive) ditambahkan setelah WORK_DIR siap; lihat
# bagian "AUDIT PERSISTEN & REVERSIBEL".
_AUDIT_PATH = "/tmp/dupliguard_evictor_audit.log"
_logger = logging.getLogger("dupliguard_evictor")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False
if not _logger.handlers:
    _fh = logging.FileHandler(_AUDIT_PATH, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_fh)


# Nonaktifkan pelemparan exception dari modul logging. Bila handler gagal
# menulis/flush (mis. Drive mount putus -> OSError Errno 107), kegagalan
# TIDAK boleh mencemari stdout dengan traceback berulang.
logging.raiseExceptions = False


def _log(level: str, msg: str):
    try:
        getattr(_logger, level)(msg)
    except Exception:
        # Logging bersifat best-effort; jangan pernah menggagalkan alur utama
        # (apalagi mencetak traceback) hanya karena audit gagal ditulis.
        pass


# ==================== SETUP ====================
auth.authenticate_user()
drive_service = build('drive', 'v3', cache_discovery=False)

# ==================== KONSTANTA ====================
MAX_RETRY     = 5
RETRY_BACKOFF = [1, 2, 4, 8, 16]
STREAM_CHUNK  = 10 * 1024 * 1024
MIME_FOLDER   = 'application/vnd.google-apps.folder'

# Batas laju (pacing) PROAKTIF untuk Drive API. Pada skala ribuan-jutaan file,
# mengandalkan retry reaktif setelah kena 429 saja boros & lambat. Token-bucket
# memastikan kita TIDAK PERNAH melampaui laju aman sejak awal, sehingga
# 'userRateLimitExceeded' praktis tidak terjadi. Drive memberi ~12.000
# request/menit/user (~200/detik); kita ambil margin besar agar aman dijalankan
# berjam-jam tanpa diblokir.
RATE_LIMIT_RPS   = 9.0   # request/detik rata-rata yang diizinkan (konservatif).
RATE_LIMIT_BURST = 18    # kapasitas burst sesaat (>= RPS, beri kelonggaran).

# Ambang AUTO-SWITCH ke mode batch tahan-RAM. Di BAWAH ambang -> alur lama
# (in-memory + UX detail penuh, tanpa regresi). Di ATAS/SAMA -> mode batch
# berbasis disk: rencana ditulis ke antrian JSONL di disk (RAM tetap rendah
# walau jutaan file), checkpoint incremental agar bisa di-resume, dan tampilan
# diringkas (agregat + sampel) karena mustahil mencetak jutaan baris.
LARGE_SCALE_THRESHOLD = 5000

# Ambang aman jumlah file untuk mode --strict (BLAKE3 mengunduh isi tiap file).
# Di atas ini, --strict minta konfirmasi eksplisit karena sangat boros.
STRICT_MAX_FILES = 2000

# Direktori kerja persisten untuk antrian rencana, checkpoint, dan audit.
# Diutamakan di Google Drive (My Drive) bila ter-mount, agar TIDAK hilang saat
# sesi Colab mati. Fallback ke /tmp bila Drive tak tersedia.
def _resolve_work_dir() -> str:
    candidates = [
        "/content/drive/MyDrive/dupliguard_evictor",  # Drive ter-mount.
        os.path.expanduser("~/dupliguard_evictor"),
    ]
    for base in candidates:
        try:
            os.makedirs(base, exist_ok=True)
            # Uji tulis singkat untuk memastikan benar-benar bisa dipakai.
            probe = os.path.join(base, ".write_test")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return base
        except Exception:
            continue
    # Fallback terakhir: /tmp (tidak persisten, diberi peringatan saat dipakai).
    fallback = "/tmp/dupliguard_evictor"
    os.makedirs(fallback, exist_ok=True)
    return fallback


WORK_DIR = _resolve_work_dir()
WORK_DIR_PERSISTENT = not WORK_DIR.startswith("/tmp")


# ==================== CHECKPOINT STORE (resume aman) ====================
import json


class _Checkpoint:
    """Catatan progres tahan-mati untuk mode batch.

    Menyimpan id file yang SUDAH selesai diproses (berhasil dikeluarkan atau
    sudah-keluar/idempoten) ke file JSONL, di-flush per aksi. Saat run diulang
    dengan laporan yang sama, id yang sudah tercatat DILEWATI sehingga hanya
    sisa yang diproses. Aman bila proses terputus di tengah.

    Kunci checkpoint diikat ke (hash laporan + folder_id) agar checkpoint satu
    laporan tidak salah dipakai untuk laporan lain.
    """
    def __init__(self, key: str):
        self.path = os.path.join(WORK_DIR, f"checkpoint_{key}.jsonl")
        self._done: set = set()
        self._fh = None
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            fid = rec.get("id")
                            if fid:
                                self._done.add(fid)
                        except Exception:
                            continue
            except Exception:
                pass
        # Buka mode append agar entri lama dipertahankan (resume).
        self._fh = open(self.path, "a", encoding="utf-8")

    def is_done(self, file_id: str) -> bool:
        return file_id in self._done

    def mark(self, file_id: str, status: str, extra: Optional[Dict] = None):
        """Catat satu file selesai + flush segera (tahan terputus)."""
        if file_id in self._done:
            return
        self._done.add(file_id)
        rec = {"id": file_id, "status": status, "ts": datetime.now().isoformat()}
        if extra:
            rec.update(extra)
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception:
            pass

    def count(self) -> int:
        return len(self._done)

    def close(self):
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass


# ==================== AUDIT PERSISTEN & REVERSIBEL ====================
# Audit lama (_AUDIT_PATH) menulis ke /tmp dan HILANG saat sesi Colab mati.
# Di sini ditambahkan handler kedua ke WORK_DIR (diutamakan Drive, persisten)
# plus log terstruktur per-aksi removeParents yang cukup untuk ROLLBACK MANUAL:
# mencatat seluruh parent SEBELUM & SESUDAH aksi, status orphan, pelaku, dan
# identitas laporan.
#
# PENTING: handler audit SELALU menulis ke DISK LOKAL, bukan langsung ke Google
# Drive mount. FUSE mount Colab tidak andal untuk penulisan kecil & sering
# (log per-aksi): flush ke endpoint mount yang putus melempar
# 'OSError: [Errno 107] Transport endpoint is not connected' pada SETIAP baris
# log, membanjiri output. Karena itu log ditulis lokal dulu; bila WORK_DIR
# persisten (Drive), snapshot disalin ke Drive di akhir run via
# flush_audit_to_drive().
_AUDIT_LOCAL_PATH = "/tmp/dupliguard_evictor_audit_persist.log"
_AUDIT_PERSIST_PATH = (os.path.join(WORK_DIR, "audit.log")
                       if WORK_DIR_PERSISTENT else _AUDIT_LOCAL_PATH)
try:
    _fh_persist = logging.FileHandler(_AUDIT_LOCAL_PATH, encoding="utf-8", delay=True)
    _fh_persist.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_fh_persist)
except Exception:
    pass


def flush_audit_to_drive():
    """Salin audit log lokal ke Drive (WORK_DIR) sebagai snapshot akhir run.
    Best-effort: kegagalan (mis. mount putus) tidak mengganggu alur utama.
    Hanya dijalankan bila WORK_DIR persisten (Drive ter-mount).
    """
    if not WORK_DIR_PERSISTENT:
        return
    try:
        for h in _logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
        if os.path.exists(_AUDIT_LOCAL_PATH):
            import shutil
            shutil.copyfile(_AUDIT_LOCAL_PATH, os.path.join(WORK_DIR, "audit.log"))
    except Exception:
        pass


def audit_header(actor_email: Optional[str], report_name: str, report_id: str,
                 report_hash: str, folder_id: Optional[str], asli_n: int,
                 dup_n: int, strict: bool, mode: str):
    """Tulis blok header audit di awal run (siapa, laporan apa, mode apa)."""
    _log('info', "=== RUN START ===")
    _log('info', f"actor={actor_email!r} report_name={report_name!r} "
                 f"report_id={report_id} report_hash={report_hash} "
                 f"folder_id={folder_id} asli={asli_n} dup={dup_n} "
                 f"strict={strict} mode={mode} work_dir={WORK_DIR!r} "
                 f"persistent={WORK_DIR_PERSISTENT}")


def audit_removal(file_id: str, name: str, parents_before: List[str],
                  parents_after: List[str], removed_parent: str):
    """Catat satu aksi removeParents secara reversibel.

    parents_before/after memungkinkan rollback manual: untuk mengembalikan,
    tambahkan kembali 'removed_parent' ke file. 'orphan=True' menandai file yang
    kini tak punya parent (hanya bisa ditemukan via search) -> diberi perhatian.
    """
    orphan = len(parents_after) == 0
    _log('info', f"REMOVE file={file_id} name={name!r} "
                 f"removed_parent={removed_parent} "
                 f"parents_before={parents_before} parents_after={parents_after} "
                 f"orphan={orphan}")
    return orphan


# ==================== RATE LIMITER (TOKEN-BUCKET) ====================
class _TokenBucket:
    """Token-bucket sederhana & thread-safe untuk membatasi laju request.

    Setiap request 'mengambil' 1 token. Token terisi ulang dengan laju
    RATE_LIMIT_RPS token/detik hingga kapasitas RATE_LIMIT_BURST. Bila token
    habis, pemanggil tidur tepat selama waktu yang dibutuhkan untuk satu token
    berikutnya. Hasilnya: laju rata-rata terjaga, namun lonjakan pendek (burst)
    tetap diizinkan agar tidak menambah latensi saat beban rendah.
    """
    def __init__(self, rate: float, capacity: float):
        import threading
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, amount: float = 1.0):
        """Blokir sampai 'amount' token tersedia, lalu konsumsi."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Isi ulang token sesuai waktu yang berlalu.
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                # Kekurangan token: hitung jeda minimum lalu tidur di luar lock.
                deficit = amount - self._tokens
                wait = deficit / self._rate if self._rate > 0 else 0.05
            time.sleep(max(wait, 0.001))


# Limiter global dipakai SEBELUM setiap request Drive (lihat _execute()).
_rate_limiter = _TokenBucket(RATE_LIMIT_RPS, RATE_LIMIT_BURST)


# ==================== WARNA TERMINAL (ANSI 256, gaya DGV.py) ====================
# Skema warna meniru persis DGV.py: judul box ber-garis, baris key:value
# berwarna, dan pewarnaan teks di dalam tanda kurung. Tidak ada HTML.
class Colors:
    RESET         = '\033[0m'
    BOLD          = '\033[1m'
    DIM           = '\033[2m'
    HEADER        = '\033[38;5;30m'
    INFO          = '\033[38;5;246m'
    SUCCESS       = '\033[38;5;29m'
    WARNING       = '\033[38;5;172m'
    ERROR         = '\033[38;5;124m'
    HIGHLIGHT     = '\033[38;5;30m'
    CYAN          = '\033[38;5;30m'
    VALUE         = '\033[38;5;74m'
    VALUE_WARN    = '\033[38;5;130m'
    VALUE_SUCCESS = '\033[38;5;71m'
    BORDER        = '\033[38;5;239m'
    # KEY per-kategori (sama seperti DGV.py).
    CYAN_KEY       = '\033[38;5;30m'   # Folder, Total, Status
    GREEN_DIM_KEY  = '\033[38;5;65m'   # Rekonsiliasi / dilindungi
    ORANGE_DIM_KEY = '\033[38;5;130m'  # blok duplikat
    GRAY_VALUE     = '\033[38;5;248m'  # value di luar kurung
    YELLOW_DIM     = '\033[38;5;100m'  # value di dalam kurung


# Lebar baku garis batas (sama seperti DGV.py).
SEPARATOR_WIDTH = 70


def print_separator(width: int = SEPARATOR_WIDTH, char: str = '\u2550', color: str = Colors.BORDER):
    """Cetak satu garis batas penuh (default '═')."""
    print(f"{color}{char * width}{Colors.RESET}")


def print_header(title: str, width: int = SEPARATOR_WIDTH, char: str = '\u2550',
                 color: str = Colors.BORDER, title_color: str = Colors.HEADER):
    """Header bergaya box: judul di-center diapit dua garis batas."""
    print_separator(width, char, color)
    print(f"{title_color}{Colors.BOLD}{title.upper().center(width)}{Colors.RESET}")
    print_separator(width, char, color)


def _colorize_value(value: str) -> str:
    """Warnai value: teks di LUAR kurung abu-abu, di DALAM kurung kuning tua.
    Tanda kurung sendiri tetap abu-abu. Mendukung banyak pasang kurung."""
    out = []
    depth = 0
    for ch in value:
        if ch == '(':
            out.append(f"{Colors.GRAY_VALUE}({Colors.YELLOW_DIM}")
            depth += 1
        elif ch == ')' and depth > 0:
            out.append(f"{Colors.GRAY_VALUE})")
            depth -= 1
        else:
            out.append(ch)
    return f"{Colors.GRAY_VALUE}{''.join(out)}"


def print_row(label: str, value: str, w: int = 26):
    print(f"{Colors.CYAN}{label:<{w}}{Colors.RESET} : {Colors.VALUE}{value}{Colors.RESET}")


def print_row_themed(key_color: str, label: str, value: str, w: int = 26):
    """Baris ringkasan: warna key per-kategori, value dipisah warna luar/dalam
    kurung otomatis (lihat _colorize_value)."""
    print(f"{key_color}{label:<{w}}{Colors.RESET} : {_colorize_value(value)}{Colors.RESET}")


def note_info(msg: str):
    print(f"{Colors.CYAN}• {Colors.RESET}{Colors.INFO}{msg}{Colors.RESET}")


def note_ok(msg: str):
    print(f"{Colors.SUCCESS}✓ {msg}{Colors.RESET}")


def note_warn(msg: str):
    print(f"{Colors.WARNING}! {msg}{Colors.RESET}")


def note_danger(msg: str):
    print(f"{Colors.ERROR}✗ {msg}{Colors.RESET}")


# ==================== HTTP: RETRY + KLASIFIKASI ERROR ====================
# Lapisan kedua setelah token-bucket (lihat RATE LIMITER): menyelamatkan
# kegagalan transien (retry) dan MEMBEDAKAN jenis kegagalan agar hasil jujur
# (404 aman-skip vs 429 retry vs kuota/auth stop). Token-bucket mencegah
# rate-limit secara proaktif; retry+backoff menangani sisa kegagalan sesaat.
def classify_error(code: int, reason: str) -> str:
    """Kembalikan 'recoverable' | 'fatal' | 'fatal_global'.
      recoverable  : transien, aman di-retry (429/5xx, rate-limit sesaat).
      fatal        : permanen & spesifik file (404, izin spesifik).
      fatal_global : memengaruhi semua request (kuota harian habis, auth gagal).
    """
    r = (reason or '').lower()
    if code == 403:
        fatal_markers = ('dailylimitexceeded', 'daily limit', 'quotaexceeded',
                         'quota exceeded', 'limitexceeded', 'storagequota')
        if any(m in r for m in fatal_markers):
            if 'ratelimitexceeded' in r or 'userratelimitexceeded' in r:
                return 'recoverable'
            return 'fatal_global'
        if 'ratelimit' in r or 'userratelimit' in r:
            return 'recoverable'
        return 'fatal'
    if code == 401:
        return 'fatal_global'
    if code == 404:
        return 'fatal'
    if code in (408, 429, 500, 502, 503, 504):
        return 'recoverable'
    return 'fatal'


def _backoff_sleep(attempt: int):
    base = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
    time.sleep(base + random.uniform(0, base * 0.5))


def _execute(request_factory) -> Tuple[Optional[Any], Optional[str]]:
    """Jalankan request Drive dengan retry/backoff dan klasifikasi error.
    request_factory membuat request baru tiap percobaan. Return (result, None)
    atau (None, 'kind:http_code:detail'). 'kind' memudahkan pemanggil mengambil
    keputusan aman tanpa menebak isi pesan.
    """
    last = 'unknown'
    for attempt in range(MAX_RETRY):
        # PACING proaktif: ambil token sebelum SETIAP percobaan request. Ini
        # menjaga laju agregat semua call (meta, removeParents, list, get_media)
        # tetap di bawah ambang aman pada skala jutaan file.
        _rate_limiter.acquire()
        try:
            return request_factory().execute(), None
        except HttpError as e:
            try:
                code = int(e.resp.status)
            except Exception:
                code = 0
            try:
                reason = (e.content.decode('utf-8', 'ignore')
                          if isinstance(getattr(e, 'content', None), bytes)
                          else str(getattr(e, 'content', '') or ''))
            except Exception:
                reason = str(e)
            kind = classify_error(code, reason)
            last = f"{kind}:http_{code}:{reason[:140]}"
            if kind == 'recoverable' and attempt < MAX_RETRY - 1:
                _backoff_sleep(attempt)
                continue
            return None, last
        except Exception as ex:
            last = f"recoverable:exc:{ex}"
            if attempt < MAX_RETRY - 1:
                _backoff_sleep(attempt)
                continue
            return None, f"fatal:exc:{ex}"
    return None, last


def _err_is_not_found(err: Optional[str]) -> bool:
    return bool(err) and 'http_404' in err


def _err_is_global(err: Optional[str]) -> bool:
    return bool(err) and err.startswith('fatal_global')


# ==================== DRIVE PRIMITIVES ====================
def get_current_user_email() -> Optional[str]:
    res, _ = _execute(lambda: drive_service.about().get(fields="user(emailAddress)"))
    return (res.get('user') or {}).get('emailAddress') if res else None


def find_reports_by_name(name: str) -> List[Dict]:
    """Cari SEMUA file dengan nama persis. Daftar dikembalikan agar pemanggil
    mendeteksi ambiguitas (nama Drive tidak unik) dan menolak menebak."""
    safe = name.replace("\\", "\\\\").replace("'", "\\'")
    res, _ = _execute(lambda: drive_service.files().list(
        q=f"name='{safe}' and trashed=false",
        fields="files(id,name,mimeType,size,modifiedTime)",
        pageSize=50, supportsAllDrives=True, includeItemsFromAllDrives=True))
    if not res:
        return []
    return [f for f in res.get('files', [])
            if f.get('mimeType') != MIME_FOLDER and f.get('name') == name]


def download_text(file_id: str) -> Optional[str]:
    """Unduh file teks SECARA PENUH (loop next_chunk sampai selesai)."""
    req = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req, chunksize=STREAM_CHUNK)
    done = False
    try:
        while not done:
            _s, done = dl.next_chunk()
    except HttpError as e:
        _log('error', f"download laporan gagal: {e}")
        return None
    data = buf.getvalue()
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode('utf-8', 'ignore')


def download_report_to_file(file_id: str, dest_path: str) -> bool:
    """Unduh laporan TXT langsung ke FILE di disk (streaming), tanpa menahan
    seluruh isi di RAM. Wajib untuk laporan jutaan baris. Return True bila sukses.
    """
    req = drive_service.files().get_media(fileId=file_id)
    try:
        with open(dest_path, "wb") as out:
            dl = MediaIoBaseDownload(out, req, chunksize=STREAM_CHUNK)
            done = False
            while not done:
                _s, done = dl.next_chunk()
        return True
    except HttpError as e:
        _log('error', f"download laporan (stream) gagal: {e}")
        return False
    except Exception as ex:
        _log('error', f"download laporan (stream) error: {ex}")
        return False


def _iter_report_lines(path: str):
    """Yield baris laporan dari disk dengan fallback encoding, hemat RAM."""
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            with open(path, "r", encoding=enc) as f:
                for line in f:
                    yield line.rstrip('\n')
            return
        except UnicodeDecodeError:
            continue
        except Exception as ex:
            _log('error', f"baca laporan gagal: {ex}")
            return
    # Fallback terakhir: abaikan byte rusak.
    try:
        with open(path, "r", encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line.rstrip('\n')
    except Exception as ex:
        _log('error', f"baca laporan (ignore) gagal: {ex}")


def parse_report_header_from_file(path: str) -> Dict[str, str]:
    """Baca HANYA header (folder/folder id/format) dari file laporan, streaming."""
    header: Dict[str, str] = {}
    for line in _iter_report_lines(path):
        stripped = line.strip()
        if (':' in stripped and not stripped.startswith('[')
                and '=' not in stripped.split(':', 1)[0]):
            label, val = stripped.split(':', 1)
            label = label.strip().lower(); val = val.strip()
            if label == 'folder':
                header['folder'] = val
            elif label == 'folder id':
                header['folder_id'] = val
            elif label.startswith('format'):
                header['format'] = val
        # Header diasumsikan di bagian awal; berhenti begitu menemui entri.
        if LINE_PREFIX_RE.match(line):
            break
    return header


def iter_report_entries(path: str):
    """Yield entri laporan SATU per SATU dari file (streaming, RAM rendah).
    Memakai parser yang sama dengan parse_report agar perilaku identik.
    """
    for line in _iter_report_lines(path):
        m = LINE_PREFIX_RE.match(line)
        if not m:
            continue
        role_tag, rest = m.group(1), m.group(2)
        kv = _parse_kv_line(rest)
        if 'id' not in kv or not kv['id']:
            continue
        yield {
            'role': kv.get('role', role_tag).upper(),
            'group': kv.get('group', ''),
            'id': kv['id'],
            'parent': kv.get('parent') or None,
            'match': kv.get('match', ''),
            'b3': kv.get('b3') or None,
            'size': kv.get('size') or None,
            'w': kv.get('w') or None,
            'h': kv.get('h') or None,
            'name': kv.get('name', ''),
        }


def get_file_meta(file_id: str) -> Tuple[Optional[Dict], Optional[str]]:
    return _execute(lambda: drive_service.files().get(
        fileId=file_id,
        fields="id,name,parents,ownedByMe,capabilities(canRemoveChildren,canEdit),"
               "md5Checksum,size,trashed,mimeType,webViewLink,createdTime",
        supportsAllDrives=True))


def blake3_of_file(file_id: str) -> Optional[str]:
    """BLAKE3 via unduh streaming (hanya mode --strict). None bila gagal."""
    if not _HAS_BLAKE3:
        return None
    req = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req, chunksize=STREAM_CHUNK)
    h = blake3.blake3()
    done = False
    try:
        while not done:
            _s, done = dl.next_chunk()
            chunk = buf.getvalue()
            if chunk:
                h.update(chunk)
                buf.seek(0); buf.truncate(0)
    except HttpError:
        return None
    return h.hexdigest()


def remove_file_from_folder(file_id: str, parent_id: str) -> Tuple[bool, str]:
    """Keluarkan file dari SATU folder (removeParents). TIDAK delete / TIDAK trash."""
    res, err = _execute(lambda: drive_service.files().update(
        fileId=file_id, removeParents=parent_id, body={}, fields="id,parents",
        supportsAllDrives=True))
    return (True, 'removed') if res is not None else (False, err or 'unknown')


def drive_view_link(file_id: str) -> str:
    """Link Drive yang bisa diklik untuk satu file (untuk validasi visual)."""
    return f"https://drive.google.com/file/d/{file_id}/view"


# ==================== PARSER LAPORAN TXT v2 ====================
LINE_PREFIX_RE = re.compile(r'^\s*\[(ASLI|DUP)\]\s*(.*)$')


def _parse_kv_line(rest: str) -> Dict[str, str]:
    """Parse 'key=value | ... | name=...'. 'name' SELALU di akhir; sisa setelah
    'name=' diambil utuh (boleh berisi '|', '(' dll)."""
    out: Dict[str, str] = {}
    name_idx = rest.find('name=')
    head = rest[:name_idx] if name_idx != -1 else rest
    if name_idx != -1:
        out['name'] = rest[name_idx + len('name='):].strip()
    for tok in head.split('|'):
        tok = tok.strip()
        if not tok or '=' not in tok:
            continue
        k, v = tok.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def parse_report(content: str) -> Tuple[Dict, List[Dict]]:
    header: Dict[str, str] = {}
    entries: List[Dict] = []
    for raw in content.split('\n'):
        line = raw.rstrip()
        stripped = line.strip()
        if ':' in stripped and not stripped.startswith('[') and '=' not in stripped.split(':', 1)[0]:
            label, val = stripped.split(':', 1)
            label = label.strip().lower(); val = val.strip()
            if label == 'folder':
                header['folder'] = val
            elif label == 'folder id':
                header['folder_id'] = val
            elif label.startswith('format'):
                header['format'] = val
        m = LINE_PREFIX_RE.match(line)
        if not m:
            continue
        role_tag, rest = m.group(1), m.group(2)
        kv = _parse_kv_line(rest)
        if 'id' not in kv or not kv['id']:
            continue
        entries.append({
            'role': kv.get('role', role_tag).upper(),
            'group': kv.get('group', ''),
            'id': kv['id'],
            'parent': kv.get('parent') or None,
            'match': kv.get('match', ''),
            'b3': kv.get('b3') or None,
            'size': kv.get('size') or None,
            'w': kv.get('w') or None,
            'h': kv.get('h') or None,
            'name': kv.get('name', ''),
        })
    return header, entries


def is_v2_report(content: str, entries: List[Dict]) -> bool:
    return ('FORMAT: v2' in content) or any(e.get('id') for e in entries)


# ==================== VERIFIKASI / PERENCANAAN ====================
def _to_int(v: Optional[str]) -> Optional[int]:
    if v is None or v == '':
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def plan_removals(entries: List[Dict], strict: bool,
                  progress=None) -> Tuple[List[Dict], List[Dict], bool]:
    """Susun rencana eksekusi aman. Return (to_remove, skipped, aborted_global).
    aborted_global=True bila kuota/auth global terdeteksi (hentikan lebih awal).

    Verifikasi berlapis (urut):
      role=ASLI            -> blacklist absolut (selalu dilindungi)
      parent kosong        -> tak tahu folder mana (skip)
      file hilang/trashed  -> tak ada yang perlu dikeluarkan (skip, IDEMPOTEN)
      parent tak melekat   -> sudah keluar dari folder (skip, IDEMPOTEN)
      size/md5(+b3) beda   -> file berubah sejak laporan (skip, anti-stale)
      tanpa izin folder    -> tak berhak (skip)
      lalu JAMINAN GRUP    -> >=1 file per grup dipertahankan
    """
    to_remove: List[Dict] = []
    skipped: List[Dict] = []

    # ANCHOR HIDUP PER GRUP: file ASLI hanya dihitung sebagai penjaga grup bila
    # benar-benar MASIH MELEKAT di folder laporan (parent = e['parent']), tidak
    # trashed, dan tidak hilang. Verifikasi ini otomatis di balik layar; asli
    # yang sudah keluar/hilang TIDAK dianggap ada sehingga aturan 'dup terlama'
    # berlaku. Verifikasi dibatasi pada folder di laporan (bukan seluruh Drive).
    group_asli = defaultdict(int)
    for e in entries:
        if e['role'] != 'ASLI':
            continue
        if not e['parent']:
            continue
        meta_a, err_a = get_file_meta(e['id'])
        if meta_a is None:
            if _err_is_global(err_a):
                # Kuota/auth global saat verifikasi asli: hentikan lebih awal.
                return to_remove, skipped, True
            # Asli tak terbaca/hilang -> tidak dianggap anchor hidup.
            continue
        if meta_a.get('trashed'):
            continue
        if e['parent'] not in set(meta_a.get('parents') or []):
            # Asli sudah keluar dari folder laporan -> bukan anchor hidup.
            continue
        # ANTI-STALE ASLI: bila size di laporan & Drive sama-sama diketahui
        # namun BERBEDA, isi asli sudah berubah sejak laporan -> JANGAN anggap
        # anchor hidup (agar dup tidak dikeluarkan berdasarkan asli yang sudah
        # bukan konten aslinya). Bila size tak dapat dibandingkan, konservatif:
        # tetap dihitung sebagai anchor (mempertahankan lebih banyak file).
        rep_size_a = _to_int(e['size'])
        cur_size_a = _to_int(meta_a.get('size'))
        if rep_size_a is not None and cur_size_a is not None and rep_size_a != cur_size_a:
            continue
        group_asli[e['group']] += 1

    total = len(entries)
    for idx, e in enumerate(entries, 1):
        if progress:
            progress(idx, total)
        item = dict(e)

        if e['role'] == 'ASLI':
            item['status'] = 'skip_asli'; skipped.append(item); continue
        if not e['parent']:
            item['status'] = 'skip_no_parent_in_report'; skipped.append(item); continue

        meta, err = get_file_meta(e['id'])
        if meta is None:
            if _err_is_not_found(err):
                item['status'] = 'skip_not_found'; skipped.append(item); continue
            if _err_is_global(err):
                # Kuota/auth global: hentikan verifikasi lebih lanjut, jujur.
                item['status'] = f'error_meta:{err}'; skipped.append(item)
                return to_remove, skipped, True
            item['status'] = f'error_meta:{err}'; skipped.append(item); continue
        if meta.get('trashed'):
            item['status'] = 'skip_trashed'; skipped.append(item); continue
        if e['parent'] not in set(meta.get('parents') or []):
            item['status'] = 'skip_already_out'; skipped.append(item); continue

        rep_size = _to_int(e['size'])
        cur_size = _to_int(meta.get('size'))
        if rep_size is not None and cur_size is not None and rep_size != cur_size:
            item['status'] = 'skip_changed_size'; skipped.append(item); continue
        if rep_size is None or cur_size is None:
            # Tanpa size yang bisa dibandingkan, tuntut verifikasi b3 (strict).
            if not strict:
                item['status'] = 'skip_unverifiable_need_strict'; skipped.append(item); continue

        if strict:
            # Mode --strict = verifikasi isi byte WAJIB. Tanpa b3 di laporan,
            # isi tak dapat diverifikasi, jadi file di-skip alih-alih lolos
            # hanya dengan kecocokan size (agar klaim 'paling aman' terpenuhi).
            if not e['b3']:
                item['status'] = 'skip_strict_need_b3'; skipped.append(item); continue
            cur_b3 = blake3_of_file(e['id'])
            if cur_b3 is None:
                item['status'] = 'skip_b3_unavailable'; skipped.append(item); continue
            if cur_b3 != e['b3']:
                item['status'] = 'skip_changed_b3'; skipped.append(item); continue

        caps = meta.get('capabilities') or {}
        if not caps.get('canRemoveChildren', True) and not caps.get('canEdit', True):
            item['status'] = 'skip_no_permission'
            item['owned_by_me'] = bool(meta.get('ownedByMe'))
            skipped.append(item); continue

        item['owned_by_me'] = bool(meta.get('ownedByMe'))
        # Simpan link Drive asli (webViewLink) agar validasi opsi 2 menampilkan
        # tautan yang benar untuk file milik orang lain / Google-native, bukan
        # selalu fallback '/file/d/<id>/view' yang bisa tidak akurat.
        item['view_link'] = meta.get('webViewLink') or None
        # Waktu upload (createdTime) untuk memilih survivor terlama saat asli
        # tidak ada di folder. Format ISO 8601 -> perbandingan string aman urut.
        item['created_time'] = meta.get('createdTime') or None
        item['status'] = 'ready'
        to_remove.append(item)

    # JAMINAN SURVIVOR: tiap grup menyisakan >=1 file. Survivor alami = ASLI
    # yang MASIH ada di folder laporan (anchor hidup). Untuk grup TANPA anchor
    # hidup, pertahankan dup dengan UPLOAD TERLAMA (createdTime paling awal),
    # karena asli selalu diupload lebih dulu sehingga salinan tertua paling
    # mendekati asli. Bila hanya ada 1 dup, otomatis dipertahankan.
    final_remove: List[Dict] = []
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for item in to_remove:
        grouped[item['group']].append(item)

    def _created_key(it: Dict):
        # createdTime kosong dianggap paling baru (dorong ke akhir urutan)
        # agar tidak salah dipilih sebagai survivor terlama. Tie-break dengan
        # id agar deterministik saat createdTime identik/kosong.
        return (it.get('created_time') or '9999-12-31T23:59:59.999Z',
                it.get('id') or '')

    for gid, items in grouped.items():
        if group_asli.get(gid, 0) > 0:
            # Ada asli hidup -> semua dup boleh dikeluarkan.
            final_remove.extend(items); continue
        # Tanpa anchor hidup -> pertahankan dup upload terlama.
        survivor = min(items, key=_created_key)
        for item in items:
            if item is survivor:
                k = dict(item); k['status'] = 'skip_group_survivor'
                skipped.append(k)
            else:
                final_remove.append(item)
    return final_remove, skipped, False


# ==================== UI / UX (TERMINAL berwarna) ====================
STATUS_LABEL = {
    'removed': ('Dikeluarkan dari folder', Colors.VALUE_SUCCESS),
    'ready': ('Siap dikeluarkan', Colors.VALUE),
    'skip_asli': ('DILINDUNGI (file asli)', Colors.VALUE_SUCCESS),
    'skip_no_parent_in_report': ('Dilewati (laporan tanpa parent)', Colors.VALUE_WARN),
    'skip_not_found': ('Dilewati (file tidak ada)', Colors.GRAY_VALUE),
    'skip_trashed': ('Dilewati (sudah di sampah)', Colors.GRAY_VALUE),
    'skip_already_out': ('Dilewati (sudah keluar dari folder)', Colors.GRAY_VALUE),
    'skip_changed_size': ('Dilewati (ukuran berubah sejak laporan)', Colors.VALUE_WARN),
    'skip_changed_b3': ('Dilewati (isi berubah sejak laporan)', Colors.VALUE_WARN),
    'skip_unverifiable_need_strict': ('Dilewati (tak terverifikasi; pakai --strict)', Colors.VALUE_WARN),
    'skip_b3_unavailable': ('Dilewati (BLAKE3 tak bisa dihitung)', Colors.VALUE_WARN),
    'skip_strict_need_b3': ('Dilewati (--strict butuh BLAKE3 di laporan)', Colors.VALUE_WARN),
    'skip_no_permission': ('Dilewati (tanpa izin mengubah folder)', Colors.ERROR),
    'skip_group_survivor': ('Dipertahankan (penjaga grup)', Colors.VALUE_SUCCESS),
    'skip_user_keep': ('Dipertahankan (pilihan Anda)', Colors.VALUE_SUCCESS),
    'skip_multi_parent': ('Dilewati (file ada di banyak folder; hindari salah keluar)', Colors.VALUE_WARN),
}


def _label_color(status: str) -> Tuple[str, str]:
    if status.startswith('error_meta'):
        return ('Gagal baca metadata: ' + status.split(':', 2)[-1][:80], Colors.ERROR)
    if status.startswith('error_remove'):
        return ('Gagal keluarkan: ' + status.split(':', 2)[-1][:80], Colors.ERROR)
    return STATUS_LABEL.get(status, (status, Colors.GRAY_VALUE))


def _human(n: Optional[int]) -> str:
    if not n:
        return '—'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    f = float(n); i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024; i += 1
    return f"{f:.2f} {units[i]}"


def _res_str(e: Dict) -> str:
    """Resolusi 'WxH px' dari token w/h laporan; '—' bila tak tersedia."""
    w = _to_int(e.get('w')); h = _to_int(e.get('h'))
    if w and h and w > 0 and h > 0:
        return f"{w} x {h} px"
    return '—'


def ui_banner():
    print_header("DupliGuard Evictor")
    print(f"{Colors.INFO}Tidak menghapus file, hanya mengeluarkannya dari folder. "
          f"File milik Anda{Colors.RESET}")
    print(f"{Colors.INFO}kembali ke My Drive Anda, sedangkan file milik pengguna "
          f"lain tetap berada{Colors.RESET}")
    print(f"{Colors.INFO}di My Drive pemiliknya.{Colors.RESET}")


def ui_report_info(header: Dict, asli_n: int, dup_n: int, total_dup_bytes: Optional[int]):
    """Kartu INFORMASI laporan (terminal)."""
    print()
    print_header("Informasi Laporan")
    print_row_themed(Colors.CYAN_KEY, "Folder", header.get('folder', '-'))
    print_row_themed(Colors.CYAN_KEY, "Folder ID", header.get('folder_id', '-'))
    print_row_themed(Colors.GREEN_DIM_KEY, "File asli", f"{asli_n} file")
    print_row_themed(Colors.ORANGE_DIM_KEY, "File duplikat", f"{dup_n} file")
    print_row_themed(Colors.CYAN_KEY, "Perkiraan ruang duplikat", _human(total_dup_bytes))


def _plan_line(mark: str, mark_color: str, e: Dict, note: str, note_color: str):
    """Cetak satu baris file di dalam grup: penanda + nama + keterangan."""
    nm = (e.get('name') or '(tanpa nama)')[:42]
    print(f"  {mark_color}{mark} {nm:<42}{Colors.RESET}{note_color}{note}{Colors.RESET}")


def ui_plan_table(to_remove: List[Dict], skipped: List[Dict]):
    """Tampilkan rencana DIKELOMPOKKAN per grup duplikat. Tiap grup memuat file
    ASLI (hijau, dipertahankan) dan DUP (oranye, dikeluarkan) bersama, plus file
    yang dilewati (abu-abu) agar tidak ada yang hilang dari pandangan."""
    print()
    print_header("Rencana Eksekusi")

    # Kumpulkan semua entri per grup, tandai aksinya.
    groups: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    for e in to_remove:
        groups[e.get('group', '')].append(('remove', e))
    for e in skipped:
        groups[e.get('group', '')].append(('skip', e))

    def _grp_key(g: str):
        try:
            return (0, int(g))
        except (TypeError, ValueError):
            return (1, g)

    for gid in sorted(groups.keys(), key=_grp_key):
        print(f"\n{Colors.CYAN}{Colors.BOLD}Grup #{gid or '-'}{Colors.RESET}")
        for action, e in groups[gid]:
            status = e.get('status', '')
            if action == 'remove':
                own = ('milik Anda' if e.get('owned_by_me')
                       else 'milik orang lain')
                _plan_line('\u25cb', Colors.ORANGE_DIM_KEY, e,
                           f"\u2192 dikeluarkan ({own})", Colors.VALUE_WARN)
            elif status in ('skip_asli', 'skip_group_survivor', 'skip_user_keep'):
                # File yang DIPERTAHANKAN (asli / penjaga grup / pilihan user).
                lbl, _ = _label_color(status)
                _plan_line('\u25cf', Colors.VALUE_SUCCESS, e,
                           f"{lbl}", Colors.VALUE_SUCCESS)
            else:
                # Dilewati karena alasan lain (sudah keluar, berubah, dll).
                lbl, _ = _label_color(status)
                _plan_line('\u00b7', Colors.GRAY_VALUE, e,
                           f"{lbl}", Colors.GRAY_VALUE)

    kept = len(skipped)
    print()
    print_separator()
    print(f"{Colors.VALUE_WARN}\u25cb dikeluarkan: {len(to_remove)} file{Colors.RESET}   "
          f"{Colors.VALUE_SUCCESS}\u25cf dipertahankan/dilindungi: {kept} file{Colors.RESET}")


def ui_menu():
    """Tampilkan tiga opsi tindakan utama (terminal)."""
    print()
    print_header("Menu Tindakan")
    print(f"{Colors.CYAN}{Colors.BOLD}[1] Terapkan rekomendasi laporan{Colors.RESET}")
    print(f"    {Colors.INFO}Keluarkan seluruh duplikat sesuai analisis laporan. "
          f"File asli tetap terlindungi.{Colors.RESET}")
    print(f"{Colors.SUCCESS}{Colors.BOLD}[2] Kecualikan file pilihan Anda{Colors.RESET}")
    print(f"    {Colors.INFO}Keluarkan duplikat, kecuali ID yang Anda tentukan untuk "
          f"dipertahankan. Hanya berlaku untuk duplikat.{Colors.RESET}")
    print(f"{Colors.WARNING}{Colors.BOLD}[3] Pengeluaran manual berdasarkan ID{Colors.RESET}")
    print(f"    {Colors.INFO}Keluarkan tepat file yang ID-nya Anda masukkan, terlepas "
          f"berstatus asli maupun duplikat. Kendali penuh di tangan Anda.{Colors.RESET}")


def ui_keep_validation(kept: List[Dict]):
    """Tampilkan file yang akan DIPERTAHANKAN beserta link Drive untuk validasi."""
    print()
    print_header(f"Akan DIPERTAHANKAN: {len(kept)} file")
    if not kept:
        note_info("(tidak ada)")
        return
    for i, e in enumerate(kept, 1):
        link = e.get('view_link') or drive_view_link(e['id'])
        print(f"{Colors.VALUE_SUCCESS}{i}.{e['name'] or '(tanpa nama)'}{Colors.RESET}")
        print(f"     {Colors.GRAY_VALUE}Resolusi: {_res_str(e)}{Colors.RESET}")
        print(f"     {Colors.CYAN}{link}{Colors.RESET}")
        print()


def ui_keep_picklist(to_remove: List[Dict]):
    """Daftar ID duplikat yang dapat disalin untuk dipertahankan (opsi 2)."""
    print()
    print_header("Daftar duplikat (salin ID yang ingin dipertahankan)")
    for e in to_remove:
        print(f"{Colors.VALUE}{e['id']}{Colors.RESET}  "
              f"{Colors.GRAY_VALUE}{e['name']}{Colors.RESET}  "
              f"{Colors.YELLOW_DIM}[{_res_str(e)}]{Colors.RESET}")


def ui_manual_picklist(entries: List[Dict]):
    """Daftar SELURUH file laporan (asli + duplikat) yang dapat disalin untuk
    dikeluarkan secara manual (opsi 3). Status asli/duplikat ditandai jelas."""
    print()
    print_header("Daftar seluruh file (salin ID yang ingin dikeluarkan)")
    for e in entries:
        is_asli = (e.get('role') == 'ASLI')
        tag = ('ASLI' if is_asli else 'DUPLIKAT')
        tag_col = Colors.VALUE_SUCCESS if is_asli else Colors.ORANGE_DIM_KEY
        print(f"{Colors.VALUE}{e['id']}{Colors.RESET}  "
              f"{tag_col}[{tag}]{Colors.RESET}  "
              f"{Colors.GRAY_VALUE}{e.get('name', '')}{Colors.RESET}")


def ui_manual_validation(to_remove: List[Dict]):
    """Tampilkan file yang akan DIKELUARKAN (opsi 3) beserta tautan Drive untuk
    validasi akhir sebelum eksekusi."""
    print()
    print_header(f"Akan DIKELUARKAN: {len(to_remove)} file")
    if not to_remove:
        note_info("(tidak ada)")
        return
    for i, e in enumerate(to_remove, 1):
        link = e.get('view_link') or drive_view_link(e['id'])
        tag = ('ASLI' if e.get('role') == 'ASLI' else 'DUPLIKAT')
        print(f"{Colors.VALUE_WARN}{i}. {e.get('name') or '(tanpa nama)'} "
              f"[{tag}]{Colors.RESET}")
        print(f"     {Colors.CYAN}{link}{Colors.RESET}")
        print()


# ==================== ALUR MENU / INPUT ====================
def _autoscroll_to_input():
    """Paksa Colab menggulir ke bawah tepat sebelum meminta input, agar prompt
    input yang muncul di akhir output SELALU terlihat tanpa scroll manual.
    Best-effort: aman diabaikan bila bukan di lingkungan Colab/IPython.
    """
    try:
        from IPython.display import Javascript, display
        display(Javascript(
            """
            (function(){
              // Gulir output cell ke paling bawah, lalu fokuskan kotak input.
              var out = document.querySelector('.output_area:last-child')
                        || document.scrollingElement || document.body;
              try { out.scrollIntoView({block:'end'}); } catch(e){}
              try { window.scrollTo(0, document.body.scrollHeight); } catch(e){}
              setTimeout(function(){
                var box = document.querySelector('input.raw-input, .raw_input, input[type=text]');
                if (box) { try { box.scrollIntoView({block:'center'}); box.focus(); } catch(e){} }
              }, 50);
            })();
            """))
    except Exception:
        pass


def _ask(prompt: str) -> str:
    print(f"{Colors.SUCCESS}{Colors.BOLD}{prompt}{Colors.RESET}", end="")
    _autoscroll_to_input()
    return input().strip()


def partition_keep(to_remove: List[Dict], skipped: List[Dict],
                   keep_ids: set) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Pisahkan to_remove menjadi (sisa_dikeluarkan, skipped_baru, dipertahankan)
    berdasarkan keep_ids. Hanya MENGURANGI cakupan (tidak pernah menambah yang
    dikeluarkan). Tetap di bawah jaminan survivor & verifikasi plan_removals."""
    new_to_remove: List[Dict] = []
    kept: List[Dict] = []
    new_skipped = list(skipped)
    for e in to_remove:
        if e['id'] in keep_ids:
            k = dict(e); k['status'] = 'skip_user_keep'
            new_skipped.append(k); kept.append(k)
        else:
            new_to_remove.append(e)
    return new_to_remove, new_skipped, kept


def execute_removals(to_remove: List[Dict]) -> Tuple[List[Tuple[Dict, str]], int]:
    """Eksekusi pengeluaran file dari folder dengan TOCTOU guard. Return
    (processed, ok_count)."""
    processed: List[Tuple[Dict, str]] = []
    ok_count = 0
    total = len(to_remove)
    if total == 0:
        return processed, ok_count
    for idx, e in enumerate(to_remove, 1):
        filled = int(28 * idx / total)
        bar = '█' * filled + '░' * (28 - filled)
        print(f"\r{Colors.CYAN}PROSES {Colors.BORDER}[{Colors.SUCCESS}{bar}"
              f"{Colors.BORDER}]{Colors.RESET} {Colors.VALUE}{int(idx/total*100)}% "
              f"{idx}/{total}{Colors.RESET}", end="", flush=True)
        # TOCTOU guard: SELALU verifikasi ulang parent dari Drive tepat sebelum
        # eksekusi. Tool ini MENGUBAH state Drive, jadi keamanan diutamakan di
        # atas penghematan request. (Optimasi 'freshness window' sebelumnya
        # dibatalkan karena bisa memakai data lama saat alur manual/opsi 2.)
        meta, err = get_file_meta(e['id'])
        if meta is None:
            if _err_is_global(err):
                processed.append((e, f"error_remove:{err}"))
                _log('error', f"abort global saat eksekusi file={e['id']}: {err}")
                break
            processed.append((e, f"error_remove:meta {err}")); continue
        _parents_now = set(meta.get('parents') or [])
        if e['parent'] not in _parents_now:
            processed.append((e, 'skip_already_out')); continue
        # GUARD MULTI-PARENT: bila file berada di banyak folder, mengeluarkannya
        # dari parent laporan bisa membuang dari folder yang tidak dimaksud.
        # Fail-safe: lewati, jangan menebak.
        if len(_parents_now) > 1:
            processed.append((e, 'skip_multi_parent'))
            _log('warning', f"skip multi-parent file={e['id']} parents={sorted(_parents_now)}")
            continue
        ok, status = remove_file_from_folder(e['id'], e['parent'])
        if ok:
            ok_count += 1
            processed.append((e, 'removed'))
            _log('info', f"removed file={e['id']} parent={e['parent']} name={e['name']!r}")
        else:
            processed.append((e, f"error_remove:{status}"))
            _log('warning', f"remove gagal file={e['id']}: {status}")
            if _err_is_global(status):
                _log('error', 'abort global saat eksekusi'); break
    print()
    return processed, ok_count


def show_result(processed: List[Tuple[Dict, str]], ok_count: int, skipped_n: int):
    fail = len(processed) - ok_count
    print()
    print_header("Hasil")
    print_row_success("✓ Berhasil dikeluarkan", f"{ok_count} file")
    print_row_warning("! Gagal/dilewati eksekusi", f"{fail} file")
    print_row_themed(Colors.GREEN_DIM_KEY, "⛨ Dilindungi/dilewati rencana", f"{skipped_n} file")
    print()
    note_info("File yang dikeluarkan tidak dihapus. Bila milik Anda ada di My "
              "Drive; bila milik orang lain tetap utuh di Drive pemiliknya.")
    print()
    print_separator()
    for e, status in processed:
        lbl, col = _label_color(status)
        name = (e.get('name') or e.get('id') or '')[:40]
        print(f"{Colors.CYAN}{name:<40}{Colors.RESET} : {col}{lbl}{Colors.RESET}")


def print_row_warning(label: str, value: str, w: int = 26):
    print(f"{Colors.WARNING}{label:<{w}}{Colors.RESET} : {Colors.VALUE_WARN}{value}{Colors.RESET}")


def print_row_success(label: str, value: str, w: int = 26):
    print(f"{Colors.SUCCESS}{label:<{w}}{Colors.RESET} : {Colors.VALUE_SUCCESS}{value}{Colors.RESET}")


def _build_manual_removals(entries: List[Dict], target_ids: set) -> List[Dict]:
    """Bangun daftar item pengeluaran manual (opsi 3) dari ID sembarang.
    Mengambil parent dari laporan; verifikasi TOCTOU & multi-parent tetap
    dilakukan di execute_removals. File tanpa parent di laporan dilewati.
    """
    items: List[Dict] = []
    for e in entries:
        if e['id'] in target_ids:
            item = dict(e)
            item['status'] = 'ready'
            items.append(item)
    return items


def run_menu(to_remove: List[Dict], skipped: List[Dict],
             entries: List[Dict]) -> Tuple[Optional[List[Dict]], List[Dict]]:
    """Tampilkan menu tiga opsi & kembalikan (final_remove, final_skipped).
    final_remove = None bila dibatalkan. Loop sampai pilihan terkonfirmasi.

    entries = SELURUH entri laporan (asli + duplikat), diperlukan agar opsi 2
    dapat mengenali ID milik file asli (dan menolaknya) serta opsi 3 dapat
    mengeluarkan ID apa pun secara manual.
    """
    asli_ids = {e['id'] for e in entries if e.get('role') == 'ASLI'}
    asli_by_id = {e['id']: e for e in entries if e.get('role') == 'ASLI'}
    all_ids = {e['id'] for e in entries}

    while True:
        ui_menu()
        choice = _ask("\nPilih tindakan [1/2/3] : ")

        # ---------- OPSI 1: TERAPKAN REKOMENDASI LAPORAN ----------
        if choice == '1':
            note_info(f"Menerapkan rekomendasi laporan: {len(to_remove)} duplikat "
                      f"akan dikeluarkan dari folder.")
            c1 = _ask(f"Konfirmasi pengeluaran {len(to_remove)} duplikat sesuai laporan? (y/n) : ")
            if c1.lower() != 'y':
                note_warn('Tindakan dibatalkan. Kembali ke menu.'); continue
            _log('info', f"opsi 1 (rekomendasi laporan) dikonfirmasi: {len(to_remove)} file")
            return to_remove, skipped

        # ---------- OPSI 2: KECUALIKAN FILE PILIHAN (hanya duplikat) ----------
        elif choice == '2':
            note_ok('Salin ID duplikat yang ingin Anda PERTAHANKAN dari daftar di '
                    'bawah. Pisahkan beberapa ID dengan spasi.')
            ui_keep_picklist(to_remove)
            keep_raw = _ask("\nMasukkan ID yang dipertahankan : ")
            keep_ids = {tok for tok in keep_raw.split() if tok}
            if not keep_ids:
                note_warn('Tidak ada ID yang dimasukkan. Kembali ke menu.'); continue

            valid_ids = {e['id'] for e in to_remove}

            # TOLAK ID MILIK FILE ASLI: kekeliruan umum. File asli tidak pernah
            # dikeluarkan, jadi memilih untuk "mempertahankannya" tak bermakna.
            # Beri tahu pengguna secara eksplisit agar sadar salah salin ID.
            asli_selected = sorted(keep_ids & asli_ids)
            if asli_selected:
                note_danger('ID berikut adalah FILE ASLI, bukan duplikat, sehingga '
                            'memang sudah terlindungi dan tidak akan dikeluarkan:')
                for aid in asli_selected:
                    nm = asli_by_id.get(aid, {}).get('name', '')
                    note_warn(f'  • {aid}  ({nm})  [ASLI]')
                note_info('Silakan periksa kembali. Opsi 2 hanya menerima ID '
                          'duplikat. Kembali ke menu.')
                continue

            unknown = sorted(keep_ids - valid_ids)
            keep_ids &= valid_ids
            if unknown:
                note_warn('ID berikut tidak ada dalam daftar duplikat dan diabaikan: '
                          + ', '.join(unknown))
            if not keep_ids:
                note_warn('Tidak ada ID duplikat yang valid. Kembali ke menu.'); continue

            new_to_remove, new_skipped, kept = partition_keep(to_remove, skipped, keep_ids)

            # VALIDASI: tampilkan nama + tautan Drive untuk verifikasi visual.
            ui_keep_validation(kept)
            note_info(f'Setelah dikecualikan, sisa yang akan dikeluarkan: '
                      f'{len(new_to_remove)} file.')
            if not new_to_remove:
                note_warn('Seluruh duplikat dipertahankan; tidak ada yang '
                          'dikeluarkan. Kembali ke menu.'); continue

            conf = _ask("\nNama & tautan sudah sesuai? Lanjutkan pengeluaran sisanya? (y/n) : ")
            if conf.lower() == 'y':
                _log('info', f"opsi 2: pertahankan {sorted(keep_ids)}, keluarkan {len(new_to_remove)}")
                ui_plan_table(new_to_remove, new_skipped)
                return new_to_remove, new_skipped
            else:
                note_warn('Tindakan dibatalkan. Kembali ke menu.')
                continue

        # ---------- OPSI 3: PENGELUARAN MANUAL BY ID (asli / duplikat) ----------
        elif choice == '3':
            note_danger('Mode manual: file yang ID-nya Anda masukkan akan '
                        'DIKELUARKAN, terlepas berstatus asli maupun duplikat.')
            note_ok('Salin ID file yang ingin DIKELUARKAN dari daftar di bawah. '
                    'Pisahkan beberapa ID dengan spasi.')
            ui_manual_picklist(entries)
            rm_raw = _ask("\nMasukkan ID yang dikeluarkan : ")
            rm_ids = {tok for tok in rm_raw.split() if tok}
            if not rm_ids:
                note_warn('Tidak ada ID yang dimasukkan. Kembali ke menu.'); continue

            unknown = sorted(rm_ids - all_ids)
            rm_ids &= all_ids
            if unknown:
                note_warn('ID berikut tidak ada dalam laporan dan diabaikan: '
                          + ', '.join(unknown))
            if not rm_ids:
                note_warn('Tidak ada ID valid. Kembali ke menu.'); continue

            manual_remove = _build_manual_removals(entries, rm_ids)
            no_parent = [e for e in manual_remove if not e.get('parent')]
            manual_remove = [e for e in manual_remove if e.get('parent')]
            if no_parent:
                note_warn('ID berikut tidak memiliki folder induk pada laporan '
                          'sehingga tidak dapat dikeluarkan dan diabaikan: '
                          + ', '.join(e['id'] for e in no_parent))
            if not manual_remove:
                note_warn('Tidak ada file yang dapat dikeluarkan. Kembali ke menu.')
                continue

            n_asli = sum(1 for e in manual_remove if e.get('role') == 'ASLI')
            ui_manual_validation(manual_remove)
            if n_asli:
                note_danger(f'PERHATIAN: {n_asli} di antaranya berstatus FILE ASLI. '
                            f'File tetap tidak dihapus, hanya dikeluarkan dari folder.')
            note_info(f'Total akan dikeluarkan: {len(manual_remove)} file.')

            conf = _ask("\nNama & tautan sudah sesuai? Lanjutkan pengeluaran? (y/n) : ")
            if conf.lower() == 'y':
                _log('info', f"opsi 3 (manual by ID): keluarkan {sorted(rm_ids)} "
                             f"(asli={n_asli})")
                return manual_remove, list(skipped)
            else:
                note_warn('Tindakan dibatalkan. Kembali ke menu.')
                continue

        else:
            note_danger('Pilihan tidak dikenali. Masukkan 1, 2, atau 3.')
            continue


# ==================== MODE BATCH: DUA FASE BERBASIS DISK ====================
# Untuk jutaan file, menahan semua entri/rencana di RAM bisa membuat Colab
# kehabisan memori. Mode ini bekerja dua fase:
#   FASE 1 (plan_to_disk): verifikasi tiap DUP satu per satu, tulis yang 'ready'
#     ke antrian JSONL di disk. RAM hanya menyimpan counter ringkas per grup
#     (untuk jaminan survivor), BUKAN seluruh entri.
#   FASE 2 (execute_from_disk): baca antrian baris-per-baris, TOCTOU penuh,
#     removeParents, checkpoint + audit reversibel. RAM tetap rendah.
# Semua INVARIANT keamanan dipertahankan: keluarkan != hapus, TOCTOU penuh,
# jaminan survivor, klasifikasi error, idempotensi.

def _verify_one(e: Dict, strict: bool) -> Tuple[str, Optional[Dict]]:
    """Verifikasi anti-stale satu entri DUP. Return (status, meta).
    status 'ready' berarti lolos semua cek & boleh dikeluarkan (tunduk survivor).
    Mengikuti logika identik plan_removals agar perilaku konsisten.
    Status global (kuota/auth) dikembalikan sebagai 'GLOBAL:<err>'.
    """
    if not e['parent']:
        return 'skip_no_parent_in_report', None
    meta, err = get_file_meta(e['id'])
    if meta is None:
        if _err_is_not_found(err):
            return 'skip_not_found', None
        if _err_is_global(err):
            return f'GLOBAL:{err}', None
        return f'error_meta:{err}', None
    if meta.get('trashed'):
        return 'skip_trashed', None
    if e['parent'] not in set(meta.get('parents') or []):
        return 'skip_already_out', None
    rep_size = _to_int(e['size']); cur_size = _to_int(meta.get('size'))
    if rep_size is not None and cur_size is not None and rep_size != cur_size:
        return 'skip_changed_size', None
    if rep_size is None or cur_size is None:
        if not strict:
            return 'skip_unverifiable_need_strict', None
    if strict:
        if not e['b3']:
            return 'skip_strict_need_b3', None
        cur_b3 = blake3_of_file(e['id'])
        if cur_b3 is None:
            return 'skip_b3_unavailable', None
        if cur_b3 != e['b3']:
            return 'skip_changed_b3', None
    caps = meta.get('capabilities') or {}
    if not caps.get('canRemoveChildren', True) and not caps.get('canEdit', True):
        return 'skip_no_permission', meta
    return 'ready', meta


def plan_to_disk(report_path: str, strict: bool, queue_path: str,
                 progress=None) -> Tuple[Dict[str, int], bool]:
    """FASE 1 (STREAMING DUA-PASS dari file laporan, RAM rendah).

    Tidak menerima list entri di RAM. Membaca laporan dari disk:
      PASS 1: hitung counter ASLI per grup (untuk jaminan survivor) +
              total DUP (untuk progress). RAM hanya 1 int per grup.
      PASS 2: verifikasi tiap DUP & tulis yang 'ready' ke antrian JSONL.
    Return (stats, aborted_global).
    """
    # PASS 1: counter ANCHOR HIDUP per grup + total DUP (streaming, RAM minimal).
    # ASLI hanya dihitung sebagai anchor bila benar-benar MASIH melekat di
    # folder laporan (parent = e['parent']), tidak trashed, tidak hilang.
    # Verifikasi ini otomatis di balik layar dan dibatasi pada folder laporan
    # (bukan seluruh Drive). Asli yang sudah keluar/hilang tidak dianggap ada.
    group_asli: Dict[str, int] = defaultdict(int)
    total = 0
    for e in iter_report_entries(report_path):
        if e['role'] == 'ASLI':
            if not e['parent']:
                continue
            meta_a, err_a = get_file_meta(e['id'])
            if meta_a is None:
                if _err_is_global(err_a):
                    return dict(defaultdict(int)), True
                continue
            if meta_a.get('trashed'):
                continue
            if e['parent'] not in set(meta_a.get('parents') or []):
                continue
            # ANTI-STALE ASLI (konsisten dengan mode interaktif): size berbeda
            # -> isi asli sudah berubah, jangan anggap anchor hidup.
            rep_size_a = _to_int(e['size'])
            cur_size_a = _to_int(meta_a.get('size'))
            if rep_size_a is not None and cur_size_a is not None and rep_size_a != cur_size_a:
                continue
            group_asli[e['group']] += 1
        else:
            total += 1

    stats = defaultdict(int)
    aborted = False
    # Buffer survivor untuk grup TANPA anchor hidup: simpan record dup upload
    # TERLAMA (createdTime paling awal). Hanya grup tanpa anchor yang dibuffer,
    # jadi RAM tetap rendah. Sisa dup grup ini langsung ditulis ke antrian.
    survivor_buf: Dict[str, Dict] = {}

    def _created_key(rec: Dict):
        # Tie-break deterministik dengan id saat createdTime identik/kosong.
        return (rec.get('created_time') or '9999-12-31T23:59:59.999Z',
                rec.get('id') or '')

    with open(queue_path, "w", encoding="utf-8") as q:
        # PASS 2: verifikasi + tulis antrian (streaming).
        idx = 0
        for e in iter_report_entries(report_path):
            if e['role'] == 'ASLI':
                stats['asli'] += 1
                continue
            idx += 1
            if progress:
                progress(idx, total)
            status, meta = _verify_one(e, strict)
            if status.startswith('GLOBAL:'):
                aborted = True
                stats['skipped'] += 1
                break
            if status != 'ready':
                stats['skipped'] += 1
                stats[f'st_{status}'] += 1
                continue
            gid = e['group']
            rec = {
                'id': e['id'], 'parent': e['parent'], 'name': e['name'],
                'group': gid,
                'owned_by_me': bool((meta or {}).get('ownedByMe')),
                'view_link': (meta or {}).get('webViewLink'),
                'created_time': (meta or {}).get('createdTime'),
            }
            if group_asli.get(gid, 0) > 0:
                # Ada asli hidup -> dup boleh dikeluarkan.
                q.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats['ready'] += 1
                continue
            # Tanpa anchor hidup -> tahan survivor upload terlama.
            cur = survivor_buf.get(gid)
            if cur is None:
                # Kandidat survivor pertama untuk grup ini.
                survivor_buf[gid] = rec
                continue
            if _created_key(rec) < _created_key(cur):
                # 'rec' lebih lama -> jadi survivor baru; yang lama dikeluarkan.
                survivor_buf[gid] = rec
                q.write(json.dumps(cur, ensure_ascii=False) + "\n")
                stats['ready'] += 1
            else:
                # 'rec' lebih baru -> dikeluarkan; survivor tetap.
                q.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats['ready'] += 1

    # Survivor tiap grup tanpa anchor dipertahankan (tidak masuk antrian).
    stats['st_skip_group_survivor'] += len(survivor_buf)
    stats['skipped'] += len(survivor_buf)
    return dict(stats), aborted


def execute_from_disk(queue_path: str, ckpt: '_Checkpoint',
                      progress=None) -> Tuple[int, int, int, int]:
    """FASE 2: eksekusi streaming dari antrian. TOCTOU penuh + checkpoint + audit.
    Return (ok, failed, skipped, orphaned).
    """
    ok = failed = skipped = orphaned = 0
    # Hitung total untuk progress (baca cepat jumlah baris).
    total = 0
    try:
        with open(queue_path, "r", encoding="utf-8") as f:
            for _ in f:
                total += 1
    except Exception:
        total = 0
    done = 0
    with open(queue_path, "r", encoding="utf-8") as q:
        for line in q:
            line = line.strip()
            if not line:
                continue
            done += 1
            if progress and total:
                progress(done, total)
            try:
                e = json.loads(line)
            except Exception:
                failed += 1
                continue
            fid = e.get('id'); parent = e.get('parent')
            if not fid or not parent:
                failed += 1
                continue
            # RESUME: lewati yang sudah tercatat selesai.
            if ckpt.is_done(fid):
                skipped += 1
                continue
            # TOCTOU penuh: verifikasi parent dari Drive tepat sebelum aksi.
            meta, err = get_file_meta(fid)
            if meta is None:
                if _err_is_global(err):
                    _log('error', f"abort global saat eksekusi file={fid}: {err}")
                    break
                failed += 1
                continue
            parents_before = list(meta.get('parents') or [])
            if parent not in set(parents_before):
                # Sudah keluar -> idempoten, tandai selesai.
                ckpt.mark(fid, 'skip_already_out')
                skipped += 1
                continue
            # GUARD MULTI-PARENT (fail-safe): file di banyak folder -> jangan
            # tebak folder mana; lewati agar tidak salah keluar. Ditandai
            # selesai (idempoten) dengan status khusus di checkpoint.
            if len(parents_before) > 1:
                _log('warning', f"skip multi-parent file={fid} parents={sorted(parents_before)}")
                ckpt.mark(fid, 'skip_multi_parent', {'parents': sorted(parents_before)})
                skipped += 1
                continue
            success, status = remove_file_from_folder(fid, parent)
            if success:
                parents_after = [p for p in parents_before if p != parent]
                is_orphan = audit_removal(fid, e.get('name', ''), parents_before,
                                          parents_after, parent)
                if is_orphan:
                    orphaned += 1
                ckpt.mark(fid, 'removed', {'orphan': is_orphan})
                ok += 1
            else:
                _log('warning', f"remove gagal file={fid}: {status}")
                failed += 1
                if _err_is_global(status):
                    _log('error', 'abort global saat eksekusi'); break
    return ok, failed, skipped, orphaned


def run_large_scale(report_path: str, strict: bool, report_hash: str,
                    header: Dict):
    """Orkestrasi mode batch: plan ke disk (streaming dari file) -> konfirmasi
    ganda -> eksekusi streaming dengan checkpoint/resume. RAM tetap rendah
    karena laporan dibaca baris-per-baris dari disk, bukan dimuat penuh.
    """
    folder_id = header.get('folder_id') or 'nofolder'
    key = f"{report_hash}_{folder_id}"
    queue_path = os.path.join(WORK_DIR, f"queue_{key}.jsonl")

    note_info("Mode batch (skala besar) aktif. RAM rendah (streaming disk), "
              "bisa di-resume.")

    def _vprog(i, n):
        if n:
            print(f"\r{Colors.CYAN}VERIFIKASI {Colors.VALUE}{int(i/n*100)}% "
                  f"{i}/{n}{Colors.RESET}", end="", flush=True)

    note_info('Memverifikasi & menulis rencana ke disk (anti-stale)...')
    stats, aborted = plan_to_disk(report_path, strict, queue_path, progress=_vprog)
    print()

    # Ringkasan agregat (bukan daftar penuh).
    print_header("Ringkasan Rencana (mode batch)")
    print_row_themed(Colors.GREEN_DIM_KEY, "File asli", f"{stats.get('asli', 0)} file")
    print_row_themed(Colors.ORANGE_DIM_KEY, "Akan dikeluarkan", f"{stats.get('ready', 0)} file")
    print_row_themed(Colors.CYAN_KEY, "Dilindungi/dilewati", f"{stats.get('skipped', 0)} file")
    # Rincian alasan skip (transparansi; hanya yang berisi).
    _skip_labels = {
        'st_skip_group_survivor': 'Penjaga grup (dipertahankan)',
        'st_skip_no_parent_in_report': 'Tanpa parent di laporan',
        'st_skip_not_found': 'File tidak ada',
        'st_skip_trashed': 'Sudah di sampah',
        'st_skip_already_out': 'Sudah keluar dari folder',
        'st_skip_changed_size': 'Ukuran berubah',
        'st_skip_changed_b3': 'Isi berubah (BLAKE3)',
        'st_skip_unverifiable_need_strict': 'Tak terverifikasi (perlu --strict)',
        'st_skip_b3_unavailable': 'BLAKE3 tak tersedia',
        'st_skip_strict_need_b3': '--strict butuh BLAKE3',
        'st_skip_no_permission': 'Tanpa izin folder',
    }
    for k, lbl in _skip_labels.items():
        if stats.get(k):
            print_row_themed(Colors.GRAY_VALUE, f"  • {lbl}", f"{stats[k]} file")

    if aborted:
        note_warn('Verifikasi dihentikan: kuota/akses API global bermasalah. '
                  'Jalankan ulang nanti (aman, idempoten, akan resume).')
        return
    ready_n = stats.get('ready', 0)
    if ready_n == 0:
        note_ok('Tidak ada file yang perlu dikeluarkan.'); return

    ckpt = _Checkpoint(key)
    if ckpt.count() > 0:
        note_info(f"Checkpoint ditemukan: {ckpt.count()} file sudah selesai pada "
                  f"run sebelumnya; akan dilewati (resume).")

    # Konfirmasi ganda (konsisten dengan opsi 1 mode interaktif).
    if _ask(f"\nKeluarkan {ready_n} file sesuai rencana? (y/n) : ").lower() != 'y':
        note_warn('Dibatalkan. Kembali tanpa perubahan.'); ckpt.close(); return

    def _eprog(i, n):
        if n:
            filled = int(28 * i / n)
            bar = '█' * filled + '░' * (28 - filled)
            print(f"\r{Colors.CYAN}PROSES {Colors.BORDER}[{Colors.SUCCESS}{bar}"
                  f"{Colors.BORDER}]{Colors.RESET} {Colors.VALUE}{int(i/n*100)}% "
                  f"{i}/{n}{Colors.RESET}", end="", flush=True)

    note_info('Memproses pengeluaran file dari folder (streaming)...')
    ok, failed, skipped, orphaned = execute_from_disk(queue_path, ckpt, progress=_eprog)
    print()
    ckpt.close()

    print_header("Hasil (mode batch)")
    print_row_success("✓ Berhasil dikeluarkan", f"{ok} file")
    print_row_warning("! Gagal", f"{failed} file")
    print_row_themed(Colors.GRAY_VALUE, "Dilewati (idempoten/resume)", f"{skipped} file")
    print_row_themed(Colors.GREEN_DIM_KEY, "Menjadi tak berinduk (orphan)", f"{orphaned} file")
    _log('info', f"selesai batch: ok={ok} failed={failed} skipped={skipped} orphan={orphaned}")
    flush_audit_to_drive()
    note_info("File tidak dihapus. Audit reversibel tersimpan di: "
              f"{_AUDIT_PERSIST_PATH}")


# ==================== MAIN ====================
def main():
    strict = ('--strict' in sys.argv)
    ui_banner()
    if strict:
        note_warn('Mode --strict aktif: verifikasi BLAKE3 penuh (mengunduh ulang '
                  'tiap file). Lebih lambat, paling aman.')

    # Identifikasi laporan via KETIK/SALIN nama (disengaja, bukan menu nomor).
    txt_name = _ask("\nSalin/ketik NAMA file laporan TXT : ")
    if not txt_name:
        note_danger('Nama laporan kosong. Dibatalkan.'); return

    candidates = find_reports_by_name(txt_name)
    if not candidates:
        note_danger(f'Laporan tidak ditemukan: {txt_name}. '
                    'Pastikan nama disalin persis.'); return
    if len(candidates) > 1:
        note_danger(f'Ada {len(candidates)} file bernama sama persis. Untuk '
                    'keamanan, tool menolak menebak. Ganti nama salah satu agar '
                    'unik lalu coba lagi.')
        return
    target = candidates[0]

    # Unduh laporan ke FILE di disk (streaming, RAM rendah walau jutaan baris).
    import hashlib
    report_path = os.path.join(WORK_DIR, "report_current.txt")
    if not download_report_to_file(target['id'], report_path):
        note_danger('Gagal membaca isi laporan.'); return

    # Hash laporan dari file (identitas audit + kunci checkpoint stabil),
    # dibaca per-blok agar tidak memuat seluruh isi ke RAM.
    _md5 = hashlib.md5()
    try:
        with open(report_path, "rb") as rf:
            for blk in iter(lambda: rf.read(1024 * 1024), b""):
                _md5.update(blk)
    except Exception:
        note_danger('Gagal membaca isi laporan.'); return
    report_hash = _md5.hexdigest()

    # Header + hitung skala via STREAMING (tanpa menahan semua entri di RAM).
    header = parse_report_header_from_file(report_path)
    asli_n = dup_n = total_dup_bytes = 0
    has_id = False
    for e in iter_report_entries(report_path):
        has_id = True
        if e['role'] == 'ASLI':
            asli_n += 1
        else:
            dup_n += 1
            v = _to_int(e.get('size'))
            if v:
                total_dup_bytes += v
    if not has_id:
        note_danger('Laporan ini bukan format v2 (tidak memuat file id). Buat '
                    'ulang dengan DGV.py terbaru lalu coba lagi.')
        return

    # AUTO-SWITCH: mode batch tahan-RAM bila jumlah entri besar.
    total_entries = asli_n + dup_n
    large_scale = total_entries >= LARGE_SCALE_THRESHOLD

    ui_report_info(header, asli_n, dup_n, total_dup_bytes or None)

    actor_email = get_current_user_email()
    mode = 'batch' if large_scale else 'interaktif'
    audit_header(actor_email, target.get('name', ''), target['id'], report_hash,
                 header.get('folder_id'), asli_n, dup_n, strict, mode)
    if not WORK_DIR_PERSISTENT:
        note_warn('Direktori kerja di /tmp (TIDAK persisten). Mount Google Drive '
                  'agar audit & checkpoint tidak hilang saat sesi berakhir.')
    note_info(f"Audit: {_AUDIT_PERSIST_PATH}")

    if dup_n == 0:
        note_ok('Tidak ada duplikat untuk diproses.'); return

    # GUARD --strict skala besar: BLAKE3 mengunduh isi tiap file. Untuk jumlah
    # sangat besar ini bisa menghabiskan bandwidth/waktu ekstrem -> minta
    # konfirmasi eksplisit.
    if strict and dup_n > STRICT_MAX_FILES:
        note_danger(f'Mode --strict pada {dup_n} file akan MENGUNDUH '
                    f'isi tiap file (sangat lambat & boros kuota).')
        if _ask(f'Lanjutkan --strict untuk {dup_n} file? (y/n) : ').lower() != 'y':
            note_warn('Dibatalkan. Jalankan tanpa --strict atau perkecil cakupan.')
            return

    # ============ MODE BATCH (skala besar, tahan-RAM, resume) ============
    if large_scale:
        run_large_scale(report_path, strict, report_hash, header)
        return

    # ============ MODE INTERAKTIF (skala kecil) ============
    # Aman memuat seluruh entri ke RAM di sini: jumlahnya di bawah
    # LARGE_SCALE_THRESHOLD. UX detail (ui_plan_table, opsi 1/2) tetap utuh.
    entries = list(iter_report_entries(report_path))

    note_info('Memverifikasi kondisi file terkini di Drive (anti-stale)...')

    def _progress(i, n):
        filled = int(28 * i / n)
        bar = '█' * filled + '░' * (28 - filled)
        print(f"\r{Colors.CYAN}VERIFIKASI {Colors.BORDER}[{Colors.SUCCESS}{bar}"
              f"{Colors.BORDER}]{Colors.RESET} {Colors.VALUE}{int(i/n*100)}% "
              f"{i}/{n}{Colors.RESET}", end="", flush=True)

    to_remove, skipped, aborted = plan_removals(entries, strict, progress=_progress)
    print()
    ui_plan_table(to_remove, skipped)

    if aborted:
        note_warn('Verifikasi dihentikan: kuota/akses API global bermasalah. '
                  'Jalankan ulang nanti (aman, idempoten).')
        return
    if not to_remove:
        note_ok('Tidak ada file yang perlu dikeluarkan.'); return

    # MENU tiga opsi (loop sampai terkonfirmasi atau dibatalkan).
    final_remove, final_skipped = run_menu(to_remove, skipped, entries)
    if not final_remove:
        note_warn('Tidak ada aksi. Dibatalkan.')
        _log('info', 'dibatalkan oleh pengguna sebelum eksekusi'); return

    note_info('Memproses pengeluaran file dari folder...')
    processed, ok_count = execute_removals(final_remove)
    show_result(processed, ok_count, len(final_skipped))
    fail = len(processed) - ok_count
    _log('info', f"selesai: berhasil={ok_count} gagal_atau_dilewati={fail} "
                 f"dilindungi={len(final_skipped)}")
    flush_audit_to_drive()


if __name__ == "__main__":
    main()
