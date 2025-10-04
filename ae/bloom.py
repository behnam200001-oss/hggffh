import os
import sqlite3
import math
import time
from typing import Optional
import mmh3
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from threading import Lock
import string
import random

# اضافه کردن ایمپورت توابع کمکی
from .utils import normalize_address

DEFAULT_ERROR_RATE = 1e-4
DEFAULT_HEADROOM = 2
DEFAULT_DB_BATCH = 100_000

class SharedBloom:
    def __init__(self, num_bits: int, num_hashes: int, shm_name: Optional[str] = None):
        self.num_bits = int(num_bits)
        self.num_hashes = int(num_hashes)
        self.num_bytes = (self.num_bits + 7) // 8

        if shm_name is None:
            self.shm = SharedMemory(create=True, size=self.num_bytes)
            self.shm_name = self.shm.name
            self.buf = np.frombuffer(self.shm.buf, dtype=np.uint8, count=self.num_bytes)
            self.buf[:] = 0
            self._owner = True
        else:
            self.shm = SharedMemory(name=shm_name)
            self.shm_name = shm_name
            self.buf = np.frombuffer(self.shm.buf, dtype=np.uint8, count=self.num_bytes)
            self._owner = False

    def _set_bit(self, pos: int):
        idx, bit = pos >> 3, pos & 7
        self.buf[idx] |= (1 << bit)

    def _get_bit(self, pos: int) -> bool:
        idx, bit = pos >> 3, pos & 7
        return (self.buf[idx] & (1 << bit)) != 0

    def _positions(self, item_bytes: bytes):
        for i in range(self.num_hashes):
            yield mmh3.hash(item_bytes, i, signed=False) % self.num_bits

    def add(self, item: bytes):
        for pos in self._positions(item):
            self._set_bit(pos)

    def contains(self, item: bytes) -> bool:
        for pos in self._positions(item):
            if not self._get_bit(pos):
                return False
        return True

    def release_buffer(self):
        self.buf = None

    def close(self):
        if hasattr(self, 'shm'):
            self.shm.close()

    def unlink(self):
        if hasattr(self, '_owner') and self._owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class BloomManager:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self._bloom: Optional[SharedBloom] = None
        self._loaded = False
        self._shm_name: Optional[str] = None
        self._num_bits = 0
        self._num_hashes = 0

    @classmethod
    def instance(cls) -> "BloomManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = BloomManager()
            return cls._instance

    def get(self) -> SharedBloom:
        if not self._loaded:
            self._load()
        return self._bloom

    def shm_name(self) -> Optional[str]:
        return self._shm_name

    @property
    def num_bits(self) -> int:
        return self._num_bits

    @property
    def num_hashes(self) -> int:
        return self._num_hashes

    def _calc_bits(self, n: int, p: float) -> int:
        return max(8, int(-n * math.log(max(p, 1e-16)) / (math.log(2) ** 2)))

    def _calc_hashes(self, m: int, n: int) -> int:
        if n <= 0:
            return 1
        return max(1, int((m / n) * math.log(2)))

    def _db_path(self) -> str:
        return 'data/addresses.db'

    def _db_count(self) -> int:
        db_path = self._db_path()
        if not os.path.exists(db_path):
            return 0
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM addresses')
            total = cur.fetchone()[0] or 0
            return int(total)
        finally:
            conn.close()

    def _build_shared(self, total: int):
        capacity = max(1, int(total * DEFAULT_HEADROOM))
        p = DEFAULT_ERROR_RATE
        m = self._calc_bits(capacity, p)
        k = self._calc_hashes(m, capacity)

        # ایجاد بلوم فیلتر جدید
        bloom = SharedBloom(num_bits=m, num_hashes=k, shm_name=None)
        self._bloom = bloom
        self._shm_name = bloom.shm_name
        self._num_bits = m
        self._num_hashes = k

        db_path = self._db_path()
        approx_mb = m / 8 / 1024 / 1024
        
        if os.path.exists(db_path) and total > 0:
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute('SELECT address FROM addresses')
                batch_size = DEFAULT_DB_BATCH
                loaded = 0
                start = time.time()
                last_report = start

                print(f"Bloom init: bits={m:,}, hashes={k}, ~{approx_mb:.1f} MB, target FP={p}")

                while True:
                    batch = cur.fetchmany(batch_size)
                    if not batch:
                        break
                    for (addr,) in batch:
                        # استفاده از آدرس نرمال‌شده - رفع مشکل کدگذاری
                        normalized_addr = normalize_address(addr)
                        bloom.add(normalized_addr)
                        loaded += 1
                    
                    now = time.time()
                    if (now - last_report) >= 0.5:
                        elapsed = now - start
                        pct = (loaded / total * 100.0)
                        rate = loaded / elapsed if elapsed > 0 else 0.0
                        eta = (total - loaded) / rate if rate > 0 else float('inf')
                        eta_txt = f"{eta:.1f}s" if eta != float('inf') else "inf"
                        print(f"Bloom load: {loaded:,}/{total:,} ({pct:.2f}%) | {rate:,.0f} addr/s | ETA {eta_txt}")
                        last_report = now

                total_time = time.time() - start
                print(f"✅ Bloom loaded {loaded:,} in {total_time:.1f}s | ~{approx_mb:.1f} MB")
            finally:
                conn.close()
        else:
            print(f"⚠️ DB empty/missing. Created empty bloom: bits={m:,}, hashes={k}, ~{approx_mb:.1f} MB, target FP={p}")

        self._loaded = True

    def _load(self):
        total = self._db_count()
        if total == 0:
            total = 100_000
        self._build_shared(total)

    def rebuild(self):
        self.shutdown()
        self._load()

    def shutdown(self):
        if self._bloom is not None:
            try:
                self._bloom.release_buffer()
            except Exception:
                pass
            try:
                self._bloom.close()
            except Exception:
                pass
            try:
                if hasattr(self._bloom, '_owner') and self._bloom._owner:
                    self._bloom.unlink()
            except Exception:
                pass
        self._bloom = None
        self._loaded = False
        self._shm_name = None
        self._num_bits = 0
        self._num_hashes = 0


def attach_shared_bloom(shm_name: str, num_bits: int, num_hashes: int) -> SharedBloom:
    return SharedBloom(num_bits=int(num_bits), num_hashes=int(num_hashes), shm_name=shm_name)


class BloomFilterMonitor:
    """مانیتور برای ردیابی عملکرد و سلامت Bloom Filter"""
    
    def __init__(self, bloom_manager, test_interval=1000000):
        self.bloom_manager = bloom_manager
        self.test_interval = test_interval
        self.false_positives = 0
        self.checks_since_last_test = 0
        self.total_checks = 0

    def calculate_real_false_positive_rate(self, test_size=100000) -> float:
        """محاسبه نرخ خطای مثبت واقعی"""
        false_positives = 0
        bloom = self.bloom_manager.get()
        
        for _ in range(test_size):
            test_addr = self._generate_random_address()
            if bloom.contains(normalize_address(test_addr)):
                false_positives += 1
                
        fpp = false_positives / test_size
        print(f"[BLOOM FILTER DIAGNOSTICS] Real False Positive Rate: {fpp:.8f} ({false_positives}/{test_size})")
        return fpp

    def _generate_random_address(self, length=34):
        """تولید آدرس شبه-تصادفی برای تست"""
        prefixes = ['1', 'bc1', '0x']
        chars = string.ascii_letters + string.digits
        prefix = random.choice(prefixes)
        return prefix + ''.join(random.choice(chars) for _ in range(length - len(prefix)))

    def periodic_check(self, force=False):
        """بررسی دوره‌ی سلامت Bloom Filter"""
        self.checks_since_last_test += 1
        self.total_checks += 1
        
        if force or self.checks_since_last_test >= self.test_interval:
            current_fpp = self.calculate_real_false_positive_rate()
            if current_fpp > 0.02:
                print(f"⚠️ [WARNING] Bloom Filter FPP is high: {current_fpp:.4f}. Consider rebuilding.")
            self.checks_since_last_test = 0