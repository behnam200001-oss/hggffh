import time
import concurrent.futures
import multiprocessing as mp
from typing import Iterable, Tuple, Optional, Any, List
from collections import deque
import base58
import random

# Compatible bech32 import
try:
    from bech32 import bech32_encode, bech32_decode, convertbits, Encoding
    HAVE_ENCODING = True
except ImportError:
    import bech32
    bech32_encode = bech32.bech32_encode
    bech32_decode = bech32.bech32_decode
    convertbits = bech32.convertbits
    Encoding = None
    HAVE_ENCODING = False

from eth_utils import is_checksum_address

from ..bloom import BloomManager, attach_shared_bloom, BloomFilterMonitor  # اضافه کردن BloomFilterMonitor
from ..address import AddressGenerator
from ..utils import safe_mp_context, should_use_threads, normalize_address, validate_address_format
from ..results import ResultsManager


class AddressValidator:
    """اعتبارسنج پیشرفته برای آدرس‌های Bitcoin و Ethereum"""
    
    @staticmethod
    def _validate_btc_address(addr_type: str, address: str) -> bool:
        """اعتبارسنجی دقیق آدرس بیت‌کوین"""
        try:
            if addr_type in ('p2pkh_c', 'p2pkh_u', 'p2sh_p2wpkh'):
                # اعتبارسنجی Base58Check
                decoded = base58.b58decode_check(address)
                return len(decoded) == 21
            elif addr_type == 'bech32':
                hrp, data = bech32_decode(address)
                return hrp == 'bc' and data is not None and data[0] == 0
            elif addr_type == 'taproot':
                hrp, data = bech32_decode(address)
                return hrp == 'bc' and data is not None and data[0] == 1
        except Exception:
            return False
        return False

    @staticmethod
    def _validate_eth_address(address: str) -> bool:
        """اعتبارسنجی آدرس اتریوم"""
        from eth_utils import is_checksum_address, is_hex_address
        return is_hex_address(address) and is_checksum_address(address)

    def validate_before_check(self, currency: str, addr_type: str, address: str) -> bool:
        """اعتبارسنجی نهایی قبل از بررسی در Bloom Filter"""
        if not address or not isinstance(address, str):
            return False
            
        # اعتبارسنجی اولیه فرمت
        if not validate_address_format(currency, addr_type, address):
            return False
            
        try:
            if currency == 'Bitcoin':
                return self._validate_btc_address(addr_type, address)
            elif currency == 'Ethereum':
                return self._validate_eth_address(address)
        except Exception:
            return False
        return False


def _scan_worker_init(currencies, shm, bits, hashes):
    if not currencies:
        currencies = ['Bitcoin']
    global _BLOOM, _ADDR, _VALIDATOR
    _BLOOM = attach_shared_bloom(shm, bits, hashes)
    _ADDR = AddressGenerator(currencies)
    _VALIDATOR = AddressValidator()


def _scan_worker_process(keys: Iterable[bytes]) -> Tuple[int, int, list, list, list]:
    found = 0
    total = 0
    last_keys = []
    found_keys = []
    bloom_matches = []
    
    for k in keys:
        total += 1
        addresses = [
            (cur, atype, addr.decode() if isinstance(addr, bytes) else addr)
            for cur, atype, addr in _ADDR.iter_addresses_for_key(k)
        ]
        
        # فیلتر کردن آدرس‌های نامعتبر قبل از بررسی در Bloom Filter
        valid_addresses = []
        for cur, atype, addr_str in addresses:
            if _VALIDATOR.validate_before_check(cur, atype, addr_str):
                valid_addresses.append((cur, atype, addr_str))
        
        last_keys.append((k.hex(), valid_addresses))
        if len(last_keys) > 2:
            last_keys.pop(0)
        
        # بررسی آدرس‌های معتبر در Bloom Filter
        matched_addresses = []
        for cur, atype, addr_str in valid_addresses:
            # استفاده از آدرس نرمال‌شده برای بررسی یکنواخت
            normalized_addr = normalize_address(addr_str)
            if _BLOOM.contains(normalized_addr):
                matched_addresses.append((cur, atype, addr_str))
        
        if matched_addresses:
            found += len(matched_addresses)
            for cur, atype, addr_str in matched_addresses:
                found_keys.append((k, cur, atype, addr_str))
                
                if len(bloom_matches) < 4:
                    bloom_matches.append((k.hex(), cur, atype, addr_str))
    
    return found, total, last_keys, found_keys, bloom_matches


class BaseScanner:
    def __init__(
        self,
        currencies: List[str] = ['Bitcoin'],
        batch_size: int = 5000,
        max_workers: Optional[int] = None,
        results_dir: str = "results"
    ):
        self.currencies = currencies or ['Bitcoin']
        self.batch_size = batch_size
        self.max_workers = max(1, mp.cpu_count() - 1) if max_workers is None else max(1, max_workers)
        bm = BloomManager.instance()
        bloom = bm.get()
        self._shm = bloom.shm_name
        self._bits = bloom.num_bits
        self._hashes = bloom.num_hashes
        self._validator = AddressValidator()
        self.results_manager = ResultsManager(results_dir)
        self.last_keys_global = deque(maxlen=2)
        self.bloom_matches_global = deque(maxlen=4)
        
        # اضافه کردن مانیتور - با هندل کردن خطا برای سازگاری
        try:
            self.monitor = BloomFilterMonitor(bm)
        except Exception as e:
            print(f"⚠️ Bloom filter monitor initialization warning: {e}")
            self.monitor = None
        
        # مدیریت حافظه
        self._memory_limit = 1024 * 1024 * 1024
        self._check_memory_interval = 100
        self._processed_count = 0

    def _check_memory_usage(self):
        """بررسی و مدیریت مصرف حافظه"""
        if self._processed_count % self._check_memory_interval == 0:
            try:
                import psutil
                import gc
                process = psutil.Process()
                if process.memory_info().rss > self._memory_limit:
                    gc.collect()
            except ImportError:
                pass

    def _executor(self):
        if should_use_threads():
            return concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=safe_mp_context(),
            initializer=_scan_worker_init,
            initargs=(self.currencies, self._shm, self._bits, self._hashes)
        )

    def _print_live(self, total_p, total_f, rate, last_keys, bloom_matches):
        print(f"\033[16A", end="")
        print(f"\033[K[PROGRESS] Processed: {total_p}  Found: {total_f}  Rate: {rate:.2f}/s")

        print(f"\033[K[GENERATED KEYS (Format Check)]")
        for priv_hex, addrs in last_keys:
            print(f"\033[K  PrivKey: {priv_hex}")
            if addrs:
                for cur, atype, addr in addrs:
                    valid = self._validator.validate_before_check(cur, atype, addr)
                    status = " [valid]" if valid else " [invalid]"
                    print(f"\033[K    {cur} ({atype}): {addr}{status}")
            else:
                print("\033[K    [no valid addresses generated]")

        print(f"\033[K[FOUND IN YOUR TARGET LIST]")
        if bloom_matches:
            for priv_hex, cur, atype, addr in bloom_matches:
                print(f"\033[K  PrivKey: {priv_hex}")
                print(f"\033[K    {cur} ({atype}): {addr} <<< TARGET FOUND!")
        else:
            print("\033[K  [No addresses from your target list in this batch]")

        for _ in range(2 - len(last_keys)):
            print("\033[K  PrivKey: [none]")
            print("\033[K    [no addresses]")

        print("\033[K")
        print("\033[K")

    def run_stream(self, key_stream: Iterable[bytes], max_batches: Optional[int] = None):
        print(f"[SCAN] Starting scan: {self.__class__.__name__}")
        total_f = 0
        total_p = 0
        start = time.time()
        run = 0
        pending: List[Any] = []
        last_report_time = start
        self.last_keys_global = deque(maxlen=2)
        self.bloom_matches_global = deque(maxlen=4)

        print("\n" * 16, end="")
        bm = BloomManager.instance()

        with self._executor() as pool:
            try:
                while max_batches is None or run < max_batches:
                    if should_use_threads():
                        batch = [next(key_stream) for _ in range(self.batch_size)]
                        found, proc, last_keys, found_keys, bloom_matches = _scan_worker_process(batch)
                        total_f += found
                        total_p += proc
                        self.last_keys_global.clear()
                        self.last_keys_global.extend(last_keys)
                        self.bloom_matches_global.clear()
                        self.bloom_matches_global.extend(bloom_matches)
                        
                        for key_data in found_keys:
                            self.results_manager.save_found_key(*key_data)
                            
                        self._processed_count += 1
                        self._check_memory_usage()
                        
                        # بررسی سلامت دوره‌ای Bloom Filter (با چک کردن وجود مانیتور)
                        if self.monitor and total_p % 100000 == 0:
                            self.monitor.periodic_check(force=True)
                            
                        now = time.time()
                        if now - last_report_time >= 1:
                            rate = total_p / max(1e-6, (now - start))
                            self._print_live(total_p, total_f, rate, list(self.last_keys_global), list(self.bloom_matches_global))
                            last_report_time = now
                        run += 1
                        continue

                    # پردازش حالت عادی...
                    while len(pending) >= self.max_workers * 2:
                        done, _ = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for f in list(done):
                            try:
                                found, proc, last_keys, found_keys, bloom_matches = f.result()
                                total_f += found
                                total_p += proc
                                self.last_keys_global.clear()
                                self.last_keys_global.extend(last_keys)
                                self.bloom_matches_global.clear()
                                self.bloom_matches_global.extend(bloom_matches)
                                
                                for key_data in found_keys:
                                    self.results_manager.save_found_key(*key_data)
                                    
                                self._processed_count += 1
                                self._check_memory_usage()
                                
                                # بررسی سلامت دوره‌ای (با چک کردن وجود مانیتور)
                                if self.monitor and total_p % 100000 == 0:
                                    self.monitor.periodic_check(force=True)
                                    
                            finally:
                                pending.remove(f)

                        now = time.time()
                        if now - last_report_time >= 1:
                            rate = total_p / max(1e-6, (now - start))
                            self._print_live(total_p, total_f, rate, list(self.last_keys_global), list(self.bloom_matches_global))
                            last_report_time = now

                    batch = [next(key_stream) for _ in range(self.batch_size)]
                    fut = pool.submit(_scan_worker_process, batch)
                    pending.append(fut)
                    run += 1

                    done = [f for f in pending if f.done()]
                    for f in done:
                        try:
                            found, proc, last_keys, found_keys, bloom_matches = f.result()
                            total_f += found
                            total_p += proc
                            self.last_keys_global.clear()
                            self.last_keys_global.extend(last_keys)
                            self.bloom_matches_global.clear()
                            self.bloom_matches_global.extend(bloom_matches)
                            
                            for key_data in found_keys:
                                self.results_manager.save_found_key(*key_data)
                                
                            self._processed_count += 1
                            self._check_memory_usage()
                                
                        finally:
                            pending.remove(f)

                    now = time.time()
                    if now - last_report_time >= 1:
                        rate = total_p / max(1e-6, (now - start))
                        self._print_live(total_p, total_f, rate, list(self.last_keys_global), list(self.bloom_matches_global))
                        last_report_time = now

            finally:
                if pending:
                    concurrent.futures.wait(pending)
                    for f in pending:
                        try:
                            ff, pp, last_keys, found_keys, bloom_matches = f.result()
                            total_f += ff
                            total_p += pp
                            self.last_keys_global.clear()
                            self.last_keys_global.extend(last_keys)
                            self.bloom_matches_global.clear()
                            self.bloom_matches_global.extend(bloom_matches)
                            
                            for key_data in found_keys:
                                self.results_manager.save_found_key(*key_data)
                                
                        except Exception:
                            pass
                try:
                    bm.shutdown()
                except Exception:
                    pass

        print(f"[DONE] Scan finished. Processed: {total_p} Found: {total_f}")