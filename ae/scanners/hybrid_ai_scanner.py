import time
import concurrent.futures
import multiprocessing as mp
from typing import Optional, Iterator, Dict, Any
import random

from ..keygen import RandomKeyGenerator
from ..strategies import AdvancedKeyGenerator
from ..ai_engine import AILearner
from ..utils import safe_mp_context, should_use_threads
from ..bloom import BloomManager, attach_shared_bloom
from ..address import AddressGenerator
from .base import BaseScanner, _scan_worker_init, _scan_worker_process


class HybridAIScanner(BaseScanner):
    """
    AI-guided hybrid scanner: adaptively mixes random keys and weak-key strategies
    using online learning (UCB, Thompson Sampling, or DRL). 
    """

    def __init__(
        self,
        currencies=['Bitcoin'],
        batch_size=5000,
        max_workers: Optional[int] = None,
        ai_algo: str = "ucb",
        use_gpu: bool = True,
        force_gpu: bool = True,  # MODIFIED: Added force_gpu parameter
        results_dir: str = "results"
    ):
        super().__init__(currencies, batch_size, max_workers, results_dir)
        # MODIFIED: Pass GPU settings to the key generator
        self.rand = RandomKeyGenerator(use_gpu=use_gpu, force_gpu=force_gpu)
        self.adv = AdvancedKeyGenerator()
        arms = ["random"] + self.adv.list_strategies()
        self.learner = AILearner(arms=arms, algo=ai_algo)
        self.ai_algo = ai_algo

    def _arm_stream(self, arm: str) -> Iterator[bytes]:
        if arm == "random":
            return self.rand.random_keys(batch_size=1)
        return self.adv.stream(arm)

    def run(self, max_batches: Optional[int] = None):
        print(f"[INFO] Starting AI-guided search with {self.ai_algo} algorithm...")
        streams: Dict[str, Iterator[bytes]] = {a: self._arm_stream(a) for a in self.learner.arms}

        total_f = 0
        total_p = 0
        start = time.time()
        last_report_time = start
        run = 0
        bm = BloomManager.instance()

        # Thread mode (Jupyter/Windows-safe fallback)
        if should_use_threads():
            print("[INFO] Running in thread mode...")
            addr = AddressGenerator(self.currencies)
            b = bm.get()
            local_bloom = attach_shared_bloom(b.shm_name, b.num_bits, b.num_hashes)
            try:
                while max_batches is None or run < max_batches:
                    arm = self.learner.select_arm(run + 1)
                    s = streams[arm]
                    batch = [next(s) for _ in range(self.batch_size)]

                    found, proc = 0, 0
                    last_keys = []
                    found_keys = []
                    bloom_matches = []
                    
                    for k in batch:
                        proc += 1
                        addresses = list(addr.iter_addresses_for_key(k))
                        last_keys.append((k.hex(), addresses))
                        if len(last_keys) > 2:
                            last_keys.pop(0)
                            
                        # Check all addresses
                        matched_addresses = []
                        for cur, atype, addr_str in addresses:
                            if isinstance(addr_str, str):
                                a_bytes = addr_str.encode('utf-8', errors='ignore')
                            else:
                                a_bytes = addr_str
                                
                            if local_bloom.contains(a_bytes):
                                matched_addresses.append((cur, atype, addr_str))
                        
                        if matched_addresses:
                            found += len(matched_addresses)
                            for cur, atype, addr_str in matched_addresses:
                                found_keys.append((k, cur, atype, addr_str))
                                
                                if len(bloom_matches) < 4:
                                    bloom_matches.append((k.hex(), cur, atype, addr_str))

                    total_f += found
                    total_p += proc
                    
                    # Update last keys for display
                    self.last_keys_global.clear()
                    self.last_keys_global.extend(last_keys)
                    self.bloom_matches_global.clear()
                    self.bloom_matches_global.extend(bloom_matches)
                    
                    for key_data in found_keys:
                        self.results_manager.save_found_key(*key_data)
                        
                    self.learner.update(arm, hits=found, total=proc, near=0)
                    
                    # Memory management
                    self._processed_count += 1
                    self._check_memory_usage()

                    now = time.time()
                    if now - last_report_time >= 1:
                        rate = total_p / max(1e-6, (now - start))
                        self._print_live(total_p, total_f, rate, list(self.last_keys_global), list(self.bloom_matches_global))
                        last_report_time = now

                    run += 1
            finally:
                try:
                    bm.shutdown()
                except Exception:
                    pass

            print(f"[DONE] AI search finished. Processed: {total_p} Found: {total_f}")
            return

        # Process pool mode with improved future management
        print("[INFO] Running in process mode...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=safe_mp_context(),
            initializer=_scan_worker_init,
            initargs=(self.currencies, self._shm, self._bits, self._hashes),
        ) as pool:
            futures = {}
            try:
                while max_batches is None or run < max_batches:
                    arm = self.learner.select_arm(run + 1)
                    s = streams[arm]
                    batch = [next(s) for _ in range(self.batch_size)]
                    
                    future = pool.submit(_scan_worker_process, batch)
                    futures[future] = arm
                    
                    # Process completed futures
                    completed = []
                    for f in list(futures.keys()):
                        if f.done():
                            try:
                                found, proc, last_keys, found_keys, bloom_matches = f.result()
                                total_f += found
                                total_p += proc
                                arm_name = futures.pop(f)
                                
                                # Update last keys for display
                                self.last_keys_global.clear()
                                self.last_keys_global.extend(last_keys)
                                self.bloom_matches_global.clear()
                                self.bloom_matches_global.extend(bloom_matches)
                                
                                self.learner.update(arm_name, hits=found, total=proc, near=0)
                                
                                for key_data in found_keys:
                                    self.results_manager.save_found_key(*key_data)
                                    
                                # Memory management
                                self._processed_count += 1
                                self._check_memory_usage()
                            except Exception as e:
                                print(f"[ERROR] Processing future: {e}")
                            completed.append(f)
                    
                    # Limit pending futures to prevent memory issues
                    while len(futures) >= self.max_workers * 2:
                        done, not_done = concurrent.futures.wait(
                            list(futures.keys())[:2],
                            return_when=concurrent.futures.FIRST_COMPLETED,
                            timeout=30
                        )
                        for f in done:
                            try:
                                found, proc, last_keys, found_keys, bloom_matches = f.result()
                                total_f += found
                                total_p += proc
                                arm_name = futures.pop(f)
                                
                                # Update last keys for display
                                self.last_keys_global.clear()
                                self.last_keys_global.extend(last_keys)
                                self.bloom_matches_global.clear()
                                self.bloom_matches_global.extend(bloom_matches)
                                
                                self.learner.update(arm_name, hits=found, total=proc, near=0)
                                
                                for key_data in found_keys:
                                    self.results_manager.save_found_key(*key_data)
                                    
                                # Memory management
                                self._processed_count += 1
                                self._check_memory_usage()
                            except Exception as e:
                                print(f"[ERROR] In waiting future: {e}")
                    
                    now = time.time()
                    if now - last_report_time >= 1:
                        rate = total_p / max(1e-6, (now - start))
                        self._print_live(total_p, total_f, rate, list(self.last_keys_global), list(self.bloom_matches_global))
                        last_report_time = now
                    
                    run += 1
            finally:
                # Wait for all pending futures
                if futures:
                    done, not_done = concurrent.futures.wait(futures.keys(), timeout=30)
                    for f in done:
                        try:
                            found, proc, last_keys, found_keys, bloom_matches = f.result()
                            total_f += found
                            total_p += proc
                            arm_name = futures[f]
                            
                            # Update last keys for display
                            self.last_keys_global.clear()
                            self.last_keys_global.extend(last_keys)
                            self.bloom_matches_global.clear()
                            self.bloom_matches_global.extend(bloom_matches)
                            
                            self.learner.update(arm_name, hits=found, total=proc, near=0)
                            for key_data in found_keys:
                                self.results_manager.save_found_key(*key_data)
                        except Exception as e:
                            print(f"[ERROR] In final future processing: {e}")
                try:
                    bm.shutdown()
                except Exception:
                    pass

        print(f"[DONE] AI search finished. Processed: {total_p} Found: {total_f}")