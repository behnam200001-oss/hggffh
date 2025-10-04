from typing import Optional
from ..strategies import AdvancedKeyGenerator
from .base import BaseScanner

class WeakScanner(BaseScanner):
    def __init__(self, currencies=['Bitcoin'], batch_size=5000, max_workers: Optional[int] = None, results_dir: str = "results"):
        super().__init__(currencies, batch_size, max_workers, results_dir)
        self.adv = AdvancedKeyGenerator()

    def run(self, strategy: str = "tiny_keys", max_batches: Optional[int] = None):
        stream = self.adv.stream(strategy)
        return self.run_stream(stream, max_batches=max_batches)