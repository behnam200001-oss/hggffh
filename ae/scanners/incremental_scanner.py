from typing import Optional
from ..keygen import IncrementalKeyGenerator
from .base import BaseScanner

class IncrementalScanner(BaseScanner):
    def __init__(
        self,
        start: int,
        end: int,
        currencies=['Bitcoin'],
        batch_size=5000,
        max_workers: Optional[int] = None,
        results_dir: str = "results"
    ):
        # MODIFIED: Added a warning for CPU-bound generation
        print("[WARNING] IncrementalScanner uses CPU-bound key generation.")
        super().__init__(currencies, batch_size, max_workers, results_dir)
        self.keygen = IncrementalKeyGenerator(start, end)

    def run(self, max_batches: Optional[int] = None):
        stream = self.keygen.keys_in_range()
        return self.run_stream(stream, max_batches=max_batches)