from typing import Optional
from ..keygen import RandomKeyGenerator
from .base import BaseScanner

class RandomScanner(BaseScanner):
    def __init__(
        self, 
        currencies=['Bitcoin'], 
        batch_size=5000, 
        max_workers: Optional[int] = None,
        results_dir: str = "results",
        use_gpu: bool = True,
        force_gpu: bool = True  # MODIFIED: Added force_gpu parameter
    ):
        super().__init__(currencies, batch_size, max_workers, results_dir)
        # MODIFIED: Pass GPU settings to the key generator
        self.keygen = RandomKeyGenerator(use_gpu=use_gpu, force_gpu=force_gpu)

    def run(self, max_batches: Optional[int] = None):
        stream = self.keygen.random_keys(batch_size=1)
        return self.run_stream(stream, max_batches=max_batches)