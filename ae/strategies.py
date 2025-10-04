import os
import json
import random
from typing import Dict, List, Tuple, Iterator

from .keygen import CURVE_ORDER
from .ai_store import ModelStore

DEFAULT_WEAK_PATH = 'config/weak_key_strategies.json'

class AdvancedKeyGenerator:
    """
    Generates weak/structured keys from configured ranges and patterns with improved random sampling.
    """

    def __init__(self, config_path: str = DEFAULT_WEAK_PATH):
        self.strategies: Dict[str, Dict] = {}
        self._load_defaults()
        self._load_config(config_path)
        self.model_store = ModelStore()

    def _load_defaults(self):
        self.strategies = {
            'tiny_keys': {'ranges': [(1, 500000)], 'priority': 10},
            'high_range_keys': {'ranges': [(CURVE_ORDER - 500000, CURVE_ORDER - 1)], 'priority': 7},
            'patterned_keys': {'ranges': [((1 << 124), (1 << 124) + 500000)], 'priority': 8},
            'prime_numbers': {'ranges': [(2, 500000)], 'priority': 9}
        }

    def _load_config(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                cfg = json.load(f)
            for n, c in cfg.items():
                self.strategies[n] = {
                    'ranges': [(int(a), int(b)) for a, b in c.get('ranges', [])],
                    'priority': int(c.get('priority', 5))
                }

    def list_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def stream(self, name: str) -> Iterator[bytes]:
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy: {name}")
        ranges: List[Tuple[int, int]] = self.strategies[name]['ranges']
        while True:
            r = random.choice(ranges)
            val = random.randint(r[0], r[1])
            yield val.to_bytes(32, 'big')

    def stream_weighted_mix(self) -> Iterator[bytes]:
        items = []
        for name, cfg in self.strategies.items():
            w = max(1, int(cfg.get('priority', 1)))
            items.extend([name] * w)
        
        while True:
            pick = random.choice(items)
            strategy_ranges = self.strategies[pick]['ranges']
            selected_range = random.choice(strategy_ranges)
            val = random.randint(selected_range[0], selected_range[1])
            yield val.to_bytes(32, 'big')