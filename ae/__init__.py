"""
ae package: fast address scanning with shared Bloom, AI-guided search, and clean multiprocessing.
"""

from .database import setup_environment, build_address_database, vacuum_database
from .bloom import BloomManager, SharedBloom, attach_shared_bloom
from .address import AddressGenerator
from .keygen import RandomKeyGenerator
from .strategies import AdvancedKeyGenerator
from .ai_store import ModelStore
from .ai_engine import AILearner
from .drl_learner import DRLearner

from .scanners.random_scanner import RandomScanner
from .scanners.weak_scanner import WeakScanner
from .scanners.hybrid_ai_scanner import HybridAIScanner

__all__ = [
    "setup_environment", "build_address_database", "vacuum_database",
    "BloomManager", "SharedBloom", "attach_shared_bloom",
    "AddressGenerator", "RandomKeyGenerator", "AdvancedKeyGenerator",
    "ModelStore", "AILearner", "DRLearner",
    "RandomScanner", "WeakScanner", "HybridAIScanner"
]