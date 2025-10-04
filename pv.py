#!/usr/bin/env python3
import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if __package__ is None or __package__ == "":
    sys.path.append(BASE_DIR)
    from ae.scanners.base import BaseScanner
    from ae.scanners.hybrid_ai_scanner import HybridAIScanner
    from ae.scanners.incremental_scanner import IncrementalScanner
    from ae.scanners.random_scanner import RandomScanner
    from ae.database import build_address_database
    from ae.bloom import BloomManager
else:
    from ..ae.scanners.base import BaseScanner
    from ..ae.scanners.hybrid_ai_scanner import HybridAIScanner
    from ..ae.scanners.incremental_scanner import IncrementalScanner
    from ..ae.scanners.random_scanner import RandomScanner
    from ..ae.database import build_address_database
    from ..ae.bloom import BloomManager

# MODIFIED: GPU check function added
def check_gpu_availability():
    """Check for CUDA availability and exit if not found."""
    print("[INFO] Checking for GPU availability...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("="*60)
            print("[FATAL ERROR] CUDA is not available on this system!")
            print("This project is configured to run exclusively on GPU.")
            print("Please install PyTorch with CUDA support:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("="*60)
            sys.exit(1)
        
        print(f"[OK] GPU Found: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA Version: {torch.version.cuda}")
        print("[INFO] Project will run exclusively on GPU.")
        print("-"*60)
    except ImportError:
        print("="*60)
        print("[FATAL ERROR] PyTorch is not installed!")
        print("Please install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("="*60)
        sys.exit(1)

def random_key_stream():
    """Generate endless random 32-byte keys using GPU."""
    # MODIFIED: Now uses the GPU-accelerated generator
    from ae.keygen import RandomKeyGenerator
    keygen = RandomKeyGenerator(use_gpu=True, force_gpu=True)
    yield from keygen.random_keys(batch_size=1)

def build_database():
    """
    Build SQLite database from addresses.txt and then build Bloom filter.
    """
    print("\n[INFO] Building database...")

    addresses_file = os.path.join(DATA_DIR, "addresses.txt")
    if not os.path.exists(addresses_file):
        print(f"[ERROR] File not found: {addresses_file}")
        print("[HINT] Create addresses.txt in pv/data and put one address per line.")
        return

    try:
        build_address_database(addresses_file, batch_size=100000, vacuum_after=True)
        print("[OK] SQLite database created successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to build SQLite database: {e}")
        return

    try:
        bm = BloomManager.instance()
        bm.rebuild()
        print("[OK] Bloom filter built successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to build Bloom filter: {e}")
        return

    print("[DONE] Database build complete.\n")

def choose_currencies():
    """Ask user which currencies to scan."""
    currencies = []
    print("Select currencies to scan:")
    print("1) Bitcoin")
    print("2) Ethereum")
    print("3) Both")
    choice = input("Enter choice: ").strip()
    if choice == "1":
        currencies.append("Bitcoin")
    elif choice == "2":
        currencies.append("Ethereum")
    elif choice == "3":
        currencies.extend(["Bitcoin", "Ethereum"])
    else:
        print("Invalid choice, defaulting to Bitcoin.")
        currencies.append("Bitcoin")
    return currencies

def choose_workers_and_batch():
    """Ask user for worker count and batch size."""
    try:
        workers = int(input("Enter number of workers (default: auto): ").strip() or 0)
    except ValueError:
        workers = 0
    try:
        batch_size = int(input("Enter batch size (default: 5000): ").strip() or 5000)
    except ValueError:
        batch_size = 5000
    return workers if workers > 0 else None, batch_size

def choose_ai_strategy():
    """Ask user which AI strategy to use."""
    print("Select AI strategy:")
    print("1) UCB (Upper Confidence Bound)")
    print("2) Thompson Sampling")
    print("3) DRL (Deep Reinforcement Learning)")
    choice = input("Enter choice: ").strip()
    if choice == "1":
        return "ucb"
    elif choice == "2":
        return "thompson"
    elif choice == "3":
        return "drl"
    else:
        print("Invalid choice, defaulting to UCB.")
        return "ucb"

def get_incremental_range():
    """Ask user for incremental search range."""
    print("Enter the search range (hexadecimal format):")
    
    while True:
        start_hex = input("Start value (hex): ").strip()
        if not start_hex:
            print("Please enter a valid hex value.")
            continue
        
        try:
            start_val = int(start_hex, 16)
            break
        except ValueError:
            print("Invalid hex format. Please try again.")
    
    while True:
        end_hex = input("End value (hex): ").strip()
        if not end_hex:
            print("Please enter a valid hex value.")
            continue
        
        try:
            end_val = int(end_hex, 16)
            if end_val <= start_val:
                print("End value must be greater than start value.")
                continue
            break
        except ValueError:
            print("Invalid hex format. Please try again.")
    
    return start_val, end_val

def random_search():
    print("\n[INFO] Starting random search...")
    currencies = choose_currencies()
    workers, batch_size = choose_workers_and_batch()
    # MODIFIED: Scanner is now initialized with forced GPU settings
    scanner = RandomScanner(
        currencies=currencies, 
        batch_size=batch_size, 
        max_workers=workers,
        use_gpu=True,
        force_gpu=True
    )
    scanner.run()

def ai_search():
    print("\n[INFO] Starting AI-guided search...")
    currencies = choose_currencies()
    workers, batch_size = choose_workers_and_batch()
    strategy = choose_ai_strategy()
    # MODIFIED: Scanner is now initialized with forced GPU settings
    scanner = HybridAIScanner(
        currencies=currencies,
        batch_size=batch_size,
        max_workers=workers,
        ai_algo=strategy,
        use_gpu=True,
        force_gpu=True
    )
    scanner.run()

def incremental_search():
    print("\n[INFO] Starting incremental search...")
    print("[NOTE] Incremental key generation is CPU-bound, but the scanning process will use GPU workers where applicable.")
    currencies = choose_currencies()
    workers, batch_size = choose_workers_and_batch()
    start_val, end_val = get_incremental_range()
    
    print(f"Searching range: 0x{start_val:X} to 0x{end_val:X}")
    
    # MODIFIED: Scanner is now initialized with forced GPU settings
    scanner = IncrementalScanner(
        start=start_val,
        end=end_val,
        currencies=currencies,
        batch_size=batch_size,
        max_workers=workers
    )
    scanner.run()

def main_menu():
    # MODIFIED: GPU check is called first
    check_gpu_availability()
    
    while True:
        print("\n=== Main Menu ===")
        print("1) Build database")
        print("2) Random search (GPU)")
        print("3) AI search (GPU)")
        print("4) Incremental search (CPU Keygen)")
        print("0) Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            build_database()
        elif choice == "2":
            random_search()
        elif choice == "3":
            ai_search()
        elif choice == "4":
            incremental_search()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()