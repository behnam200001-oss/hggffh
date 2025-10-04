#!/usr/bin/env python3
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if __package__ is None or __package__ == "":
    sys.path.append(BASE_DIR)
    from ae.scanners.base import BaseScanner
    from ae.scanners.hybrid_ai_scanner import HybridAIScanner
    from ae.scanners.incremental_scanner import IncrementalScanner
    from ae.database import build_address_database
    from ae.bloom import BloomManager
else:
    from ..ae.scanners.base import BaseScanner
    from ..ae.scanners.hybrid_ai_scanner import HybridAIScanner
    from ..ae.scanners.incremental_scanner import IncrementalScanner
    from ..ae.database import build_address_database
    from ..ae.bloom import BloomManager

def random_key_stream():
    """Generate endless random 32-byte keys."""
    while True:
        yield os.urandom(32)

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
    scanner = BaseScanner(currencies=currencies, batch_size=batch_size, max_workers=workers)
    scanner.run_stream(random_key_stream())

def ai_search():
    print("\n[INFO] Starting AI-guided search...")
    currencies = choose_currencies()
    workers, batch_size = choose_workers_and_batch()
    strategy = choose_ai_strategy()
    scanner = HybridAIScanner(
        currencies=currencies,
        batch_size=batch_size,
        max_workers=workers,
        ai_algo=strategy
    )
    scanner.run()

def incremental_search():
    print("\n[INFO] Starting incremental search...")
    currencies = choose_currencies()
    workers, batch_size = choose_workers_and_batch()
    start_val, end_val = get_incremental_range()
    
    print(f"Searching range: 0x{start_val:X} to 0x{end_val:X}")
    
    scanner = IncrementalScanner(
        start=start_val,
        end=end_val,
        currencies=currencies,
        batch_size=batch_size,
        max_workers=workers
    )
    scanner.run()

def main_menu():
    while True:
        print("\n=== Main Menu ===")
        print("1) Build database")
        print("2) Random search")
        print("3) AI search")
        print("4) Incremental search")
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