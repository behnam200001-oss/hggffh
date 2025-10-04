import os
import sqlite3

DB_PATH = 'data/addresses.db'
TABLE_SQL = 'CREATE TABLE IF NOT EXISTS addresses (address TEXT PRIMARY KEY)'

def setup_environment():
    os.makedirs('data', exist_ok=True)
    create_addresses_database()
    print("Environment setup completed.")

def create_addresses_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(TABLE_SQL)
    conn.commit()
    conn.close()

def build_address_database(
    input_file: str,
    batch_size: int = 100_000,
    show_progress_every: int = 100_000,
    fast_mode: bool = True,
    vacuum_after: bool = False
):
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(TABLE_SQL)

    if fast_mode:
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=OFF")
            cur.execute("PRAGMA temp_store=MEMORY")
            cur.execute("PRAGMA cache_size=-200000")
        except Exception as e:
            print(f"PRAGMA tuning warning: {e}")

    total_lines = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            total_lines += 1

    inserted = 0
    skipped = 0
    seen = 0
    tx_count = 0
    conn.execute("BEGIN")
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                addr = line.strip()
                if not addr:
                    continue
                cur.execute("INSERT OR IGNORE INTO addresses(address) VALUES(?)", (addr,))
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    skipped += 1
                seen += 1
                tx_count += 1

                if tx_count >= batch_size:
                    conn.commit()
                    conn.execute("BEGIN")
                    tx_count = 0

                if show_progress_every and (i % show_progress_every == 0):
                    pct = (i/total_lines*100) if total_lines else 0
                    print(f"Processed {i:,}/{total_lines:,} ({pct:.2f}%) | Insert: {inserted:,} | Skip: {skipped:,}")

        conn.commit()
    finally:
        if fast_mode:
            try:
                cur.execute("PRAGMA synchronous=NORMAL")
            except Exception:
                pass
        conn.close()

    print(f"Database import finished. Seen: {seen:,}, Inserted: {inserted:,}, Duplicates: {skipped:,}")

    if vacuum_after:
        vacuum_database()

def vacuum_database():
    print("Running VACUUM...")
    conn = sqlite3.connect(DB_PATH)
    conn.isolation_level = None
    cur = conn.cursor()
    cur.execute("VACUUM")
    conn.close()
    print("VACUUM completed.")
