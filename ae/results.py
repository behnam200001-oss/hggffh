import os
from datetime import datetime
from typing import List, Dict, Any

class ResultsManager:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # فایل تکست برای ذخیره نتایج
        self.txt_path = os.path.join(output_dir, "found_keys.txt")
        
        # اگر فایل وجود ندارد، هدر اضافه کن
        if not os.path.exists(self.txt_path):
            with open(self.txt_path, 'w', encoding='utf-8') as f:
                f.write("# Key Search Results\n")
                f.write("# Format: Timestamp | Private Key (hex) | Currency | Address Type | Address\n")
                f.write("# " + "="*80 + "\n")
    
    def save_found_key(self, private_key: bytes, currency: str, address_type: str, address: str):
        """ذخیره کلید یافت شده در فایل تکست"""
        private_key_hex = private_key.hex()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # ذخیره در فایل تکست
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} | {private_key_hex} | {currency} | {address_type} | {address}\n")
        
        # نمایش در کنسول
        print(f"\n✅ کلید یافت شده و ذخیره شد:")
        print(f"   Private Key: {private_key_hex}")
        print(f"   Address: {address}")
        print(f"   Currency: {currency} - Type: {address_type}")
    
    def get_found_keys(self) -> List[Dict[str, Any]]:
        """دریافت تمام کلیدهای یافت شده از فایل"""
        keys = []
        if os.path.exists(self.txt_path):
            with open(self.txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split(' | ')
                    if len(parts) >= 5:
                        keys.append({
                            'timestamp': parts[0],
                            'private_key': parts[1],
                            'currency': parts[2],
                            'address_type': parts[3],
                            'address': parts[4]
                        })
        return keys