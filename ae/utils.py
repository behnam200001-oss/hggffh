import sys
import multiprocessing as mp
import hashlib

def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

def safe_mp_context():
    try:
        return mp.get_context("spawn")
    except Exception:
        return mp

def should_use_threads() -> bool:
    return in_notebook()

def normalize_address(addr: str) -> bytes:
    """
    نرمال‌سازی آدرس برای کدگذاری یکنواخت و قطعی
    این تابع تضمین می‌کند که نمایش بایتی هر آدرس همیشه یکسان است
    """
    if not addr or not isinstance(addr, str):
        return b''
    
    normalized_addr = addr.strip().lower()
    byte_representation = normalized_addr.encode('utf-8', errors='replace')
    return byte_representation

def validate_address_format(currency: str, addr_type: str, address: str) -> bool:
    """اعتبارسنجی اولیه فرمت آدرس"""
    if not address or not isinstance(address, str):
        return False
    
    if currency == 'Bitcoin':
        if addr_type in ('p2pkh_c', 'p2pkh_u', 'p2sh_p2wpkh'):
            return len(address) >= 26 and len(address) <= 35
        elif addr_type == 'bech32':
            return address.startswith('bc1') and len(address) >= 14
        elif addr_type == 'taproot':
            return address.startswith('bc1p') and len(address) >= 14
    elif currency == 'Ethereum':
        return address.startswith('0x') and len(address) == 42
    
    return True