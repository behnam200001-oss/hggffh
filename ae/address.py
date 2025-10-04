import hashlib
import base58
from coincurve import PrivateKey
from eth_keys import keys

# Compatible bech32 import
try:
    from bech32 import bech32_encode, bech32_decode, convertbits, Encoding
    HAVE_ENCODING = True
except ImportError:
    import bech32
    bech32_encode = bech32.bech32_encode
    bech32_decode = bech32.bech32_decode
    convertbits = bech32.convertbits
    Encoding = None
    HAVE_ENCODING = False


class AddressGenerator:
    def __init__(self, currencies=['Bitcoin']):
        self.currencies = currencies

    def iter_addresses_for_key(self, priv_bytes):
        if 'Bitcoin' in self.currencies:
            yield from self._btc(priv_bytes)
        if 'Ethereum' in self.currencies:
            yield ('Ethereum', 'eth', str(self._eth(priv_bytes)))

    def _btc(self, priv):
        sk = PrivateKey(priv)

        # Compressed P2PKH
        pub_c = sk.public_key.format(compressed=True)
        h160 = self._hash160(pub_c)
        yield ('Bitcoin', 'p2pkh_c', base58.b58encode_check(b'\x00' + h160).decode())

        # P2SH-P2WPKH
        redeem = b'\x00\x14' + h160
        rs = hashlib.new('ripemd160', hashlib.sha256(redeem).digest()).digest()
        yield ('Bitcoin', 'p2sh_p2wpkh', base58.b58encode_check(b'\x05' + rs).decode())

        # Bech32 (SegWit v0)
        yield ('Bitcoin', 'bech32', self._bech32('bc', 0, h160))

        # Uncompressed P2PKH
        pub_u = sk.public_key.format(compressed=False)
        yield ('Bitcoin', 'p2pkh_u', base58.b58encode_check(b'\x00' + self._hash160(pub_u)).decode())

        # Taproot (Bech32m v1)
        xonly = pub_c[1:]
        yield ('Bitcoin', 'taproot', self._bech32('bc', 1, xonly, bech32m=True))

    def _hash160(self, b):
        return hashlib.new('ripemd160', hashlib.sha256(b).digest()).digest()

    def _bech32(self, hrp, ver, prog, bech32m=False):
        data = [ver] + list(convertbits(prog, 8, 5, True))
        if HAVE_ENCODING:
            enc = Encoding.BECH32M if (bech32m or ver == 1) else Encoding.BECH32
            return bech32_encode(hrp, data, enc)
        else:
            return bech32_encode(hrp, data)

    def _eth(self, priv):
        return keys.PrivateKey(priv).public_key.to_checksum_address()
