"""Reversible decimal-text storage for UPF files; no floating-point conversion.

This is a storage codec, not a PAW Hamiltonian adapter. Decoding reproduces the
original bytes, including every decimal digit, sign, exponent and whitespace.
"""
from __future__ import annotations

from array import array
import hashlib
import json
import re
import struct
import sys

MAGIC = b'SDFTUPFT1\0'
NUMBER = re.compile(rb'([+-]?)([0-9]+)\.([0-9]+)([EeDd])([+-]?)([0-9]+)')
MASK = (1 << 64) - 1


def _little_endian(values: array) -> bytes:
    copy = array(values.typecode, values)
    if sys.byteorder != 'little':
        copy.byteswap()
    return copy.tobytes()


def _shuffle(raw: bytes, width: int) -> bytes:
    return b''.join(raw[i::width] for i in range(width))


def _unshuffle(raw: bytes, width: int) -> bytes:
    if len(raw) % width:
        raise ValueError('invalid shuffled array size')
    result = bytearray(len(raw))
    count = len(raw) // width
    for i in range(width):
        result[i::width] = raw[i * count:(i + 1) * count]
    return bytes(result)


def encode_upf(raw: bytes) -> bytes:
    if b'\0' in raw:
        raise ValueError('UPF contains NUL bytes')
    formats: list[tuple] = []
    format_index: dict[tuple, int] = {}
    ids = array('H')
    mantissa_delta = array('Q')
    exponent = array('H')
    skeleton = bytearray()
    last = 0
    prior = 0
    for match in NUMBER.finditer(raw):
        sign, before, after, marker, esign, edigits = match.groups()
        digits = before + after
        # Leave exceptional tokens in the original skeleton unchanged.
        if len(digits) > 19 or len(edigits) > 4:
            continue
        mantissa = int(digits)
        power = int(edigits)
        if mantissa > MASK or power > 65535:
            continue
        template = (sign.decode(), len(before), len(after), marker.decode(),
                    esign.decode(), len(edigits))
        if template not in format_index:
            format_index[template] = len(formats)
            formats.append(template)
        if len(formats) > 65535:
            raise ValueError('too many decimal formats')
        skeleton.extend(raw[last:match.start()])
        skeleton.append(0)
        ids.append(format_index[template])
        mantissa_delta.append((mantissa - prior) & MASK)
        exponent.append(power)
        prior = mantissa
        last = match.end()
    skeleton.extend(raw[last:])
    metadata = json.dumps({'formats': formats,
                           'original_sha256': hashlib.sha256(raw).hexdigest(),
                           'original_bytes': len(raw)}, separators=(',', ':')).encode()
    header = struct.pack('<IQQ', len(metadata), len(ids), len(skeleton))
    return (MAGIC + header + metadata + skeleton + _shuffle(_little_endian(ids), 2)
            + _shuffle(_little_endian(mantissa_delta), 8)
            + _shuffle(_little_endian(exponent), 2))


def decode_upf(encoded: bytes, *, max_original_bytes: int = 64 * 1024**2) -> bytes:
    if not encoded.startswith(MAGIC):
        raise ValueError('unsupported UPF storage format')
    offset = len(MAGIC)
    metadata_size, count, skeleton_size = struct.unpack_from('<IQQ', encoded, offset)
    offset += 20
    if metadata_size > 1024**2 or skeleton_size > max_original_bytes:
        raise ValueError('UPF storage exceeds decode bounds')
    if offset + metadata_size + skeleton_size + count * 12 != len(encoded):
        raise ValueError('UPF storage length mismatch')
    metadata = json.loads(encoded[offset:offset + metadata_size])
    if metadata['original_bytes'] > max_original_bytes:
        raise ValueError('restored UPF exceeds decode bound')
    offset += metadata_size
    skeleton = encoded[offset:offset + skeleton_size]
    offset += skeleton_size
    arrays = []
    for code, width in [('H', 2), ('Q', 8), ('H', 2)]:
        values = array(code)
        values.frombytes(_unshuffle(encoded[offset:offset + count * width], width))
        if sys.byteorder != 'little':
            values.byteswap()
        arrays.append(values)
        offset += count * width
    ids, deltas, powers = arrays
    chunks = skeleton.split(b'\0')
    if len(chunks) != count + 1:
        raise ValueError('UPF numeric token count mismatch')
    result = bytearray(chunks[0])
    mantissa = 0
    for i in range(count):
        mantissa = (mantissa + deltas[i]) & MASK
        sign, before, after, marker, esign, ewidth = metadata['formats'][ids[i]]
        digits = str(mantissa).zfill(before + after)
        token = sign + digits[:before] + '.' + digits[before:] + marker + esign + str(powers[i]).zfill(ewidth)
        result.extend(token.encode())
        result.extend(chunks[i + 1])
        if len(result) > max_original_bytes:
            raise ValueError('decoded UPF exceeds bound')
    raw = bytes(result)
    if (len(raw) != metadata['original_bytes'] or
            hashlib.sha256(raw).hexdigest() != metadata['original_sha256']):
        raise ValueError('restored UPF byte identity failed')
    return raw
