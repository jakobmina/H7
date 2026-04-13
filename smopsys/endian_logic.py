"""
endian_logic.py — Big-Endian Topological Serializer
Part of the H7 Metriplectic Framework.

Encodes physical states (momentum, winding, ternary weights) into 
standardized Big-Endian hex strings.
"""

import struct
from typing import Dict, Any

class HexadecimalSpecifications:
    FORMATS = {
        'uint8':   {'bits': 8,   'bytes': 1, 'hex_chars': 2,  'min_value': 0, 'max_value': 255},
        'uint16':  {'bits': 16,  'bytes': 2, 'hex_chars': 4,  'min_value': 0, 'max_value': 65535},
        'uint32':  {'bits': 32,  'bytes': 4, 'hex_chars': 8,  'min_value': 0, 'max_value': 4294967295},
        'uint64':  {'bits': 64,  'bytes': 8, 'hex_chars': 16, 'min_value': 0, 'max_value': 18446744073709551615},
        'uint128': {'bits': 128, 'bytes': 16,'hex_chars': 32, 'min_value': 0, 'max_value': 340282366920938463463374607431768211455},
    }

class BigEndianHexadecimalEncoder:
    @staticmethod
    def to_hex_uint8(value: int)   -> str: return f"{value & 0xFF:02X}"
    @staticmethod
    def to_hex_uint16(value: int)  -> str: return f"{value & 0xFFFF:04X}"
    @staticmethod
    def to_hex_uint32(value: int)  -> str: return f"{value & 0xFFFFFFFF:08X}"
    @staticmethod
    def to_hex_uint64(value: int)  -> str: return f"{value & 0xFFFFFFFFFFFFFFFF:016X}"
    @staticmethod
    def to_hex_uint128(value: int) -> str: return f"{value & ((1 << 128) - 1):032X}"
    
    @staticmethod
    def from_hex_uint16(hex_str: str) -> int: return int(hex_str, 16)

class TopologicalBigEndianEncoder:
    """
    Packs topology into 14 bits (MOD 6 Information Logistics).
    Bit Mapping:
      [0-2]: index - 1 (3 bits)
      [3-5]: pair - 1 (3 bits)
      [6]:   winding / 2 (1 bit)
      [7]:   reserved (0)
      [8]:   mapping (1 bit)
      [9-10]: ternary_weight bits (2 bits: -1->00, 0->01, +1->10)
      [11-13]: discrete_phase_fragment (3 bits)
    """
    
    TERNARY_TO_BITS = {-1: 0b00, 0: 0b01, 1: 0b10}
    BITS_TO_TERNARY = {0b00: -1, 0b01: 0, 0b10: 1}

    @staticmethod
    def pack_topology(index, pair, winding, mapping, ternary_weight, discrete_phase_fragment) -> int:
        packed = 0
        packed |= ((index - 1) & 0x7) << 0
        packed |= ((pair - 1) & 0x7) << 3
        packed |= ((winding // 2) & 0x1) << 6
        packed |= (mapping & 0x1) << 8
        packed |= (TopologicalBigEndianEncoder.TERNARY_TO_BITS[ternary_weight] & 0x3) << 9
        packed |= (discrete_phase_fragment & 0x7) << 11
        return packed

    @staticmethod
    def unpack_topology(value: int) -> Dict[str, Any]:
        return {
            'index': ((value >> 0) & 0x7) + 1,
            'pair':  ((value >> 3) & 0x7) + 1,
            'winding': ((value >> 6) & 0x1) * 2,
            'mapping': (value >> 8) & 0x1,
            'ternary_weight': TopologicalBigEndianEncoder.BITS_TO_TERNARY.get((value >> 9) & 0x3, 0),
            'discrete_phase_fragment': (value >> 11) & 0x7,
        }

    # Standard topology entries for MOD 6 system (n=0..6)
    topology_entries = [
        {'index': 1, 'pair': 6, 'winding': 0, 'mapping': 0, 'ternary_weight': 1,  'discrete_phase_fragment': 0},
        {'index': 5, 'pair': 2, 'winding': 0, 'mapping': 0, 'ternary_weight': 1,  'discrete_phase_fragment': 1},
        {'index': 3, 'pair': 4, 'winding': 0, 'mapping': 0, 'ternary_weight': 1,  'discrete_phase_fragment': 6},
        {'index': 4, 'pair': 3, 'winding': 2, 'mapping': 1, 'ternary_weight': -1, 'discrete_phase_fragment': 5},
        {'index': 5, 'pair': 2, 'winding': 2, 'mapping': 1, 'ternary_weight': -1, 'discrete_phase_fragment': 2},
        {'index': 6, 'pair': 1, 'winding': 2, 'mapping': 1, 'ternary_weight': -1, 'discrete_phase_fragment': 3},
        {'index': 2, 'pair': 3, 'winding': 0, 'mapping': 0, 'ternary_weight': 0,  'discrete_phase_fragment': 4},
    ]

class LittleEndianHexadecimalEncoder:
    """Encodes values to hex with little-endian byte ordering."""
    @staticmethod
    def to_hex_uint16(value: int) -> str:
        # Swap bytes: 0x1234 -> 0x3412
        return f"{((value & 0xFF) << 8) | ((value >> 8) & 0xFF):04X}"
    
    @staticmethod
    def from_hex_uint16(hex_str: str) -> int:
        val = int(hex_str, 16)
        return ((val & 0xFF) << 8) | ((val >> 8) & 0xFF)

class InformationLogisticsBridge:
    """The Bridge: Swaps endianness as a physical logistics operation."""
    @staticmethod
    def bridge_be_to_le(be_hex: str) -> str:
        val = BigEndianHexadecimalEncoder.from_hex_uint16(be_hex)
        return LittleEndianHexadecimalEncoder.to_hex_uint16(val)
    
    @staticmethod
    def bridge_le_to_be(le_hex: str) -> str:
        val = LittleEndianHexadecimalEncoder.from_hex_uint16(le_hex)
        return BigEndianHexadecimalEncoder.to_hex_uint16(val)
