import os
import struct
import lzma
import pytest
from scripts.ingest_replays import ReplayParser

def create_mock_yrp(filepath):
    """Creates a fake valid .yrp file for testing."""
    magic = 0x31707279 # "yrp1"
    version = 0x133d
    flag = 0
    seed = 12345
    
    # Fake uncompressed data:
    # Player 0 deck: main: 1 card, extra: 0
    # Player 1 deck: main: 1 card, extra: 0
    # Then some actions
    uncompressed = bytearray()
    # P0 Main deck
    uncompressed.extend(struct.pack('<I', 1)) # size
    uncompressed.extend(struct.pack('<I', 89631139)) # BEWD
    # P0 Extra deck
    uncompressed.extend(struct.pack('<I', 0))
    
    # P1 Main deck
    uncompressed.extend(struct.pack('<I', 1))
    uncompressed.extend(struct.pack('<I', 89631139))
    # P1 Extra deck
    uncompressed.extend(struct.pack('<I', 0))
    
    # action 1 (length 1, value 0x5)
    uncompressed.extend(struct.pack('<B', 1))
    uncompressed.extend(struct.pack('<B', 0x5))
    
    compressed_data = lzma.compress(uncompressed, format=lzma.FORMAT_ALONE)
    
    data_size = len(compressed_data)
    hash_val = 0
    props = b'\x00' * 8 # YGOPro sets 8 bytes
    
    header = struct.pack('<IIIIII', magic, version, flag, seed, data_size, hash_val) + props
    
    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(compressed_data)

def test_parse_yrp(tmp_path):
    yrp_path = tmp_path / "test.yrp"
    create_mock_yrp(yrp_path)
    
    parser = ReplayParser(str(yrp_path))
    parser.parse()
    
    assert parser.header['seed'] == 12345
    assert len(parser.player_decks[0]['main']) == 1
    assert parser.player_decks[0]['main'][0] == 89631139
    assert len(parser.actions) == 1
    assert parser.actions[0] == b'\x05'
