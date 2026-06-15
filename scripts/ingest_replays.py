import os
import struct
import lzma
import io
import logging
import ctypes
import numpy as np

from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import GameTransition, DuelStats
from core.ygoenv.env import YgoEnv

class ReplayParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = {}
        self.player_decks = {0: {'main': [], 'extra': []}, 1: {'main': [], 'extra': []}}
        self.actions = []
        
    def parse(self):
        with open(self.filepath, 'rb') as f:
            data = f.read()
            
        # Parse header (32 bytes)
        magic, version, flag, seed, data_size, hash_val = struct.unpack('<IIIIII', data[:24])
        props = data[24:32]
        
        self.header = {
            'magic': magic,
            'version': version,
            'flag': flag,
            'seed': seed,
            'data_size': data_size,
            'hash': hash_val,
            'props': props
        }
        
        compressed_data = data[32:]
        try:
            uncompressed = lzma.decompress(compressed_data, format=lzma.FORMAT_ALONE)
        except lzma.LZMAError:
            uncompressed = lzma.decompress(compressed_data)
            
        self._parse_uncompressed(uncompressed)
        
    def _parse_uncompressed(self, data):
        reader = io.BytesIO(data)
        
        for player in [0, 1]:
            # Main deck
            main_size_bytes = reader.read(4)
            if not main_size_bytes: return
            main_size = struct.unpack('<I', main_size_bytes)[0]
            for _ in range(main_size):
                code = struct.unpack('<I', reader.read(4))[0]
                self.player_decks[player]['main'].append(code)
                
            # Extra deck
            extra_size_bytes = reader.read(4)
            if not extra_size_bytes: return
            extra_size = struct.unpack('<I', extra_size_bytes)[0]
            for _ in range(extra_size):
                code = struct.unpack('<I', reader.read(4))[0]
                self.player_decks[player]['extra'].append(code)
                
        # Read actions
        while True:
            length_byte = reader.read(1)
            if not length_byte:
                break
            length = struct.unpack('<B', length_byte)[0]
            if length == 0:
                continue
            action_data = reader.read(length)
            self.actions.append(action_data)


class ReplayIngester:
    def __init__(self, db_session: Session):
        self.db = db_session
        
    def ingest(self, filepath: str):
        parser = ReplayParser(filepath)
        parser.parse()
        
        # 1. Create a DuelStats entry to track this replay
        duel_stats = DuelStats(
            player_1_deck="ReplayDeck1",
            player_2_deck="ReplayDeck2",
            winner=None
        )
        self.db.add(duel_stats)
        self.db.commit()
        
        duel_id = duel_stats.id
        
        # 2. Setup YgoEnv
        env = YgoEnv(omniscience=True) # Full observation for ingestion
        env.reset()
        
        # We need to recreate the duel with the exact seed and decks from the replay
        env.engine.destroy_duel()
        
        # Override seed
        options = env.engine.lib.OCG_DuelOptions # Wait, options is local in create_duel
        
        # A bit hacky: we can recreate create_duel logic here or just rely on env methods
        # Let's recreate the duel correctly
        duel_ptr = ctypes.c_void_p()
        from core.ygoenv.wrapper import OCG_DuelOptions
        options = OCG_DuelOptions()
        options.seed[0] = parser.header['seed']
        options.team1.startingLP = 8000
        options.team1.startingDrawCount = 5
        options.team1.drawCountPerTurn = 1
        options.team2.startingLP = 8000
        options.team2.startingDrawCount = 5
        options.team2.drawCountPerTurn = 1
        
        options.cardReader = ctypes.cast(env.engine._cb_card_reader, ctypes.c_void_p)
        options.scriptReader = ctypes.cast(env.engine._cb_script_reader, ctypes.c_void_p)
        options.logHandler = ctypes.cast(env.engine._cb_log_handler, ctypes.c_void_p)
        
        env.engine.lib.OCG_CreateDuel(ctypes.byref(duel_ptr), ctypes.byref(options))
        env.engine.duel_ptr = duel_ptr
        env.engine._duel_valid = True
        
        # Add cards
        for player in [0, 1]:
            for loc_type, loc_val in [('main', 0x01), ('extra', 0x40)]:
                for seq, code in enumerate(parser.player_decks[player][loc_type]):
                    env.engine.add_card(code, player, loc_val, seq, 0x8)
                    
        env.engine.start_duel()
        env._current_state = {"phase": "START", "turn": 1}
        env._cached_mask = None
        
        # Step through the replay actions
        step_idx = 0
        
        for raw_action in parser.actions:
            # Advance engine to get current state observation
            legal_actions = env.get_legal_actions()
            
            current_player = 0
            if env._current_state_actions and len(env._current_state_actions) > 0:
                current_player = env._current_state_actions[0].get("player", 0)
                
            obs = env._get_observation(player=current_player)
            
            # Save transition (observation, raw action info)
            # Since action_data is binary, we can store it as hex or json array
            action_info = {"raw_bytes": raw_action.hex()}
            
            # Format observation for SQLite (JSON)
            obs_list = obs.tolist() if isinstance(obs, np.ndarray) else obs
            
            transition = GameTransition(
                duel_id=duel_id,
                step=step_idx,
                state={"observation": obs_list, "current_player": current_player},
                action=action_info,
                reward=0.0
            )
            self.db.add(transition)
            
            # Push response to engine
            buffer = ctypes.create_string_buffer(raw_action, len(raw_action))
            env.engine.lib.OCG_DuelSetResponse(env.engine.duel_ptr, buffer, len(raw_action))
            
            # Invalidate cached mask so it reads next state
            env._cached_mask = None
            step_idx += 1
            
        self.db.commit()
        env.close()

if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Ingest YGOPro replays (.yrp) into SQLite")
    parser.add_argument("--dir", type=str, default="data", help="Directory containing .yrp files")
    args = parser.parse_args()
    
    with SessionLocal() as db:
        ingester = ReplayIngester(db)
        # Find all replay files in dir
        yrp_files = []
        for ext in ["*.yrp", "*.yrpx", "*.bytes"]:
            yrp_files.extend(glob.glob(os.path.join(args.dir, "**", ext), recursive=True))
        
        logging.info(f"Found {len(yrp_files)} .yrp files to ingest.")
        for f in yrp_files:
            try:
                logging.info(f"Ingesting {f}...")
                ingester.ingest(f)
                logging.info(f"Successfully ingested {f}")
            except Exception as e:
                logging.error(f"Failed to ingest {f}: {e}")
