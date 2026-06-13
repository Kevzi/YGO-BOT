import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.ygoenv.wrapper import YgoEngine

logging.basicConfig(level=logging.INFO)

def test_translator():
    engine = YgoEngine()
    
    # Simulate current_state_actions as returned by wrapper._extract_actions
    current_state_actions = [{
        "action_type": 1,
        "source": "engine_process",
        "msg": "MSG_SELECT_IDLECMD",
        "player": 0,
        "choices": [
            # 1er Normal Summon disponible
            {"type": "SUMMON", "engine_type": 0, "engine_index": 0, "action_idx": 0, "code": 89631139, "loc": 0x02, "seq": 0},
            # 2e SpSummon disponible
            {"type": "SPSUMMON", "engine_type": 1, "engine_index": 1, "action_idx": 21, "code": 46986414, "loc": 0x02, "seq": 1},
            # To End Phase
            {"type": "END_PHASE", "engine_type": 7, "engine_index": 0, "action_idx": 151}
        ]
    }]
    
    # Mocking self.lib.OCG_DuelSetResponse
    set_response_called_with = None
    
    class MockLib:
        def OCG_DuelSetResponse(self, ptr, buffer):
            nonlocal set_response_called_with
            set_response_called_with = buffer.raw
            
    engine.lib = MockLib()
    engine._duel_valid = True
    engine.duel_ptr = None
    
    import struct
    
    # Test 1: SUMMON (engine_type 0, engine_index 0)
    # Expected value: (0 << 16) | 0 = 0
    engine.translate_and_set_response(0, current_state_actions)
    val = struct.unpack("<I", set_response_called_with[:4])[0]
    logging.info(f"Test 1 (SUMMON) response value: {val} (expected: 0)")
    assert val == 0
    
    # Test 2: SPSUMMON (engine_type 1, engine_index 1)
    # Expected value: (1 << 16) | 1 = 65537
    engine.translate_and_set_response(21, current_state_actions)
    val = struct.unpack("<I", set_response_called_with[:4])[0]
    logging.info(f"Test 2 (SPSUMMON) response value: {val} (expected: 65537)")
    assert val == 65537
    
    # Test 3: END_PHASE (engine_type 7, engine_index 0)
    # Expected value: (0 << 16) | 7 = 7
    engine.translate_and_set_response(151, current_state_actions)
    val = struct.unpack("<I", set_response_called_with[:4])[0]
    logging.info(f"Test 3 (END_PHASE) response value: {val} (expected: 7)")
    assert val == 7
    
    # Simulate MSG_SELECT_BATTLECMD
    current_state_actions_battle = [{
        "action_type": 1,
        "source": "engine_process",
        "msg": "MSG_SELECT_BATTLECMD",
        "player": 0,
        "choices": [
            # 1er Attaquant
            {"type": "ATTACK", "engine_type": 1, "engine_index": 0, "action_idx": 153, "code": 89631139, "loc": 0x04, "seq": 0, "direct": 1},
            # To M2
            {"type": "MAIN_PHASE_2", "engine_type": 2, "engine_index": 0, "action_idx": 190}
        ]
    }]
    
    # Test 4: ATTACK (engine_type 1, engine_index 0)
    # Expected value: (0 << 16) | 1 = 1
    engine.translate_and_set_response(153, current_state_actions_battle)
    val = struct.unpack("<I", set_response_called_with[:4])[0]
    logging.info(f"Test 4 (ATTACK) response value: {val} (expected: 1)")
    assert val == 1
    
    # Test 5: TO_M2 (engine_type 2, engine_index 0)
    # Expected value: (0 << 16) | 2 = 2
    engine.translate_and_set_response(190, current_state_actions_battle)
    val = struct.unpack("<I", set_response_called_with[:4])[0]
    logging.info(f"Test 5 (TO_M2) response value: {val} (expected: 2)")
    assert val == 2
    
    # Simulate MSG_SELECT_CARD
    current_state_actions_select_card = [{
        "action_type": 1,
        "source": "engine_process",
        "msg": "MSG_SELECT_CARD",
        "player": 0,
        "min": 1,
        "max": 1,
        "cancelable": 0,
        "choices": [
            # 1ere Cible
            {"type": "SELECT_CARD", "engine_type": 0, "engine_index": 0, "action_idx": 191, "code": 89631139, "loc": 0x04, "seq": 0, "pos": 1},
            # 2e Cible
            {"type": "SELECT_CARD", "engine_type": 0, "engine_index": 1, "action_idx": 192, "code": 3201284, "loc": 0x04, "seq": 1, "pos": 1}
        ]
    }]
    
    # Test 6: SELECT_CARD (action 192, min=1)
    # Expected value: Un buffer avec \x01
    engine.translate_and_set_response(192, current_state_actions_select_card)
    val = struct.unpack("<1B", set_response_called_with[:1])[0]
    logging.info(f"Test 6 (SELECT_CARD) response value: {val} (expected: 1)")
    assert val == 1
    
    # Test 7: SELECT_CARD min=2
    current_state_actions_select_card[0]["min"] = 2
    engine.translate_and_set_response(191, current_state_actions_select_card)
    val = struct.unpack("<2B", set_response_called_with[:2])
    logging.info(f"Test 7 (SELECT_CARD min=2) response value: {val} (expected: (0, 1))")
    assert val == (0, 1)

    logging.info("Action Translator works successfully!")

if __name__ == "__main__":
    test_translator()
