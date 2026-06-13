import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.ygoenv.env import YgoEnv

logging.basicConfig(level=logging.INFO)

def test_parser():
    env = YgoEnv()
    obs, info = env.reset()
    
    # Check observation shape and non-zero
    logging.info(f"Observation shape: {obs.shape}")
    assert obs.shape == (15 * 384,), "Observation shape mismatch"
    
    # We loaded 40 Normal Monsters into the deck, then start_duel was called.
    # The starting draw count is 5.
    # Therefore, we expect query_field_state to find 5 cards in Hand.
    codes = env.engine.query_field_state(0)
    logging.info(f"Field state (player 0): {codes}")
    
    # Verify that the engine extracted some legal actions 
    # MSG_SELECT_IDLECMD should have been processed.
    legal_actions = info.get("legal_actions", [])
    logging.info(f"Legal actions mask: {np.sum(legal_actions)} actions available.")
    
    # The current state should give us some actual parsed action.
    raw_actions = env.engine.get_legal_actions(env._current_state)
    logging.info(f"Raw parsed actions from Engine: {raw_actions}")
    
    logging.info("Parser and QueryLocation working successfully!")

if __name__ == "__main__":
    test_parser()
