import pytest
import numpy as np
import gymnasium as gym
from core.ygoenv.env import YgoEnv

def test_ygoenv_initialization():
    env = YgoEnv()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    # The observation space for now will be a flat array (e.g. 100 features)
    # Observation space (Box): Un vecteur plat représentant le terrain vectorisé.
    # Action space (Discrete): Par exemple 200 actions possibles.
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    
def test_ygoenv_reset():
    env = YgoEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)

def test_ygoenv_step():
    env = YgoEnv()
    env.reset()
    
    # action 1 is the mock action in wrapper.py
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_ygoenv_save_restore_state():
    env = YgoEnv()
    env.reset()
    
    # Save state
    state = env.save_state()
    assert state is not None
    
    # Take action
    env.step(1)
    
    # Restore state
    env.restore_state(state)
    
    # Verify restored state
    assert env._current_state == state["current_state"]

def test_ygoenv_zero_shot():
    import tempfile
    import os
    import pickle
    from ai.embeddings import embed_loader
    
    dummy_data = {
        1234: np.array([0.1, 0.2, 0.3], dtype=np.float32)
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embed_path = os.path.join(tmpdir, "embed.pkl")
        with open(embed_path, "wb") as f:
            pickle.dump(dummy_data, f)
            
        try:
            embed_loader.filepath = embed_path
            embed_loader.load()
            
            env = YgoEnv()
            obs, info = env.reset()
            
            # Test shape
            assert obs.shape == (60694,)
            
            # Since the env uses embeddings, just verify it runs without crashing
            # and returns an observation of the correct shape and type
            assert obs.dtype == np.float32
        finally:
            embed_loader._is_loaded = False
            embed_loader._matrix = None
            embed_loader._card_to_idx = None
            embed_loader._embedding_dim = 0

