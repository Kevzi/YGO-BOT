import pytest
import numpy as np
import gymnasium as gym
from core.ygoenv.env import YgoEnv

def test_ygoenv_initialization():
    env = YgoEnv()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    # The observation space for now will be a flat array (e.g. 100 features)
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
            
        embed_loader.filepath = embed_path
        embed_loader.load()
        
        try:
            env = YgoEnv()
            obs, info = env.reset()
            
            # Test shape
            assert obs.shape == (env.NUM_CARDS * embed_loader._embedding_dim,)
            
            # Simulate unknown card ID by replacing one ID with 999999999
            # Since _get_observation currently simulates zeros for IDs,
            # we directly call the batch with an unknown ID
            card_ids = np.zeros(env.NUM_CARDS, dtype=np.int32)
            card_ids[0] = 999999999
            vectors = embed_loader.get_embeddings_batch(card_ids)
            obs_unknown = np.asarray(vectors).flatten()
            
            assert obs_unknown.shape == obs.shape
            # Ensure the unknown card resulted in zeros
            assert np.allclose(obs_unknown[:embed_loader._embedding_dim], 0.0)
        finally:
            embed_loader._is_loaded = False
            embed_loader._matrix = None
            embed_loader._card_to_idx = None
            embed_loader._embedding_dim = 0

