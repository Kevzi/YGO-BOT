import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ai.mcts import MCTS, MCTSNode
from ai.ppo import PPOAgent

class MockEnv:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
    def save_state(self):
        return {"mock_state": 1}
        
    def restore_state(self, state):
        pass
        
    def get_obs(self):
        return np.ones((self.obs_dim,), dtype=np.float32)
        
    def get_legal_actions(self):
        # Mask the first action to test masking logic
        mask = np.ones((self.act_dim,), dtype=np.bool_)
        if self.act_dim > 0:
            mask[0] = False
        return mask
        
    def step(self, action):
        obs = np.ones((self.obs_dim,), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"legal_actions": self.get_legal_actions()}
        return obs, reward, terminated, truncated, info

def test_mcts_initialization():
    mcts = MCTS(num_simulations=8, c_puct=1.0)
    assert mcts.num_simulations == 8
    assert mcts.c_puct == 1.0

def test_mcts_search():
    obs_dim = 10
    act_dim = 5
    hidden_dim = 16
    
    env = MockEnv(obs_dim, act_dim)
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    rng = jax.random.PRNGKey(42)
    params = agent.init_params(rng)
    hidden_state = agent.init_hidden_state(1)
    
    # Extract the single state since we're operating without batch dimension in MCTS loop usually
    h, c = hidden_state
    hidden_state_single = (h[0], c[0])
    
    mcts = MCTS(num_simulations=4)
    
    # Test search returns an action and improved policy
    action, policy_improved = mcts.search(
        env=env,
        agent=agent,
        params=params,
        hidden_state=hidden_state_single,
        obs=env.get_obs(),
        legal_actions=env.get_legal_actions()
    )
    
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < act_dim
    assert policy_improved.shape == (act_dim,)
    assert np.isclose(np.sum(policy_improved), 1.0)
