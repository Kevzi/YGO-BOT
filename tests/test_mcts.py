import pytest
import numpy as np
import jax.numpy as jnp
from ai.mcts import MCTS, MCTSNode

class MockActorCritic:
    def __init__(self, act_dim=200):
        self.act_dim = act_dim
        
    def apply(self, params, carry, obs, prev_action, action_mask):
        # Fake evaluation
        new_carry = (carry[0] + 0.1, carry[1] + 0.1)
        logits = np.zeros((1, self.act_dim), dtype=np.float32)
        # On favorise l'action 5 si elle est légale
        if action_mask[0, 5]:
            logits[0, 5] = 2.0
        value = np.array([[0.5]], dtype=np.float32)
        return new_carry, logits, value

class MockEnv:
    def __init__(self):
        self.action_history = []
        self.step_count = 0
        
    def get_action_mask(self):
        mask = np.zeros(200, dtype=np.bool_)
        mask[0] = True
        mask[5] = True
        return mask
        
    def get_legal_actions(self):
        return self.get_action_mask()
        
    def clone(self):
        new_env = MockEnv()
        new_env.action_history = list(self.action_history)
        new_env.step_count = self.step_count
        return new_env
        
    def step(self, action):
        self.action_history.append(action)
        self.step_count += 1
        obs = np.ones((60694,), dtype=np.float32)
        
        terminated = self.step_count >= 3
        reward = 1.0 if terminated else 0.0
        truncated = False
        return obs, reward, terminated, truncated, {}

def test_mcts_search_independence():
    env = MockEnv()
    actor = MockActorCritic(act_dim=200)
    mcts = MCTS(actor, params=None, c_puct=1.25)
    
    carry = (jnp.zeros((1, 512)), jnp.zeros((1, 512)))
    obs = np.ones((60694,), dtype=np.float32)
    prev_action = 0
    
    root = mcts.search(env, obs, carry, prev_action, num_simulations=15)
    
    # Le réseau doit avoir simulé des chemins
    assert root.N == 15
    assert len(root.children) > 0
    
    # L'environnement de base doit rester intact (State Cloning isolation)
    assert env.step_count == 0
    assert len(env.action_history) == 0

def test_mcts_action_probs():
    env = MockEnv()
    actor = MockActorCritic(act_dim=200)
    mcts = MCTS(actor, params=None, c_puct=1.25)
    
    carry = (jnp.zeros((1, 512)), jnp.zeros((1, 512)))
    obs = np.ones((60694,), dtype=np.float32)
    
    root = mcts.search(env, obs, carry, prev_action=0, num_simulations=20)
    probs = mcts.get_action_probs(root, temperature=1.0)
    
    # Les probabilités doivent sommer à 1
    assert np.abs(np.sum(probs) - 1.0) < 1e-5
    
    # Le réseau favorisait l'action 5, PUCT devrait l'avoir explorée davantage
    assert probs[5] > probs[0]
