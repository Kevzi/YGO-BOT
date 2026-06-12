import pytest
import jax
import jax.numpy as jnp
import numpy as np
from ai.agent import DummyAgent, MCTSAgent
from ai.ppo import PPOAgent

@pytest.fixture
def dummy_agent():
    return DummyAgent()

def test_dummy_agent_chooses_valid_action(dummy_agent):
    legal_actions = [
        {"action_type": 1, "card_id": 1234},
        {"action_type": 2, "card_id": 5678}
    ]
    chosen = dummy_agent.select_action(legal_actions)
    assert chosen in legal_actions

def test_dummy_agent_empty_actions_raises_error(dummy_agent):
    with pytest.raises(ValueError):
        dummy_agent.select_action([])

@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)

@pytest.fixture
def ppo_agent():
    return PPOAgent(obs_dim=100, act_dim=200, hidden_dim=64)

@pytest.fixture
def ppo_params(ppo_agent, rng):
    return ppo_agent.init_params(rng)

def test_ppo_agent_forward_pass(ppo_agent, ppo_params, rng):
    obs = jax.random.normal(rng, (4, ppo_agent.obs_dim), dtype=jnp.float32)
    hidden_state = ppo_agent.init_hidden_state(4)
    
    probs, values, next_hidden = ppo_agent.forward(ppo_params, hidden_state, obs)
    
    assert probs.shape == (4, 200)
    assert values.shape == (4,)
    assert probs.dtype == jnp.float32
    assert values.dtype == jnp.float32
    
    sums = jnp.sum(probs, axis=-1)
    assert jnp.allclose(sums, jnp.ones_like(sums), atol=1e-5)

def test_ppo_loss_computation(ppo_agent, ppo_params, rng):
    k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    batch_size = 4
    time_steps = 5
    
    obs = jax.random.normal(k1, (batch_size, time_steps, ppo_agent.obs_dim), dtype=jnp.float32)
    actions = jax.random.randint(k2, (batch_size, time_steps), minval=0, maxval=200, dtype=jnp.int32)
    old_log_probs = jax.random.uniform(k3, (batch_size, time_steps), minval=-2.0, maxval=0.0, dtype=jnp.float32)
    advantages = jax.random.normal(k4, (batch_size, time_steps), dtype=jnp.float32)
    returns = jax.random.normal(k5, (batch_size, time_steps), dtype=jnp.float32)
    dones = jax.random.randint(k6, (batch_size, time_steps), minval=0, maxval=2, dtype=jnp.int32).astype(jnp.float32)
    
    loss_fn = jax.jit(ppo_agent.compute_loss)
    loss, metrics = loss_fn(ppo_params, obs, actions, old_log_probs, advantages, returns, dones)
    
    assert not jnp.isnan(loss)
    assert not jnp.isinf(loss)
    assert loss.dtype == jnp.float32
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics

def test_ppo_update_params(ppo_agent, ppo_params, rng):
    k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    batch_size = 4
    time_steps = 5
    
    obs = jax.random.normal(k1, (batch_size, time_steps, ppo_agent.obs_dim), dtype=jnp.float32)
    actions = jax.random.randint(k2, (batch_size, time_steps), minval=0, maxval=200, dtype=jnp.int32)
    old_log_probs = jax.random.uniform(k3, (batch_size, time_steps), minval=-2.0, maxval=0.0, dtype=jnp.float32)
    advantages = jax.random.normal(k4, (batch_size, time_steps), dtype=jnp.float32)
    returns = jax.random.normal(k5, (batch_size, time_steps), dtype=jnp.float32)
    dones = jax.random.randint(k6, (batch_size, time_steps), minval=0, maxval=2, dtype=jnp.int32).astype(jnp.float32)
    
    update_fn = jax.jit(ppo_agent.update_params)
    new_params, metrics = update_fn(ppo_params, obs, actions, old_log_probs, advantages, returns, dones)
    
    assert "total_loss" in metrics
    # Verify params updated
    w1_old = ppo_params["policy"]["w"]
    w1_new = new_params["policy"]["w"]
    assert not jnp.allclose(w1_old, w1_new)

def test_ppo_zero_shot_observation():
    # Test that PPO agent can handle the flattened zero-shot observation from YgoEnv
    from core.ygoenv.env import YgoEnv
    from ai.embeddings import embed_loader
    import tempfile
    import os
    import pickle
    
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
            obs, _ = env.reset()
            
            # Initialize PPO with the correct obs_dim
            obs_dim = obs.shape[0]
            agent = PPOAgent(obs_dim=obs_dim, act_dim=200, hidden_dim=64)
            rng = jax.random.PRNGKey(42)
            params = agent.init_params(rng)
            hidden_state = agent.init_hidden_state(1)
            
            # PPO forward pass with zero-shot observation
            obs_jax = jnp.asarray(obs, dtype=jnp.float32)
            probs, value, next_hidden = agent.forward(params, hidden_state, obs_jax)
            
            # Should not crash, shape should be correct
            assert probs.shape == (200,)
            assert probs.dtype == jnp.float32
        finally:
            # Reset global state
            embed_loader._is_loaded = False
            embed_loader._matrix = None
            embed_loader._card_to_idx = None
            embed_loader._embedding_dim = 0


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

def test_mcts_agent_initialization():
    obs_dim = 10
    act_dim = 5
    hidden_dim = 16
    
    ppo = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    agent = MCTSAgent(ppo, num_simulations=4)
    
    assert agent.mcts.num_simulations == 4
    assert agent.ppo_agent == ppo

def test_mcts_agent_select_action():
    obs_dim = 10
    act_dim = 5
    hidden_dim = 16
    
    ppo = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    agent = MCTSAgent(ppo, num_simulations=4)
    
    env = MockEnv(obs_dim, act_dim)
    rng = jax.random.PRNGKey(42)
    params = ppo.init_params(rng)
    hidden_state = ppo.init_hidden_state(1)
    
    obs = env.get_obs()
    
    action, policy, value, next_hidden = agent.select_action(env, params, hidden_state, obs)
    
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < act_dim
    assert policy.shape == (act_dim,)
    assert policy[0] == 0.0  # Masked action must have 0 probability
    assert isinstance(value, float)
    
    # hidden state has h and c
    h, c = next_hidden
    assert h.shape == (1, hidden_dim)
    assert c.shape == (1, hidden_dim)


