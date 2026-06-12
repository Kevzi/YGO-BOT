import pytest
import jax
import jax.numpy as jnp
from ai.ppo import PPOAgent

def test_ppo_lstm_initialization():
    obs_dim = 10
    act_dim = 5
    hidden_dim = 32
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    
    # Check init_hidden_state
    batch_size = 4
    h, c = agent.init_hidden_state(batch_size)
    assert h.shape == (batch_size, hidden_dim)
    assert c.shape == (batch_size, hidden_dim)
    assert jnp.all(h == 0)
    assert jnp.all(c == 0)

    # Check params
    rng = jax.random.PRNGKey(42)
    params = agent.init_params(rng)
    assert "lstm" in params
    assert "w_i" in params["lstm"]
    assert "w_h" in params["lstm"]
    assert "b" in params["lstm"]

def test_ppo_lstm_forward():
    obs_dim = 10
    act_dim = 5
    hidden_dim = 32
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    
    batch_size = 4
    rng = jax.random.PRNGKey(42)
    params = agent.init_params(rng)
    hidden_state = agent.init_hidden_state(batch_size)
    
    obs = jax.random.normal(rng, (batch_size, obs_dim))
    
    probs, value, next_hidden_state = agent.forward(params, hidden_state, obs)
    
    assert probs.shape == (batch_size, act_dim)
    assert value.shape == (batch_size,)
    
    next_h, next_c = next_hidden_state
    assert next_h.shape == (batch_size, hidden_dim)
    assert next_c.shape == (batch_size, hidden_dim)
    
    # Ensure hidden state actually changed
    h, c = hidden_state
    assert not jnp.allclose(next_h, h)
    assert not jnp.allclose(next_c, c)
