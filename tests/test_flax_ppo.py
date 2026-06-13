import jax
import jax.numpy as jnp
from ai.ppo import PPOAgent

def test_flax_ppo_forward() -> None:
    batch_size = 2
    seq_length = 5
    obs_dim = 100  # Réduit pour les tests unitaires
    act_dim = 20
    hidden_dim = 64
    
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    
    rng = jax.random.PRNGKey(42)
    params, opt_state = agent.init_params(rng)
    
    # --- Test Forward Single Step ---
    dummy_obs = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
    dummy_prev_action = jnp.zeros((batch_size,), dtype=jnp.int32)
    dummy_mask = jnp.ones((batch_size, act_dim), dtype=jnp.bool_)
    dummy_mask = dummy_mask.at[0, 0].set(False)
    
    hidden_state = agent.init_hidden_state(batch_size)
    
    probs, value, next_hidden = agent.forward(params, hidden_state, dummy_obs, dummy_prev_action, dummy_mask)
    
    assert probs.shape == (batch_size, act_dim), f"Expected {(batch_size, act_dim)}, got {probs.shape}"
    assert value.shape == (batch_size,), f"Expected {(batch_size,)}, got {value.shape}"
    assert float(probs[0, 0]) == 0.0, f"Expected 0.0 for masked action, got {probs[0, 0]}"
    assert float(probs[0, 1]) > 0.0, "Expected positive probability for legal action"
    
    # --- Test Forward Sequence (BPTT) ---
    dummy_obs_seq = jnp.zeros((batch_size, seq_length, obs_dim), dtype=jnp.float32)
    dummy_prev_act_seq = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    dummy_mask_seq = jnp.ones((batch_size, seq_length, act_dim), dtype=jnp.bool_)
    dummy_dones_seq = jnp.zeros((batch_size, seq_length), dtype=jnp.bool_)
    dummy_dones_seq = dummy_dones_seq.at[0, 2].set(True)
    
    logits_seq, value_seq = agent._forward_sequence(params, dummy_obs_seq, dummy_prev_act_seq, dummy_mask_seq, dummy_dones_seq)
    
    assert logits_seq.shape == (batch_size, seq_length, act_dim), f"Expected {(batch_size, seq_length, act_dim)}, got {logits_seq.shape}"
    assert value_seq.shape == (batch_size, seq_length), f"Expected {(batch_size, seq_length)}, got {value_seq.shape}"
    
    # --- Test Compute Loss & Update Params ---
    actions = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    old_log_probs = jnp.zeros((batch_size, seq_length), dtype=jnp.float32)
    advantages = jnp.ones((batch_size, seq_length), dtype=jnp.float32)
    returns = jnp.ones((batch_size, seq_length), dtype=jnp.float32)
    
    loss, metrics = agent.compute_loss(params, dummy_obs_seq, dummy_mask_seq, actions, old_log_probs, advantages, returns, dummy_dones_seq)
    
    assert loss.ndim == 0, "Loss must be a scalar"
    assert "policy_loss" in metrics and "value_loss" in metrics and "entropy" in metrics
    
    new_params, new_opt_state, update_metrics = agent.update_params(params, opt_state, dummy_obs_seq, dummy_mask_seq, actions, old_log_probs, advantages, returns, dummy_dones_seq)
    
    assert update_metrics["total_loss"] == loss

if __name__ == "__main__":
    test_flax_ppo_forward()
    print("All tests passed.")
