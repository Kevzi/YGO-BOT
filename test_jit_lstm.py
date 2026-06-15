import jax
import jax.numpy as jnp
from ai.ppo import PPOAgent

def test_lstm_compilation():
    print("Testing LSTM Initialization and Forward Pass compilation...")
    obs_dim = 60694
    act_dim = 250
    hidden_dim = 512
    batch_size = 4
    time_steps = 10
    
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    
    rng = jax.random.PRNGKey(42)
    params, opt_state = agent.init_params(rng)
    
    # Test single step forward pass
    obs_single = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
    action_mask_single = jnp.ones((batch_size, act_dim), dtype=jnp.bool_)
    hidden_state = agent.init_hidden_state(batch_size)
    
    # Compile single step
    print("Compiling single step forward...")
    forward_jit = jax.jit(agent.forward)
    probs, value, next_hidden = forward_jit(params, hidden_state, obs_single, action_mask_single)
    
    print(f"Forward pass completed. Probs shape: {probs.shape}, Value shape: {value.shape}, Next hidden H shape: {next_hidden[0].shape}")
    
    # Test sequence compute_loss
    print("Compiling sequence compute_loss (BPTT)...")
    obs_seq = jnp.zeros((batch_size, time_steps, obs_dim), dtype=jnp.float32)
    action_mask_seq = jnp.ones((batch_size, time_steps, act_dim), dtype=jnp.bool_)
    actions_seq = jnp.zeros((batch_size, time_steps), dtype=jnp.int32)
    old_log_probs = jnp.zeros((batch_size, time_steps), dtype=jnp.float32)
    advantages = jnp.zeros((batch_size, time_steps), dtype=jnp.float32)
    returns = jnp.zeros((batch_size, time_steps), dtype=jnp.float32)
    dones = jnp.zeros((batch_size, time_steps), dtype=jnp.float32)
    
    # Set a mock done in the middle to see if shape is fine
    dones = dones.at[:, 5].set(1.0)
    
    update_jit = jax.jit(agent.update_params)
    
    new_params, new_opt_state, metrics = update_jit(
        params, opt_state, obs_seq, action_mask_seq, actions_seq, 
        old_log_probs, advantages, returns, dones
    )
    
    print(f"Update completed. Metrics: loss={metrics['total_loss']}, policy_loss={metrics['policy_loss']}, entropy={metrics['entropy']}")
    print("All JIT compilations passed successfully!")

if __name__ == "__main__":
    test_lstm_compilation()
