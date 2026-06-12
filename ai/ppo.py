import jax
import jax.numpy as jnp

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        assert obs_dim > 0 and act_dim > 0 and hidden_dim > 0, "Dimensions must be strictly positive"
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
    def init_hidden_state(self, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Initialise l'état caché (h, c) pour un batch de séquences."""
        return (jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32), 
                jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32))

    def init_params(self, rng: jax.random.PRNGKey) -> dict:
        """Initialise les poids du réseau de neurones en JAX pur."""
        keys = jax.random.split(rng, 5)
        
        glorot_init = jax.nn.initializers.glorot_uniform()
        zeros_init = jax.nn.initializers.zeros
        
        # Feature Extractor (shared MLP before LSTM)
        w1_f = glorot_init(keys[0], (self.obs_dim, self.hidden_dim), dtype=jnp.float32)
        b1_f = zeros_init(keys[1], (self.hidden_dim,), dtype=jnp.float32)
        
        # LSTM layer (takes feature + h_prev -> 4 * hidden_dim)
        lstm_w_i = glorot_init(keys[2], (self.hidden_dim, 4 * self.hidden_dim), dtype=jnp.float32)
        lstm_w_h = glorot_init(keys[3], (self.hidden_dim, 4 * self.hidden_dim), dtype=jnp.float32)
        lstm_b = zeros_init(keys[4], (4 * self.hidden_dim,), dtype=jnp.float32)

        # Split keys for policy and value heads
        keys_heads = jax.random.split(keys[0], 4)
        
        # Policy Head
        w_p = glorot_init(keys_heads[0], (self.hidden_dim, self.act_dim), dtype=jnp.float32)
        b_p = zeros_init(keys_heads[1], (self.act_dim,), dtype=jnp.float32)
        
        # Value Head
        w_v = glorot_init(keys_heads[2], (self.hidden_dim, 1), dtype=jnp.float32)
        b_v = zeros_init(keys_heads[3], (1,), dtype=jnp.float32)
        
        return {
            "feature": {"w": w1_f, "b": b1_f},
            "lstm": {"w_i": lstm_w_i, "w_h": lstm_w_h, "b": lstm_b},
            "policy": {"w": w_p, "b": b_p},
            "value": {"w": w_v, "b": b_v}
        }

    def _lstm_step(self, params: dict, hidden_state: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Exécute une étape LSTM. x: (batch, hidden_dim), hidden_state: (h, c)."""
        h, c = hidden_state
        
        lstm_params = params["lstm"]
        # W_i * x + W_h * h + b
        gates = jnp.dot(x, lstm_params["w_i"]) + jnp.dot(h, lstm_params["w_h"]) + lstm_params["b"]
        
        # i, f, o, g
        i, f, o, g = jnp.split(gates, 4, axis=-1)
        
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        o = jax.nn.sigmoid(o)
        g = jax.nn.tanh(g)
        
        c_new = f * c + i * g
        h_new = o * jax.nn.tanh(c_new)
        
        return (h_new, c_new), h_new

    def forward(self, params: dict, hidden_state: tuple[jnp.ndarray, jnp.ndarray], obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass public pour l'inférence. Retourne les probabilités, la Value et le nouvel état caché."""
        if obs.ndim not in (1, 2):
            raise ValueError(f"Invalid observation shape: {obs.shape}. Expected 1D or 2D array.")
            
        is_single = obs.ndim == 1
        if is_single:
            obs = jnp.expand_dims(obs, axis=0)
            
        # Extract features
        x = jnp.maximum(0, jnp.dot(obs, params["feature"]["w"]) + params["feature"]["b"])
        
        # LSTM Step
        next_hidden_state, h_new = self._lstm_step(params, hidden_state, x)
        
        # Policy & Value heads
        logits = jnp.dot(h_new, params["policy"]["w"]) + params["policy"]["b"]
        probs = jax.nn.softmax(logits, axis=-1)
        
        value = jnp.dot(h_new, params["value"]["w"]) + params["value"]["b"]
        
        if is_single:
            return probs[0], value[0, 0], next_hidden_state
        return probs, value[..., 0], next_hidden_state

    def _forward_sequence(self, params: dict, obs_seq: jnp.ndarray, dones_seq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calcul de la séquence complète pour BPTT via jax.lax.scan.
        obs_seq: (batch, time, obs_dim)
        dones_seq: (batch, time)
        """
        if obs_seq.shape[1] == 0:
            batch_size = obs_seq.shape[0]
            return (jnp.zeros((batch_size, 0, self.act_dim)), 
                    jnp.zeros((batch_size, 0)))
                    
        batch_size = obs_seq.shape[0]
        init_state = self.init_hidden_state(batch_size)
        
        def scan_fn(carry, step_inputs):
            h, c = carry
            obs_t, done_t = step_inputs
            
            # Reset hidden state where episode ended in previous step
            done_t = jnp.expand_dims(done_t, axis=-1)
            h = h * (1.0 - done_t)
            c = c * (1.0 - done_t)
            
            # Feature extraction
            x = jnp.maximum(0, jnp.dot(obs_t, params["feature"]["w"]) + params["feature"]["b"])
            
            # LSTM step
            (h_new, c_new), h_out = self._lstm_step(params, (h, c), x)
            
            # Heads
            logits = jnp.dot(h_out, params["policy"]["w"]) + params["policy"]["b"]
            value = jnp.dot(h_out, params["value"]["w"]) + params["value"]["b"]
            
            return (h_new, c_new), (logits, value[..., 0])

        # Swap batch and time axes for scan: (time, batch, ...)
        obs_seq_t = jnp.swapaxes(obs_seq, 0, 1)
        dones_seq_t = jnp.swapaxes(dones_seq, 0, 1)
        
        # Padding dones to shift it by 1 step (reset state on step AFTER done)
        # dones_seq_t has shape (time, batch).
        # We want to reset AT step t if done was true at step t-1.
        dones_shifted = jnp.concatenate([jnp.zeros((1, batch_size), dtype=jnp.float32), dones_seq_t[:-1]], axis=0)

        _, (logits_seq_t, value_seq_t) = jax.lax.scan(scan_fn, init_state, (obs_seq_t, dones_shifted))
        
        # Swap back to (batch, time, ...)
        logits_seq = jnp.swapaxes(logits_seq_t, 0, 1)
        value_seq = jnp.swapaxes(value_seq_t, 0, 1)
        
        return logits_seq, value_seq

    def compute_loss(self, params: dict, obs: jnp.ndarray, actions: jnp.ndarray, 
                    old_log_probs: jnp.ndarray, advantages: jnp.ndarray, returns: jnp.ndarray,
                    dones: jnp.ndarray,
                    clip_ratio: float = 0.2, value_coef: float = 1.0, entropy_coef: float = 0.01) -> tuple[jnp.ndarray, dict]:
        """Calcule la perte PPO avec BPTT sur une séquence temporelle."""
        assert obs.ndim == 3, "obs must have shape (batch, time, dim)"
        
        actions = jnp.clip(actions, 0, self.act_dim - 1)
        
        # Unroll sequence
        logits_seq, values_seq = self._forward_sequence(params, obs, dones)
        
        log_probs_all = jax.nn.log_softmax(logits_seq, axis=-1)
        # Advanced indexing to get log_probs of chosen actions
        batch_idx = jnp.arange(obs.shape[0])[:, None]
        time_idx = jnp.arange(obs.shape[1])[None, :]
        log_probs = log_probs_all[batch_idx, time_idx, actions]
        
        ratio = jnp.exp(log_probs - old_log_probs)
        
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        obj1 = ratio * advantages
        obj2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(obj1, obj2))
        
        value_loss = jnp.mean((values_seq - returns) ** 2)
        
        probs = jax.nn.softmax(logits_seq, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_probs_all, axis=-1))
        
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy
        }
        
        return total_loss, metrics

    def update_params(self, params: dict, obs: jnp.ndarray, actions: jnp.ndarray, 
                    old_log_probs: jnp.ndarray, advantages: jnp.ndarray, returns: jnp.ndarray,
                    dones: jnp.ndarray,
                    learning_rate: float = 3e-4) -> tuple[dict, dict]:
        """Mise à jour BPTT des paramètres avec SGD."""
        loss_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, metrics), grads = loss_fn(params, obs, actions, old_log_probs, advantages, returns, dones)
        
        new_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
        
        metrics["total_loss"] = loss
        return new_params, metrics

