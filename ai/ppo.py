import jax
import jax.numpy as jnp
import optax
import math
from ai.network import ActorCriticLSTM, init_hidden_state

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512, learning_rate: float = 3e-4):
        if not (math.isfinite(learning_rate) and learning_rate > 0):
            raise ValueError(f"learning_rate must be a positive finite float, got {learning_rate}")
        assert obs_dim > 0 and act_dim > 0 and hidden_dim > 0, "Dimensions must be strictly positive"
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        self.model = ActorCriticLSTM(act_dim=act_dim, hidden_dim=hidden_dim)
        # Patch: Missing gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate)
        )
        
    def init_hidden_state(self, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Initialise l'état caché (h, c) pour un batch."""
        return init_hidden_state(batch_size, self.hidden_dim)

    def init_params(self, rng: jax.random.PRNGKey) -> tuple[dict, optax.OptState]:
        """Initialise les poids du modèle Flax et l'état de l'optimiseur."""
        dummy_obs = jnp.zeros((1, self.obs_dim), dtype=jnp.float32)
        dummy_prev_action = jnp.zeros((1,), dtype=jnp.int32)
        dummy_mask = jnp.zeros((1, self.act_dim), dtype=jnp.bool_)
        dummy_carry = self.init_hidden_state(1)
        
        variables = self.model.init(rng, dummy_carry, dummy_obs, dummy_prev_action, dummy_mask)
        params = variables['params']
        
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def forward(self, params: dict, hidden_state: tuple[jnp.ndarray, jnp.ndarray], obs: jnp.ndarray, prev_action: jnp.ndarray, action_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass public pour l'inférence. Retourne les probabilités, la Value et le nouvel état caché."""
        if obs.ndim not in (1, 2):
            raise ValueError(f"Invalid observation shape: {obs.shape}. Expected 1D or 2D array.")
        if obs.ndim != action_mask.ndim:
            raise ValueError(f"obs.ndim ({obs.ndim}) != action_mask.ndim ({action_mask.ndim})")
        if action_mask.shape[-1] != self.act_dim:
            raise ValueError(f"action_mask.shape[-1] ({action_mask.shape[-1]}) != {self.act_dim}")
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs.shape[-1] ({obs.shape[-1]}) != {self.obs_dim}")
            
        action_mask = jnp.asarray(action_mask, dtype=jnp.bool_)
        
        is_single = obs.ndim == 1
        if is_single:
            obs = jnp.expand_dims(obs, axis=0)
            prev_action = jnp.expand_dims(prev_action, axis=0)
            action_mask = jnp.expand_dims(action_mask, axis=0)
            # Ensure hidden state has batch dimension
            h, c = hidden_state
            if h.ndim == 1:
                hidden_state = (jnp.expand_dims(h, axis=0), jnp.expand_dims(c, axis=0))
            
        if obs.shape[0] != action_mask.shape[0]:
            raise ValueError(f"Batch size mismatch: obs {obs.shape[0]} != action_mask {action_mask.shape[0]}")
            
        action_mask = jnp.where(~jnp.any(action_mask, axis=-1, keepdims=True), True, action_mask)
        
        next_hidden_state, logits, value = self.model.apply({'params': params}, hidden_state, obs, prev_action, action_mask)
        
        probs = jax.nn.softmax(logits, axis=-1)
        
        if is_single:
            return probs[0], value[0, 0], (next_hidden_state[0][0], next_hidden_state[1][0])
        return probs, value[..., 0], next_hidden_state

    def _forward_sequence(self, params: dict, obs_seq: jnp.ndarray, prev_act_seq: jnp.ndarray, mask_seq: jnp.ndarray, dones_seq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calcul de la séquence complète pour BPTT via jax.lax.scan."""
        if obs_seq.shape[1] == 0:
            batch_size = obs_seq.shape[0]
            return (jnp.zeros((batch_size, 0, self.act_dim), dtype=jnp.float32), 
                    jnp.zeros((batch_size, 0), dtype=jnp.float32))
                    
        if obs_seq.shape[:2] != mask_seq.shape[:2]:
            raise ValueError(f"Sequence shape mismatch: obs {obs_seq.shape[:2]} != mask {mask_seq.shape[:2]}")
        if dones_seq.shape != obs_seq.shape[:2]:
            raise ValueError(f"Sequence shape mismatch: dones {dones_seq.shape} != obs {obs_seq.shape[:2]}")
            
        batch_size, time_steps = obs_seq.shape[:2]
        init_state = self.init_hidden_state(batch_size)
        
        # Patch: Vectorize sequence processing to avoid bottleneck
        obs_flat = obs_seq.reshape((batch_size * time_steps, -1))
        prev_act_flat = prev_act_seq.reshape((batch_size * time_steps,))
        
        encoded_flat = self.model.apply({'params': params}, obs_flat, prev_act_flat, method=self.model.encode)
        encoded_seq = encoded_flat.reshape((batch_size, time_steps, -1))
        
        def scan_fn(carry, step_inputs):
            h, c = carry
            enc_t, mask_t, done_t = step_inputs
            
            # Patch: Explicit boolean/float handling via where
            done_t_expanded = jnp.expand_dims(done_t, axis=-1)
            h = jnp.where(done_t_expanded, 0.0, h)
            c = jnp.where(done_t_expanded, 0.0, c)
            carry = (h, c)
            
            new_carry, logits, value = self.model.apply({'params': params}, carry, enc_t, mask_t, method=self.model.apply_lstm)
            
            return new_carry, (logits, value[..., 0])

        encoded_seq_t = jnp.swapaxes(encoded_seq, 0, 1)
        mask_seq_t = jnp.swapaxes(mask_seq, 0, 1)
        dones_seq_t = jnp.swapaxes(dones_seq, 0, 1)
        
        # Shift dones by 1 step (reset on step AFTER done)
        dones_shifted = jnp.concatenate([jnp.zeros((1, batch_size), dtype=jnp.bool_), dones_seq_t[:-1]], axis=0)

        _, (logits_seq_t, value_seq_t) = jax.lax.scan(scan_fn, init_state, (encoded_seq_t, mask_seq_t, dones_shifted))
        
        logits_seq = jnp.swapaxes(logits_seq_t, 0, 1)
        value_seq = jnp.swapaxes(value_seq_t, 0, 1)
        
        return logits_seq, value_seq

    def compute_loss(self, params: dict, obs: jnp.ndarray, action_masks: jnp.ndarray, actions: jnp.ndarray, 
                    old_log_probs: jnp.ndarray, advantages: jnp.ndarray, returns: jnp.ndarray,
                    dones: jnp.ndarray,
                    clip_ratio: float = 0.2, value_coef: float = 1.0, entropy_coef: float = 0.01) -> tuple[jnp.ndarray, dict]:
        """Calcule la perte PPO avec BPTT sur une séquence temporelle."""
        assert obs.ndim == 3, "obs must have shape (batch, time, dim)"
        assert actions.shape == obs.shape[:2], "actions shape mismatch"
        assert returns.shape == obs.shape[:2], "returns shape mismatch"
        assert advantages.shape == obs.shape[:2], "advantages shape mismatch"
        
        action_masks = jnp.asarray(action_masks, dtype=jnp.bool_)
        action_masks = jnp.where(~jnp.any(action_masks, axis=-1, keepdims=True), True, action_masks)
        
        actions = jnp.clip(actions, 0, self.act_dim - 1)
        
        # Shift actions to get previous actions (using 0 for the first step)
        batch_size = obs.shape[0]
        first_step_actions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        prev_actions_seq = jnp.concatenate([first_step_actions, actions[:, :-1]], axis=1)
        
        logits_seq, values_seq = self._forward_sequence(params, obs, prev_actions_seq, action_masks, dones)
        
        log_probs_all = jax.nn.log_softmax(logits_seq, axis=-1)
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
        # Patch: Fragile Entropy Calculation
        # Set safe_probs to 1.0 for masked actions so log(1.0) = 0.0, zeroing out their entropy contribution cleanly.
        safe_probs = jnp.where(action_masks, probs, 1.0)
        entropy = -jnp.mean(jnp.sum(probs * jnp.log(safe_probs), axis=-1))
        
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy
        }
        
        return total_loss, metrics

    def update_params(self, params: dict, opt_state: optax.OptState, obs: jnp.ndarray, action_masks: jnp.ndarray, actions: jnp.ndarray, 
                    old_log_probs: jnp.ndarray, advantages: jnp.ndarray, returns: jnp.ndarray,
                    dones: jnp.ndarray) -> tuple[dict, optax.OptState, dict]:
        """Mise à jour BPTT des paramètres avec Optax."""
        loss_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, metrics), grads = loss_fn(params, obs, action_masks, actions, old_log_probs, advantages, returns, dones)
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        metrics["total_loss"] = loss
        return new_params, new_opt_state, metrics
