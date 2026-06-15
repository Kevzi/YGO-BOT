import jax
import jax.numpy as jnp
import optax
import math
from functools import partial
from ai.network import PPOActorCritic

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512, learning_rate: float = 3e-4):
        if not (math.isfinite(learning_rate) and learning_rate > 0):
            raise ValueError(f"learning_rate must be a positive finite float, got {learning_rate}")
        assert obs_dim > 0 and act_dim > 0 and hidden_dim > 0, "Dimensions must be strictly positive"
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        self.model = PPOActorCritic(action_dim=act_dim, hidden_size=hidden_dim)
        # Patch: Missing gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate)
        )
        
    def init_hidden_state(self, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Initialise l'état caché (h, c) pour un batch."""
        return PPOActorCritic.initialize_carry(batch_size, self.hidden_dim)

    def init_params(self, rng: jax.random.PRNGKey) -> tuple[dict, optax.OptState]:
        """Initialise les poids du modèle Flax et l'état de l'optimiseur."""
        dummy_obs = jnp.zeros((1, 1, self.obs_dim), dtype=jnp.float32)
        dummy_prev_act = jnp.zeros((1, 1), dtype=jnp.int32)
        dummy_mask = jnp.zeros((1, 1, self.act_dim), dtype=jnp.bool_)
        dummy_dones = jnp.zeros((1, 1), dtype=jnp.float32)
        dummy_carry = self.init_hidden_state(1)
        
        variables = self.model.init(rng, dummy_carry, dummy_obs, dummy_prev_act, dummy_mask, dummy_dones)
        params = variables['params']
        
        opt_state = self.optimizer.init(params)
        return params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _forward_jit(self, params, hidden_state, obs_seq, prev_action_seq, action_mask_seq, dones_seq):
        """Passage forward compilé XLA (très rapide, pas de fuite mémoire Python)."""
        return self.model.apply(
            {'params': params}, hidden_state, obs_seq, prev_action_seq, action_mask_seq, dones_seq
        )

    def forward(self, params: dict, hidden_state: tuple[jnp.ndarray, jnp.ndarray], obs: jnp.ndarray, prev_action: jnp.ndarray, action_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass public pour l'inférence. Retourne les probabilités, la Value et le nouvel état caché."""
        if obs.ndim not in (1, 2):
            raise ValueError(f"Invalid observation shape: {obs.shape}. Expected 1D or 2D array.")
        
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
                
        action_mask = jnp.where(~jnp.any(action_mask, axis=-1, keepdims=True), True, action_mask)
        
        # Add time dimension for PPOActorCritic sequence requirement
        obs_seq = jnp.expand_dims(obs, axis=1)
        prev_action_seq = jnp.expand_dims(prev_action, axis=1)
        action_mask_seq = jnp.expand_dims(action_mask, axis=1)
        dones_seq = jnp.zeros((obs.shape[0], 1), dtype=jnp.float32)
        # Appeler la méthode JIT-compilée
        next_hidden_state, logits_seq, value_seq = self._forward_jit(
            params, hidden_state, obs_seq, prev_action_seq, action_mask_seq, dones_seq
        )
        
        # Remove time dimension
        logits = logits_seq[:, 0, :]
        value = value_seq[:, 0]
        
        probs = jax.nn.softmax(logits, axis=-1)
        
        if is_single:
            return probs[0], value[0], (next_hidden_state[0][0], next_hidden_state[1][0])
        return probs, value, next_hidden_state

    def compute_loss(self, params: dict, hidden_state: tuple[jnp.ndarray, jnp.ndarray], obs: jnp.ndarray, prev_actions: jnp.ndarray, action_masks: jnp.ndarray, actions: jnp.ndarray, 
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
        
        batch_size, time_steps = obs.shape[0], obs.shape[1]
        
        # [PATCH] Protection contre les séquences vides
        # Si la longueur de séquence est 0, on retourne une perte nulle.
        # Cela utilise jax.lax.cond pour être compatible avec JIT si nécessaire, 
        # mais on peut aussi juste utiliser une condition classique si time_steps est statique.
        # Pour une compatibilité JIT stricte, il faut éviter un return conditionnel pur sur une dimension dynamique.
        # Cependant, obs.shape est statique en JAX (sauf si on utilise jax.experimental.Dynamic).
        # Ici on suppose que time_steps est fixe.
        if time_steps == 0:
            return jnp.array(0.0), {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        dones = jnp.asarray(dones, dtype=jnp.float32)
        # Shift dones by 1 step (reset on step AFTER done)
        dones_shifted = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=jnp.float32), dones[:, :-1]], axis=1)
        
        # Apply the model on the entire sequence directly
        _, logits_seq, values_seq = self.model.apply({'params': params}, hidden_state, obs, prev_actions, action_masks, dones_shifted)
        
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
        
        is_nan = jnp.isnan(policy_loss) | jnp.isnan(value_loss) | jnp.isnan(entropy) | jnp.isnan(total_loss)
        def fail_fast_on_nan(has_nan):
            if has_nan:
                raise ValueError("Fail Fast: NaN detected in loss calculation (Policy, Value, or Entropy)!")
        jax.debug.callback(fail_fast_on_nan, is_nan)
        
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy
        }
        
        return total_loss, metrics

    def update_params(self, params: dict, opt_state: optax.OptState, hidden_state: tuple[jnp.ndarray, jnp.ndarray], obs: jnp.ndarray, prev_actions: jnp.ndarray, action_masks: jnp.ndarray, actions: jnp.ndarray, 
                    old_log_probs: jnp.ndarray, advantages: jnp.ndarray, returns: jnp.ndarray,
                    dones: jnp.ndarray) -> tuple[dict, optax.OptState, dict]:
        """Mise à jour BPTT des paramètres avec Optax."""
        loss_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, metrics), grads = loss_fn(params, hidden_state, obs, prev_actions, action_masks, actions, old_log_probs, advantages, returns, dones)
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        metrics["total_loss"] = loss
        return new_params, new_opt_state, metrics
