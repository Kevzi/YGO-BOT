import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple

class MaskedLSTMCell(nn.Module):
    """
    Cellule LSTM enveloppée pour gérer la réinitialisation de l'état caché (carry)
    lorsqu'un épisode se termine (done_t == 1.0).
    """
    hidden_size: int

    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], inputs: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        x_t, done_t = inputs
        c, h = carry
        
        # Masquage du carry précédent : si done_t == 1.0, le mask est 0.0, ce qui reset l'état.
        mask = (1.0 - done_t)
        if mask.ndim == 1:
            mask = jnp.expand_dims(mask, axis=-1)
            
        carry = (c * mask, h * mask)
        
        # Appel propre à la cellule LSTM native de Flax
        return nn.LSTMCell(features=self.hidden_size)(carry, x_t)

class PPOActorCritic(nn.Module):
    """
    Réseau de neurones Actor-Critic avec LSTM pour PPO, utilisant nn.scan pour BPTT.
    Prend en charge l'observation de 60694 dimensions, maintient un état de croyance
    (belief state), et masque les actions illégales.
    """
    action_dim: int = 250
    hidden_size: int = 512

    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], obs_seq: jnp.ndarray, prev_action_seq: jnp.ndarray, action_mask_seq: jnp.ndarray, dones_seq: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        # obs_seq : forme (batch_size, time_steps, obs_dim)
        
        # 1. Feature Extractor (appliqué sur la séquence entière via vmap/broadcasting natif)
        act_emb = nn.Embed(num_embeddings=self.action_dim, features=32, name="act_emb")(prev_action_seq)
        x_in = jnp.concatenate([obs_seq, act_emb], axis=-1)
        x = nn.Dense(self.hidden_size, name="encoder")(x_in)
        x = nn.relu(x)
        
        # 2. LSTM Cell déroulée dans le temps (BPTT)
        scanned_lstm = nn.scan(
            MaskedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,  # Itère sur l'axe du temps (index 1 de obs_seq et dones_seq)
            out_axes=1  # Conserve la séquence temporelle en sortie
        )(hidden_size=self.hidden_size, name="lstm")
        
        # Application du LSTM (prend en entrée le carry précédent et le tuple de séquences)
        new_carry, hidden_seq = scanned_lstm(carry, (x, dones_seq))
        
        # 3. Policy Head + Action Masking
        logits = nn.Dense(self.action_dim, name="actor")(hidden_seq)
        logits = jnp.where(action_mask_seq, logits, -1e9)
        
        # 4. Value Head (Critic)
        value = nn.Dense(1, name="critic")(hidden_seq)
        
        return new_carry, logits, jnp.squeeze(value, axis=-1)

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int = 512) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialise l'état caché (c, h) du LSTM pour un batch."""
        return (jnp.zeros((batch_size, hidden_size), dtype=jnp.float32),
                jnp.zeros((batch_size, hidden_size), dtype=jnp.float32))
