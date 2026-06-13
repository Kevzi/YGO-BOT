import jax
import jax.numpy as jnp
import flax.linen as nn

class ActorCriticLSTM(nn.Module):
    """
    Réseau de neurones Actor-Critic avec LSTM pour PPO.
    Prend en charge l'observation de 60694 dimensions, compresse avec un encodeur,
    maintient un état de croyance (belief state) via un LSTM,
    et masque les actions illégales.
    """
    act_dim: int = 200
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, carry: tuple[jnp.ndarray, jnp.ndarray], obs: jnp.ndarray, prev_action: jnp.ndarray, action_mask: jnp.ndarray) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        """
        Inférence sur un seul pas de temps (single step).
        """
        x = self.encode(obs, prev_action)
        return self.apply_lstm(carry, x, action_mask)

    @nn.compact
    def encode(self, obs: jnp.ndarray, prev_action: jnp.ndarray) -> jnp.ndarray:
        """Encode l'observation et l'action précédente."""
        x = nn.Dense(self.hidden_dim, name="encoder")(obs)
        x = nn.relu(x)
        
        prev_act_one_hot = jax.nn.one_hot(prev_action, self.act_dim, dtype=jnp.float32)
        act_emb = nn.Dense(self.hidden_dim, name="action_encoder")(prev_act_one_hot)
        act_emb = nn.relu(act_emb)
        
        return x + act_emb

    @nn.compact
    def apply_lstm(self, carry: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray, action_mask: jnp.ndarray) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        """Applique la cellule LSTM et les têtes Actor/Critic."""
        lstm_cell = nn.LSTMCell(features=self.hidden_dim, name="lstm")
        new_carry, h_out = lstm_cell(carry, x)
        
        value = nn.Dense(1, name="critic")(h_out)
        logits = nn.Dense(self.act_dim, name="actor")(h_out)
        
        logits = jnp.where(action_mask, logits, -1e9)
        
        return new_carry, logits, value

def init_hidden_state(batch_size: int, hidden_dim: int = 512) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialise l'état caché (h, c) du LSTM pour un batch."""
    return (jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32),
            jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32))
