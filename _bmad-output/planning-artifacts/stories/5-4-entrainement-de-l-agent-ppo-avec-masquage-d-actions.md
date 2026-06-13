# Story 5.4: Entraînement de l'Agent PPO avec Masquage d'Actions (Action Masking)

## Story Foundation

- **User story statement:** As a Chercheur en IA, I want d'entraîner le modèle PPO en utilisant le masque d'actions (Action Mask) généré par l'environnement, So that le réseau apprenne à attribuer 0% de probabilité aux actions illégales, évitant ainsi le gaspillage d'exploration et accélérant l'apprentissage des conditions de victoire.
- **Acceptance Criteria:**
  - **Given** le lot d'observations renvoyé par Gym et son mask
  - **When** le PPO met à jour ses poids ou produit ses inférences
  - **Then** il pénalise la prédiction d'actions illégales à l'aide d'un gradient "Action Masking" (en soustrayant -1e9 aux logits invalides avant softmax)
  - **And** la Value Loss et l'Entropy démontrent que l'agent apprend à explorer uniquement les actions valides (probabilities of illegal actions strictly equal to 0.000).

## Developer Context

- **Files to modify:**
  - `ai/ppo.py`: Modify the `PPOAgent.forward` method to perform Action Masking directly inside the JAX network before computing the final probabilities and log_probs.

- **Current State:**
  - Currently, `ai/ppo.py` ignores `action_mask` during the forward pass. `ai/distributed.py` tries to force illegal probabilities to 0 manually via Numpy `probs * mask_np`, but this happens *after* the network outputs its logits/probabilities. This means the gradients flowing back to the network try to increase the probabilities of illegal actions, completely confusing the learner.

- **What must be preserved:**
  - Keep the overall signature of `PPOAgent.forward(self, params, hidden_state, obs, prev_action, action_mask)` intact.
  - Do not alter the rest of the PPO loss logic or the distributed setup.

## Technical Requirements

- Replace Numpy masking in `ai/distributed.py` with pure JAX masking inside `ai/ppo.py`?
  - Actually, if we apply `-1e9` inside the JAX `ActorCriticLSTM` or `PPOAgent.forward` method before Softmax, the probabilities for illegal actions will naturally be 0.
  - In `ai/ppo.py`:
    ```python
    logits, value, next_hidden = self.model.apply({'params': params}, hidden_state, obs, prev_action, action_mask)
    # Apply action masking
    # If action_mask is True (legal), keep logit. If False (illegal), add -1e9
    masked_logits = jnp.where(action_mask, logits, -1e9)
    # Compute Softmax
    probs = jax.nn.softmax(masked_logits, axis=-1)
    ```
  - Wait, `ActorCriticLSTM` currently returns `probs`, not `logits`. We need to inspect `ai/network.py` or `ai/ppo.py` to see where `softmax` is applied and apply the mask BEFORE the softmax.

## Tasks/Subtasks
- [x] Fix `get_action_mask` in `core/ygoenv/env.py` to return the real mask
- [x] Update `ai/network.py` ActorCriticLSTM to use `-1e9` for masking
- [x] Remove old numpy fallback in `ai/distributed.py`

## Dev Agent Record
- **Debug Log**: Found that `get_action_mask()` was a mock returning `np.ones`. Fixed it to use `get_legal_actions()`.
- **Completion Notes**: Action masking is now fully integrated directly in the JAX network calculation for both stability and efficiency.

## File List
- `core/ygoenv/env.py` (Modified)
- `ai/network.py` (Modified)
- `ai/distributed.py` (Modified)

## Change Log
- Removed Numpy probability mask and integrated -1e9 mask inside ActorCriticLSTM. `env.py` now correctly provides `get_legal_actions()` as `get_action_mask()`.

## Status
Status: review
