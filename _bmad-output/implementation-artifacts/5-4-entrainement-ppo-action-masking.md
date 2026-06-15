---
baseline_commit: 76fa931985ce5dbe7a325c8747f822ba8417d466
---
# Story 5.4: Entraînement de l'Agent PPO avec Masquage d'Actions (Action Masking)

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story Foundation

As a Chercheur en IA,
I want d'entraîner le modèle PPO en utilisant le masque d'actions (Action Mask) généré par l'environnement,
So that le réseau apprenne à attribuer 0% de probabilité aux actions illégales, évitant ainsi le gaspillage d'exploration et accélérant l'apprentissage des conditions de victoire.

## Acceptance Criteria

1. **Given** le lot d'observations renvoyé par Gym
2. **When** le PPO met à jour ses poids
3. **Then** il pénalise la prédiction d'actions illégales à l'aide d'un gradient "Action Masking"
4. **And** la Value Loss et l'Entropy démontrent que l'agent apprend à explorer uniquement les actions valides.

## Developer Context & Guardrails

### Technical Requirements
- Mettre à jour la Policy du PPO (dans le fichier réseau/JAX) pour accepter un masque d'action (`action_mask`) en plus des observations.
- Appliquer l'Action Masking **avant** l'opération `softmax` (souvent en ajoutant une valeur fortement négative comme `-1e9` aux logits des actions illégales).
- Veiller à ce que l'entropie soit calculée uniquement sur les actions valides pour éviter les `NaN` (si des probabilités de 0 sont générées).
- Assurer la stabilité numérique des gradients.

### Architecture Compliance
- **JAX Pure Functions**: L'application du masque d'action doit être codée sous forme d'opérations vectorisées purement JAX, garantissant la compatibilité avec `jax.jit`.
- **Tensor Dtypes**: Le dtype du masque d'action doit être respecté (ex: booléen ou `jnp.float32`) et en conformité stricte avec les logits.
- **Fail Fast**: Si un `NaN` apparaît durant le calcul de loss (Policy, Value ou Entropy), le processus doit s'arrêter explicitement.

### Previous Story Intelligence
- **Story 5.3**: L'observation de l'environnement est maintenant un tenseur de dimension `(60694,)`. Le réseau JAX traite cette taille exacte. Le masque d'action (`(200,)`) est soit fourni avec l'observation (ex: via un dictionnaire/Dict space) soit passé séparément depuis l'environnement.
- **Bugs Connus (Epic 6 Reviews)**: 
  - `[Patch] Fragile Entropy Calculation [ai/ppo.py]` - L'entropie crashe souvent lors de l'application d'action masking si des probabilités nulles interviennent dans `p * log(p)`.
  - `[Patch] Missing Gradient Clipping [ai/ppo.py]` - S'assurer que le mask ne provoque pas de gradients explosifs.

### File Structure Requirements
- Composant impacté majeur : `ai/ppo.py` et `ai/network.py`.
- L'environnement (`core/ygoenv/`) doit correctement exposer/envoyer l'action mask à l'agent JAX.

## Project Context Reference
- **Epic**: Epic 5 - Mise à l'échelle de l'Environnement (Architecture Avancée).
- **Langue de communication**: Français.
- **But Ultime**: Sans ce masque, l'espace de 200 actions rendrait l'apprentissage de Yu-Gi-Oh! impossible, car l'agent piocherait 95% d'actions illégales. Le mask guide le PPO pour se concentrer sur la stratégie pure.

## Completion Status
Ultimate context engine analysis completed - comprehensive developer guide created.

## Tasks/Subtasks
- [x] Update act_dim to 250 in `ai/network.py`
- [x] Ensure entropy is safely computed and action masking applied in `ai/ppo.py`
- [x] Add explicit Fail Fast for NaN in `ai/ppo.py`
- [x] Check `core/ygoenv/env.py` mapping of action_mask

## Dev Agent Record
- **Debug Log**: Verified that `jnp.where(action_mask, logits, -1e9)` is applied correctly in `network.py`. Verified that entropy masking is safe using `safe_probs` in `ppo.py`. Added a `jax.debug.callback` check to fail fast if NaN is detected during loss calculation in `ppo.py`. Checked that `core/ygoenv/env.py` uses action space 250 correctly. Updated `act_dim` to 250 in `network.py`.
- **Completion Notes**: Story implemented successfully. All acceptance criteria and technical requirements are met. Action size is 250.

## File List
- `ai/network.py`
- `ai/ppo.py`

## Change Log
- Changed `act_dim` to 250 in `ai/network.py`.
- Added `jax.debug.callback` with explicit error raise to fast-fail if NaN is detected in `policy_loss`, `value_loss`, or `entropy` in `ai/ppo.py`.
