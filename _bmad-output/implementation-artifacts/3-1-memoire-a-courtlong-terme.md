---
epic: 3
story: 1
title: Mémoire à Court/Long Terme (Intégration LSTM)
status: done
baseline_commit: 142644830760e1541b0b8ff531389426e9805b97
---

# Story 3.1: Mémoire à Court/Long Terme (Intégration LSTM)

Status: done

## Story

As a Chercheur en IA,
I want d'étendre l'architecture neuronale PPO avec des couches récurrentes LSTM (Long Short-Term Memory),
so that l'agent puisse se souvenir des cartes révélées puis retournées face cachée, gérant ainsi l'information imparfaite et les dynamiques de bluff.

## Acceptance Criteria

1. **Given** des trajectoires d'environnement séquentiel
2. **When** l'agent traite la séquence
3. **Then** il maintient et met à jour un "hidden state" persistant d'un tour sur l'autre
4. **And** les gradients peuvent se propager dans le temps (BPTT) lors de l'entraînement sans erreur JAX.

## Tasks / Subtasks

- [x] Task 1 (AC: 1, 2, 3, 4): Modifier `PPOAgent` pour y intégrer un LSTM (JAX pur).
  - [x] Subtask 1.1: Ajouter une fonction `init_hidden_state(batch_size)` pour initialiser le hidden state (`h` et `c`).
  - [x] Subtask 1.2: Modifier `init_params` pour initialiser les poids LSTM (Input/Forget/Output/Cell gates).
  - [x] Subtask 1.3: Modifier le `forward` pass pour accepter `(params, hidden_state, obs)` et retourner `(probs, value, next_hidden_state)`.
- [x] Task 2 (AC: 1, 3): Mettre à jour la collecte de données (`RolloutWorker`).
  - [x] Subtask 2.1: Adapter `RolloutWorker.collect_rollout` pour maintenir le `hidden_state` au fil des étapes (step t à t+1).
  - [x] Subtask 2.2: Sauvegarder les trajectoires en conservant la structure temporelle `(time_steps, obs_dim)` ou `(batch_size, time_steps, obs_dim)` pour permettre BPTT.
  - [x] Subtask 2.3: S'assurer que le `hidden_state` est remis à zéro lors d'un `env.reset()`.
- [x] Task 3 (AC: 4): Mettre à jour la fonction d'apprentissage (`compute_loss` / `update_params`) pour gérer BPTT.
  - [x] Subtask 3.1: La perte doit être calculée en "déroulant" la séquence (unroll) sur la dimension temporelle avec `jax.lax.scan` (vivement recommandé pour les performances) ou par traitement de séquence structurée.

## Dev Notes

- **Pureté Fonctionnelle JAX** : Le LSTM doit être écrit de manière "pure". Les matrices de poids du LSTM (`W_i`, `W_f`, `W_o`, `W_g`) doivent être explicitement initialisées et gérées dans `params`. Un LSTM prend en entrée `obs_t` et `(h_{t-1}, c_{t-1})` et ressort `h_t` et `(h_t, c_t)`. 
- **Style Architectural** : L'architecture actuelle utilise JAX/Numpy pur pour le réseau (pas de Flax ou Haiku). Il faut conserver ce style de `jax.nn.initializers` comme dans la codebase existante `ai/ppo.py`.
- **dtype Obligatoire** : Les opérations doivent définir `dtype=jnp.float32`.
- **Format Tensoriel** : Pour BPTT, les séquences passées au `compute_loss` doivent avoir une forme prenant en compte la notion de séquence temporelle, ce qui affecte `distributed/train.py` lors de la concaténation de données pour le Learner.

### Project Structure Notes
- **Fichiers modifiés** : 
  - `ai/ppo.py` (L'agent PPO central : ajout LSTM et BPTT via jax.lax.scan)
  - `distributed/worker.py` (Gestion de l'état caché par environnement lors du rollout)
  - `distributed/train.py` (Adaptation du passage des batchs pour la dimension temporelle)

### References
- [Architecture JAX pure]: [Source: _bmad-output/planning-artifacts/architecture.md#Pattern Consistency]
- [PRD Vision LSTM]: [Source: _bmad-output/planning-artifacts/prds/prd-ygo-bot-2026-06-11/prd.md#FR-1.3 Mémoire et Bluff]

## Dev Agent Record

### Agent Model Used
### Agent Model Used
Gemini 2.5 Pro

### Debug Log References
- Manual tests for `distributed.train` executed via Ray and JAX BPTT passed perfectly.

### Completion Notes List
- Implemented pure JAX LSTM in `ai/ppo.py`, including `_lstm_step` and `_forward_sequence` using `jax.lax.scan`.
- Adapted `RolloutWorker` in `distributed/worker.py` to maintain hidden state during play and explicitly reset it to zero upon `env.reset()`.
- Extracted the sequence `dones` flag alongside observations to correctly mask out hidden state propagation across episodes during BPTT training.
- Updated `LeagueTrainer` in `distributed/train.py` to use `np.stack` to preserve the temporal dimension for sequences passed to `jax.lax.scan`.
- Ensured strict numerical precision using `jnp.float32`.

### File List
- `ai/ppo.py`
- `distributed/worker.py`
- `distributed/train.py`
- `tests/ai/test_ppo_lstm.py`

### Review Findings

- [x] [Review][Dismiss] Matrices de Poids LSTM Non-Explicites — Utilisateur a choisi de conserver l'optimisation JAX pour les perfs.
- [x] [Review][Patch] Conversion JAX in-loop catastrophique [`distributed/worker.py`]
- [x] [Review][Patch] Band-Aids de normalisation des probabilités [`distributed/worker.py`]
- [x] [Review][Patch] Crash sur `num_steps <= 0` dans `train_step` [`distributed/train.py`]
- [x] [Review][Patch] Bootstrapping des valeurs omis pour les trajectoires tronquées [`distributed/worker.py`]
- [x] [Review][Patch] Manque de vérification de forme (ndim) pour les observations dans `forward` [`ai/ppo.py`]
- [x] [Review][Patch] Crash dans `_forward_sequence` si la dimension temporelle est 0 [`ai/ppo.py`]
- [x] [Review][Patch] Déviation de la signature de retour du LSTM [`ai/ppo.py`]
- [x] [Review][Patch] Continuité d'épisode brisée entre les appels `collect_rollout` (reset inconditionnel) [`distributed/worker.py`]
- [x] [Review][Defer] Mise à jour SGD basique sans Adam/Clipping [ai/ppo.py] — deferred, pre-existing
- [x] [Review][Defer] Absence de Minibatches et d'Epochs PPO [distributed/train.py] — deferred, pre-existing
- [x] [Review][Defer] Absence de Generalized Advantage Estimation (GAE) [distributed/worker.py] — deferred, pre-existing
- [x] [Review][Defer] Absence de clipping sur la Value Function [ai/ppo.py] — deferred, pre-existing
- [x] [Review][Defer] Goulot d'étranglement synchrone via `ray.get` [distributed/train.py] — deferred, pre-existing
- [x] [Review][Defer] Extracteur de caractéristiques (Feature Extractor) trop superficiel [ai/ppo.py] — deferred, pre-existing
- [x] [Review][Defer] Absence d'Action Masking [ai/ppo.py] — deferred, pre-existing
- [x] [Review][Defer] Hyperparamètres codés en dur [distributed/worker.py] — deferred, pre-existing
- [x] [Review][Defer] Stub Fake pour la base de données [distributed/train.py] — deferred, pre-existing
