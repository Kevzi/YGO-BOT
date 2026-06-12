---
epic: 3
story: 4
title: Sémantique "Zero-Shot" (Modèle Hybride)
status: done
baseline_commit: 142644830760e1541b0b8ff531389426e9805b97
---

# Story 3.4: Sémantique "Zero-Shot" (Modèle Hybride)

Status: done

## Story

As a Chercheur / Joueur,
I want que le modèle PPO intègre les vecteurs d'embeddings plutôt que de simples IDs entiers pour représenter les cartes en entrée,
So that l'agent puisse transférer sa connaissance des effets de cartes validées vers des cartes qu'il n'a jamais affrontées durant son entraînement (capacité de généralisation Zero-Shot).

## Acceptance Criteria

1. **Given** l'apparition d'une carte totalement inconnue du modèle (absente de `code_list.txt` mais présente dans `embed.pkl` ou inversement)
2. **When** l'agent l'évalue via l'observation retournée par l'environnement
3. **Then** l'environnement utilise le chargeur sémantique pour résoudre le vecteur (retournant potentiellement un vecteur nul si la carte n'existe pas dans les embeddings)
4. **And** l'agent extrapole un coup de manière mathématiquement fluide, sans lever d'erreur JAX ni crasher.

## Tasks / Subtasks

- [x] Task 1: Mettre à jour l'espace d'observation de `YgoEnv`
  - [x] Subtask 1.1: Modifier `self.observation_space` pour correspondre à la dimension attendue du terrain combiné aux embeddings (ex: N cartes * embedding_dim).
  - [x] Subtask 1.2: Câbler `_get_observation` pour convertir une liste simulée d'IDs de cartes en un seul tenseur plat via `embed_loader.get_embeddings_batch`.
- [x] Task 2: Adapter les tests de l'environnement Gym
  - [x] Subtask 2.1: Créer un test unitaire simulant la présence d'un ID inconnu (ex: `999999999`) pour valider la robustesse "Zero-Shot".
- [x] Task 3: Mise à jour du réseau de neurones PPO
  - [x] Subtask 3.1: Créer un test dans `test_agent.py` vérifiant que l'agent PPO accepte cette nouvelle forme d'observation sans erreur de dimension (Shape Error).

## Dev Notes

- **Dimensionnement** : Le vecteur retourné par `YgoEnv._get_observation()` doit correspondre exactement à `obs_dim` utilisé pour initialiser l'agent PPO. Pensez à ajuster `obs_dim` dans les tests PPO en conséquence (ex: `num_cards * embedding_dim`).
- **Gestion du Zero-Shot** : La classe `EmbeddingLoader` retourne déjà un vecteur de zéros si l'ID n'est pas trouvé. Assurez-vous que cette logique est correctement déclenchée dans l'environnement.
- **Performances** : Utilisez `get_embeddings_batch` avec un tableau Numpy d'IDs de cartes plutôt qu'une boucle Python pour garantir la rapidité de la conversion avant l'envoi au cerveau JAX.

### Architecture Compliance
- *Séparation Logique* : C'est au wrapper Python (`ygoenv`) de mapper les IDs en vecteurs, pas à l'agent JAX qui reste mathématiquement "pur" (ne connaît que les floats).
- *Data Exchange Formats* : L'observation retournée doit être explicitement de type `np.float32` (convertie ensuite en `jnp.float32` dans MCTS/Agent).

### Previous Story Intelligence
*(Issues and learnings from Story 3.3 to remember for 3.4)*
- **JAX/Numpy Overheads** : La vectorisation a été optimisée dans `EmbeddingLoader` pour utiliser le masquage Numpy. L'environnement doit envoyer des tableaux Numpy (`np.array`) et non des listes Python à `get_embeddings_batch()`.
- **Fail Fast** : Si `embed_loader` n'est pas chargé, `YgoEnv` doit lever une erreur claire ou appeler `load()` plutôt que de faillir silencieusement, bien que pour les tests un comportement mocké (vecteurs vides) soit acceptable.

### File List
*(Will be populated by dev agent)*

### Change Log
*(Will be populated by dev agent)*

## Dev Agent Record

### Implementation Plan
*(Will be populated by dev agent)*

### Completion Notes
*(Will be populated by dev agent)*

### Review Findings
- [x] [Review][Patch] Inconsistent Documentation vs Reality [`tests/core/ygoenv/test_env.py`]
- [x] [Review][Patch] Silent failure when embedding loader is not initialized [`core/ygoenv/env.py`]
- [x] [Review][Patch] Observation Shape Mismatch Risk on Lazy Loading [`core/ygoenv/env.py`]
- [x] [Review][Patch] Zero-Shot Unit Test bypasses Environment Observation Logic [`tests/core/ygoenv/test_env.py`]
- [x] [Review][Patch] `embed_loader.load()` unhandled exception skips finally block [`tests/core/ygoenv/test_env.py`]
- [x] [Review][Patch] Redundant `ray.get(future)` call [`tests/distributed/test_ray_pipeline.py`]
- [x] [Review][Defer] Brittle Hardcoded Actions [`tests/core/ygoenv/test_env.py`] — deferred, pre-existing
- [x] [Review][Defer] Encapsulation Violation - State [`tests/core/ygoenv/test_env.py`] — deferred, pre-existing
- [x] [Review][Defer] Coupling of DB Contexts [`tests/distributed/test_ray_pipeline.py`] — deferred, pre-existing test environment constraint
- [x] [Review][Defer] Degraded Test Strictness [`tests/test_engine.py`] — deferred, pre-existing mock constraints
- [x] [Review][Defer] Language Inconsistency — deferred, not a blocker for this story
- [x] [Review][Defer] Test is executed without a mock database fixture injected [`tests/distributed/test_ray_pipeline.py`] — deferred, pre-existing
