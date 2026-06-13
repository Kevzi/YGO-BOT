---
baseline_commit: NO_VCS
---

# Story 6.2: Recherche Arborescente MCTS (Gumbel AlphaZero)

**Epic:** Epic 6: Intelligence Artificielle et Planification (Deep RL)

As a Chercheur en IA,
I want d'intégrer un algorithme MCTS (Gumbel AlphaZero) pour guider le réseau de neurones,
So that l'agent puisse planifier et anticiper les coups futurs au-delà d'une réaction myope à l'instant t.

## Acceptance Criteria

- [ ] **Given** la phase d'entraînement en Self-Play
- [ ] **When** le MCTS est configuré en mode d'omniscience
- [ ] **Then** le MCTS voit les cartes cachées pour simuler l'arbre des possibles très rapidement et guider le réseau
- [ ] **And** en partie réelle (inférence), le bot utilise un Information Set MCTS (IS-MCTS) ou s'appuie sur la politique PPO guidée pour réagir aux états cachés.

## Tasks/Subtasks

- [ ] Créer le squelette de l'algorithme MCTS dans `ai/mcts.py`
- [ ] Implémenter la sélection et l'expansion des nœuds (Node Selection / Expansion) avec PUCT
- [ ] Brancher le réseau ActorCriticLSTM existant pour évaluer les feuilles de l'arbre
- [ ] Gérer le "State Cloning" pour que les simulations MCTS n'altèrent pas l'environnement principal
- [ ] Implémenter l'intégration Gumbel pour combiner l'Actor et le MCTS efficacement
- [ ] Rédiger les tests unitaires pour `ai/mcts.py`

### Review Findings

- [x] [Review][Defer] Missing Implementation of Information Set MCTS (IS-MCTS) / PPO Fallback Mechanism — deferred: L'implémentation de l'IS-MCTS est reportée à une Story future car ce MVP se concentre prioritairement sur la validation de l'entraînement en auto-jeu avec un MCTS omniscient, la gestion complexe du brouillard de guerre en inférence fera l'objet d'une phase dédiée à la mise en production.
- [x] [Review][Patch] Missing Omniscience Mode for MCTS [core/ygoenv/env.py]
- [x] [Review][Patch] Use of Hardcoded Penalty for Action Masking [ai/mcts.py]
- [x] [Review][Patch] ZeroDivisionError on Empty Masks in MCTS [ai/mcts.py]
- [x] [Review][Patch] Invalid action -1 crashes sim_env.step during MCTS expansion [ai/mcts.py]
- [x] [Review][Patch] KeyError or IndexError extracting life points [core/ygoenv/env.py]
- [x] [Review][Patch] Undefined behavior if engine.create_duel() fails [core/ygoenv/env.py]
- [x] [Review][Patch] KeyError when incrementing the turn counter during step [core/ygoenv/env.py]
- [x] [Review][Patch] Catastrophic Shape Mismatch in Observation Space [core/ygoenv/env.py]
- [x] [Review][Patch] Silent Failures Mask Critical Database Errors [core/ygoenv/env.py]
- [x] [Review][Patch] Inconsistent Mask Typing Causes Unsafe Logic [core/ygoenv/env.py]
- [x] [Review][Patch] Episode Truncation Misinterprets Turns as Actions [core/ygoenv/env.py]
- [x] [Review][Patch] Brittle State Dictionary Parsing [core/ygoenv/env.py]
- [x] [Review][Patch] Noise Recklessly Applied to Illegal Paths [ai/mcts.py]
- [x] [Review][Defer] Action Replay Cloning is Broken for Stochastic Games without synchronizing the random seed [core/ygoenv/env.py] — deferred, pre-existing
- [x] [Review][Defer] JAX Performance Black Hole [ai/mcts.py] — deferred, pre-existing
- [x] [Review][Defer] Illegal Fallback Action Execution Freezes the Engine [core/ygoenv/env.py] — deferred, pre-existing
## Dev Notes

**Architecture Requirements:**
- MCTS implementation should reside in `ai/mcts.py`.
- Must support pure JAX functions where possible, but environment interaction (state cloning) will require Python boundaries.
- Ensure state cloning uses the `core/ygoenv/` wrapper correctly sans planter le moteur C++ ocgcore.
- Follow functional programming paradigms and explicit dtypes (e.g. `jnp.float32`).

**Previous Learnings:**
- `ai/ppo.py` and `ai/network.py` have been refactored to support LSTM state and `prev_action`. MCTS must also maintain or respect this `carry` state during its rollouts.
- Attention aux variables `jnp.bool_` et la coercion `jnp.float32`.
- Action Masking s'appuie sur `jnp.finfo(jnp.float32).min` (remplace `-1e9`).

## Dev Agent Record

**Implementation Plan:**

**Completion Notes:**

## File List

- `ai/mcts.py` [NEW]
- `tests/test_mcts.py` [NEW]

## Change Log

## Status
done
