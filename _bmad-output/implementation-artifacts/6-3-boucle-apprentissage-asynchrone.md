---
baseline_commit: NO_VCS
---

# Story 6.3: Boucle d'Apprentissage Asynchrone (Self-Play)

**Epic:** Epic 6: Intelligence Artificielle et Planification (Deep RL)

As a Chercheur en IA,
I want de mettre en place une boucle d'auto-jeu asynchrone (Self-Play),
So that l'agent puisse constamment affronter des versions antérieures de lui-même et collecter des trajectoires pour améliorer continuellement sa stratégie et éviter d'exploiter un adversaire aléatoire.

## Acceptance Criteria

- [ ] **Given** l'exécution du script `scripts/train.py`
- [ ] **When** les environnements parallèles sont lancés
- [ ] **Then** les agents jouent asynchronement contre des itérations passées de leur propre modèle
- [ ] **And** les trajectoires générées mettent à jour les poids du réseau JAX PPO de manière stable sans s'enfermer dans un optimum local trivial.

## Tasks/Subtasks

- [ ] Adapter `scripts/train.py` pour orchestrer le Self-Play (pool de modèles).
- [ ] Utiliser l'ActorCriticLSTM (et potentiellement le MCTS) dans la collecte des trajectoires (RolloutWorker).
- [ ] Maintenir un pool d'adversaires historiques (anciens checkpoints du modèle) contre lesquels s'entraîner.
- [ ] Gérer l'alternance des tours entre le modèle principal et le modèle historique dans l'environnement.
- [ ] Ajuster le calcul du GAE (Generalized Advantage Estimation) pour s'aligner sur la perspective du joueur apprenant.

## Dev Notes

**Architecture Requirements:**
- Refonte de `scripts/train.py` (ou création de `ai/self_play.py`) pour gérer de multiples itérations de l'agent.
- La boucle doit conserver les checkpoints périodiquement et tirer au sort un adversaire historique.
- L'environnement (`YgoEnv`) simule un duel entre player 0 et player 1. Il faut que l'Actor courant contrôle un joueur, et l'historique l'autre, tout en gérant les observations et actions depuis la bonne perspective.
- L'entraînement doit se faire avec l'omniscience activée (`YgoEnv(omniscience=True)`) pendant le MVP.

**Previous Learnings:**
- MCTS a été implémenté (`ai/mcts.py`) et l'environnement utilise `action_history` pour cloner l'état. Le MCTS peut s'avérer lent si utilisé systématiquement pour chaque pas d'entraînement sans JIT.
## Review Findings

- [x] [Review][Patch] Missing Terminal Rewards from Opponent Turns [scripts/train.py]
- [x] [Review][Patch] Flawed Truncation Handling / GAE Bootstrapping [scripts/train.py]
- [x] [Review][Patch] Zero Probability / Underflow Crash in Action Masking [scripts/train.py]
- [x] [Review][Patch] Opponent evaluates observations from the wrong perspective [scripts/train.py]
- [x] [Review][Patch] NoneType iteration crash in `env.py` [core/ygoenv/env.py]
- [x] [Review][Patch] Infinite Loop Risk in Rollouts [scripts/train.py]
- [x] [Review][Patch] Dead Code in GAE Calculation [scripts/train.py]
- [x] [Review][Defer] Inaccurate FPS Metrics [scripts/train.py] — deferred, pre-existing
- [x] [Review][Defer] Resource Leaks (env.close()) [scripts/train.py] — deferred, pre-existing
- [x] [Review][Defer] Naïve State Stacking for Recurrent Networks [scripts/train.py] — deferred, pre-existing
- [x] [Review][Defer] Brittle Path Injection [scripts/train.py] — deferred, pre-existing
- [x] [Review][Defer] Hardcoded Randomness [scripts/train.py] — deferred, pre-existing

## Dev Notes

**Implementation Plan:**

**Completion Notes:**

## File List

- `scripts/train.py` [UPDATE]
- `ai/self_play.py` [NEW]

## Change Log

## Status
ready-for-dev
