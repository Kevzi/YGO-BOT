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
- L'environnement gère désormais le mode omniscient (`omniscience=True`), ce qui lève les erreurs liées au masquage aveugle.
- Des correctifs ont été appliqués sur les calculs de probabilité (division par zéro évitée avec `max(..., 1e-8)`) et l'application stricte du `action_mask`.

## Dev Agent Record

**Implementation Plan:**

**Completion Notes:**

## File List

- `scripts/train.py` [UPDATE]
- `ai/self_play.py` [NEW]

## Change Log

## Status
ready-for-dev
