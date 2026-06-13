---
status: ready-for-dev
---

# Story 6.4: Distribution Massive et League Training via Ray

**Epic:** Epic 6: Intelligence Artificielle et Planification (Deep RL)

As a Chercheur en IA / DevOps,
I want de déployer notre boucle d'entraînement via le framework Ray,
So that l'apprentissage puisse passer à l'échelle sur des clusters multi-GPU et multi-cœurs (League Training), gérant l'espace d'états gigantesque de Yu-Gi-Oh!.

## Acceptance Criteria

- [ ] **Given** le besoin d'augmenter drastiquement le nombre de parties simulées
- [ ] **When** le script de déploiement est lancé via le cluster Ray
- [ ] **Then** Ray orchestre la distribution des calculs, équilibrant la charge des différents acteurs MCTS et du Learner central
- [ ] **And** le League Training gère dynamiquement plusieurs pools de versions d'agents (decks méta, historiques, etc.) sans goulot d'étranglement mémoire.

## Tasks/Subtasks

- [ ] Configurer un script d'entraînement distribué (`scripts/train_distributed.py`) utilisant l'API `ray.remote`.
- [ ] Séparer l'architecture en un acteur "Learner" (centralise l'optimisation des poids PPO sur GPU) et de multiples acteurs "Workers" (collectent les trajectoires en CPU ou petit GPU).
- [ ] Permettre la communication des poids (broadcast) du Learner vers les Workers.
- [ ] Implémenter le "League Training" : les Workers tirent au sort des adversaires depuis différents pools (passé, meta-decks, aléatoire).
- [ ] S'assurer que le stockage des historiques (Puffer/Replay Buffer) ne crée pas de goulot d'étranglement lors du retour des trajectoires vers le Learner.

## Dev Notes

**Architecture Requirements:**
- Ray est déjà installé et configuré localement.
- Le Learner doit faire des pas d'optimisation JAX de manière asynchrone pendant que les RolloutWorkers génèrent des données.
- Utiliser un Object Store de Ray ou un `ray.util.queue.Queue` pour transférer les batchs d'expérience.
- Les workers instancient `YgoEnv` et effectuent les `collect_rollout()`.

**Previous Learnings (Story 6.3):**
- La boucle de Self-Play locale a été validée. L'environnement doit avoir `omniscience=True` pour éviter les soucis de masquage aveugle.
- Des correctifs importants ont été faits dans `collect_rollout`:
  - `done` est intercepté si la partie se finit pendant le tour de l'adversaire (récupération du reward).
  - Le calcul GAE a été corrigé avec un `bootstrap_value` correct pour la troncature (`truncated`).
  - L'observation pour l'adversaire est bien calculée depuis sa propre perspective (`_get_observation(player=current_player)`).
  - La division par zéro sur les probabilités d'action a été résolue.
- La logique distribuée doit s'inspirer de `scripts/train.py` en s'assurant que ces correctifs soient préservés dans le code distribué.

**Tech Specifics:**
- Utiliser `ray.init(ignore_reinit_error=True)` pour la flexibilité.
- Les acteurs Worker doivent s'exécuter avec `@ray.remote(num_cpus=1)`.
- Le Learner peut s'exécuter avec `@ray.remote(num_gpus=1)` si un GPU est disponible.

## File List

- `scripts/train_distributed.py` [NEW]
- `ai/distributed.py` [NEW] (optionnel, pour l'abstraction)

## Completion Notes
Ultimate context engine analysis completed - comprehensive developer guide created.
