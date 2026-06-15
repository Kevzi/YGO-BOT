# Retrospective: Epic 6 - Intelligence Artificielle et Planification (Deep RL)

Status: done

## 1. Vue d'Ensemble
L'Epic 6 représentait le cœur de l'intelligence artificielle du bot : la transition d'un simple réseau de neurones vers un agent PPO complet avec mémoire (LSTM), capacité de planification (MCTS), et surtout, une architecture d'entraînement distribué massive (Ray).

### Objectifs Atteints :
- **Story 6.1 (LSTM & Belief State) :** Intégration réussie des couches LSTM dans le réseau JAX/Flax pour traiter la séquence d'observations temporelles et gérer l'information imparfaite (Brouillard de Guerre).
- **Story 6.2 (MCTS Gumbel AlphaZero) :** Implémentation de l'algorithme MCTS pour explorer l'arbre des possibles. 
- **Story 6.3 & 6.4 (League Training Asynchrone avec Ray) :** Construction du pipeline distribué `train_distributed.py`. Le modèle tourne désormais sur des RolloutWorkers indépendants (CPU) qui envoient leurs batchs d'expérience via une Queue au Learner central (GPU RTX 3070 Ti).

## 2. Succès Majeurs
- **Scalabilité de l'Architecture :** Le passage sur Ray permet de générer des milliers de transitions par seconde de manière asynchrone. L'utilisation du matériel est optimisée.
- **Résolution des Fuites Mémoire (OOM) :** La compilation JIT (`@jax.jit`) de la fonction d'inférence (`_forward_jit`) a radicalement réduit la consommation de RAM Python lors des rollouts, stabilisant totalement les workers.
- **Apprentissage Actif :** Les logs du Learner (Loss, Value Loss, Entropy) démontrent que les gradients s'appliquent correctement et que le réseau commence à calibrer son Critic.

## 3. Leçons Apprises (Pain Points & Pivots)
- **Le Gouffre du "Cold Start" :** Lancer un algorithme MCTS sur un réseau dont la politique initiale est purement aléatoire (1 chance sur 250) s'est avéré inefficace et beaucoup trop coûteux en temps de calcul. Le MCTS explore des branches absurdes car le réseau n'a pas "d'intuition" de base.
- **Pivot Stratégique :** Ce constat nous a forcés à prendre une décision architecturale brillante : **désactiver le MCTS temporairement** et introduire l'**Epic 7 (Behavioral Cloning)**. Avant de demander à l'IA d'inventer des stratégies par elle-même (Self-Play), nous allons utiliser des replays humains (`.yrp`) pour "bootstrapper" sa politique.

## 4. Prochaines Étapes
- Mettre en pause le Self-Play asynchrone (Epic 6).
- Exécuter l'Epic 7 (Apprentissage Supervisé) pour amorcer les poids du réseau PPO via l'Imitation de replays humains.
- Une fois le réseau amorcé, réactiver l'Epic 6 (League Training) pour l'amélioration continue.
