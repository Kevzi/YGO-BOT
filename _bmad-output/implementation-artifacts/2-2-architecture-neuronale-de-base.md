---
baseline_commit: 142644830760e1541b0b8ff531389426e9805b97
---

# Story 2.2: Architecture Neuronale de Base (PPO sous JAX)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Chercheur en IA,
I want de développer la structure réseau neuronale de base de l'agent en utilisant JAX et d'implémenter l'algorithme PPO (Proximal Policy Optimization),
So that l'agent puisse mettre à jour sa politique en fonction des récompenses (rewards) obtenues dans l'environnement Gym.

## Acceptance Criteria

1. **Given** un lot (batch) d'observations issues de l'environnement
   **When** l'agent exécute sa "forward pass"
   **Then** il produit des probabilités d'action valides (Policy) et une estimation de la valeur de l'état (Value)
   **And** les fonctions JAX de mise à jour respectent la pureté fonctionnelle et sont compilables avec `jax.jit`.

## Tasks / Subtasks

- [x] Créer le modèle neuronal (Policy & Value networks)
  - [x] Définir l'architecture du réseau de neurones en JAX (utiliser Flax ou Haiku si nécessaire, ou JAX pur)
  - [x] Implémenter la tête "Policy" (distribution de probabilité sur les actions légales)
  - [x] Implémenter la tête "Value" (estimation de la valeur de l'état)
- [x] Implémenter la "forward pass" et `jax.jit`
  - [x] Assurer que l'inférence est pure et compilable avec `@jax.jit`
  - [x] Gérer l'état RNG de JAX (PRNGKey) correctement
- [x] Implémenter l'algorithme de perte (Loss) PPO
  - [x] Calculer le ratio de politique, le clipping PPO, et l'entropie
  - [x] Calculer la perte de valeur (Value loss)
- [x] Ajouter les tests unitaires
  - [x] Valider que le modèle accepte les observations de `YgoEnv` (taille 100) et retourne des actions valides (taille 200).
  - [x] Valider que la fonction de perte ne renvoie pas de NaN sur un batch fictif.

## Developer Context

### Architecture Compliance
- **Pureté Fonctionnelle JAX:** Les fonctions modifiant ou traitant les tenseurs doivent être des fonctions "pures" sans mutation sur place.
- **Typage Explicite (dtype):** Toutes les opérations tensorielles et vecteurs d'observations doivent impérativement définir explicitement leur type de données (`jnp.float32` pour les réseaux de neurones, `jnp.int32` pour les indices d'action).
- **Component Boundaries:** Le module `ai/` est mathématiquement pur. Il ne connaît pas l'existence de l'API REST ni de la DB. Il prend des tenseurs en entrée (JAX array) et retourne des probabilités/valeurs.
- **Portabilité (NFR-3):** Le code JAX doit supporter l'inférence en mode CPU de façon transparente (fallback via `--xla_device cpu`).

### File Structure Requirements
- Les modèles PPO doivent être créés sous `ai/agent.py` ou `ai/ppo.py`.
- Les tests unitaires correspondants doivent être sous `tests/ai/test_agent.py`.

### Previous Story Intelligence (from 2.1)
- L'environnement `YgoEnv` a un espace d'observation mocké de forme `(100,)` en `np.float32`.
- L'espace d'action est `Discrete(200)`.
- Les tests de typage ont montré qu'il est crucial d'utiliser les types standards (ex: `tuple`, `dict`) au lieu de `Tuple`, `Dict` pour Python 3.9+.
- Le mock MVP de l'environnement est géré par la classe `YgoEngine`. Assurez-vous de bien respecter ces tailles et types de tenseurs lors des tests JAX.

### Testing Requirements
- Pytest est utilisé pour valider les modèles JAX.
- Vous devez valider le bon fonctionnement de `jax.jit` sur la fonction d'inférence, pour s'assurer qu'aucune erreur de shape dynamique ne survient.

## References
- [Source: _bmad-output/planning-artifacts/epics.md]
- [Source: _bmad-output/planning-artifacts/architecture.md]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List
- Implémentation du réseau PPO MLP en JAX pur.
- Le modèle prend un espace d'observations de 100 et génère des logits de taille 200 pour la "Policy", ainsi qu'une estimation de la valeur.
- JAX pure functions : la fonction `forward` et `compute_loss` sont purement fonctionnelles, compilables avec `jax.jit`.
- L'algorithme PPO a été implémenté (Policy Loss + PPO Clipping + Value Loss + Entropy Bonus).
- Les tests unitaires vérifient la création, le format `float32` pour `logits` et `values`, ainsi que l'absence de NaN après `compute_loss`.

### File List
- `ai/ppo.py` (NEW)
- `tests/ai/test_agent.py` (MODIFIED)

### Review Findings
- [x] [Review][Patch] Nettoyage du typage et de l'initialisation (supprimer Tuple/Dict, ajouter dtypes explicites, utiliser jax.nn.initializers) [ai/ppo.py:3]
- [x] [Review][Patch] Correction des critères d'acceptation (retourner les probabilités dans le forward pass, implémenter la fonction de mise à jour des paramètres) [ai/ppo.py]
- [x] [Review][Patch] Amélioration de la logique PPO (normaliser les avantages, renommer c1/c2, sécuriser jnp.squeeze) [ai/ppo.py:36]
- [x] [Review][Patch] Ajout de protections Edge Case (dimensions > 0, vérification des shapes et des limites d'actions) [ai/ppo.py:6]
- [x] [Review][Patch] Refactorisation des tests (imports au niveau module, utilisation de fixtures, assertions plus robustes avec données aléatoires) [tests/ai/test_agent.py]
- [x] [Review][Defer] Rigidité architecturale (1 seule couche cachée en dur) [ai/ppo.py:6] — deferred, pre-existing (conception "de base")
- [x] [Review][Defer] Mélange de paradigmes OOP et fonctionnel [ai/ppo.py] — deferred, pre-existing (choix structurel provisoire)

