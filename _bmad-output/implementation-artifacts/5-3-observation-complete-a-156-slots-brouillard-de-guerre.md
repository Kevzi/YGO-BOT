# Story 5.3: Observation Complète à 156 Slots (Brouillard de Guerre)

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story Foundation

As a Chercheur en IA,
I want que l'observation de l'agent passe à une représentation matricielle complète du jeu (156 emplacements pour les deux joueurs) incluant un brouillard de guerre,
So that l'agent puisse percevoir tout le terrain, sa main, les cimetières, et "cacher" les informations privées de l'adversaire (cartes face cachée, main).

## Acceptance Criteria

1. **Given** le calcul de l'observation à chaque étape
2. **When** `env.step()` ou `env.reset()` est appelé
3. **Then** l'environnement construit un tenseur `(60694,)` concaténant 156 slots sémantiques et un vecteur d'information globale (LP, Phase)
4. **And** les caractéristiques et embeddings des cartes cachées ou adverses non publiques sont masqués par des zéros.

## Developer Context & Guardrails

### Technical Requirements
- Construire un tenseur d'observation de dimension `(60694,)`.
- L'environnement `core/ygoenv/` doit générer la nouvelle matrice.
- Implémenter le "brouillard de guerre": si une carte appartient à la main de l'adversaire ou est face cachée sur son terrain (non révélée publiquement), son vecteur de caractéristiques/embedding doit être un vecteur de zéros.
- Inclure les informations globales (Life Points, Phase actuelle) en plus des 156 slots sémantiques des cartes.

### Architecture Compliance
- **Zero-Shot / Embeddings**: L'observation doit exploiter l'embedding (depuis `embed.pkl`) chargé en RAM.
- **Fail Fast**: Si un état C++ retourné ne peut être mappé sur le tenseur 60694, lever une erreur propre.
- **Tensor Dtypes**: Le tenseur d'observation renvoyé à Gym doit explicitement définir son dtype (ex: `jnp.float32` ou numpy `np.float32`) pour éviter les crashs JAX en aval.
- Les fonctions JAX en aval dépendent de cette forme exacte `(60694,)`. Toute erreur de taille (Shape Mismatch) dans l'observation crashera l'agent PPO.

### Previous Story Intelligence
- **Story 5.2**: L'environnement a été mis à jour pour mapper les requêtes de cibles sur les actions 191-199 et l'auto-complétion. Il faut s'assurer que la nouvelle génération d'observation n'interfère pas avec l'état dynamique de sélection.
- **Bugs Connus (Epic 6 Reviews)**: 
  - `[Patch] Careless Type Coercion / Dtypes [ai/ppo.py]`
  - `[Patch] Shape Mismatches in Environment Processing [ai/ppo.py]`
  - Attention accrue au respect strict des dimensions `(60694,)` et des types stricts (`float32`).

### File Structure Requirements
- Composant impacté: `core/ygoenv/` (Wrapper Gym Python interagissant avec ocgcore).
- Fichiers de Tests: Les tests liés doivent être implémentés/modifiés (ex. `tests/test_env.py` ou équivalent).

## Project Context Reference
- **Epic**: Epic 5 - Mise à l'échelle de l'Environnement (Architecture Avancée).
- **Langue de communication**: Français (les commentaires internes peuvent rester dans les standards du projet).
- L'inférence finale s'attend à recevoir une observation parfaite pour son LSTM. Le non-respect du brouillard de guerre crée de la triche ("omniscience" non voulue en dehors du Self-Play Mcripts).

## Completion Status
Ultimate context engine analysis completed - comprehensive developer guide created.

## Tasks/Subtasks

- [x] Verify existing implementation of 156-slots and fog of war in `core/ygoenv/env.py`.
- [x] Update test space and mock logic in `tests/core/ygoenv/test_env.py` to match the `(60694,)` observation space shape.

## Dev Agent Record

### Debug Log
- The `core/ygoenv/env.py` already contained the implementation logic for the 156-slots observation vector, fog of war zeros, and proper concatenations resulting in `(60694,)` dimension tensor.
- Verified test failing because it checked `NUM_CARDS * embed_loader._embedding_dim`.
- Updated test logic to properly handle new observation constraints and embedding mock. Tests pass successfully.

### Completion Notes
- ✅ Updated `test_ygoenv_zero_shot` in `test_env.py` to match the exact 60694 constraint shape logic and removed old assertions.
- ✅ All regression tests run successfully and pass, confirming existing `env.py` implementation satisfies story AC.

## File List

- `tests/core/ygoenv/test_env.py` (modified)

## Change Log

- Addressed outdated shape validation in unit tests.
- Re-certified environmental tests for exact dimensionality.
