# Story 1.2: Intégration du Moteur C++ (ygoenv / ocgcore)

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Développeur,
I want d'interfacer le moteur de règles ygopro-core (C++) via le wrapper Python `core/ygoenv/`,
So that le backend puisse valider les actions, vérifier la légalité des coups, et ne pas avoir à recalculer les règles de Yu-Gi-Oh!.

## Acceptance Criteria

1. **Given** l'envoi d'un état de jeu valide
2. **When** le moteur est sollicité
3. **Then** il retourne les actions légales possibles
4. **And** les erreurs éventuelles du moteur C++ lèvent une exception claire interceptable par l'API.

## Tasks / Subtasks

- [ ] Task 1: Setup et Compilation du Moteur C++ (AC: 1, 2)
  - [ ] Récupérer le code source de `ygopro-core` ou `ocgcore` dans `core/ocgcore` (ou configurer une dépendance binaire existante).
  - [ ] Configurer le script de build ou `pyproject.toml` pour compiler le moteur si nécessaire.
- [ ] Task 2: Création du Wrapper Python `ygoenv` (AC: 3, 4)
  - [ ] Développer l'interface (via `ctypes`, `cffi`, `pybind11` ou un package existant) pour instancier le moteur depuis Python dans `core/ygoenv/`.
  - [ ] Implémenter une méthode pour transmettre un état de duel et récupérer les actions légales.
  - [ ] Implémenter une gestion d'exceptions stricte (Fail Fast) pour intercepter les crashs du C++ et lever une erreur Python dédiée (ex: `EngineCrashError`).
- [ ] Task 3: Tests Unitaires / Intégration (AC: 1, 2, 3, 4)
  - [ ] Écrire un test Pytest dans `tests/test_engine.py` vérifiant l'initialisation du moteur et l'extraction d'une action légale (sans erreur).

## Dev Notes

### Architecture Compliance
- **Fail Fast:** L'architecture exige que toute erreur du moteur C++ ne soit pas retry, mais lève une exception qui sera interceptée plus tard (Story 1.3) pour renvoyer un JSON 500 structuré (`{"error": {"code": "ENGINE_CRASH", ...}}`).
- **Séparation Logique:** Toute la logique du jeu DOIT rester dans `ocgcore`. Le code Python ne fait que wrapper et passer les appels.
- **Interopérabilité:** Veiller à ce que les états remontés par le C++ puissent être facilement transformés en dictionnaire (pour être historisés en base via la Story 1.1).

### Library & Framework Requirements
- Outils de binding C++ pour Python (ctypes, cffi, cython, ou pybind11) selon ce qui s'intègre le mieux avec `ocgcore`.
- Python 3.11.

### File Structure Requirements
- `core/ocgcore/` : Code source ou binaire du moteur C++.
- `core/ygoenv/` : Module Python encapsulant l'appel au C++ (qui servira de base pour le wrapper Gym de l'Epic 2).

### Testing Requirements
- `pytest tests/test_engine.py` doit s'exécuter avec succès.
- Il est critique que le test déclenche intentionnellement une erreur moteur (état invalide) pour valider que l'exception Python est bien levée (Test du Fail Fast).

### Previous Story Intelligence
- La Story 1.1 a configuré SQLite, SQLAlchemy et Pydantic. Bien que non dépendant, il faut garder à l'esprit que les transitions (GameTransition) enregistreront le `state` (JSON) et `action` (JSON) renvoyés par ce wrapper C++.

### Project Context Reference
- [Source: epics.md#Epic 1: La Fondation du Sparring-Partner]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
