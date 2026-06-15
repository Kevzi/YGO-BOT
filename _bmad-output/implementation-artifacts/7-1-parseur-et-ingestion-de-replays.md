---
baseline_commit: 76fa931985ce5dbe7a325c8747f822ba8417d466
---

# Story 7.1: Parseur et Ingestion de Replays (.yrp)

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Développeur / Data Engineer,
I want lire les fichiers de replay .yrp et les faire rejouer de manière silencieuse par le moteur C++ (ocgcore / ygoenv),
so that je puisse extraire l'état (observation) et l'action jouée à chaque étape pour les insérer dans la table SQLite game_transitions.

## Acceptance Criteria

1. **Given** un fichier replay `.yrp` valide
   **When** le script d'ingestion est exécuté
   **Then** le replay est lu et rejoué par le moteur de règles C++ (sans UI)
   **And** chaque état du jeu (les 156 slots + infos) et l'action sélectionnée sont extraits et enregistrés dans la table SQLite `game_transitions`.
2. Les données insérées en base doivent être compatibles pour le futur DataLoader JAX (Story 7.2) : format tensoriel / indices corrects, ou structure facile à parser.

## Tasks / Subtasks

- [x] Task 1: Implémentation du parsing `.yrp` (AC: 1)
  - [x] Subtask 1.1: Créer le script `scripts/ingest_replays.py`
  - [x] Subtask 1.2: Lire l'en-tête et les actions du fichier binaire `.yrp`
- [x] Task 2: Connexion au moteur Gym/C++ pour simulation silencieuse (AC: 1)
  - [x] Subtask 2.1: Lancer un environnement `ygoenv` configuré pour charger le deck et les options du replay
  - [x] Subtask 2.2: Avancer l'environnement step by step en soumettant les actions du replay
- [x] Task 3: Historisation dans SQLite (AC: 1, 2)
  - [x] Subtask 3.1: Configurer la session SQLAlchemy pour se connecter à `game_transitions`
  - [x] Subtask 3.2: Insérer l'observation (`observation`) et l'action à chaque `step()` de l'environnement

## Dev Notes

- Relevant architecture patterns and constraints:
  - SQLAlchemy doit être utilisé pour écrire dans la base (fichiers dans `db/`). Noms de table en `snake_case`.
  - La logique d'observation à extraire doit être la même que l'observation de PPO (brouillard de guerre, 156 slots).
  - L'implémentation doit être typée (Type Hints).
  - JAX array conversion is not strictly needed inside the DB, maybe store as JSON array or serialized binary for fast loading.
  - Fail Fast error handling: Any parsing error on `.yrp` must properly abort without corrupted data.

### Project Structure Notes

- `scripts/ingest_replays.py`: script principal pour lancer l'ingestion massive.
- `core/ygoenv/`: peut nécessiter une adaptation mineure pour charger un duel depuis une "seed" ou un replay spécifique.
- `db/models.py`: contient le modèle `GameTransition`.
- `data/`: dossier où sont stockés / attendus les `.yrp`.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 7]
- [Source: _bmad-output/planning-artifacts/architecture.md#Data Architecture]

## Dev Agent Record

### Agent Model Used
Antigravity

### Debug Log References
- Parsing YRP files uses standard struct format with lzma decompression.
- Bypassed Gym step() to push binary actions directly via OCG_DuelSetResponse.
- Included comprehensive test tests/scripts/test_ingest_replays.py with a mock YRP file.

### Completion Notes List
- ✅ Implemented `scripts/ingest_replays.py` to parse YGOPro YRP replays and extract decks + actions.
- ✅ Connected to YgoEnv/OCG_CreateDuel with precise seed parsing and simulated steps silently.
- ✅ Saved complete observations (156 slots) and actions into SQLite `game_transitions`.
- ✅ Data is compatible for later DataLoader usage.

### File List
- `scripts/ingest_replays.py`
- `tests/scripts/test_ingest_replays.py`
