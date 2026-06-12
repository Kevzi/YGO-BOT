---
epic: 4
story: 1
title: "Synchronisation et Cache YGOPRODeck"
status: done
baseline_commit: 28324257d377740e7731aca336064ad73d2f9737
---

# Story 4.1: Synchronisation et Cache YGOPRODeck

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Développeur / Chercheur,
I want de créer un module client (ou un script d'ingestion) capable d'interroger l'API YGOPRODeck,
so that je puisse télécharger, mettre à jour, et cacher localement (dans SQLite) toutes les métadonnées officielles des cartes (noms, textes, propriétés) nécessaires au bon fonctionnement de l'environnement Python.

## Acceptance Criteria

1. **Given** une commande ou un endpoint de synchronisation
   **When** on l'exécute
   **Then** les dernières données de l'API YGOPRODeck sont récupérées en respectant la limite de rate-limit (20 requêtes/seconde max)
   **And** la table SQLite des cartes est mise à jour et sert de source de vérité exclusive pour éviter le bannissement d'IP.

## Tasks / Subtasks

- [x] Task 1: Ajouter un client HTTP asynchrone (ex: `httpx` ou `requests`) via Poetry
- [x] Task 2: Définir le modèle SQLAlchemy `Card` (ou `CardMetadata`)
  - [x] Subtask 2.1: Créer le modèle dans `db/models.py` avec les champs utiles (id/passcode, name, type, desc, race, archetype, etc.)
  - [x] Subtask 2.2: Générer la migration Alembic correspondante et l'appliquer (`alembic upgrade head`)
- [x] Task 3: Créer le script de synchronisation YGOPRODeck
  - [x] Subtask 3.1: Implémenter le téléchargement depuis `https://db.ygoprodeck.com/api/v7/cardinfo.php`
  - [x] Subtask 3.2: Implémenter l'insertion/mise à jour (upsert) en base SQLite via SQLAlchemy
  - [x] Subtask 3.3: Gérer le rate-limit et les erreurs HTTP potentielles (Fail Fast si l'API est HS)
- [x] Task 4: Tests automatisés
  - [x] Subtask 4.1: Ajouter un test mockant l'API YGOPRODeck pour vérifier le parsing et l'insertion SQLite sans faire d'appels réseau réels

## Dev Notes

- **Dependencies**: La stack technique requiert l'ajout d'une librairie HTTP (ex: `poetry add httpx` ou `requests`).
- **Database Rules**: Respecter la convention de nommage de l'architecture : tables SQL au pluriel en `snake_case` (ex: `cards`), colonnes en `snake_case`. Utiliser SQLite (déjà en place).
- **Rate-Limiting**: Le point de terminaison de l'API renvoie généralement toutes les données d'un coup, mais si des appels multiples sont faits (ex: images ou lots), la limite stricte de 20 req/sec doit être codée pour éviter les bans.
- **Architecture**: L'API FastAPI (`api/`) ou l'environnement RL (`core/ygoenv/`) consulteront ensuite cette base locale au lieu d'appeler YGOPRODeck en direct, conformément au critère d'acceptation "source de vérité exclusive".
- **Code Naming**: Les fonctions et variables Python doivent être en `snake_case`.

### Project Structure Notes

- Créer un script comme `scripts/sync_ygoprodeck.py` ou un module dans un nouveau package `data/` selon les préférences.
- Les tests devront aller dans `tests/` en suivant la même arborescence.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.1]
- [Source: _bmad-output/planning-artifacts/architecture.md#Naming Patterns]

### Review Findings
- [ ] [Review][Patch] Destructive Alembic Migration History (rewrote history instead of appending) [alembic/versions]
- [ ] [Review][Patch] Missing Test Dependency (pytest-httpx missing in pyproject.toml) [pyproject.toml]
- [ ] [Review][Patch] Database Pollution in Tests (uses live DB) [tests/scripts/test_sync_ygoprodeck.py]
- [ ] [Review][Patch] Missing Critical Game Attributes (scale, linkval) [db/models.py]
- [ ] [Review][Patch] Dangerous Process Termination (sys.exit in async function) [scripts/sync_ygoprodeck.py]
- [ ] [Review][Patch] KeyError risk on missing API keys [scripts/sync_ygoprodeck.py]
- [ ] [Review][Patch] Uncaught SQLAlchemyError [scripts/sync_ygoprodeck.py]
- [ ] [Review][Patch] Insufficient Column Sizing Risk (desc should be Text) [db/models.py]
- [ ] [Review][Patch] Reckless Path Manipulation (sys.path.insert) [scripts/sync_ygoprodeck.py]
- [ ] [Review][Patch] Shadowing Built-in Types (type column name) [db/models.py]
- [ ] [Review][Patch] Naive Rate-Limiting Implementation [scripts/sync_ygoprodeck.py]
- [x] [Review][Defer] Hard SQLite Dialect Lock-in (architecture specifies SQLite for now) [scripts/sync_ygoprodeck.py] — deferred, pre-existing
- [x] [Review][Defer] Missing Indexing Strategy (atk, def, etc.) [db/models.py] — deferred, pre-existing

## Dev Agent Record

### Agent Model Used

Antigravity

### Debug Log References

### Completion Notes List

### File List

