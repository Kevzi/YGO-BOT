---
baseline_commit: a48a14a00d6c080546cb3e524d95c043de5ed15b
---

# Story 1.1: Base de Données et Historisation (SQLite + Alembic)

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Développeur / Chercheur,
I want d'initialiser la base de données locale (SQLite) avec Alembic et créer les schémas initiaux (`duel_stats`, `game_transitions`),
so that l'historique des duels puisse être enregistré dès le premier test fonctionnel de l'API.

## Acceptance Criteria

1. **Given** une base de données vierge
2. **When** on exécute `alembic upgrade head`
3. **Then** les tables `duel_stats` et `game_transitions` (en `snake_case`) sont créées
4. **And** le modèle Pydantic de base est prêt pour l'insertion.

## Tasks / Subtasks

- [x] Task 1: Setup Alembic & SQLAlchemy (AC: 1, 2)
  - [x] Installer/Vérifier `alembic` et `sqlalchemy` via Poetry
  - [x] Initialiser l'environnement alembic (`alembic init alembic`)
  - [x] Configurer `alembic.ini` et `env.py` pour pointer vers une base SQLite locale (ex: `data/ygo.db`)
- [x] Task 2: Définition des Modèles SQLAlchemy (AC: 3)
  - [x] Créer `models/base.py` avec `DeclarativeBase`
  - [x] Créer le modèle `DuelStats` (table `duel_stats`)
  - [x] Créer le modèle `GameTransition` (table `game_transitions`)
  - [x] Générer la première migration Alembic (`alembic revision --autogenerate`)
- [x] Task 3: Définition des Modèles Pydantic (AC: 4)
  - [x] Créer les schémas Pydantic correspondants dans `schemas/` avec `alias_generator = to_camel` pour assurer l'interopérabilité.

## Dev Notes

### Architecture Compliance
- **Base de données:** SQLite est requise par l'architecture pour éviter les serveurs distants.
- **Conventions de nommage:** Strict `snake_case` dans la base de données SQLite et les fichiers Python.
- **Interopérabilité:** Pydantic `camelCase` (en utilisant `ConfigDict(alias_generator=to_camel, populate_by_name=True)`) pour les futurs échanges API, afin que les clients Omega/Neos reçoivent du `camelCase`.
- **Frameworks:** Utiliser SQLAlchemy 2.0 (style `DeclarativeBase` et `Mapped[T]`) et Pydantic V2.

### Library & Framework Requirements
- SQLAlchemy >= 2.0
- Alembic
- Pydantic >= 2.0
- La base de données SQLite (par exemple `data/ygo.db`) devra idéalement utiliser des verrous ou WAL (`PRAGMA journal_mode=WAL`) pour supporter les futures écritures concurrentes de Ray (Epic 2), même si cette optimisation pourra être peaufinée plus tard.

### Project Structure Notes
- `alembic/` et `alembic.ini` à la racine.
- `src/` ou structure racine avec:
  - `models/` (SQLAlchemy)
  - `schemas/` (Pydantic)
  - `db/` ou `core/` pour la session et la configuration DB.

### References
- [Source: architecture.md#Base de Données et Persistance]
- [Source: epics.md#Epic 1: La Fondation du Sparring-Partner]
- [Source: prd.md]

## Dev Agent Record

### Agent Model Used
Antigravity (DeepMind)

### Debug Log References
- Tests passed locally (test_db.py)

### Completion Notes List
- Initialisation d'Alembic effectuée.
- `alembic.ini` et `env.py` configurés pour SQLite local (`sqlite:///data/ygo.db`).
- Modèles SQLAlchemy `DuelStats` et `GameTransition` créés dans `db/models.py`.
- Migration générée et appliquée (base créée).
- Schémas Pydantic créés dans `schemas/duel.py` avec configuration `to_camel`.
- Tests pytest écrits dans `tests/test_db.py` et exécutés avec succès.

### File List
- `alembic.ini` (modifié)
- `alembic/env.py` (modifié)
- `alembic/versions/*_initial_schema.py` (nouveau)
- `db/__init__.py` (nouveau)
- `db/base.py` (nouveau)
- `db/models.py` (nouveau)
- `schemas/__init__.py` (nouveau)
- `schemas/duel.py` (nouveau)
- `tests/conftest.py` (nouveau)
- `tests/test_db.py` (nouveau)
