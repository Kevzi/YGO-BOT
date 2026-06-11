---
title: 'Initialisation du Dépôt et Structure du Projet'
type: 'chore'
created: '2026-06-11'
status: 'done'
baseline_commit: 'NO_VCS'
context: [
  "c:/Users/kevin/Downloads/Projet code/ygo bot/_bmad-output/planning-artifacts/architecture.md"
]
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Le projet YGO-BOT dispose d'une architecture scellée mais aucun fichier physique n'a encore été créé. Il faut initialiser le dépôt pour permettre le développement du code Python/C++.

**Approach:** Utiliser Poetry pour initialiser la gestion des dépendances (FastAPI, JAX, Ray, SQLAlchemy/Alembic, Ruff, Pytest), créer l'arborescence complète validée lors de l'Étape 6 de l'architecture, et générer les fichiers de configuration de base (pyproject.toml, .github/workflows/ci.yml, Dockerfile).

## Boundaries & Constraints

**Always:** Respecter stricement l'arborescence définie dans le document d'architecture (`api/`, `core/`, `ai/`, `db/`, `cluster/`, `tests/`).
Utiliser Python 3.10+.

**Ask First:** S'il est nécessaire de modifier les dépendances spécifiées ou l'organisation des dossiers racine par rapport au plan validé.

**Never:** Écrire de la logique métier (code python fonctionnel) dans cette étape. Seuls les fichiers de structure, `pyproject.toml` et un `main.py` vide/minimal doivent être créés.

</frozen-after-approval>

## Code Map

- `pyproject.toml` -- Configuration principale Poetry et règles Ruff.
- `.github/workflows/ci.yml` -- Pipeline d'intégration continue (Linting & Tests).
- `api/main.py` -- Point d'entrée FastAPI (fichiers vides ou boilerplates minimaux).
- `api/security.py` -- Middleware.
- `api/duel_routes.py` -- Routes API.
- `core/ocgcore/` -- Dossier pour le sous-module C++.
- `core/ygoenv/` -- Wrapper Gym.
- `ai/agent.py` -- Cerveau JAX.
- `ai/mcts.py` -- MCTS.
- `ai/embeddings.py` -- Chargement des embeddings.
- `db/alembic/` -- Migrations SQL.
- `db/models.py` -- Schémas SQLAlchemy.
- `cluster/ray_tpu_config.yaml` -- Configuration Ray.
- `tests/` -- Dossier de tests.
- `Dockerfile` -- Conteneurisation.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml` -- Créer la structure `pyproject.toml` incluant les dépendances fastapi, uvicorn, jax, jaxlib, ray, sqlalchemy, alembic, pandas, ruff, pytest, pytest-asyncio.
- [x] `api/`, `core/ocgcore/`, `core/ygoenv/`, `ai/`, `db/alembic/`, `cluster/`, `tests/`, `.github/workflows/` -- Créer les dossiers physiques.
- [x] `api/main.py`, `api/security.py`, `api/duel_routes.py`, `ai/agent.py`, `ai/mcts.py`, `ai/embeddings.py`, `db/models.py`, `cluster/ray_tpu_config.yaml`, `Dockerfile` -- Créer les fichiers vides/minimaux.
- [x] `.github/workflows/ci.yml` -- Créer le fichier CI de base.

**Acceptance Criteria:**
- Given un environnement vierge, when l'initialisation est terminée, then l'arborescence complète existe et correspond exactement au diagramme de l'Étape 6 de l'architecture.
- Given le fichier pyproject.toml, when on analyse ses dépendances, then toutes les bibliothèques requises par l'architecture (JAX, FastAPI, Ray, Alembic) sont présentes.

## Verification

**Commands:**
- `poetry check` -- expected: Le fichier pyproject.toml est valide.
- `ls -R` (ou équivalent) -- expected: L'arborescence définie est présente sur le disque.

## Suggested Review Order

**Configuration des dépendances**

- Définition du projet et de la stack (Poetry, JAX, FastAPI, Ray)
  [`pyproject.toml:1`](../../pyproject.toml#L1)

**Infrastructure CI et Docker**

- Standardisation de l'environnement pour l'API
  [`Dockerfile:1`](../../Dockerfile#L1)

- Pipeline de vérification continue
  [`ci.yml:1`](../../.github/workflows/ci.yml#L1)
