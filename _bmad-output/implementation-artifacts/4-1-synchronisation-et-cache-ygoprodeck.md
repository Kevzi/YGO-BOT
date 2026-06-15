---
epic: 4
story: 1
title: "Synchronisation et Cache YGOPRODeck"
status: ready-for-dev
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

- [ ] Task 1: Création du modèle de base de données SQLite
  - [ ] Subtask 1.1: Définir le modèle Pydantic/SQLAlchemy pour les cartes.
  - [ ] Subtask 1.2: Ajouter la migration Alembic associée.
- [ ] Task 2: Intégration API YGOPRODeck
  - [ ] Subtask 2.1: Implémenter le client HTTP (ex: httpx) avec gestion stricte du rate limit (max 20 requêtes/s).
  - [ ] Subtask 2.2: Gérer le téléchargement des métadonnées (passcode, name, text, properties).
- [ ] Task 3: Ingestion et Mise en Cache
  - [ ] Subtask 3.1: Sauvegarder les données récupérées en base SQLite.
  - [ ] Subtask 3.2: Implémenter la logique d'upsert (mise à jour si existant, création sinon).
- [ ] Task 4: Gestion des erreurs
  - [ ] Subtask 4.1: Implémenter le pattern "Fail Fast" en cas de panne réseau sans retry silencieux.

## Dev Notes

### Technical Requirements
- Language/Framework: Python (FastAPI/Scripts).
- Database: SQLite avec requêtes Alembic pour la migration.
- Error Handling: "Fail Fast", utiliser l'approche définie dans l'architecture où toute défaillance retourne une erreur formelle sans retry.
- Rate Limiting: 20 req/s strict maximum pour YGOPRODeck.

### Architecture Compliance
- Code style: Naming patterns en camelCase pour l'API REST JSON (via Pydantic `alias_generator = to_camel`) et snake_case pour le code Python interne et les tables SQLite.
- Database: Respecter l'utilisation de SQLAlchemy + Alembic existante.
- Éviter d'utiliser `sys.exit(1)` dans les fonctions internes, lever des exceptions appropriées à la place.

### File Structure Requirements
- Le modèle de base de données doit être dans les fichiers de modèles (`db/models.py`).
- Le script de synchronisation doit être dans le dossier scripts ou un module dédié (ex: `data/sync.py`).

### Testing Requirements
- Mocker l'API HTTP YGOPRODeck lors des tests (utiliser pytest-httpx ou équivalent).
- Les tests ne doivent pas modifier la base de données de production. Utiliser une base de données en mémoire (ou isolée) via pytest fixtures.

### Previous Story Intelligence
- Les migrations Alembic doivent ajouter aux existantes, ne pas écraser l'historique de l'Epic 1.

### References
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4: Intelligence du Metagame (Connaissance externe)]
- [Source: _bmad-output/planning-artifacts/architecture.md]

## Dev Agent Record

### Agent Model Used
Antigravity

### Debug Log References

### Completion Notes List
- Ultimate context engine analysis completed - comprehensive developer guide created for 4-1.

### File List
