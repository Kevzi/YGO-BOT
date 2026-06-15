---
epic: 4
story: 2
title: "Interopérabilité du Parseur de Decks (omega-api-decks)"
status: ready-for-dev
---

# Story 4.2: Interopérabilité du Parseur de Decks (omega-api-decks)

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Joueur / Chercheur,
I want de pouvoir charger un deck via un format standard (`.ydk` ou code presse-papier) en envoyant la requête au microservice `omega-api-decks`,
So that l'environnement Python reçoive systématiquement une liste d'IDs entiers (passcodes) propre et validée, peu importe la complexité du format d'entrée.

## Acceptance Criteria

1. **Given** un fichier de deck brut soumis à l'API FastAPI
   **When** FastAPI délègue la requête de conversion au microservice local `omega-api-decks`
   **Then** le microservice le décode et FastAPI extrait les passcodes (Main/Extra/Side)
   **And** la liste d'IDs validée est injectée pour initialiser correctement le deck de l'agent dans l'environnement Gym (`ygoenv`).

## Tasks / Subtasks

- [ ] Task 1: Créer les schémas Pydantic
  - [ ] Subtask 1.1: Créer `DeckImportRequest` et `DeckImportResponse` avec `alias_generator = to_camel`.
- [ ] Task 2: Implémenter le service de parsing
  - [ ] Subtask 2.1: Créer un client HTTP asynchrone dans un nouveau fichier (ex: `core/deck_parser.py`) pour appeler `omega-api-decks`.
  - [ ] Subtask 2.2: Gérer les exceptions réseau et API (Fail Fast).
- [ ] Task 3: Exposer la route API FastAPI
  - [ ] Subtask 3.1: Ajouter l'endpoint `POST /api/v1/decks/import` dans `api/deck_routes.py` ou équivalent.
  - [ ] Subtask 3.2: Intégrer la gestion d'erreur avec renvoi 500 structuré.
- [ ] Task 4: Tests unitaires
  - [ ] Subtask 4.1: Ajouter des tests dans `tests/api/` utilisant `pytest-httpx` ou `httpx_mock` pour valider le parsing en succès et en échec.

## Dev Notes

### Technical Requirements
- Le module doit interagir en tant que client avec le microservice local `omega-api-decks`.
- L'API FastAPI (`api/`) doit disposer d'une route pour l'import d'un deck (que ce soit un fichier `.ydk` uploadé, ou une string brute via JSON).
- Requête HTTP Asynchrone : interroger `omega-api-decks` pour obtenir le deck sous forme de liste d'identifiants.
- Pydantic : Utilisation stricte de l'alias generator `to_camel` pour les schémas de l'API. Modèles attendus : `DeckImportRequest` et `DeckImportResponse`.
- "Fail Fast": Si `omega-api-decks` est injoignable ou renvoie une erreur, FastAPI doit immédiatement renvoyer une erreur 500 structurée sans retry silencieux.

### Architecture Compliance
- Ne pas mixer la logique IA et la logique d'API. Le contrôleur HTTP se placera dans `api/deck_routes.py` ou un fichier équivalent.
- Les noms de variables internes doivent respecter le `snake_case`.

### Library & Framework Requirements
- L'utilisation de `httpx` (async) est requise pour effectuer l'appel au microservice externe, puisqu'elle a été introduite dans la story précédente.

### Testing Requirements
- Ajouter des tests via Pytest dans le dossier `tests/api/`.
- Les appels HTTP sortants de FastAPI vers `omega-api-decks` DOIVENT être mockés.
- Tester à la fois les scénarios de réussite (parsing valide) et les scénarios d'échec.

### Previous Story Intelligence
- Les tests ne doivent pas polluer la base de données de production/dev (utilisation correcte des fixtures).
- Toujours lever des exceptions (ex: `HTTPError` ou exception custom) et laisser la gestion à un bloc contrôleur englobant au lieu d'utiliser sys.exit().

### References
- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.2]

## Dev Agent Record

### Agent Model Used
Antigravity

### Debug Log References

### Completion Notes List
- Ultimate context engine analysis completed - comprehensive developer guide created for 4-2.

### File List
