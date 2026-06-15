# Story 5.2: Auto-complétion et Ciblage Dynamique (MSG_SELECT_CARD)

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Chercheur en IA,
I want que l'environnement puisse gérer dynamiquement les requêtes de ciblage (MSG_SELECT_CARD),
So that l'agent ne reste pas bloqué indéfiniment lorsqu'il doit choisir la cible d'une attaque ou d'un effet complexe.

## Acceptance Criteria

1. **Given** un effet ou une attaque nécessitant un ciblage
2. **When** le moteur ocgcore renvoie `MSG_SELECT_CARD` avec une liste de cibles légales
3. **Then** l'environnement Gym permet à l'agent d'utiliser les slots 191 à 199 pour choisir une cible parmi la liste
4. **And** si plusieurs cibles sont requises, l'environnement boucle avec le C++ pour auto-compléter la sélection sans crasher.

## Tasks / Subtasks

- [ ] Task 1: Gérer `MSG_SELECT_CARD` dans la boucle ocgcore
  - [ ] Subtask 1.1: Intercepter le message `MSG_SELECT_CARD` depuis l'état C++
  - [ ] Subtask 1.2: Extraire la liste des cibles légales
- [ ] Task 2: Mapping de l'espace d'actions pour le ciblage
  - [ ] Subtask 2.1: Mapper les actions (slots 191 à 199) à l'index des cibles légales de l'environnement Gym
- [ ] Task 3: Auto-complétion des cibles multiples
  - [ ] Subtask 3.1: Implémenter une boucle de sélection pour gérer dynamiquement ou auto-compléter la sélection sans crasher si le moteur requiert plusieurs cibles

## Dev Notes

- **Architecture:** Le code d'interception IPC se trouve dans `core/ygoenv/` (Wrapper Python Gym). L'agent doit utiliser l'action masking pour éviter de sélectionner des slots invalides parmi 191-199.
- **Fail Fast:** Aucune exception silencieuse ; lever proprement des exceptions si ocgcore renvoie des états incohérents lors d'une sélection.
- **Git Intelligence / Bugs connus:** Prenez en compte les corrections récentes documentées dans l'Epic 6, comme `[Patch] Shape Mismatches in Environment Processing [ai/ppo.py]`, car modifier le traitement des actions dans Gym peut impacter les dimensions gérées par JAX.

### Project Structure Notes

- Respect strict du typage Python (`Type Hints`).
- Pensez à l'interface Gym (`env.step(action)`).

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-5]
- [Source: _bmad-output/planning-artifacts/architecture.md#Integration-Points]

## Dev Agent Record

### Agent Model Used

Antigravity

### Debug Log References

### Completion Notes List

### File List
