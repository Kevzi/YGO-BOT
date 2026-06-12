## Deferred from: code review of 1-1-base-de-donnees-et-historisation (2026-06-12)

- Validation JSON : Conservé en Dict[str, Any] pour l'instant. La structure sera décidée avec YGO Omega/Neos (Story 1.3) et ygoenv (Story 2.1).
- Session DB applicative : La création de la session de base de données (get_db) est reportée à la création du serveur API (Story 1.3) pour s'intégrer avec l'injection de dépendances de FastAPI.

## Deferred from: code review of 1-2-integration-du-moteur-c (2026-06-12)

- Hardcoded Database Connections: The database URL is statically hardcoded as `sqlite:///data/ygo.db` in `alembic.ini`.
- Untested Migration Scripts: The test suite setup bypasses Alembic by using `Base.metadata.create_all(engine)`.
- Disabled Foreign Keys in Tests: The in-memory SQLite engine instantiated in `conftest.py` fails to enforce `PRAGMA foreign_keys=ON`.
- No Data Integrity on Transitions: The `GameTransition` table lacks a compound unique constraint on `(duel_id, step)`.
- Useless Pydantic Validation: The `GameTransitionBase` schema types the crucial `state` and `action` fields as `Dict[str, Any]`.
- Ambiguous Column Semantics: The `winner` column in `DuelStats` is an unconstrained `Integer` with no documentation.
- Fragile Path Injections: The `README.md` instructs developers to manually set `$env:PYTHONPATH='.'` to run tests.
- Contradictory Setup Instructions: The `README.md` proudly claims the project uses Poetry, but then immediately abandons it.
- CMake dependency unhandled: If `cmake` is not installed, `subprocess.run` raises `FileNotFoundError`.
- Zip-Slip Vulnerability: `tar.extractall` extracts without validating member paths.
- Unhandled Network Failures: Network failures during `urlretrieve` are unhandled.
- Weak State Validation: `get_legal_actions` only checks `state is None` without validating keys.

## Deferred from: code review of 1-3-serveur-api-fastapi (2026-06-12)

- Global Engine State Vulnerability: Concurrent requests might corrupt singleton if not thread-safe.
- Ignored Request Data: `duel_id` is required in schema but never passed to engine.
- **Reckless CORS Configuration**: La route gère OPTIONS manuellement pour localhost:8080 sans utiliser le middleware `CORSMiddleware` recommandé par FastAPI.

## Deferred from: code review of 1-4-agent-dummy.md (2026-06-12)
- **Race Condition / Thread-unsafe Engine Access**: Le moteur est global, ce qui causera des problèmes de concurrence. Accepté pour le MVP.
- **Phantom State Application**: L'intégration C++ n'applique pas encore le state_dict reçu. Le stub sera remplacé plus tard.
- **Missing game-over DB insertion**: L'historisation des "fin de partie" (victoire/défaite) nécessite une intégration plus poussée des événements C++.
- **Hardcoded engine output limits dummy**: Le moteur C++ stubbé retourne toujours une seule action, limitant les choix de l'agent.
- **Naive DLL Resolution / IsADirectoryError**: Le script de build `build_engine.py` a des limites de recherche de DLL.

## Deferred from: code review of 2-3-pipeline-dentrainement-distribue (2026-06-12)
- [ ] Enregistrements de statistiques factices en BDD : Accepter le bouchon actuel ("winner": 1). Raison : Implémenter un meilleur faux gagnant est une perte de temps à ce stade ; le brancher aux véritables signaux du moteur ocgcore lorsqu'il sera pleinement opérationnel.
- [ ] Allers-retours CPU/GPU (Host-Device) inefficaces par étape : Optimiser la boucle de rollout pour bufferiser sur CPU et inférer par batch. Raison : S'y attaquer maintenant rajouterait une complexité massive, l'objectif actuel est de valider la plomberie du calcul distribué. Optimisation à faire dans un second temps.

## Deferred from: story-2.4 code review (2026-06-12)

- Security Issue with `/kill` Route: The `os._exit(0)` is highly abrupt and bypasses FastAPI/Uvicorn cleanup. Should be replaced with graceful shutdown logic.

## Deferred from: code review of 3-3-chargeur-dembeddings-vectoriels-in-memory.md (2026-06-12)

- Security Vulnerability in Pickle: `pickle.load()` is insecure (prototype tool).
- Synchronous Agent Forward in MCTS: `agent.forward` is called synchronously in the simulation loop, an anti-pattern (architectural choice to batch later).
- Flawed Zero-Sum Assumption: `-value_for_backprop` assumes alternating zero-sum, which breaks for chained actions in YGO.
- Duplicated Mock Definitions: `MockEnv` is identical in test files (refactoring opportunity).
- Mélange de paradigmes OOP et fonctionnel [ai/ppo.py] — deferred, pre-existing (choix structurel provisoire)

## Deferred from: code review of 3-4-semantique-zero-shot.md (2026-06-12)
- Brittle Hardcoded Actions [`tests/core/ygoenv/test_env.py`]
- Encapsulation Violation - State [`tests/core/ygoenv/test_env.py`]
- Coupling of DB Contexts [`tests/distributed/test_ray_pipeline.py`]
- Degraded Test Strictness [`tests/test_engine.py`]
- Language Inconsistency
- Test is executed without a mock database fixture injected [`tests/distributed/test_ray_pipeline.py`]

## Deferred from: code review of 2-1-environnement-rl-standardise (2026-06-12)
- Hardcoded Action/Observation Space Dimension: Hardcoded dimensions (200, 100) are deferred as this is an MVP mock.
- Useless Output/Zeros Observation/Simplistic reset: Reward 0, terminated False, phase DRAW are MVP mocks and deferred.
- Worthless Test Assertions / Missing Teardown: Basic typing tests without resource teardown deferred for MVP phase.

## Deferred from: code review of 2-4-configuration-de-linfrastructure-cloud (2026-06-12)

- Ports réseaux Ray non restreints : Les ports 6379 et 8076 ne sont pas restreints, exposant potentiellement le cluster.
- Zone de disponibilité codée en dur pour les TPUs : us-central1-a est codé en dur, ce qui risque de causer des échecs d'autoscaling si la capacité TPU manque.

## Deferred from: code review of 3-1-memoire-a-courtlong-terme (2026-06-12)

- Mise à jour SGD basique sans Adam/Clipping [ai/ppo.py]
- Absence de Minibatches et d'Epochs PPO [distributed/train.py]
- Absence de Generalized Advantage Estimation (GAE) [distributed/worker.py]
- Absence de clipping sur la Value Function [ai/ppo.py]
- Goulot d'étranglement synchrone via `ray.get` [distributed/train.py]
- Extracteur de caractéristiques (Feature Extractor) trop superficiel [ai/ppo.py]
- Absence d'Action Masking [ai/ppo.py]
- Hyperparamètres codés en dur [distributed/worker.py]
- Stub Fake pour la base de données [distributed/train.py]
