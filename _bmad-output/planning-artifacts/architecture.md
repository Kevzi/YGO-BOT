---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
workflowType: 'architecture'
project_name: 'ygo bot'
user_name: 'Kevin'
date: '2026-06-11'
lastStep: 8
status: 'complete'
completedAt: '2026-06-11'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
- **Séparation Moteur / Décision** : La validation des actions est déléguée au moteur C++ (ocgcore), tandis que le modèle décisionnel (Python/JAX) gère la stratégie.
- **Pipeline Deep RL & Planification** : Le cerveau de l'IA s'appuie sur l'algorithme PPO avec des couches LSTM pour la mémorisation/le bluff, complété par MCTS (Gumbel AlphaZero) pour anticiper les coups futurs dans un environnement à information imparfaite (POMDP).
- **Généralisation Sémantique (Zero-Shot)** : Un LLM est utilisé pour projeter les textes complexes des cartes en vecteurs continus (`embed.pkl`), permettant de jouer des cartes inconnues.
- **Intégration d'Inférence** : API locale FastAPI pour servir des requêtes à ultra-basse latence depuis YGO Omega (client lourd via DuelBotWrapper) ou Neos (client web).
- **Entraînement Distribué** : Pipeline d'apprentissage asynchrone parallélisant des millions de parties à l'aide de Ray.

**Non-Functional Requirements:**
- Latence d'inférence très stricte (< 100ms pour garantir l'expérience utilisateur).
- Précision 100% des règles garantie de manière déterministe (hors du réseau de neurones).
- Tolérance aux contraintes matérielles : Portabilité sur GPU grand public (GTX 1650) ou CPU en mode dégradé.

**Scale & Complexity:**
Ce projet requiert une modélisation mathématique très avancée pour résoudre un problème de complexité Π11-complet.

- Primary domain: Intelligence Artificielle (Deep RL), Systèmes Distribués.
- Complexity level: Très Haute (Enterprise/Research).
- Estimated architectural components: ~6 composants majeurs (Moteur C++, Environnement Gym, Encodeur LLM, Modèle JAX, Serveur d'Inférence, Clients/Wrappers).

### Technical Constraints & Dependencies

- Intégration de technologies hétérogènes communicantes (C++, C#, Python).
- Fortes contraintes imposées par les wrappers existants (`DuelBotWrapper` pour YGO Omega).
- Dépendance à des infrastructures matérielles spécifiques (parallélisation GPU) pour la phase d'entraînement "Élite".

### Cross-Cutting Concerns Identified

- **Sérialisation et communication réseau (IPC)** : Maintenir un pipeline JSON robuste et léger pour la transmission des états de jeu entre le simulateur et l'API d'inférence.
- **Vectorisation des observations temporelles** : Assurer que l'état du terrain et l'historique soient correctement encodés pour alimenter les couches LSTM.
- **Découplage architectural** : Garantir que les changements de règles dans le moteur C++ n'entraînent pas de réécritures complètes de la politique de l'agent.

## Starter Template Evaluation

### Primary Technology Domain

**Backend IA & Infrastructure d'Entraînement Distribuée (Python)** basé sur l'analyse des besoins en apprentissage par renforcement et en inférence ultra-rapide.

### Starter Options Considered

- **Générateurs génériques (ex: Tiangolo FastAPI / Cookiecutter Data Science)** : Rejetés. Ils incluent trop de bruit (frontend web, structures non adaptées) ou manquent de support natif pour les calculs distribués (Ray) et l'accélération matérielle spécifique (JAX/TPU).
- **Boilerplate Custom (Poetry + Ruff + Docker)** : Choisi. Une structure sur-mesure construite avec les outils standards de l'industrie Python moderne. C'est la seule approche offrant la flexibilité requise pour lier le code C++ (ocgcore), l'orchestration Ray, et l'API d'inférence.

### Selected Starter: Custom Modern Python Boilerplate (Poetry)

**Rationale for Selection:**
L'architecture de l'IA est hautement spécialisée. La meilleure fondation est un squelette Python pur, vierge mais strictement configuré pour la reproductibilité (Poetry), la qualité du code (Ruff), et la portabilité cloud/locale (Docker), intégrant nativement SQLite pour l'historique.

**Initialization Command:**

```bash
poetry new ygo-bot-core
cd ygo-bot-core
poetry add fastapi uvicorn jax jaxlib ray sqlalchemy pandas
poetry add --group dev ruff pytest pytest-asyncio
```

**Architectural Decisions Provided by Starter:**

**Language & Runtime:**
Python 3.10+ avec typage statique strict (type hints) pour sécuriser les interfaces entre le moteur RL et l'API.

**Styling Solution:**
*Non applicable (Architecture Backend/IA).* Les retours visuels (PowerBI, .yrp) seront gérés via des exports CSV externes.

**Build Tooling & Optimization:**
- **Poetry** pour une résolution déterministe des dépendances (crucial pour l'écosystème JAX/Cuda/TPU).
- **Docker & Docker Compose** pour conteneuriser l'environnement d'entraînement (Google TRC) et isoler le serveur d'inférence (FastAPI) pour les joueurs.

**Testing Framework:**
**Pytest** pour la validation unitaire de l'API locale et la validation déterministe des modèles avant déploiement.

**Code Organization:**
L'arborescence imposée sépare strictement les responsabilités :
- `src/api/` : Endpoints FastAPI (Inférence locale port 3000)
- `src/rl/` : Architecture neuronale JAX (PPO, LSTM, Embeddings)
- `src/env/` : Wrapper Gym (ygoenv) et intégration du moteur C++
- `src/distributed/` : Acteurs et configuration Ray (League Training)
- `data/` : Bases de données SQLite et exports CSV/Replays
- `tests/` : Suite Pytest
- `docker/` : Dockerfiles (Environnements CPU vs GPU/TPU)

**Development Experience:**
- Linting et formatage unifiés, ultra-rapides via **Ruff**.
- Rechargement à chaud (Hot-reload) de l'API locale via Uvicorn pendant le développement.
- Fichiers `docker-compose.yml` garantissant que les contributeurs open-source peuvent lancer un environnement de dev "prêt à coder" en une seule commande.

*Note de déploiement : Le projet sera poussé et mis à jour sur le dépôt GitHub https://github.com/Kevzi/YGO-BOT.*

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
- Mécanisme de chargement des embeddings (In-memory) pour satisfaire la contrainte de latence stricte (< 100ms).
- Stratégie de résilience API (Arrêt explicite en cas d'erreur de l'ocgcore).

**Important Decisions (Shape Architecture):**
- Gestion des migrations de base de données (Alembic).
- Sécurisation des endpoints FastAPI (CORS local vs API Key distant).
- Infrastructure d'entraînement (Ray Cluster Launcher via YAML).
- Pipeline CI/CD (GitHub Actions).

**Deferred Decisions (Post-MVP):**
- Aucune pour le moment.

### Data Architecture

- **Migrations de Base de Données :** Alembic. *Raison :* Évolution prévue des schémas liés aux statistiques des parties via `ShouldUpdate`.
- **Mise en Cache (Embeddings) :** Chargement intégral en RAM/VRAM au démarrage de l'API FastAPI. *Raison :* Exigence absolue de très faible latence et volonté de ne pas alourdir l'infrastructure locale du joueur final avec des composants comme Redis.

### Authentication & Security

- **Sécurisation de l'API (Inférence) :** Stratégie hybride. Restriction CORS stricte (localhost) pour les déploiements locaux (joueurs/YGO Omega), et intégration d'une couche d'authentification par API Key/Bearer Token pour les serveurs exposés publiquement (ex: Serveur Koishi). *Raison :* Sécuriser les serveurs communautaires exposant des ressources GPU/CPU intensives contre les requêtes abusives.

### API & Communication Patterns

- **Résilience et Gestion des Erreurs :** En cas de plantage ou d'état corrompu renvoyé par le moteur de règles C++ (ocgcore), l'API FastAPI provoquera un crash élégant ("Fail Fast") sans retry, en renvoyant une erreur HTTP 500 explicite. *Raison :* Le moteur C++ est source de vérité absolue ; un retry risquerait de désynchroniser irréversiblement l'état de croyance de l'IA et l'état graphique du client.

### Frontend Architecture

- *Non applicable.* L'interface utilisateur est entièrement déléguée aux clients existants (YGO Omega, Neos Web).

### Infrastructure & Deployment

- **CI/CD :** GitHub Actions. Exécution automatique de Ruff (linting) et Pytest à chaque Pull Request/Commit. *Raison :* Assurer un standard de qualité maximal pour la communauté open-source.
- **Provisioning Cluster :** Ray Cluster Launcher (YAML). *Raison :* Adapté à l'entraînement massif ("League Training") sur les TPU Google TRC, sans la surcharge DevOps d'un cluster Kubernetes complet (KubeRay).

### Decision Impact Analysis

**Implementation Sequence:**
1. Initialisation du projet (Poetry) et configuration du dépôt GitHub.
2. Mise en place de la pipeline CI/CD (GitHub Actions : Ruff, Pytest).
3. Intégration de la base SQLite et initialisation d'Alembic.
4. Développement de l'API FastAPI avec middleware hybride (CORS/API Key).
5. Implémentation du chargeur en mémoire vive pour le modèle JAX et `embed.pkl`.
6. Ajout des handlers d'erreur HTTP 500 (Fail Fast) pour le pont C++/Python.
7. Création des templates YAML pour Ray Cluster Launcher.

**Cross-Component Dependencies:**
- L'API FastAPI dépend étroitement de la fiabilité du pont IPC avec `ocgcore`. La politique sans *retry* impose une gestion d'erreurs robuste dans le wrapper Python (ygoenv).
- La stratégie "In-memory" pour les embeddings exige que le serveur / conteneur Docker dispose de suffisamment de RAM allouée au démarrage.

## Implementation Patterns & Consistency Rules

### Pattern Categories Defined

**Critical Conflict Points Identified:**
4 zones où les agents IA pourraient introduire des incohérences dommageables (Casse JSON, organisation des tests, gestion des tenseurs JAX, formats d'erreurs).

### Naming Patterns

**Database Naming Conventions:**
- Tables SQL : Toujours au pluriel et en `snake_case` (ex: `game_transitions`, `duel_stats`).
- Colonnes : `snake_case`.
- Clés primaires/étrangères : `id` pour la PK, `[table_singulier]_id` pour les FK (ex: `duel_id`).

**API Naming Conventions:**
- Endpoints REST : Noms au pluriel (ex: `/api/v1/duels`, `/api/v1/actions`).
- Payload JSON : Bien que le code Python interne utilise le `snake_case`, toutes les réponses JSON exposées par FastAPI vers C# (Omega) ou Neos (Web) doivent utiliser le **`camelCase`** pour s'aligner avec les standards front-end (ex: `{"actionId": 4, "gameState": [...]}`). FastAPI utilisera les alias pydantic pour gérer la conversion.

**Code Naming Conventions:**
- Fichiers et dossiers : `snake_case` (ex: `duel_routes.py`).
- Classes et Modèles Pydantic/SQLAlchemy : `PascalCase` (ex: `GameState`).
- Fonctions et variables (Python) : `snake_case` (ex: `calculate_reward()`).

### Structure Patterns

**Project Organization:**
- La logique métier de l'IA ne doit JAMAIS fuiter dans les routes de l'API. Le dossier `api/` ne contient que les schémas Pydantic et l'injection de dépendances.
- Les tests (`tests/`) doivent refléter l'arborescence du projet (ex: `tests/api/test_duel_routes.py`, `tests/ai/test_mcts.py`).

### Format Patterns

**API Response Formats:**
- **Erreurs (Fail Fast)** : L'API ne retourne JAMAIS de traces Python brutes. En cas d'échec de l'ocgcore, la réponse doit toujours être un JSON structuré avec un status `500` :
  `{"error": {"code": "ENGINE_CRASH", "detail": "La validation C++ a échoué à l'étape X"}}`

**Data Exchange Formats (JAX / Numpy) :**
- Toutes les opérations tensorielles JAX et vecteurs d'observations doivent impérativement définir explicitement leur type de données (dtype) pour éviter les instabilités mémoires (généralement `jnp.float32` pour les réseaux de neurones, `jnp.int32` pour les indices d'action).

### Process Patterns

**Error Handling Patterns:**
- Interdiction d'utiliser des `try/except Exception: pass` silencieux, particulièrement dans le pont C++/Python (`core/ygoenv/`). Toute exception doit être logguée proprement (via `logging`) avant de lever une erreur 500 pour arrêter le duel.

### Enforcement Guidelines

**All AI Agents MUST:**
- Utiliser la librairie `pydantic` avec `alias_generator = to_camel` pour toute communication vers l'extérieur.
- Déclarer des Type Hints (Typage strict Python) obligatoires pour chaque paramètre de fonction et valeur de retour.
- Assurer que chaque fonction JAX modifiant l'état est "pure" (pas de mutation sur place).

**Anti-Patterns (À ÉVITER ABSOLUMENT) :**
- Renvoyer du JSON en `snake_case` au client C#.
- Hardcoder des identifiants de cartes (passcodes) directement dans les scripts Python d'inférence (doit passer par le fichier constant ou la DB).

## Project Structure & Boundaries

### Complete Project Directory Structure

```text
ygo-bot/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD : Exécution automatique de Ruff et Pytest
├── api/                           # Serveur d'inférence FastAPI
│   ├── main.py                    # Point d'entrée (Port 3000), initialise la RAM
│   ├── security.py                # Middleware hybride (CORS strict / API Key)
│   └── duel_routes.py             # Endpoints HTTP (gestion du Fail Fast 500)
├── core/                          # Intégration du Moteur C++
│   ├── ocgcore/                   # Sous-module du moteur ygopro-core en C++
│   └── ygoenv/                    # Wrapper Python Gym pour l'IPC et le masking
├── ai/                            # Cerveau de l'Agent (JAX)
│   ├── agent.py                   # Logique globale et interactions PPO/LSTM
│   ├── mcts.py                    # Algorithmes de recherche (Gumbel AlphaZero)
│   └── embeddings.py              # Chargeur In-Memory ultra-rapide pour embed.pkl
├── db/                            # Persistance et Données
│   ├── alembic/                   # Fichiers de migration de schéma Alembic
│   └── models.py                  # Schémas SQLAlchemy (historique, ShouldUpdate)
├── cluster/                       # Infrastructure
│   └── ray_tpu_config.yaml        # Template pour le Ray Cluster Launcher (League Training)
├── tests/                         # Tests unitaires et de validation (Pytest)
├── Dockerfile                     # Standardisation du déploiement
└── pyproject.toml                 # Gestionnaire de dépendances Poetry et config Ruff
```

### Architectural Boundaries

**API Boundaries:**
- Le trafic externe ne pénètre que via `api/main.py`. Les requêtes d'inférence sont validées par les schémas Pydantic avant de jamais toucher les tenseurs JAX.
- Les erreurs levées par `core/ygoenv/` ne franchissent pas la frontière de l'API brutes : elles sont formatées en JSON 500 sécurisé dans `duel_routes.py`.

**Component Boundaries:**
- Le dossier `ai/` est mathématiquement pur. Il ne connaît pas l'existence de l'API REST ni de la base de données. Il prend des tenseurs en entrée (JAX array) et recrache des probabilités/valeurs.
- Le dossier `core/` est le seul habilité à communiquer avec le moteur C++ `ocgcore`.

**Data Boundaries:**
- Seul le dossier `db/` interagit avec SQLite et Alembic. Les statistiques d'entraînement générées par `ai/` ou le cluster Ray doivent passer par les sessions SQLAlchemy définies dans `db/models.py`.

### Requirements to Structure Mapping

**Feature/Epic Mapping:**
- *Généralisation Zero-Shot (Embeddings)* → `ai/embeddings.py` (chargement) et `api/main.py` (initialisation mémoire).
- *Planification Stratégique* → `ai/mcts.py` (Gumbel AlphaZero) couplé à `ai/agent.py` (PPO).
- *Validation des Règles* → `core/ocgcore/` et `core/ygoenv/`.
- *Inférence Temps Réel* → `api/duel_routes.py`.

**Cross-Cutting Concerns:**
- *Qualité du Code* → `pyproject.toml` (dépendances Poetry + règles Ruff) et `.github/workflows/ci.yml`.
- *Distribution* → `cluster/ray_tpu_config.yaml`.

### Integration Points

**Internal Communication:**
L'API HTTP (dans `api/`) invoque le wrapper Gym (dans `core/`), qui met à jour l'état depuis le C++, puis passe cet état aux modèles JAX (dans `ai/`) pour obtenir une décision.

**Data Flow:**
1. Le Client (C#) envoie l'état du duel (JSON).
2. FastAPI (`api/`) le parse via Pydantic (`camelCase` -> `snake_case`).
3. Le Wrapper (`core/`) vectorise l'état en tenseurs `jnp.float32`.
4. Le Cerveau (`ai/`) injecte les embeddings (`embed.pkl`), effectue l'inférence JAX, et retourne l'Action ID.
5. L'Action ID redescend jusqu'à FastAPI qui répond en JSON au Client.

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
Toutes les décisions technologiques (FastAPI, JAX, Ray, SQLite, **Alembic**, C++) sont parfaitement compatibles. Le choix du `Fail Fast` (Erreur 500) est en totale cohérence avec l'absence d'effets de bord attendue par JAX et l'état immuable du MCTS.

**Pattern Consistency:**
L'utilisation de `camelCase` (via Pydantic) pour l'API externe et de `snake_case` pour le code Python interne élimine le conflit cognitif principal entre les normes du web et celles du Machine Learning. Les contraintes de tenseurs stricts (dtype) assurent la cohérence mémoire requise par JAX.

**Structure Alignment:**
La structure du projet isole le moteur C++ (`core/`), l'API (`api/`) et l'IA (`ai/`). Cette isolation physique garantit que les frontières architecturales décidées ne seront pas franchies par erreur lors de l'ajout de nouvelles fonctionnalités.

### Requirements Coverage Validation ✅

**Epic/Feature Coverage:**
L'architecture prend en charge tous les aspects nécessaires : traduction NLP (Embeddings in-memory), apprentissage par renforcement (PPO/MCTS), exécution du duel (ygoenv/ocgcore), et service temps-réel (FastAPI).

**Functional Requirements Coverage:**
Le fonctionnement zero-shot est pris en charge sans coder de passcodes en dur. La validation absolue des actions est sécurisée par le moteur C++ centralisé.

**Non-Functional Requirements Coverage:**
La latence (inférieure à 100 ms) est garantie par le chargement en RAM. L'entraînement massif est soutenu nativement par la configuration YAML du Ray Cluster Launcher pour Google TRC. La sécurité est assurée par le middleware CORS/API Key.
Pour garantir la **Portabilité** (NFR-3), l'architecture supporte explicitement l'inférence en mode dégradé sur processeur (CPU) via l'argument `--xla_device cpu` de JAX, permettant aux joueurs sans GPU dédié d'utiliser le bot localement avec fluidité.

### Implementation Readiness Validation ✅

**Decision Completeness:**
Les versions des dépendances (Python **3.10+**, FastAPI ^0.100.0, JAX ^0.4.13, SQLAlchemy/Alembic, etc.) et le Starter (Poetry) sont entièrement spécifiés et prêts à être initialisés.

**Structure Completeness:**
L'arborescence complète jusqu'au niveau du fichier (ex: `api/duel_routes.py`, `ai/agent.py`) est définie, supprimant toute ambiguïté pour le développeur ou l'agent implémenteur.

**Pattern Completeness:**
Les règles de formatage (JSON), d'interopérabilité IPC et de résilience (Arrêt explicite sur erreur) sont gravées dans le marbre.

### Gap Analysis Results

Aucun gap critique ou bloquant n'a été identifié.
*(Remarque mineure : L'intégration éventuelle d'outils de MLOps avancés comme Weights & Biases pour la télémétrie des entraînements sur Ray reste ouverte, mais peut être ajoutée post-MVP de manière non-intrusive).*

### Validation Issues Addressed

Le risque de désynchronisation de l'état entre le client YGO Omega et l'IA JAX en cas de crash du moteur Lua/C++ a été formellement adressé via la politique de Fail Fast (Code 500 structuré) à l'Étape 4.

### Architecture Completeness Checklist

**Requirements Analysis**
- [x] Project context thoroughly analyzed
- [x] Scale and complexity assessed
- [x] Technical constraints identified
- [x] Cross-cutting concerns mapped

**Architectural Decisions**
- [x] Critical decisions documented with versions
- [x] Technology stack fully specified
- [x] Integration patterns defined
- [x] Performance considerations addressed

**Implementation Patterns**
- [x] Naming conventions established
- [x] Structure patterns defined
- [x] Communication patterns specified
- [x] Process patterns documented

**Project Structure**
- [x] Complete directory structure defined
- [x] Component boundaries established
- [x] Integration points mapped
- [x] Requirements to structure mapping complete

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION

**Confidence Level:** High

**Key Strengths:**
- Séparation stricte des responsabilités (Pureté de l'IA vs Isolation du C++).
- Infrastructure extrêmement performante optimisée pour la vitesse pure et le Deep RL à grande échelle.
- Standardisation moderne du socle Python (Poetry, Ruff, Pydantic).

**Areas for Future Enhancement:**
- Intégration optionnelle d'outils MLOps (WandB/MLflow) pour le monitoring des métriques d'entraînement complexes sur le cluster Ray.

### Implementation Handoff

**AI Agent Guidelines:**
- Follow all architectural decisions exactly as documented.
- Use implementation patterns consistently across all components (pure JAX functions, explicit dtypes).
- Respect project structure and boundaries (no business logic in `api/`).
- Refer to this document for all architectural questions.

**First Implementation Priority:**
Initialisation du projet via Poetry et configuration du dépôt GitHub avec l'arborescence définie à l'Étape 6.
