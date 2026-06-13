---
stepsCompleted: [1, 2]
inputDocuments: [
  "c:/Users/kevin/Downloads/Projet code/ygo bot/_bmad-output/planning-artifacts/prds/prd-ygo-bot-2026-06-11/prd.md",
  "c:/Users/kevin/Downloads/Projet code/ygo bot/_bmad-output/planning-artifacts/architecture.md"
]
---

# ygo bot - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for ygo bot, decomposing the requirements from the PRD, UX Design if it exists, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

FR-1.1: Compréhension Sémantique et "Zero-Shot" Hybride (Traduction des textes complexes des cartes en embeddings vectoriels `embed.pkl` et capacité d'extrapolation zero-shot).
FR-1.2: Apprentissage par Renforcement (RL) utilisant l'algorithme PPO sous le framework JAX.
FR-1.3: Mémoire et Bluff via l'intégration de réseaux LSTM pour la mémoire à court/long terme.
FR-1.4: Planification et Anticipation via l'implémentation de MCTS (Gumbel AlphaZero).
FR-2.1: Séparation Logique (Délégation de la validation des actions et des chaînes au moteur C++ ocgcore/ygopro-core).
FR-2.2: Environnement Gym (ygoenv) (Interface Python standardisée pour l'agent).
FR-3.1: API d'Inférence (Serveur local Python FastAPI exposant les endpoints pour l'action optimale).
FR-3.2: Interception Simulateur (Wrapper C# DuelBotWrapper ou scripts pour Omega/EDOPro).
FR-3.3: Clients Supportés (Intégration sur YGO Omega et Neos).
FR-4.1: Intégration API YGOPRODeck (Récupération des données, images et prix).
FR-4.2: Parsing de Decks (Support des codes de deck via omega-api-decks).

### NonFunctional Requirements

NFR-1: Latence d'inférence (Sélection d'action < 100 ms).
NFR-2: Fidélité des règles (Respect à 100% des règles via ocgcore).
NFR-3: Portabilité de l'inférence (Support GPU grand public et fallback CPU `--xla_device cpu`).

### Additional Requirements

- **Starter Template:** Custom Modern Python Boilerplate (Poetry) incluant FastAPI, JAX, Ray, SQLAlchemy/Alembic, Ruff, Pytest. (Déjà partiellement implémenté, mais structurant pour l'Epic 1).
- Base de données locale (SQLite) avec migrations Alembic pour historiser les transitions.
- Mise en cache intégrale In-Memory (RAM/VRAM) pour `embed.pkl` au démarrage de l'API.
- Sécurisation de l'API par middleware hybride (CORS strict pour localhost, API Key pour distant).
- Politique d'erreur "Fail Fast" : l'API renvoie une erreur JSON 500 structurée (`{"error": {"code": "ENGINE_CRASH", ...}}`) sans retry en cas de plantage du moteur C++.
- Les payloads JSON de l'API doivent être formatés en camelCase (via Pydantic `alias_generator = to_camel`) pour l'interopérabilité client.
- Opérations tensorielles JAX avec typage explicite (`dtype`).
- Fonctions JAX "pures" sans mutation sur place.
- Infrastructure d'entraînement via Ray Cluster Launcher (YAML) pour Google TRC.
- Pipeline CI/CD automatisée avec GitHub Actions (Ruff + Pytest).

### UX Design Requirements

Aucune exigence de design UX formelle (interface déléguée aux clients Omega/Neos).

### FR Coverage Map

FR-1.1: Epic 3 - Compréhension sémantique
FR-1.2: Epic 2 - Apprentissage PPO
FR-1.3: Epic 3 - Mémoire et Bluff (LSTM)
FR-1.4: Epic 3 - Planification (MCTS)
FR-2.1: Epic 1 - Séparation logique (Moteur C++)
FR-2.2: Epic 2 - Environnement Gym
FR-3.1: Epic 1 - API d'Inférence
FR-3.2: Epic 1 - Interception Simulateur
FR-3.3: Epic 1 - Clients Supportés
FR-4.1: Epic 4 - Intégration YGOPRODeck
FR-4.2: Epic 4 - Parsing de Decks

## Epic List

### Epic 1: La Fondation du Sparring-Partner (Jouable en local)
Permettre à un joueur de se connecter à l'IA via un client (Omega/Neos) et de lancer un duel où le bot répond avec des coups 100% légaux. *(Note: Inclure l'implémentation de la DB SQLite gérée via Alembic pour l'enregistrement des historiques).*
**FRs covered:** FR-2.1, FR-3.1, FR-3.2, FR-3.3

### Epic 2: Le Pipeline d'Apprentissage Deep RL (Pour la recherche)
Fournir aux chercheurs et ingénieurs l'environnement standardisé et l'algorithme de base permettant au bot d'apprendre de ses erreurs en jouant des millions de parties.
**FRs covered:** FR-1.2, FR-2.2

### Epic 3: Cognition Avancée et Niveau Surhumain (Maîtrise Stratégique)
Doter l'IA de la capacité d'anticiper plusieurs tours à l'avance, de se souvenir des cartes cachées (bluff), et de réagir intelligemment face à des cartes inconnues.
**FRs covered:** FR-1.1, FR-1.3, FR-1.4

### Epic 4: Intelligence du Metagame (Connaissance externe)
Permettre au joueur/chercheur de charger n'importe quel deck méta depuis des sources externes pour tester le bot contre des stratégies spécifiques.
**FRs covered:** FR-4.1, FR-4.2

### Epic 5: Mise à l'échelle de l'Environnement (Architecture Avancée)
Refonte et optimisation de l'environnement Gym (ygoenv) pour permettre à l'agent PPO de comprendre l'intégralité du jeu (Espace d'Actions et Observation Space), et résoudre les blocages complexes du moteur C++.
**FRs covered:** FR-5.1, FR-5.2, FR-5.3, FR-5.4

### Epic 6: Intelligence Artificielle et Planification (Deep RL)
Connecter l'environnement à un modèle JAX complet capable d'apprendre par lui-même (Self-Play) en respectant les actions légales, en retenant l'information cachée via LSTM, en planifiant avec MCTS et en s'entraînant massivement via Ray.
**FRs covered:** FR-6.1, FR-6.2, FR-6.3, FR-6.4

## Epic 1: La Fondation du Sparring-Partner (Jouable en local)

Permettre à un joueur de se connecter à l'IA via un client (Omega/Neos) et de lancer un duel où le bot répond avec des coups 100% légaux.

### Story 1.1: Base de Données et Historisation (SQLite + Alembic)

As a Développeur / Chercheur,
I want d'initialiser la base de données locale (SQLite) avec Alembic et créer les schémas initiaux (`duel_stats`, `game_transitions`),
So that l'historique des duels puisse être enregistré dès le premier test fonctionnel de l'API.

**Acceptance Criteria:**

**Given** une base de données vierge
**When** on exécute `alembic upgrade head`
**Then** les tables `duel_stats` et `game_transitions` (en `snake_case`) sont créées
**And** le modèle Pydantic de base est prêt pour l'insertion.

### Story 1.2: Intégration du Moteur C++ (ygoenv / ocgcore)

As a Développeur,
I want d'interfacer le moteur de règles ygopro-core (C++) via le wrapper Python `core/ygoenv/`,
So that le backend puisse valider les actions, vérifier la légalité des coups, et ne pas avoir à recalculer les règles de Yu-Gi-Oh!.

**Acceptance Criteria:**

**Given** l'envoi d'un état de jeu valide
**When** le moteur est sollicité
**Then** il retourne les actions légales possibles
**And** les erreurs éventuelles du moteur C++ lèvent une exception claire interceptable par l'API.

### Story 1.3: Serveur API FastAPI (Inférence In-Memory & Fail Fast)

As a Joueur (via Client),
I want de pouvoir envoyer l'état du duel à une API locale rapide qui répond avec la structure JSON attendue (`camelCase`),
So that le client (YGO Omega/Neos) puisse s'y connecter et que toute erreur du moteur C++ me retourne proprement un code 500 structuré.

**Acceptance Criteria:**

**Given** une requête du client avec l'état en `camelCase`
**When** elle est reçue
**Then** elle est parsée en `snake_case` via Pydantic
**And** en cas de plantage d'ocgcore (Fail Fast), une erreur 500 stricte `{"error": {"code": "ENGINE_CRASH", ...}}` est retournée sans *retry*.

### Story 1.4: Agent "Dummy" (Boucle de bout-en-bout)

As a Joueur,
I want que le serveur API puisse retourner un coup légal aléatoire (ou basé sur une heuristique très simple),
So that je puisse tester la connexion et le déroulement complet d'un duel avant même l'entraînement du vrai modèle JAX.

**Acceptance Criteria:**

**Given** l'API qui tourne et le moteur C++ branché
**When** je lance un duel via YGO Omega
**Then** le bot Dummy parvient à piocher, jouer une carte, et passer son tour (sans planter) jusqu'à la fin de la partie
**And** les transitions et la fin de partie sont bien insérées en base de données (lien avec Story 1.1).

## Epic 2: Le Pipeline d'Apprentissage Deep RL (Pour la recherche)

Fournir aux chercheurs et ingénieurs l'environnement standardisé et l'algorithme de base permettant au bot d'apprendre de ses erreurs en jouant des millions de parties.

### Story 2.1: Environnement RL Standardisé (Wrapper Gym / ygoenv)

As a Chercheur en IA,
I want d'encapsuler la boucle logique C++ (de la Story 1.4) dans une interface standardisée de type OpenAI Gym / PettingZoo (`core/ygoenv/`),
So that je puisse y brancher n'importe quel algorithme d'apprentissage par renforcement standard sans me soucier de la complexité du moteur de Yu-Gi-Oh!.

**Acceptance Criteria:**

**Given** l'environnement Gym initialisé
**When** on appelle les méthodes standard `env.reset()` et `env.step(action)`
**Then** il retourne correctement l'état sous le format `(observation, reward, done, info)`
**And** les tenseurs d'observation retournés sont strictement typés (ex: `jnp.float32`), en conformité avec nos conventions.

### Story 2.2: Architecture Neuronale de Base (PPO sous JAX)

As a Chercheur en IA,
I want de développer la structure réseau neuronale de base de l'agent en utilisant JAX et d'implémenter l'algorithme PPO (Proximal Policy Optimization),
So that l'agent puisse mettre à jour sa politique en fonction des récompenses (rewards) obtenues dans l'environnement Gym.

**Acceptance Criteria:**

**Given** un lot (batch) d'observations issues de l'environnement
**When** l'agent exécute sa "forward pass"
**Then** il produit des probabilités d'action valides (Policy) et une estimation de la valeur de l'état (Value)
**And** les fonctions JAX de mise à jour respectent la pureté fonctionnelle et sont compilables avec `jax.jit`.

### Story 2.3: Pipeline d'Entraînement Distribué (Ray)

As a Chercheur,
I want d'orchestrer l'entraînement en parallèle sur plusieurs threads/machines via la librairie Ray,
So that je puisse simuler massivement des milliers de parties simultanément et accélérer de façon exponentielle la vitesse d'apprentissage (League Training).

**Acceptance Criteria:**

**Given** l'exécution du script d'entraînement principal
**When** Ray initialise le cluster local
**Then** de multiples acteurs ("Workers") exécutent l'environnement Gym en parallèle et renvoient leurs trajectoires au processus central ("Learner")
**And** un mécanisme de file d'attente (queue) est mis en place pour l'enregistrement SQLite afin d'éviter les verrous (locks) de la base de données.

### Story 2.4: Configuration de l'Infrastructure Cloud (Ray Cluster Launcher)

As a Chercheur / Ops,
I want de définir la configuration YAML pour le Ray Cluster Launcher,
So that le déploiement de l'entraînement massif sur des infrastructures distantes (comme Google TRC / TPU) soit automatisé et parfaitement reproductible.

**Acceptance Criteria:**

**Given** le fichier `cluster/ray_tpu_config.yaml` rempli
**When** on exécute `ray up cluster/ray_tpu_config.yaml`
**Then** l'infrastructure cloud peut s'initialiser et installer correctement nos dépendances via Poetry.

## Epic 3: Cognition Avancée et Niveau Surhumain (Maîtrise Stratégique)

Doter l'IA de la capacité d'anticiper plusieurs tours à l'avance, de se souvenir des cartes cachées (bluff), et de réagir intelligemment face à des cartes inconnues.

### Story 3.1: Mémoire à Court/Long Terme (Intégration LSTM)

As a Chercheur en IA,
I want d'étendre l'architecture neuronale PPO avec des couches récurrentes LSTM (Long Short-Term Memory),
So that l'agent puisse se souvenir des cartes révélées puis retournées face cachée, gérant ainsi l'information imparfaite et les dynamiques de bluff.

**Acceptance Criteria:**

**Given** des trajectoires d'environnement séquentiel
**When** l'agent traite la séquence
**Then** il maintient et met à jour un "hidden state" persistant d'un tour sur l'autre
**And** les gradients peuvent se propager dans le temps (BPTT) lors de l'entraînement sans erreur JAX.

### Story 3.2: Planification Arborescente (MCTS / Gumbel AlphaZero)

As a Chercheur en IA,
I want d'implémenter l'algorithme Monte Carlo Tree Search (MCTS), spécifiquement la variante "Gumbel AlphaZero",
So that l'agent puisse simuler virtuellement plusieurs scénarios futurs avant de choisir son coup final, améliorant drastiquement sa prise de décision.

**Acceptance Criteria:**

**Given** un état de jeu actuel
**When** le MCTS est sollicité
**Then** il simule $N$ trajectoires virtuelles via l'environnement Gym et retourne une politique améliorée (Policy Improvement)
**And** les appels de simulation envoyés au moteur C++ (ocgcore) s'exécutent sur des états "clonés" (state cloning au niveau du wrapper) afin de ne jamais corrompre l'état du duel principal.

### Story 3.3: Chargeur d'Embeddings Vectoriels In-Memory

As a Développeur IA,
I want de développer un chargeur ultra-rapide qui monte en mémoire vive (RAM/VRAM) le dictionnaire sémantique (`embed.pkl`) contenant toutes les cartes du jeu au démarrage du serveur API,
So that la conversion de l'ID d'une carte vers son vecteur sémantique se fasse de façon instantanée (contrainte < 100 ms).

**Acceptance Criteria:**

**Given** le démarrage de l'API FastAPI
**When** le service s'initialise
**Then** le fichier `embed.pkl` est désérialisé et chargé intégralement
**And** le modèle JAX accède à ces vecteurs directement sans jamais nécessiter de lecture disque (I/O) lors d'un duel.

### Story 3.4: Sémantique "Zero-Shot" (Modèle Hybride)

As a Chercheur / Joueur,
I want que le modèle PPO intègre les vecteurs d'embeddings plutôt que de simples IDs entiers pour représenter les cartes en entrée,
So that l'agent puisse transférer sa connaissance des effets de cartes validées vers des cartes qu'il n'a jamais affrontées durant son entraînement (capacité de généralisation Zero-Shot).

**Acceptance Criteria:**

**Given** l'apparition d'une carte totalement inconnue du modèle (absente de `code_list.txt` mais présente dans `embed.pkl`)
**When** l'agent l'évalue
**Then** il utilise son vecteur sémantique pour extrapoler un coup de manière mathématiquement fluide, sans lever d'erreur.

## Epic 4: Intelligence du Metagame (Connaissance externe)

Permettre au joueur/chercheur de charger n'importe quel deck méta depuis des sources externes pour tester le bot contre des stratégies spécifiques.

### Story 4.1: Synchronisation et Cache YGOPRODeck

As a Développeur / Chercheur,
I want de créer un module client (ou un script d'ingestion) capable d'interroger l'API YGOPRODeck,
So that je puisse télécharger, mettre à jour, et cacher localement (dans SQLite) toutes les métadonnées officielles des cartes (noms, textes, propriétés) nécessaires au bon fonctionnement de l'environnement Python.

**Acceptance Criteria:**

**Given** une commande ou un endpoint de synchronisation
**When** on l'exécute
**Then** les dernières données de l'API YGOPRODeck sont récupérées en respectant la limite de rate-limit (20 requêtes/seconde max)
**And** la table SQLite des cartes est mise à jour et sert de source de vérité exclusive pour éviter le bannissement d'IP.

### Story 4.2: Interopérabilité du Parseur de Decks (omega-api-decks)

As a Joueur / Chercheur,
I want de pouvoir charger un deck via un format standard (`.ydk` ou code presse-papier) en envoyant la requête au microservice `omega-api-decks`,
So that l'environnement Python reçoive systématiquement une liste d'IDs entiers (passcodes) propre et validée, peu importe la complexité du format d'entrée.

**Acceptance Criteria:**

**Given** un fichier de deck brut soumis à l'API FastAPI
**When** FastAPI délègue la requête de conversion au microservice local `omega-api-decks`
**Then** le microservice le décode et FastAPI extrait les passcodes (Main/Extra/Side)
**And** la liste d'IDs validée est injectée pour initialiser correctement le deck de l'agent dans l'environnement Gym (`ygoenv`).

## Epic 5: Mise à l'échelle de l'Environnement (Architecture Avancée)

Refonte et optimisation de l'environnement Gym (ygoenv) pour permettre à l'agent PPO de comprendre l'intégralité du jeu (Espace d'Actions et Observation Space), et résoudre les blocages complexes du moteur C++.

### Story 5.1: Espace d'Actions Discretisé (200 actions statiques)

As a Chercheur en IA,
I want d'abandonner l'espace d'actions dynamique au profit d'un espace discretisé fixe (`Discrete(200)`),
So that le réseau de neurones PPO puisse projeter ses probabilités d'actions sur des neurones de sortie statiques (Action Masking).

**Acceptance Criteria:**

**Given** l'initialisation de l'environnement
**When** on interroge `env.action_space`
**Then** il retourne un `spaces.Discrete(200)`
**And** les actions générées par le moteur C++ (Invocations, Poses, Activations, Battle) sont mappées de manière déterministe sur les 200 slots sémantiques.

### Story 5.2: Auto-complétion et Ciblage Dynamique (MSG_SELECT_CARD)

As a Chercheur en IA,
I want que l'environnement puisse gérer dynamiquement les requêtes de ciblage (MSG_SELECT_CARD),
So that l'agent ne reste pas bloqué indéfiniment lorsqu'il doit choisir la cible d'une attaque ou d'un effet complexe.

**Acceptance Criteria:**

**Given** un effet ou une attaque nécessitant un ciblage
**When** le moteur ocgcore renvoie `MSG_SELECT_CARD` avec une liste de cibles légales
**Then** l'environnement Gym permet à l'agent d'utiliser les slots 191 à 199 pour choisir une cible parmi la liste
**And** si plusieurs cibles sont requises, l'environnement boucle avec le C++ pour auto-compléter la sélection sans crasher.

### Story 5.3: Observation Complète à 156 Slots (Brouillard de Guerre)

As a Chercheur en IA,
I want que l'observation de l'agent passe à une représentation matricielle complète du jeu (156 emplacements pour les deux joueurs) incluant un brouillard de guerre,
So that l'agent puisse percevoir tout le terrain, sa main, les cimetières, et "cacher" les informations privées de l'adversaire (cartes face cachée, main).

**Acceptance Criteria:**

**Given** le calcul de l'observation à chaque étape
**When** `env.step()` ou `env.reset()` est appelé
**Then** l'environnement construit un tenseur `(60694,)` concaténant 156 slots sémantiques et un vecteur d'information globale (LP, Phase)
**And** les caractéristiques et embeddings des cartes cachées ou adverses non publiques sont masqués par des zéros.

### Story 5.4: Entraînement de l'Agent PPO avec Masquage d'Actions (Action Masking)

As a Chercheur en IA,
I want d'entraîner le modèle PPO en utilisant le masque d'actions (Action Mask) généré par l'environnement,
So that le réseau apprenne à attribuer 0% de probabilité aux actions illégales, évitant ainsi le gaspillage d'exploration et accélérant l'apprentissage des conditions de victoire.

**Acceptance Criteria:**

**Given** le lot d'observations renvoyé par Gym
**When** le PPO met à jour ses poids
**Then** il pénalise la prédiction d'actions illégales à l'aide d'un gradient "Action Masking"
**And** la Value Loss et l'Entropy démontrent que l'agent apprend à explorer uniquement les actions valides.

## Epic 6: Intelligence Artificielle et Planification (Deep RL)

Connecter l'environnement à un modèle JAX complet capable d'apprendre par lui-même (Self-Play) en respectant les actions légales, en retenant l'information cachée via LSTM, en planifiant avec MCTS et en s'entraînant massivement via Ray.

### Story 6.1: Intégration LSTM et État de Croyance (Belief State)

As a Chercheur en IA,
I want d'intégrer des couches LSTM au sein de la politique PPO sous JAX,
So that l'agent puisse traiter la séquence temporelle de l'observation (60694 dimensions) pour construire un état de croyance (belief state) et se souvenir des cartes cachées ou tutorisées par l'adversaire (bluff).

**Acceptance Criteria:**

**Given** un tenseur d'observation masqué par le brouillard de guerre
**When** l'agent LSTM traite la séquence temporelle des observations et des actions passées
**Then** le réseau maintient et met à jour un "hidden state" persistant
**And** l'agent utilise efficacement la mémoire à court/long terme pour prendre des décisions face à des informations incomplètes (POMDP).

### Review Findings

- [x] [Review][Patch] Sequence Processing Bottleneck [ai/ppo.py]
- [x] [Review][Patch] Missing Gradient Clipping [ai/ppo.py]
- [x] [Review][Patch] Fake Tests [tests/test_flax_ppo.py]
- [x] [Review][Patch] Careless Type Coercion / Dtypes [ai/ppo.py]
- [x] [Review][Patch] Unsafe Single-Batch Dimensions [ai/ppo.py]
- [x] [Review][Patch] Fragile Entropy Calculation [ai/ppo.py]
- [x] [Review][Patch] Optimizer Edge Cases [ai/ppo.py]
- [x] [Review][Patch] Shape Mismatches in Environment Processing [ai/ppo.py]
- [x] [Review][Patch] Missing Past Actions in LSTM Input [ai/network.py]
- [x] [Review][Patch] Missing Return Type Hints [ai/network.py]
- [x] [Review][Defer] Brittle Shared Network Topology [ai/ppo.py] — deferred, pre-existing
- [x] [Review][Defer] Masking Critical Environment Bugs [ai/ppo.py] — deferred, pre-existing
- [x] [Review][Defer] Dangerous Magic Numbers [ai/network.py] — deferred, pre-existing
- [x] [Review][Defer] Rigid Optimizer Configuration [ai/ppo.py] — deferred, pre-existing
- [x] [Review][Defer] Naive Massive Dense Layer [ai/network.py] — deferred, pre-existing

### Story 6.2: Recherche Arborescente MCTS (Gumbel AlphaZero)

As a Chercheur en IA,
I want d'intégrer un algorithme MCTS (Gumbel AlphaZero) pour guider le réseau de neurones,
So that l'agent puisse planifier et anticiper les coups futurs au-delà d'une réaction myope à l'instant t.

**Acceptance Criteria:**

**Given** la phase d'entraînement en Self-Play
**When** le MCTS est configuré en mode d'omniscience
**Then** le MCTS voit les cartes cachées pour simuler l'arbre des possibles très rapidement et guider le réseau
**And** en partie réelle (inférence), le bot utilise un Information Set MCTS (IS-MCTS) ou s'appuie sur la politique PPO guidée pour réagir aux états cachés.

### Story 6.3: Boucle d'Apprentissage Asynchrone (Self-Play)

As a Chercheur en IA,
I want de mettre en place une boucle d'auto-jeu asynchrone (Self-Play),
So that l'agent puisse constamment affronter des versions antérieures de lui-même et collecter des trajectoires pour améliorer continuellement sa stratégie et éviter d'exploiter un adversaire aléatoire.

**Acceptance Criteria:**

**Given** l'exécution du script `scripts/train.py`
**When** les environnements parallèles sont lancés
**Then** les agents jouent asynchronement contre des itérations passées de leur propre modèle
**And** les trajectoires générées mettent à jour les poids du réseau JAX PPO de manière stable sans s'enfermer dans un optimum local trivial.

### Story 6.4: Distribution Massive et League Training via Ray

As a Chercheur en IA / DevOps,
I want de déployer notre boucle d'entraînement via le framework Ray,
So that l'apprentissage puisse passer à l'échelle sur des clusters multi-GPU et multi-cœurs (League Training), gérant l'espace d'états gigantesque de Yu-Gi-Oh!.

**Acceptance Criteria:**

**Given** le besoin d'augmenter drastiquement le nombre de parties simulées
**When** le script de déploiement est lancé via le cluster Ray
**Then** Ray orchestre la distribution des calculs, équilibrant la charge des différents acteurs MCTS et du Learner central
**And** le League Training gère dynamiquement plusieurs pools de versions d'agents (decks méta, historiques, etc.) sans goulot d'étranglement mémoire.

