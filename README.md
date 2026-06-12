# YGO-BOT 🐉

> **Une Intelligence Artificielle Autonome (Deep RL) pour Yu-Gi-Oh!**

YGO-BOT est un projet de recherche et d'ingénierie visant à développer un agent intelligent capable d'atteindre un niveau de jeu surhumain au jeu de cartes à collectionner Yu-Gi-Oh!, en s'appuyant sur l'Apprentissage par Renforcement Profond (Deep Reinforcement Learning) et des modèles de fondation (LLMs).

## 🚀 Fonctionnalités Principales

- **Moteur de Règles C++** : Intégration du moteur officiel `ocgcore` encapsulé via un environnement OpenAI Gym (`ygoenv`) pour une simulation parfaite des duels.
- **Deep Reinforcement Learning** : Algorithme PPO (Proximal Policy Optimization) implémenté en **JAX** pour une exécution ultra-rapide et parallélisée.
- **Entraînement Distribué** : Orchestration via **Ray** pour le "League Training" (Self-Play et agents spécialisés) à grande échelle sur grappes CPU/TPU.
- **Architecture Cognitive Avancée** :
  - *Belief State (LSTM)* pour la gestion de l'information imparfaite (cartes face cachée, bluff).
  - *Recherche MCTS / Gumbel AlphaZero* pour la projection et la planification des tours.
  - *Moteur Zero-Shot via Embeddings* chargé en RAM (<100ms de latence) pour comprendre sémantiquement les nouvelles cartes inédites.
- **API Performante** : Serveur REST en **FastAPI** communiquant en `camelCase` avec les clients externes (ex: interfaces Omega/Neos).
- **Historisation et Statistiques** : Persistance robuste via **SQLite**, **SQLAlchemy 2.0**, et **Alembic** avec des modèles stricts (snake_case en DB).

## 🏗️ Architecture du Projet

```text
ygo-bot/
├── alembic/              # Migrations de base de données
├── data/                 # Base de données SQLite locale (ygo.db)
├── db/                   # Modèles SQLAlchemy (snake_case)
├── schemas/              # Schémas Pydantic (camelCase pour l'API)
├── tests/                # Suite de tests Pytest
├── _bmad-output/         # Artefacts de planification et suivi de sprints
└── pyproject.toml        # Dépendances (Poetry)
```

## 🛠️ Installation et Démarrage Rapide

Ce projet utilise [Poetry](https://python-poetry.org/) ou un environnement virtuel Python classique.

**Via environnement virtuel (recommandé) :**

```bash
# 1. Cloner le dépôt
git clone https://github.com/Kevzi/YGO-BOT.git
cd YGO-BOT

# 2. Créer un environnement virtuel
python -m venv .venv

# 3. Activer l'environnement
# Sous Windows (PowerShell) :
.\.venv\Scripts\Activate.ps1
# Sous Linux/Mac :
source .venv/bin/activate

# 4. Installer les dépendances
pip install sqlalchemy alembic pydantic fastapi uvicorn jax jaxlib ray pandas pytest

# 5. Mettre à jour la base de données locale (SQLite)
alembic upgrade head
```

## 🧪 Lancer les Tests

Pour s'assurer que les modèles de base de données et de schémas sont correctement configurés :

```bash
# S'assurer que le chemin racine est dans le PYTHONPATH
$env:PYTHONPATH='.'  # Windows PowerShell
export PYTHONPATH='.' # Linux/Mac

pytest tests/ -v
```

## 📈 Suivi du Développement

Le développement est orchestré via la méthode BMad (Agents Autonomes). Le cycle de vie est documenté dans le dossier `_bmad-output/`, incluant les épopées (`epics.md`) et le suivi de sprint (`sprint-status.yaml`).

### Roadmap actuelle (Epic 1 : Sparring-Partner Local)
- [x] Story 1.1 : Base de Données et Historisation (SQLite + Alembic)
- [ ] Story 1.2 : Intégration du Moteur C++ (ygoenv / ocgcore)
- [ ] Story 1.3 : Serveur API FastAPI
- [ ] Story 1.4 : Agent Dummy (Boucle complète)

## 📄 Licence

(À définir)
