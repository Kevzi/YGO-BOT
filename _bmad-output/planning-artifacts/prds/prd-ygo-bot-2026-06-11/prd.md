---
title: "PRD: ygo bot (Agent Intelligent Autonome Yu-Gi-Oh!)"
status: final
created: "2026-06-11"
updated: "2026-06-11"
---

# PRD: ygo bot (Agent Intelligent Autonome Yu-Gi-Oh!)

## 1. Vision et Objectifs
**Vision :** Développer une intelligence artificielle autonome capable d'atteindre un niveau de performance surhumain au jeu de cartes Yu-Gi-Oh!. En s'appuyant sur les dernières avancées en apprentissage par renforcement (Deep RL) et en traitement du langage naturel (LLMs), le bot est conçu pour maîtriser un environnement stochastique à information imparfaite avec un espace d'états gigantesque. 

**Objectifs principaux :**
1. **Recherche et Développement :** Fournir une plateforme d'expérimentation open-source (ygoenv, ygo-agent) pour repousser les limites de l'IA.
2. **Produit pour les joueurs :** Offrir un "sparring-partner" de très haut niveau, intégré de manière transparente dans des clients comme YGO Omega ou Neos.
3. **Généralisation :** Maintenir une stabilité surhumaine sur un pool de "méta-decks" validés tout en gardant une capacité "zero-shot" pour interpréter de nouvelles cartes grâce aux embeddings.

## 2. Public Cible
1. **Chercheurs et Ingénieurs en IA :** Utilisent l'environnement pour tester des algorithmes (PPO, MCTS, NEAT).
2. **Développeurs de la communauté :** Créent des bots ou intègrent l'IA dans différents environnements via les wrappers et APIs fournis.
3. **Joueurs (Compétitifs et Amateurs) :** Affrontent l'IA pour s'entraîner, tester des decks (deck-testing) ou s'amuser, via une interface utilisateur familière (Neos, Omega).

## 3. Cas d'Utilisation (User Journeys)
- **UJ-1: Le joueur compétitif s'entraîne contre la méta.** Un joueur lance YGO Omega. Il spécifie le chemin de son fichier de deck (`.ydk`) via le paramètre de configuration `DeckFile`. Le bot tourne localement via un serveur FastAPI. Le joueur affronte un deck méta contrôlé de main de maître par l'IA.
- **UJ-2: Le chercheur entraîne un nouveau modèle.** Un ingénieur lance un entraînement distribué sur GPU en utilisant JAX. L'environnement Python génère des millions de parties simulées à l'aide du moteur C++ sous-jacent.
- **UJ-3: L'intégration Web Neos.** Un joueur occasionnel se connecte au client web Neos. L'IA est exécutée par le joueur en tâche de fond sur sa machine locale, exposant une API sur `127.0.0.1:3000`. Dans Neos, il construit son deck via l'interface standard, renseigne l'adresse locale dans les paramètres, et lance le duel. Le front-end web se branche sur la puissance du "cerveau" local. (Des serveurs communautaires comme Koishi peuvent aussi héberger l'IA à distance).

## 4. Fonctionnalités Requises (Features)
### 4.1. Architecture Décisionnelle (Le Cerveau)
- **FR-1.1 Compréhension Sémantique et "Zero-Shot" Hybride :** Traduction des textes complexes des cartes en embeddings vectoriels (`embed.pkl`). Le bot offre une performance surhumaine garantie sur une liste validée de cartes (`scripts/code_list.txt`). Face à des cartes inconnues, l'IA tente d'extrapoler l'action via sa capacité "zero-shot", avec une marge d'erreur inhérente à l'hyper-complexité du jeu.
- **FR-1.2 Apprentissage par Renforcement (RL) :** Utilisation de l'algorithme PPO sous le framework JAX.
- **FR-1.3 Mémoire et Bluff :** Intégration de réseaux LSTM pour la mémoire à court/long terme.
- **FR-1.4 Planification et Anticipation :** Implémentation de MCTS (Gumbel AlphaZero) pour anticiper les coups futurs face aux informations cachées.

### 4.2. Moteur de Règles et Environnement
- **FR-2.1 Séparation Logique :** Le bot ne calcule pas les règles. Il interroge un moteur C++ ultra-rapide (ocgcore/ygopro-core, scripté en Lua) pour valider les actions légales et résoudre les chaînes.
- **FR-2.2 Environnement Gym (ygoenv) :** Interface Python standardisée pour faciliter l'entraînement de l'agent.

### 4.3. Déploiement et Intégration (Sparring-Partner)
- **FR-3.1 API d'Inférence :** Serveur local Python (FastAPI/Uvicorn) exposant des endpoints pour recevoir l'état du jeu et retourner l'action optimale.
- **FR-3.2 Interception Simulateur :** Wrapper C# (DuelBotWrapper) ou scripts pour intercepter l'état graphique/mémoire sur YGO Omega et EDOPro.
- **FR-3.3 Clients Supportés :** Intégration transparente pour l'utilisateur final sur YGO Omega (client lourd) et Neos (client web).

### 4.4. Analyse et Données
- **FR-4.1 Intégration API YGOPRODeck :** Récupération automatique des données de cartes, images et prix.
- **FR-4.2 Parsing de Decks :** Support de la lecture des codes de deck via omega-api-decks.

## 5. Exigences Non-Fonctionnelles (NFR)
- **NFR-1 Latence d'inférence :** Le temps de calcul pour la sélection d'une action par l'IA doit idéalement être inférieur à 100 ms en production pour ne pas frustrer le joueur humain.
- **NFR-2 Fidélité des règles :** Le bot doit respecter 100% des règles officielles du jeu, garanti par l'ocgcore.
- **NFR-3 Portabilité de l'inférence :** Pour une expérience fluide, une carte graphique NVIDIA d'entrée de gamme (ex: GTX 1650) est recommandée. Cependant, l'inférence stricte sur CPU est fonctionnellement supportée en mode dégradé (via `--xla_device cpu`).
