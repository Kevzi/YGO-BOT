---
epic: "Epic 5: Mise à l'échelle de l'Environnement (Architecture Avancée)"
story: "Story 5.3: Observation Complète à 156 Slots (Brouillard de Guerre)"
status: "in-progress"
---

# Story 5.3: Observation Complète à 156 Slots (Brouillard de Guerre)

## Business Value
Pour qu'un réseau de neurones PPO apprenne à jouer à un niveau compétitif à Yu-Gi-Oh!, il doit percevoir l'intégralité du jeu, et non un résumé abstrait. Cela inclut la main, le cimetière, la zone bannie, l'extra deck et le deck. De plus, il doit apprendre à composer avec l'incertitude : les cartes face cachées ou dans la main de l'adversaire (information privée) doivent lui être masquées pour forcer l'émergence d'une mémoire et du bluff.

## Acceptance Criteria

1. **Extraction C++ Exhaustive**
   - **Given** une étape de duel (step ou reset)
   - **When** l'environnement appelle `query_field_state`
   - **Then** le C++ est interrogé sur toutes les localisations : Deck, Hand, MZone, SZone, Grave, Removed, Extra.

2. **Brouillard de Guerre (Fog of War)**
   - **Given** l'extraction des données d'une carte
   - **When** cette carte appartient à l'adversaire ET n'est pas publique (ex: Posée face cachée, dans la main, etc.)
   - **Then** ses statistiques (ATK, DEF, Level, etc.) et son vecteur sémantique sont remplacés par des zéros.

3. **Format du Tenseur d'Observation (156 Slots + Global)**
   - **Given** la construction finale de l'observation
   - **When** le wrapper concatène toutes les données
   - **Then** il crée un tenseur de dimension fixe `(60694,)` contenant 156 emplacements de cartes plus un vecteur global (LP, Phase).

4. **Validité JAX**
   - **Given** l'observation finale
   - **When** elle est transmise au réseau PPO
   - **Then** son type est strictement `jnp.float32` ou `np.float32`.

## Technical Context
- **Constants** : Mettre à jour `constants.py` avec les drapeaux manquants `QUERY_ALIAS`, `QUERY_TYPE`, et `QUERY_END`.
- **Wrapper C++** : Modifier `wrapper.py` (`query_field_state`) pour extraire et parser tous ces nouveaux champs en mémoire binaire (via la méthode `read_uint...` de `BinaryReader`). Ajouter une méthode `get_global_info()` utilisant `OCG_DuelQueryField`.
- **Environment Gym** : Dans `env.py`, redéfinir la construction du tableau d'observation. Assigner chaque carte à son emplacement précis (`[:20]` pour Grave, etc.), injecter le brouillard de guerre, et l'information globale (LP et encodeurs de Phase).
