import gymnasium as gym
import numpy as np
from core.ygoenv.wrapper import YgoEngine, EngineCrashError

class YgoEnv(gym.Env):
    """
    Environnement Gym standardisé encapsulant le moteur C++ (ocgcore) via YgoEngine.
    """
    
    def __init__(self, omniscience=False):
        super(YgoEnv, self).__init__()
        self.omniscience = omniscience
        
        # Dimensions pour le MVP:
        # Espace d'action: Par exemple, 200 actions discrètes possibles
        self.action_space = gym.spaces.Discrete(200)
        
        # Espace d'observation: Un vecteur plat
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(60694,), dtype=np.float32)
        
        self.engine = YgoEngine()
        self._current_state = None
        self.max_turns = 100
        self.action_history = []
        self.step_count = 0
        
        # Charger les embeddings
        try:
            from ai.embeddings import EmbeddingLoader
            self.embed_loader = EmbeddingLoader()
            self.embed_loader.load()
            embed_dim = self.embed_loader._embedding_dim
            if embed_dim > 0:
                self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(60694,), dtype=np.float32)
            import logging
            logging.info(f"Embeddings chargés avec dimension {embed_dim}.")
        except Exception as e:
            import logging
            logging.error(f"Erreur lors du chargement des embeddings: {e}")
            self.embed_loader = None

    def get_action_mask(self):
        """
        Retourne un masque booléen (ou de 0/1) indiquant quelles actions sont légales.
        Pour l'instant, c'est un mock où toutes les actions sont légales.
        """
        import numpy as np
        return np.ones(self.action_space.n, dtype=np.bool_)

    def set_deck(self, deck: dict):
        """Injecte le deck de l'agent dans l'environnement."""
        self.agent_deck = deck
        
    def _get_observation(self, mock_card_ids: np.ndarray = None) -> np.ndarray:
        # Appeler l'engine pour extraire les informations riches des cartes
        cards_info = self.engine.query_field_state(0) # on interroge l'état pour le joueur 0
        global_info = self.engine.get_global_info() if hasattr(self.engine, 'get_global_info') else {"lp": [8000, 8000]}
        
        # Brouillard de Guerre et Filtres
        for card in cards_info:
            player = card["player"]
            loc = card["location"]
            pos = card.get("position", 0)
            
            # Condition pour masquer:
            # - Appartient à l'adversaire (player == 1)
            # - ET (dans la main/deck/extra OU face cachée sur le terrain)
            is_opponent = (player == 1)
            is_hidden_location = loc in [0x01, 0x02, 0x40] # Deck, Hand, Extra
            is_facedown = (pos & 0xA) != 0 # POS_FACEDOWN_ATTACK = 2 | 8, POS_FACEDOWN_DEFENSE = 8
            
            if is_opponent and (is_hidden_location or is_facedown) and not self.omniscience:
                card["code"] = 0
                card["level"] = 0
                card["type"] = 0
                card["attack"] = 0
                card["defense"] = 0
                
        # Structure de 156 slots
        slots = []
        
        def filter_loc(p, l, max_len):
            found = [c for c in cards_info if c["player"] == p and c["location"] == l]
            # Pour la MZone/SZone, la séquence est importante (0 à 4 pour MZone, 0 à 5 pour EMZone, etc.)
            # Mais pour simplifier, on pad juste à max_len
            while len(found) < max_len:
                found.append({"code": 0, "position": 0, "type": 0, "level": 0, "attack": 0, "defense": 0})
            return found[:max_len]
            
        for player in [0, 1]:
            slots.extend(filter_loc(player, 0x04, 5)) # MZone (5)
            slots.extend(filter_loc(player, 0x08, 5)) # SZone (5)
            slots.extend(filter_loc(player, 0x100, 1)) # FZone (1) - YGOPro FZone est souvent seq=5 dans SZone, on gèrera plus tard, pour le moment on pad
            slots.extend(filter_loc(player, 0x200, 1)) # EMZone (1)
            slots.extend(filter_loc(player, 0x02, 15)) # Hand (15)
            slots.extend(filter_loc(player, 0x10, 20)) # Grave (20)
            slots.extend(filter_loc(player, 0x20, 15)) # Removed (15)
            slots.extend(filter_loc(player, 0x40, 15)) # Extra (15)
            
            # Le Deck (1 slot spécial)
            deck_cards = [c for c in cards_info if c["player"] == player and c["location"] == 0x01]
            deck_slot = {"code": 0, "position": 0, "type": 0, "level": len(deck_cards), "attack": 0, "defense": 0} # On stocke la taille dans "level"
            slots.append(deck_slot)

        # Résolution des embeddings et normalisation
        features = []
        for i, slot in enumerate(slots):
            cid = slot.get("code", 0)
            if cid is None:
                cid = 0
            
            # Normalisation
            pos = float(slot.get("position", 0))
            ctype = float(slot.get("type", 0))
            atk = float(slot.get("attack", 0)) / 5000.0
            def_val = float(slot.get("defense", 0)) / 5000.0
            
            # Traitement spécial pour le Deck Slot (index 77 et 155)
            if i == 77 or i == 155:
                lvl = float(slot.get("level", 0)) / 60.0 # Taille du deck
            else:
                lvl = float(slot.get("level", 0)) / 12.0
            
            stats = np.array([pos, lvl, ctype, atk, def_val], dtype=np.float32)
            
            if cid > 0 and self.embed_loader and hasattr(self.embed_loader, 'get_embedding'):
                try:
                    emb = self.embed_loader.get_embedding(int(cid))
                except Exception:
                    emb = np.zeros(384, dtype=np.float32)
            else:
                emb = np.zeros(384, dtype=np.float32) # Fallback
                
            features.append(stats)
            features.append(emb)
            
        # Ajout des Informations Globales (LP, Phase)
        lp_info = global_info.get("lp", [8000, 8000]) if isinstance(global_info, dict) else [8000, 8000]
        if not isinstance(lp_info, list) or len(lp_info) < 2:
            lp_info = [8000, 8000]
            
        lp_0 = float(lp_info[0]) / 8000.0
        lp_1 = float(lp_info[1]) / 8000.0
        
        # Encodage One-Hot de la Phase (8 dimensions)
        phase_map = {"DRAW": 0, "STANDBY": 1, "MAIN1": 2, "BATTLE": 3, "MAIN2": 4, "END": 5, "START": 6}
        phase_idx = phase_map.get(self._current_state.get("phase", "START"), 7)
        phase_one_hot = np.zeros(8, dtype=np.float32)
        phase_one_hot[phase_idx] = 1.0
        
        global_vec = np.array([lp_0, lp_1], dtype=np.float32)
        features.append(global_vec)
        features.append(phase_one_hot)
            
        # Concaténer en un seul vecteur (60694)
        obs = np.concatenate(features)
        return obs
        
    def get_legal_actions(self) -> np.ndarray:
        """
        Retourne un masque des actions légales depuis le moteur C++.
        """
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        
        if self._current_state is None:
            mask[0] = True
            return mask
            
        actions = self.engine.get_legal_actions(self._current_state)
        self._current_state_actions = actions # Save for step translation
        
        for a in actions:
            if a.get("msg") in ["MSG_SELECT_IDLECMD", "MSG_SELECT_BATTLECMD", "MSG_SELECT_CARD"]:
                for choice in a.get("choices", []):
                    idx = choice.get("action_idx")
                    if idx is not None and 0 <= idx < self.action_space.n:
                        mask[idx] = True
            else:
                # fallback old behaviour
                idx = a.get("action_type", 0)
                if 0 <= idx < self.action_space.n:
                    mask[idx] = True
                
        # Fallback de sécurité si l'engine ne retourne rien d'exploitable
        if not np.any(mask):
            mask[0] = True
                
        return mask
        
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        
        self.action_history = []
        
        # Détruire le duel précédent s'il existe
        self.engine.destroy_duel()
        
        # Initialiser un nouveau duel C++
        self._current_state = {"phase": "DRAW", "turn": 1}
        self._current_state_actions = []
        duel_ok = self.engine.create_duel()
        
        if not duel_ok:
            raise RuntimeError("Impossible de créer un duel C++.")
        else:
            # Charger un deck Beatdown pour les 2 joueurs
            try:
                from db.session import SessionLocal
                from db.models import Card
                with SessionLocal() as db:
                    # Blue-Eyes White Dragon (89631139) et Gene-Warped Warwolf (03201284)
                    beatdown_codes = [89631139, 3201284]
                    cards = db.query(Card).filter(Card.id.in_(beatdown_codes)).all()
                    codes = [c.id for c in cards]
                    
                    if len(codes) > 0:
                        # Remplir le Deck avec ce qu'on a trouvé (alterné pour faire 40 cartes)
                        deck = (codes * 20)[:40]
                        
                        for player in (0, 1):
                            for seq, code in enumerate(deck):
                                self.engine.add_card(code, player, 0x01, seq, 0x8)
                                
                        self.engine.start_duel()
                    else:
                        raise ValueError("Les decks sont vides, impossible de démarrer le duel.")
            except Exception as e:
                raise RuntimeError(f"Erreur lors du chargement du deck: {e}")
                
        # Le state n'est pas None, il faut au moins un dict pour bypasser le if is None
        self._current_state = {"phase": "START", "turn": 1}
        
        # On doit d'abord appeler get_legal_actions pour avancer le moteur (tirage des mains, etc.)
        legal_actions = self.get_legal_actions()
        obs = self._get_observation()
        
        current_player = 0
        if self._current_state_actions and len(self._current_state_actions) > 0:
            current_player = self._current_state_actions[0].get("player", 0)
            
        return obs, {"legal_actions": legal_actions, "current_player": current_player}
        
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._current_state is None:
            raise RuntimeError("L'environnement doit être réinitialisé via reset() avant le premier step().")
            
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide: {action}")
            
        self.action_history.append(action)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Traduction de l'action
        self.engine.translate_and_set_response(action, self._current_state_actions)
        
        # Le moteur simule le reste jusqu'à la prochaine attente
        legal_actions_mask = self.get_legal_actions()
        
        # Parser les messages pour chercher MSG_WIN (5) ou autres events importants
        for a in self._current_state_actions:
            if a.get("msg") == "WIN":
                terminated = True
                # player 0 = nous, player 1 = adversaire, player 2 = draw
                win_player = a.get("player", 2)
                if win_player == 0:
                    reward = 1.0
                elif win_player == 1:
                    reward = -1.0
                else:
                    reward = 0.0
                break
        
        # Mettre à jour l'état
        self.step_count += 1
        if self.step_count >= self.max_turns:
            truncated = True
            
        obs = self._get_observation()
        
        # Déterminer le joueur courant
        current_player = 0
        if self._current_state_actions and len(self._current_state_actions) > 0:
            current_player = self._current_state_actions[0].get("player", 0)
            
        info = {
            "legal_actions": legal_actions_mask,
            "current_player": current_player
        }
        
        return obs, reward, terminated, truncated, info

    def save_state(self) -> dict:
        """Sauvegarde l'état complet du duel (clone) pour le MCTS."""
        import copy
        engine_state = self.engine.save_state() if hasattr(self.engine, 'save_state') else None
        return {
            "current_state": copy.deepcopy(self._current_state),
            "engine_state": engine_state
        }

    def restore_state(self, state: dict):
        """Restaure un état sauvegardé du duel."""
        import copy
        self._current_state = copy.deepcopy(state["current_state"])
        if state.get("engine_state") is not None and hasattr(self.engine, 'restore_state'):
            self.engine.restore_state(state["engine_state"])

    def clone(self) -> 'YgoEnv':
        """
        Clone l'environnement via Action Replay.
        Crée un nouvel environnement et rejoue l'historique complet des actions.
        """
        new_env = YgoEnv(omniscience=self.omniscience)
        if hasattr(self, 'agent_deck'):
            new_env.set_deck(self.agent_deck)
            
        new_env.reset()
        
        for act in self.action_history:
            new_env.step(act)
            
        return new_env
