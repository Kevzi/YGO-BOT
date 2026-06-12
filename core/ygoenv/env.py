import gymnasium as gym
import numpy as np
from core.ygoenv.wrapper import YgoEngine, EngineCrashError

class YgoEnv(gym.Env):
    """
    Environnement Gym standardisé encapsulant le moteur C++ (ocgcore) via YgoEngine.
    """
    
    def __init__(self):
        super(YgoEnv, self).__init__()
        
        # Dimensions pour le MVP:
        # - Action space: par exemple 200 actions discrètes possibles (id de l'action)
        self.action_space = gym.spaces.Discrete(200)
        
        self.NUM_CARDS = 15
        self.DEFAULT_EMBEDDING_DIM = 64
        
        from ai.embeddings import embed_loader
        self.embedding_dim = embed_loader._embedding_dim if embed_loader.is_loaded() and embed_loader._embedding_dim > 0 else self.DEFAULT_EMBEDDING_DIM
        
        # - Observation space: tenseur plat float32 de taille fixe (NUM_CARDS * embedding_dim) représentant le terrain vectorisé
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.NUM_CARDS * self.embedding_dim,), dtype=np.float32)
        
        self.engine = YgoEngine()
        self._current_state = None
        
    def _get_observation(self, mock_card_ids: np.ndarray = None) -> np.ndarray:
        # TODO: Appeler l'engine pour extraire les identifiants réels des cartes sur le terrain
        # Pour le MVP, on simule une liste de cartes. La plupart sont vides (0, donc inconnues).
        from ai.embeddings import embed_loader
        
        card_ids = mock_card_ids if mock_card_ids is not None else np.zeros(self.NUM_CARDS, dtype=np.int32)
        
        if not embed_loader.is_loaded():
            import logging
            logging.warning("Embedding loader non chargé ! Appel implicite à load() ou retour de zéros si mock.")
            # Fallback en silence uniquement parce qu'on est en dev/test. Dans un vrai moteur : embed_loader.load()
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        if embed_loader._embedding_dim != self.embedding_dim:
            raise ValueError(f"Shape Mismatch Lazy-Loading: YgoEnv initialisé avec {self.embedding_dim} mais embed_loader a {embed_loader._embedding_dim}.")
            
        vectors = embed_loader.get_embeddings_batch(card_ids)
        return np.asarray(vectors, dtype=np.float32).flatten()
        
    def get_legal_actions(self) -> np.ndarray:
        if self._current_state is None:
            return np.ones(self.action_space.n, dtype=np.bool_)
            
        actions = self.engine.get_legal_actions(self._current_state)
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        for a in actions:
            # MVP: on utilise 'action_type' comme index d'action
            idx = a.get("action_type", 0)
            if 0 <= idx < self.action_space.n:
                mask[idx] = True
                
        # Fallback de sécurité pour éviter un masque complètement vide
        if not np.any(mask):
            mask[0] = True
            
        return mask
        
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        
        # Initialiser un duel ou réinitialiser l'état
        self._current_state = {"phase": "DRAW", "turn": 1}
        
        obs = self._get_observation()
        info = {"legal_actions": self.get_legal_actions()}
        
        return obs, info
        
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._current_state is None:
            raise RuntimeError("L'environnement doit être réinitialisé via reset() avant le premier step().")
            
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide: {action}")
            
        # Appliquer l'action au state
        self._current_state["last_action"] = action
            
        # Simuler un passage au tour suivant ou fin de partie
        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        info = {"legal_actions": self.get_legal_actions()}
        
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
