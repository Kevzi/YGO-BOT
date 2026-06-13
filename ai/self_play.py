import ray
import numpy as np
import random
import copy

@ray.remote
class SelfPlayManager:
    """
    Gestionnaire centralisé pour le League Training.
    Stocke les snapshots historiques du modèle et sert de Parameter Server pour les poids récents.
    """
    def __init__(self, max_history=50):
        self.history = []
        self.max_history = max_history
        self.latest_params = None

    def set_latest_params(self, params):
        """Met à jour les poids les plus récents (utilisé par le Learner)."""
        self.latest_params = params

    def get_latest_params(self):
        """Retourne les poids les plus récents (utilisé par le Worker pour l'agent principal)."""
        return self.latest_params

    def add_snapshot(self, params):
        """Ajoute un snapshot de poids dans l'historique."""
        self.history.append(params)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        print(f"[SelfPlayManager] Snapshot ajouté. Taille de l'historique: {len(self.history)}")

    def get_match_params(self):
        """
        Retourne les poids d'un adversaire.
        50% du temps le modèle le plus récent,
        50% du temps un modèle historique aléatoire.
        """
        if not self.history or random.random() < 0.5:
            return self.latest_params
            
        idx = random.randint(0, len(self.history) - 1)
        return self.history[idx]
        
    def get_history_size(self):
        return len(self.history)
