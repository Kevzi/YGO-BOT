import ray
import numpy as np
import random
import copy

@ray.remote
class SelfPlayManager:
    """
    Gestionnaire centralisé pour le League Training.
    Stocke les snapshots historiques du modèle et fournit des adversaires.
    """
    def __init__(self, max_history=50):
        self.history = []
        self.max_history = max_history

    def add_snapshot(self, params):
        """Ajoute un snapshot de poids dans l'historique."""
        # On copie profondément pour éviter les mutations
        # params est un PyTree de ndarrays JAX. copy.deepcopy fonctionne généralement.
        # Mais pour être sûr avec JAX, on peut juste stocker la référence si le Learner 
        # recrée un nouvel arbre à chaque update, ce qui est le cas avec optax.
        self.history.append(params)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        print(f"[SelfPlayManager] Snapshot ajouté. Taille de l'historique: {len(self.history)}")

    def get_opponent(self, latest_params):
        """
        Retourne les poids d'un adversaire.
        50% du temps le modèle le plus récent (latest_params),
        50% du temps un modèle historique aléatoire.
        """
        if not self.history or random.random() < 0.5:
            return latest_params
            
        idx = random.randint(0, len(self.history) - 1)
        return self.history[idx]
        
    def get_history_size(self):
        return len(self.history)
