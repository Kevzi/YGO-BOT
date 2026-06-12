import os
import pickle
import logging
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Exception levée pour les erreurs liées au chargement des embeddings."""
    pass

class EmbeddingLoader:
    def __init__(self, filepath: str = "data/embed.pkl"):
        self.filepath = filepath
        self._card_to_idx = None
        self._matrix = None
        self._embedding_dim = 0
        self._is_loaded = False

    def load(self):
        """Désérialise et charge le dictionnaire d'embeddings en mémoire vive."""
        if not os.path.exists(self.filepath):
            raise EmbeddingError(f"Le fichier d'embeddings est introuvable : {self.filepath}")

        try:
            with open(self.filepath, "rb") as f:
                data = pickle.load(f)
                
            if not isinstance(data, dict):
                raise EmbeddingError(f"Le fichier {self.filepath} doit contenir un dictionnaire Python.")
            if not data:
                self._embedding_dim = 0
                self._is_loaded = True
                return
                
            self._embedding_dim = len(next(iter(data.values())))
            num_cards = len(data)
            
            # Matrice continue Numpy pour lookups rapides. Dernière ligne = unknown (zeros)
            matrix = np.zeros((num_cards + 1, self._embedding_dim), dtype=np.float32)
            card_to_idx = {}
            
            for idx, (card_id, vector) in enumerate(data.items()):
                matrix[idx] = np.array(vector, dtype=np.float32)
                card_to_idx[int(card_id)] = idx
                
            self._matrix = matrix
            self._card_to_idx = card_to_idx
            self._is_loaded = True
            logger.info(f"Loaded {num_cards} card embeddings from {self.filepath} into memory.")
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Erreur lors de la désérialisation de {self.filepath}: {e}")

    def is_loaded(self) -> bool:
        """Indique si les embeddings sont chargés."""
        return self._is_loaded

    def get_embedding(self, card_id: int) -> jnp.ndarray:
        """Retourne l'embedding JAX pour un ID de carte. O(1)."""
        if not self.is_loaded():
            raise EmbeddingError("Les embeddings n'ont pas été chargés. Appelez load() d'abord.")
            
        if self._embedding_dim == 0:
            return jnp.zeros((0,), dtype=jnp.float32)
            
        idx = self._card_to_idx.get(card_id, len(self._card_to_idx))
        return jnp.asarray(self._matrix[idx])

    def get_embeddings_batch(self, card_ids: jnp.ndarray) -> jnp.ndarray:
        """Retourne un batch d'embeddings pour un tableau d'IDs."""
        if not self.is_loaded():
            raise EmbeddingError("Les embeddings n'ont pas été chargés.")
        if self._embedding_dim == 0:
            return jnp.zeros((len(card_ids), 0), dtype=jnp.float32)
            
        card_ids_np = np.asarray(card_ids)
        unknown_idx = len(self._card_to_idx)
        
        # Fast vectorized lookup via map
        # Vectorisation JAX-compatible : utilisation de Numpy avancé
        indices = np.array([self._card_to_idx.get(int(cid), unknown_idx) for cid in card_ids_np], dtype=np.int32)
        batch_np = self._matrix[indices]
                
        return jnp.asarray(batch_np)

# Global instance for the API
embed_loader = EmbeddingLoader()
