import os
import tempfile
import pickle
import numpy as np
import jax.numpy as jnp
import pytest

from ai.embeddings import EmbeddingLoader, EmbeddingError

def test_embedding_loader_deserialization():
    # Arrange
    # Create a dummy embed.pkl file
    dummy_data = {
        89631139: np.array([0.1, 0.2, 0.3], dtype=np.float32), # Blue-Eyes White Dragon
        46986414: np.array([0.4, 0.5, 0.6], dtype=np.float32), # Dark Magician
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embed_path = os.path.join(tmpdir, "embed.pkl")
        with open(embed_path, "wb") as f:
            pickle.dump(dummy_data, f)
            
        # Act
        loader = EmbeddingLoader(embed_path)
        loader.load()
        
        # Assert
        assert loader.is_loaded() is True
        
        # JAX arrays conversion check
        be_embed = loader.get_embedding(89631139)
        assert isinstance(be_embed, jnp.ndarray)
        assert be_embed.dtype == jnp.float32
        assert jnp.allclose(be_embed, jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))

def test_embedding_loader_missing_file():
    # Arrange
    loader = EmbeddingLoader("non_existent_file.pkl")
    
    # Act & Assert
    with pytest.raises(EmbeddingError):
        loader.load()

def test_embedding_loader_unknown_card():
    # Arrange
    dummy_data = {
        89631139: np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embed_path = os.path.join(tmpdir, "embed.pkl")
        with open(embed_path, "wb") as f:
            pickle.dump(dummy_data, f)
            
        loader = EmbeddingLoader(embed_path)
        loader.load()
        
        # Act & Assert
        # For a zero-shot model, we might want it to return a zero vector 
        # or raise an error. Let's say it returns a zero vector with a warning.
        # But wait, "La conversion de l'ID d'une carte vers son vecteur sémantique se fasse de façon instantanée"
        # If the card is unknown (meaning not in embed.pkl), we can return zeros.
        unknown_embed = loader.get_embedding(12345678)
        assert unknown_embed.shape == (3,)
        assert jnp.allclose(unknown_embed, jnp.zeros(3, dtype=jnp.float32))

def test_embedding_loader_batch_retrieval():
    # Arrange
    dummy_data = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embed_path = os.path.join(tmpdir, "embed.pkl")
        with open(embed_path, "wb") as f:
            pickle.dump(dummy_data, f)
            
        loader = EmbeddingLoader(embed_path)
        loader.load()
        
        # Act
        batch_ids = jnp.array([1, 2, 3]) # 3 is unknown
        batch_embeds = loader.get_embeddings_batch(batch_ids)
        
        # Assert
        assert batch_embeds.shape == (3, 2)
        assert jnp.allclose(batch_embeds[0], jnp.array([1.0, 0.0]))
        assert jnp.allclose(batch_embeds[1], jnp.array([0.0, 1.0]))
        assert jnp.allclose(batch_embeds[2], jnp.array([0.0, 0.0])) # Zeros for unknown
