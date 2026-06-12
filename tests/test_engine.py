import pytest
import ctypes
import os
from pathlib import Path

# Nous allons créer le wrapper dans `core.ygoenv.wrapper`
from core.ygoenv.wrapper import EngineCrashError, YgoEngine

def test_engine_initialization():
    """Test que le moteur s'initialise correctement et charge ocgcore.dll."""
    engine = YgoEngine()
    assert engine is not None
    assert getattr(engine, 'lib', None) is not None, "La librairie C n'a pas été chargée"

def test_engine_legal_actions():
    """Test qu'un état initial valide retourne des actions possibles (ou du moins ne crashe pas)."""
    engine = YgoEngine()
    
    # On simule un état très simple ou même vide.
    # Dans un vrai scénario, `state` serait un dictionnaire.
    mock_state = {"duel_id": 1, "step": 0, "player": 0}
    
    # On s'attend à ce que le moteur retourne une liste d'actions ou lève une erreur si c'est invalide.
    # On a implémenté un système Fail Fast, donc si le state est None ça plante, mais avec un dict,
    # le système l'accepte et renvoie les actions.
    actions = engine.get_legal_actions(mock_state)
    assert isinstance(actions, list)
    assert len(actions) > 0

def test_engine_fail_fast_exception():
    """Test spécifique pour valider le Fail Fast : un état corrompu doit lever EngineCrashError."""
    engine = YgoEngine()
    
    with pytest.raises(EngineCrashError):
        # Envoyer un état totalement invalide pour forcer une erreur
        engine.get_legal_actions(None)
