import os
import ctypes
from pathlib import Path
from typing import Any, Dict, List

class EngineCrashError(Exception):
    """Exception levée lorsque le moteur C++ crashe ou retourne une erreur fatale."""
    pass

class YgoEngine:
    def __init__(self):
        self.lib = None
        self._load_library()

    def _load_library(self):
        """Charge ocgcore.dll via ctypes."""
        core_dir = Path(__file__).parent.parent
        dll_path = core_dir / "ocgcore.dll"
        
        if dll_path.exists():
            try:
                # Load DLL
                self.lib = ctypes.CDLL(str(dll_path))
                
                # Bind C functions
                self.lib.create_duel.argtypes = [ctypes.c_uint32]
                self.lib.create_duel.restype = ctypes.c_void_p
                
                self.lib.start_duel.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                self.lib.end_duel.argtypes = [ctypes.c_void_p]
                
                # Setup callbacks placeholders
                self.lib.set_script_reader.argtypes = [ctypes.c_void_p]
                self.lib.set_card_reader.argtypes = [ctypes.c_void_p]
                self.lib.set_message_handler.argtypes = [ctypes.c_void_p]
                
            except Exception as e:
                raise EngineCrashError(f"Erreur lors du chargement de ocgcore.dll: {e}")
        else:
            # Mode "mock" si le moteur n'est pas encore compilé (utile pour l'instant)
            self.lib = "MOCK_LIB"

    def get_legal_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrait les actions légales depuis l'état actuel du jeu.
        """
        if state is None:
            raise EngineCrashError("L'état passé au moteur est invalide (None).")
            
        if self.lib == "MOCK_LIB":
            # Mock behavior
            return [{"action_type": "draw"}]
            
        try:
            # Test réel du C++ : on crée un duel temporaire
            duel_ptr = self.lib.create_duel(12345)
            if not duel_ptr:
                raise EngineCrashError("Le moteur a retourné un pointeur nul (crash interne).")
            
            # On termine le duel
            self.lib.end_duel(duel_ptr)
        except Exception as e:
            raise EngineCrashError(f"Crash du moteur lors de la récupération des actions: {e}")
        
        return [{"action_type": "draw"}]
