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
        """Charge ocgcore via ctypes."""
        import sys
        core_dir = Path(__file__).parent.parent
        
        if sys.platform.startswith("win"):
            dll_path = core_dir / "ocgcore.dll"
        elif sys.platform.startswith("darwin"):
            dll_path = core_dir / "libocgcore.dylib"
        else:
            dll_path = core_dir / "libocgcore.so"
            
        if dll_path.is_file():
            try:
                # Load DLL
                self.lib = ctypes.CDLL(str(dll_path))
                
                # Bind C functions
                self.lib.create_duel.argtypes = [ctypes.c_uint32]
                self.lib.create_duel.restype = ctypes.c_void_p
                
                self.lib.start_duel.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                self.lib.end_duel.argtypes = [ctypes.c_void_p]
                
                self.lib.process.argtypes = [ctypes.c_void_p]
                self.lib.process.restype = ctypes.c_uint32
                
                # Setup callbacks placeholders
                self.lib.set_script_reader.argtypes = [ctypes.c_void_p]
                self.lib.set_card_reader.argtypes = [ctypes.c_void_p]
                self.lib.set_message_handler.argtypes = [ctypes.c_void_p]
                
            except Exception as e:
                raise EngineCrashError(f"Erreur lors du chargement de ocgcore: {e}")
        else:
            raise EngineCrashError(f"Le moteur compilé est introuvable à l'emplacement: {dll_path}")

    def get_legal_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrait les actions légales depuis l'état actuel du jeu.
        """
        if state is None:
            raise EngineCrashError("L'état passé au moteur est invalide (None).")
            
        try:
            # Test réel du C++ : on crée un duel temporaire
            duel_ptr = self.lib.create_duel(12345)
            if not duel_ptr:
                raise EngineCrashError("Le moteur a retourné un pointeur nul (crash interne).")
            
            try:
                # Transmission de l'état
                self._apply_state(duel_ptr, state)
                
                # Traitement moteur
                self.lib.process(duel_ptr)
                
                # Extraction des actions
                actions = self._extract_actions(duel_ptr)
                
                return actions
            finally:
                # On termine le duel
                self.lib.end_duel(duel_ptr)
        except Exception as e:
            if isinstance(e, EngineCrashError):
                raise
            raise EngineCrashError(f"Crash du moteur lors de la récupération des actions: {e}")

    def _apply_state(self, duel_ptr, state: Dict[str, Any]):
        # TODO: Appeler set_player_info, new_card, etc. en fonction de l'état
        pass

    def _extract_actions(self, duel_ptr) -> List[Dict[str, Any]]:
        # TODO: Appeler get_message, query_field_info pour populer les actions réelles
        return [{"action_type": 1, "source": "engine_process"}]

    def save_state(self):
        """Retourne un buffer binaire de l'état du duel C++."""
        # TODO: Appeler une fonction C++ pour extraire la mémoire du duel
        return None

    def restore_state(self, state_buffer):
        """Restaure l'état C++ depuis un buffer binaire."""
        # TODO: Appeler une fonction C++ pour restaurer la mémoire du duel
        pass
