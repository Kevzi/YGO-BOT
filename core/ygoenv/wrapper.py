import os
import ctypes
from pathlib import Path
from typing import Any, Dict, List

class EngineCrashError(Exception):
    """Exception levée lorsque le moteur C++ crashe ou retourne une erreur fatale."""
    pass

class OCG_CardData(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint32),
        ("alias", ctypes.c_uint32),
        ("setcodes", ctypes.POINTER(ctypes.c_uint16)),
        ("type", ctypes.c_uint32),
        ("level", ctypes.c_int32),
        ("attribute", ctypes.c_uint32),
        ("race", ctypes.c_uint64),
        ("attack", ctypes.c_int32),
        ("defense", ctypes.c_int32),
        ("lscale", ctypes.c_uint32),
        ("rscale", ctypes.c_uint32),
        ("link_marker", ctypes.c_uint32),
        ("ot", ctypes.c_uint32),
    ]

class OCG_Player(ctypes.Structure):
    _fields_ = [
        ("startingLP", ctypes.c_uint32),
        ("startingDrawCount", ctypes.c_uint32),
        ("drawCountPerTurn", ctypes.c_uint32),
    ]

class OCG_DuelOptions(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint64 * 4),
        ("flags", ctypes.c_uint64),
        ("team1", OCG_Player),
        ("team2", OCG_Player),
        ("cardReader", ctypes.c_void_p),
        ("payload1", ctypes.c_void_p),
        ("scriptReader", ctypes.c_void_p),
        ("payload2", ctypes.c_void_p),
        ("logHandler", ctypes.c_void_p),
        ("payload3", ctypes.c_void_p),
        ("cardReaderDone", ctypes.c_void_p),
        ("payload4", ctypes.c_void_p),
        ("payload5", ctypes.c_void_p),
        ("enableUnsafeLibraries", ctypes.c_uint8),
    ]

class OCG_NewCardInfo(ctypes.Structure):
    _fields_ = [
        ("team", ctypes.c_uint8),
        ("duelist", ctypes.c_uint8),
        ("code", ctypes.c_uint32),
        ("con", ctypes.c_uint8),
        ("loc", ctypes.c_uint32),
        ("seq", ctypes.c_uint32),
        ("pos", ctypes.c_uint32),
    ]

class OCG_QueryInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("con", ctypes.c_uint8),
        ("loc", ctypes.c_uint32),
        ("seq", ctypes.c_uint32),
        ("overlay_seq", ctypes.c_uint32),
    ]

# Callback definitions
CARD_READER_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(OCG_CardData))
SCRIPT_READER_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)

def my_card_reader(payload, code, data_ptr):
    try:
        data_ptr.contents.code = code
        data_ptr.contents.alias = 0
        data_ptr.contents.lscale = 0
        data_ptr.contents.rscale = 0
        data_ptr.contents.link_marker = 0
        data_ptr.contents.ot = 0
        
        global _dummy_setcodes
        if '_dummy_setcodes' not in globals():
            _dummy_setcodes = (ctypes.c_uint16 * 1)(0)
        data_ptr.contents.setcodes = ctypes.cast(_dummy_setcodes, ctypes.POINTER(ctypes.c_uint16))
        
        # Load from cache
        from core.ygoenv.constants import YGO_TYPE_MAPPING, YGO_ATTRIBUTE_MAPPING, YGO_RACE_MAPPING
        card_info = YgoEngine._CARD_DB_CACHE.get(code)
        if card_info:
            data_ptr.contents.attack = card_info.get("atk", 0)
            data_ptr.contents.defense = card_info.get("def", 0)
            data_ptr.contents.level = card_info.get("level", 0)
            
            # Map type, attribute, race
            t = card_info.get("type", "Normal Monster")
            data_ptr.contents.type = YGO_TYPE_MAPPING.get(t, YGO_TYPE_MAPPING["Normal Monster"])
            
            a = card_info.get("attribute", "LIGHT")
            data_ptr.contents.attribute = YGO_ATTRIBUTE_MAPPING.get(a, YGO_ATTRIBUTE_MAPPING["LIGHT"])
            
            r = card_info.get("race", "Dragon")
            data_ptr.contents.race = YGO_RACE_MAPPING.get(r, YGO_RACE_MAPPING["Dragon"])
        else:
            # Fallback
            data_ptr.contents.attack = 0
            data_ptr.contents.defense = 0
            data_ptr.contents.level = 0
            data_ptr.contents.type = YGO_TYPE_MAPPING["Normal Monster"]
            data_ptr.contents.attribute = YGO_ATTRIBUTE_MAPPING["LIGHT"]
            data_ptr.contents.race = YGO_RACE_MAPPING["Dragon"]
            
    except Exception as e:
        import logging
        logging.error(f"Erreur dans card_reader: {e}")

_engine_lib = None

def my_script_reader(payload, duel_ptr, name):
    try:
        script_name = name.decode('utf-8') if isinstance(name, bytes) else name
        
        import os
        from pathlib import Path
        filename = os.path.basename(script_name)
        
        scripts_dir = Path(__file__).parent.parent.parent / "data" / "scripts"
        path = scripts_dir / filename
        
        import logging
        logging.info(f"Checking script at: {path}, exists: {path.exists()}")
        
        if path.exists():
            with open(path, "rb") as f:
                content = f.read()
                
            if _engine_lib:
                # Appeler OCG_LoadScript (duel_ptr, buffer, length, name)
                _engine_lib.OCG_LoadScript(duel_ptr, content, len(content), name)
                return 1 # Succès
                
        # Le script n'a pas été trouvé
        return 0
    except Exception as e:
        import logging
        logging.error(f"Erreur dans script_reader: {e}")
        return 0

# Keep references to avoid garbage collection
_global_card_reader = CARD_READER_CB(my_card_reader)
_global_script_reader = SCRIPT_READER_CB(my_script_reader)

class YgoEngine:
    _CARD_DB_CACHE = None

    @classmethod
    def _load_card_db_cache(cls):
        if cls._CARD_DB_CACHE is not None:
            return
            
        import logging
        logging.info("Chargement du cache des cartes en mémoire...")
        try:
            import os
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from db.session import SessionLocal
            from db.models import Card
            
            with SessionLocal() as db:
                cards = db.query(Card).all()
                cls._CARD_DB_CACHE = {}
                for c in cards:
                    cls._CARD_DB_CACHE[c.id] = {
                        "type": c.type,
                        "atk": c.atk or 0,
                        "def": c.def_ or 0,
                        "level": c.level or 0,
                        "attribute": c.attribute,
                        "race": c.race
                    }
            logging.info(f"Cache de {len(cls._CARD_DB_CACHE)} cartes chargé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du cache des cartes: {e}")
            cls._CARD_DB_CACHE = {}

    def __init__(self):
        self._load_card_db_cache()
        self.lib = None
        self.duel_ptr = None  # Pointeur persistant vers le duel actif
        self._duel_valid = False
        self._load_library()

    def _load_library(self):
        """Charge ocgcore via ctypes."""
        import sys
        core_dir = Path(__file__).parent
        
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
                
                global _engine_lib
                _engine_lib = self.lib
                
                # Bind C functions for ocgcore-KCG
                self.lib.OCG_CreateDuel.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(OCG_DuelOptions)]
                self.lib.OCG_CreateDuel.restype = ctypes.c_int
                
                self.lib.OCG_LoadScript.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p]
                self.lib.OCG_LoadScript.restype = ctypes.c_int
                
                self.lib.OCG_StartDuel.argtypes = [ctypes.c_void_p]
                self.lib.OCG_StartDuel.restype = None
                
                self.lib.OCG_DestroyDuel.argtypes = [ctypes.c_void_p]
                self.lib.OCG_DestroyDuel.restype = None
                
                self.lib.OCG_DuelProcess.argtypes = [ctypes.c_void_p]
                self.lib.OCG_DuelProcess.restype = ctypes.c_int
                
                self.lib.OCG_DuelNewCard.argtypes = [ctypes.c_void_p, ctypes.POINTER(OCG_NewCardInfo)]
                self.lib.OCG_DuelNewCard.restype = None
                
                self.lib.OCG_DuelQueryCount.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint32]
                self.lib.OCG_DuelQueryCount.restype = ctypes.c_uint32
                
                self.lib.OCG_DuelGetMessage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                self.lib.OCG_DuelGetMessage.restype = ctypes.c_void_p
                
                self.lib.OCG_DuelSetResponse.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self.lib.OCG_DuelSetResponse.restype = None
                
                self.lib.OCG_DuelQueryLocation.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(OCG_QueryInfo)]
                self.lib.OCG_DuelQueryLocation.restype = ctypes.c_void_p
                
                self.lib.OCG_DuelQueryField.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                self.lib.OCG_DuelQueryField.restype = ctypes.c_void_p
                
            except Exception as e:
                raise EngineCrashError(f"Erreur lors du chargement de ocgcore: {e}")
        else:
            raise EngineCrashError(f"Le moteur compilé est introuvable à l'emplacement: {dll_path}")

    # --- Duel Lifecycle ---

    def create_duel(self) -> bool:
        """
        Crée un nouveau duel C++. Retourne True si la création a réussi.
        Détruit tout duel précédent avant d'en créer un nouveau.
        """
        self.destroy_duel()
        
        duel_ptr = ctypes.c_void_p()
        options = OCG_DuelOptions()
        options.seed[0] = 12345
        options.team1.startingLP = 8000
        options.team1.startingDrawCount = 5
        options.team1.drawCountPerTurn = 1
        options.team2.startingLP = 8000
        options.team2.startingDrawCount = 5
        options.team2.drawCountPerTurn = 1
        
        # Attacher les callbacks
        self._cb_card_reader = CARD_READER_CB(my_card_reader)
        self._cb_script_reader = SCRIPT_READER_CB(my_script_reader)
        
        # Log handler pour voir les erreurs C++
        LOG_HANDLER_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int)
        def my_log_handler(payload, message, type):
            try:
                msg = message.decode('utf-8', errors='ignore')
                import logging
                logging.error(f"[OCG_LOG] {msg}")
            except:
                pass
        self._cb_log_handler = LOG_HANDLER_CB(my_log_handler)
        
        options.cardReader = ctypes.cast(self._cb_card_reader, ctypes.c_void_p)
        options.scriptReader = ctypes.cast(self._cb_script_reader, ctypes.c_void_p)
        options.logHandler = ctypes.cast(self._cb_log_handler, ctypes.c_void_p)
        
        res = self.lib.OCG_CreateDuel(ctypes.byref(duel_ptr), ctypes.byref(options))
        
        # OCG_DUEL_CREATION_SUCCESS = 0
        if res == 0 and duel_ptr.value is not None and duel_ptr.value != 0:
            self.duel_ptr = duel_ptr
            self._duel_valid = True
            import logging
            logging.info(f"Duel C++ créé avec succès (ptr={duel_ptr.value:#x})")
            return True
        else:
            import logging
            logging.warning(f"OCG_CreateDuel a échoué avec le code: {res}")
            self.duel_ptr = None
            self._duel_valid = False
            return False

    def destroy_duel(self):
        """Détruit le duel courant s'il est valide."""
        if self._duel_valid and self.duel_ptr is not None and self.duel_ptr.value:
            try:
                self.lib.OCG_DestroyDuel(self.duel_ptr)
            except Exception as e:
                import logging
                logging.error(f"Erreur lors de la destruction du duel C++ : {e}")
        self.duel_ptr = None
        self._duel_valid = False

    def add_card(self, code: int, owner: int, location: int, sequence: int, position: int):
        """Ajoute une carte au duel courant."""
        if not self._duel_valid:
            return
            
        info = OCG_NewCardInfo()
        info.team = owner
        info.duelist = 0 # MUST BE 0 (sinon ça utilise extra_lists_main qui a un bug de destructor)
        info.code = code
        info.con = owner
        info.loc = location
        info.seq = sequence
        info.pos = position
        
        self.lib.OCG_DuelNewCard(self.duel_ptr, ctypes.byref(info))
        
    def start_duel(self):
        """Démarre le duel après avoir ajouté toutes les cartes."""
        if not self._duel_valid:
            return
            
        import os
        from pathlib import Path
        scripts_dir = Path(__file__).parent.parent.parent / "data" / "scripts"
        
        # Charger les scripts système obligatoires avant de démarrer
        for script_name in ["constant.lua", "utility.lua"]:
            path = scripts_dir / script_name
            if path.exists():
                with open(path, "rb") as f:
                    content = f.read()
                    buf = ctypes.create_string_buffer(content, len(content))
                    self.lib.OCG_LoadScript(self.duel_ptr, buf, len(content), script_name.encode('utf-8'))
            else:
                import logging
                logging.warning(f"Script système manquant : {script_name}")
        
        # OCG_StartDuel receives OCG_Duel
        self.lib.OCG_StartDuel(self.duel_ptr)

    # --- Actions ---

    def get_legal_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrait les actions légales depuis le duel actif.
        Si le duel n'est pas valide, retourne un fallback minimal.
        """
        if not self._duel_valid:
            return [{"action_type": 0, "source": "no_valid_duel"}]
            
        try:
            status = 2 # OCG_DUEL_STATUS_CONTINUE
            # Boucler tant que le moteur indique qu'il doit continuer (2)
            while status == 2:
                status = self.lib.OCG_DuelProcess(self.duel_ptr)
            
            # 1 = AWAITING, 0 = END
            actions = self._extract_actions(self.duel_ptr)
            return actions
        except Exception as e:
            import logging
            logging.error(f"Erreur dans get_legal_actions: {e}")
            return [{"action_type": 0, "source": "engine_error"}]

    def _apply_state(self, state: Dict[str, Any]):
        """Applique un état au duel actif (ajout de cartes, etc.)."""
        # TODO: Appeler OCG_DuelNewCard, set_player_info, etc.
        pass

    def translate_and_set_response(self, action_idx: int, current_state_actions: List[Dict[str, Any]]):
        """Traduit un index RL en un buffer binaire pour l'engine."""
        if not self._duel_valid:
            return
            
        import struct
        
        # Parcourir les actions légales stockées dans l'état pour trouver le mapping de l'action choisie
        response_value = 0
        response_len = 0
        
        # Prototype simple: on cherche l'action dans le message IDLECMD, BATTLECMD ou CARD
        response_bytes = None
        for msg_block in current_state_actions:
            if msg_block.get("msg") in ["MSG_SELECT_IDLECMD", "MSG_SELECT_BATTLECMD"]:
                for choice in msg_block.get("choices", []):
                    if choice.get("action_idx") == action_idx:
                        t = choice.get("engine_type", 0)
                        s = choice.get("engine_index", 0)
                        
                        # (index_du_choix << 16) | type_action
                        response_value = (s << 16) | (t & 0xFFFF)
                        response_bytes = struct.pack("<I", response_value)
                        break
                if response_bytes: break
            
            elif msg_block.get("msg") == "MSG_SELECT_CARD":
                for choice in msg_block.get("choices", []):
                    if choice.get("action_idx") == action_idx:
                        s = choice.get("engine_index", 0)
                        min_c = msg_block.get("min", 1)
                        
                        # Remplissage des bytes
                        bytes_list = [s]
                        
                        # Auto-complétion pour min > 1 (MVP)
                        auto_idx = 0
                        while len(bytes_list) < min_c:
                            if auto_idx != s:
                                bytes_list.append(auto_idx)
                            auto_idx += 1
                            
                        response_bytes = struct.pack(f"<{len(bytes_list)}B", *bytes_list)
                        break
                if response_bytes: break
                
        # Si c'est pas géré, fallback: on envoie action_idx tel quel
        if response_bytes is None:
            response_bytes = struct.pack("<I", action_idx)
            
        buffer = ctypes.create_string_buffer(response_bytes, 64)
        self.lib.OCG_DuelSetResponse(self.duel_ptr, buffer)

    def _extract_actions(self, duel_ptr) -> List[Dict[str, Any]]:
        """Extrait les actions depuis les messages du moteur via BinaryReader."""
        if not self._duel_valid:
            return [{"action_type": 1, "source": "no_valid_duel"}]
            
        length = ctypes.c_uint32()
        msg_ptr = self.lib.OCG_DuelGetMessage(duel_ptr, ctypes.byref(length))
        
        if not msg_ptr or length.value == 0:
            return []
            
        # Lire les données en bytes
        buffer = ctypes.string_at(msg_ptr, length.value)
        
        from core.ygoenv.utils import BinaryReader
        from core.ygoenv.constants import MSG_WIN, MSG_SELECT_IDLECMD, MSG_SELECT_BATTLECMD, MSG_SELECT_CARD
        reader = BinaryReader(buffer)
        
        actions = []
        
        try:
            while not reader.eof():
                msg_type = reader.read_uint8()
                
                if msg_type == MSG_WIN:
                    player = reader.read_uint8()
                    reason = reader.read_uint8()
                    actions.append({"action_type": 0, "source": "engine_process", "msg": "WIN", "player": player, "reason": reason})
                    continue
                
                elif msg_type == MSG_SELECT_BATTLECMD:
                    player = reader.read_uint8()
                    battle_actions = []
                    
                    OFFSET_ATTACK = 153
                    OFFSET_BATTLE_ACTIVATE = 173
                    
                    # 1. Activatable (select_chains, type: 0)
                    count = reader.read_uint32()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        desc = reader.read_uint64()
                        client_mode = reader.read_uint8()
                        idx = OFFSET_BATTLE_ACTIVATE + i
                        battle_actions.append({"type": "BATTLE_ACTIVATE", "engine_type": 0, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq})
                        
                    # 2. Attackable (type: 1)
                    count = reader.read_uint32()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        dir_atk = reader.read_uint8()
                        idx = OFFSET_ATTACK + i
                        battle_actions.append({"type": "ATTACK", "engine_type": 1, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq, "direct": dir_atk})
                        
                    # 3. M2 & EP
                    m2 = reader.read_uint8()
                    ep = reader.read_uint8()
                    
                    if m2: battle_actions.append({"type": "MAIN_PHASE_2", "engine_type": 2, "engine_index": 0, "action_idx": 190})
                    if ep: battle_actions.append({"type": "END_PHASE", "engine_type": 3, "engine_index": 0, "action_idx": 151}) # Note: 151 est To EP dans notre plan global
                    
                    actions.append({
                        "action_type": 1, 
                        "source": "engine_process", 
                        "msg": "MSG_SELECT_BATTLECMD", 
                        "player": player, 
                        "choices": battle_actions
                    })
                    break
                    
                elif msg_type == MSG_SELECT_CARD:
                    player = reader.read_uint8()
                    cancelable = reader.read_uint8()
                    min_c = reader.read_uint32()
                    max_c = reader.read_uint32()
                    count = reader.read_uint32()
                    
                    card_actions = []
                    OFFSET_SELECT_CARD = 191
                    
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        pos = reader.read_uint8()
                        
                        idx = OFFSET_SELECT_CARD + i
                        if idx <= 199:
                            card_actions.append({
                                "type": "SELECT_CARD", 
                                "engine_type": 0, 
                                "engine_index": i, 
                                "action_idx": idx, 
                                "code": code, 
                                "loc": loc, 
                                "seq": seq, 
                                "pos": pos
                            })
                            
                    actions.append({
                        "action_type": 1, 
                        "source": "engine_process", 
                        "msg": "MSG_SELECT_CARD", 
                        "player": player, 
                        "min": min_c,
                        "max": max_c,
                        "cancelable": cancelable,
                        "choices": card_actions
                    })
                    break

                elif msg_type == MSG_SELECT_IDLECMD:
                    player = reader.read_uint8()
                    idle_actions = []
                    
                    # Offsets pour notre Action Space statique
                    OFFSET_SUMMON = 0
                    OFFSET_SPSUMMON = 20
                    OFFSET_REPOS = 40
                    OFFSET_MSET = 60
                    OFFSET_SSET = 80
                    OFFSET_ACTIVATE = 100
                    
                    # 1. Summonable (type: 0)
                    count = reader.read_uint8()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        idx = OFFSET_SUMMON + i
                        idle_actions.append({"type": "SUMMON", "engine_type": 0, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq})
                        
                    # 2. SPSummonable (type: 1)
                    count = reader.read_uint8()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        idx = OFFSET_SPSUMMON + i
                        idle_actions.append({"type": "SPSUMMON", "engine_type": 1, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq})
                        
                    # 3. Reposition (type: 2)
                    count = reader.read_uint8()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        idx = OFFSET_REPOS + i
                        idle_actions.append({"type": "REPOS", "engine_type": 2, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq})
                        
                    # 4. MSet (type: 3)
                    count = reader.read_uint8()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        idx = OFFSET_MSET + i
                        idle_actions.append({"type": "MSET", "engine_type": 3, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq})
                        
                    # 5. SSet (type: 4)
                    count = reader.read_uint8()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        idx = OFFSET_SSET + i
                        idle_actions.append({"type": "SSET", "engine_type": 4, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq})
                        
                    # 6. Activate (type: 5)
                    count = reader.read_uint8()
                    for i in range(count):
                        code = reader.read_uint32()
                        con = reader.read_uint8()
                        loc = reader.read_uint8()
                        seq = reader.read_uint8()
                        idx = OFFSET_ACTIVATE + i
                        idle_actions.append({"type": "ACTIVATE", "engine_type": 5, "engine_index": i, "action_idx": idx, "code": code, "loc": loc, "seq": seq})
                        
                    bp = reader.read_uint8()
                    ep = reader.read_uint8()
                    shuffle = reader.read_uint8()
                    
                    if bp: idle_actions.append({"type": "BATTLE_PHASE", "engine_type": 6, "engine_index": 0, "action_idx": 150})
                    if ep: idle_actions.append({"type": "END_PHASE", "engine_type": 7, "engine_index": 0, "action_idx": 151})
                    if shuffle: idle_actions.append({"type": "SHUFFLE", "engine_type": 8, "engine_index": 0, "action_idx": 152})
                    
                    actions.append({
                        "action_type": 1, 
                        "source": "engine_process", 
                        "msg": "MSG_SELECT_IDLECMD", 
                        "player": player, 
                        "choices": idle_actions
                    })
                    break
                
                # Pour le prototype, on ignore les autres messages complexes (ça demande d'implanter tout le protocole)
                actions.append({"action_type": 1, "source": "engine_process", "msg": msg_type})
                continue
                
        except EOFError:
            pass
            
        return actions

    def save_state(self):
        """Retourne un buffer binaire de l'état du duel C++."""
        # TODO: Appeler une fonction C++ pour extraire la mémoire du duel
        return None

    def get_global_info(self) -> Dict[str, Any]:
        """
        Récupère les informations globales du duel (LP, etc.) via OCG_DuelQueryField.
        """
        if not self._duel_valid:
            return {"lp": [8000, 8000]}
            
        length = ctypes.c_uint32()
        msg_ptr = self.lib.OCG_DuelQueryField(self.duel_ptr, ctypes.byref(length))
        
        info = {"lp": [8000, 8000]}
        if msg_ptr and length.value > 0:
            buffer = ctypes.string_at(msg_ptr, length.value)
            from core.ygoenv.utils import BinaryReader
            reader = BinaryReader(buffer)
            
            try:
                # Format OCG_DuelQueryField :
                # uint32 : duel_options
                reader.read_uint32()
                
                for p in range(2):
                    info["lp"][p] = reader.read_uint32()
                    
                    # Ignorer mzone (7 slots)
                    for _ in range(7):
                        if reader.read_uint8() == 1:
                            reader.read_uint8()
                            reader.read_uint32()
                            
                    # Ignorer szone (8 slots)
                    for _ in range(8):
                        if reader.read_uint8() == 1:
                            reader.read_uint8()
                            reader.read_uint32()
                            
                    # Ignorer les tailles de deck/hand/grave/etc (6 * uint32)
                    for _ in range(6):
                        reader.read_uint32()
            except EOFError:
                pass
                
        return info

    def query_field_state(self, player_id: int) -> List[Dict[str, Any]]:
        """
        Récupère l'état complet du plateau pour chaque emplacement.
        Renvoie une liste de dictionnaires contenant code, position, type, level, atk, def.
        """
        if not self._duel_valid:
            return []
            
        cards_info = []
        from core.ygoenv.constants import (
            QUERY_CODE, QUERY_POSITION, QUERY_TYPE, 
            QUERY_LEVEL, QUERY_ATTACK, QUERY_DEFENSE
        )
        query_flags = QUERY_CODE | QUERY_POSITION | QUERY_TYPE | QUERY_LEVEL | QUERY_ATTACK | QUERY_DEFENSE
        
        # Locations standards YGOPro: 
        # 0x01: Deck, 0x02: Hand, 0x04: MZone, 0x08: SZone, 0x10: Grave, 0x20: Removed, 0x40: Extra
        locations = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40] 
        
        for p in [player_id, 1 - player_id]:
            for loc in locations:
                info = OCG_QueryInfo()
                info.flags = query_flags
                info.con = p
                info.loc = loc
                info.seq = 0
                info.overlay_seq = 0
                
                length = ctypes.c_uint32()
                msg_ptr = self.lib.OCG_DuelQueryLocation(self.duel_ptr, ctypes.byref(length), ctypes.byref(info))
                
                if msg_ptr and length.value > 0:
                    buffer = ctypes.string_at(msg_ptr, length.value)
                    from core.ygoenv.utils import BinaryReader
                    reader = BinaryReader(buffer)
                    
                    try:
                        total_len = reader.read_uint32()
                        while not reader.eof():
                            block_len = reader.read_uint16()
                            if block_len == 0:
                                continue # Empty card slot
                                
                            card_data = {
                                "player": p,
                                "location": loc,
                                "code": 0,
                                "position": 0,
                                "type": 0,
                                "level": 0,
                                "attack": 0,
                                "defense": 0
                            }
                            
                            from core.ygoenv.constants import QUERY_END
                            # Lecture de toutes les propriétés de la carte jusqu'à QUERY_END
                            while True:
                                if block_len > 0:
                                    flag = reader.read_uint32()
                                    if flag == QUERY_END:
                                        break
                                        
                                    if flag == QUERY_CODE:
                                        card_data["code"] = reader.read_uint32()
                                    elif flag == QUERY_POSITION:
                                        card_data["position"] = reader.read_uint32()
                                    elif flag == QUERY_TYPE:
                                        card_data["type"] = reader.read_uint32()
                                    elif flag == QUERY_LEVEL:
                                        card_data["level"] = reader.read_uint32()
                                    elif flag == QUERY_ATTACK:
                                        card_data["attack"] = reader.read_uint32()
                                    elif flag == QUERY_DEFENSE:
                                        card_data["defense"] = reader.read_uint32()
                                    else:
                                        # Skip unknown flag payload
                                        reader.read_bytes(block_len - 4)
                                        
                                block_len = reader.read_uint16()
                                    
                            if card_data["code"] > 0:
                                cards_info.append(card_data)
                                
                    except EOFError:
                        pass
        return cards_info

    def restore_state(self, state_buffer):
        """Restaure l'état C++ depuis un buffer binaire."""
        # TODO: Appeler une fonction C++ pour restaurer la mémoire du duel
        pass

