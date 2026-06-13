import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.ygoenv.wrapper import YgoEngine
from core.ygoenv.constants import YGO_TYPE_MAPPING, YGO_ATTRIBUTE_MAPPING, YGO_RACE_MAPPING

logging.basicConfig(level=logging.INFO)

def test_card_reader():
    engine = YgoEngine()
    
    # Check if Blue-Eyes White Dragon is in DB cache
    blue_eyes_code = 89631139
    card_info = engine._CARD_DB_CACHE.get(blue_eyes_code)
    
    if card_info:
        logging.info(f"Found Blue-Eyes in cache: {card_info}")
        assert card_info["atk"] == 3000
        assert card_info["def"] == 2500
        assert card_info["level"] == 8
        assert "Dragon" in card_info["race"]
        
        c_type = YGO_TYPE_MAPPING.get(card_info["type"])
        assert c_type == YGO_TYPE_MAPPING["Normal Monster"]
        
        logging.info("Card mapping logic looks correct!")
    else:
        logging.error("Blue-Eyes not found in cache. Did the DB sync run?")
        sys.exit(1)

if __name__ == "__main__":
    test_card_reader()
