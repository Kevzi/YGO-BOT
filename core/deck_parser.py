import httpx
from typing import Dict, List, Any

import os

class DeckParserError(Exception):
    pass

def parse_deck_sync(deck_data: str) -> Dict[str, List[int]]:
    """
    Parse localement un deck au format .ydk (version synchrone).
    """
    deck = {"main": [], "extra": [], "side": []}
    current_section = None
    
    for line in deck_data.splitlines():
        line = line.strip()
        if not line or line.startswith("!") and not line.startswith("!side"):
            continue
            
        if line.startswith("#main"):
            current_section = "main"
        elif line.startswith("#extra"):
            current_section = "extra"
        elif line.startswith("!side"):
            current_section = "side"
        elif line.isdigit():
            if current_section:
                deck[current_section].append(int(line))
            else:
                # If no section is specified, assume main deck
                deck["main"].append(int(line))
                
    if not deck["main"]:
        raise DeckParserError("Le deck fourni ne contient aucune carte valide dans le Main Deck.")
        
    return deck

async def parse_deck(deck_data: str, parser_url: str = None) -> Dict[str, List[int]]:
    """
    Parse localement un deck au format .ydk.
    Retourne un dictionnaire avec 'main', 'extra', et 'side' contenants les ID des cartes.
    """
    return parse_deck_sync(deck_data)
