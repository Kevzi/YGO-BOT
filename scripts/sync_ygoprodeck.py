import asyncio
import httpx
import logging
import sys
import time
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import func

# Local imports
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.models import Card
from db.session import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

async def fetch_cards():
    logger.info(f"Fetching cards from {API_URL}...")
    try:
        async with httpx.AsyncClient() as client:
            await asyncio.sleep(0.05) # Basic rate limit hygiene
            response = await client.get(API_URL, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except httpx.HTTPError as e:
        logger.error(f"HTTP Error while fetching cards: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def sync_cards_to_db(cards_data):
    logger.info(f"Syncing {len(cards_data)} cards to database...")
    db = SessionLocal()
    try:
        values_to_insert = []
        for c in cards_data:
            values_to_insert.append({
                "id": c["id"],
                "name": c["name"],
                "type": c["type"],
                "desc": c["desc"],
                "race": c.get("race"),
                "archetype": c.get("archetype"),
                "atk": c.get("atk"),
                "def_": c.get("def"),
                "level": c.get("level"),
                "attribute": c.get("attribute")
            })
            
        chunk_size = 1000
        for i in range(0, len(values_to_insert), chunk_size):
            chunk = values_to_insert[i:i+chunk_size]
            stmt = insert(Card).values(chunk)
            
            update_dict = {
                "name": stmt.excluded.name,
                "type": stmt.excluded.type,
                "desc": stmt.excluded.desc,
                "race": stmt.excluded.race,
                "archetype": stmt.excluded.archetype,
                "atk": stmt.excluded.atk,
                "def_": stmt.excluded.def_,
                "level": stmt.excluded.level,
                "attribute": stmt.excluded.attribute,
                "updated_at": func.now()
            }
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_=update_dict
            )
            
            db.execute(stmt)
            
        db.commit()
        logger.info("Sync complete.")
    finally:
        db.close()

async def main():
    start_time = time.time()
    try:
        cards_data = await fetch_cards()
        if cards_data:
            sync_cards_to_db(cards_data)
    except Exception:
        sys.exit(1)
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
