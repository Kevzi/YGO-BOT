import os
import httpx
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:3000/api/v1/decks/import"
DECKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'decks'))

async def import_deck(file_path):
    filename = os.path.basename(file_path)
    logger.info(f"Importing {filename}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        deck_content = f.read()
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                API_URL,
                json={"deckData": deck_content},
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            main_count = len(data.get("main", []))
            extra_count = len(data.get("extra", []))
            side_count = len(data.get("side", []))
            
            logger.info(f"Success [{filename}]: {main_count} Main, {extra_count} Extra, {side_count} Side")
            return True
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP Error for {filename}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
    except Exception as e:
        logger.error(f"Error importing {filename}: {e}")
        
    return False

async def main():
    if not os.path.exists(DECKS_DIR):
        logger.error(f"Decks directory not found: {DECKS_DIR}")
        return
        
    ydk_files = [os.path.join(DECKS_DIR, f) for f in os.listdir(DECKS_DIR) if f.endswith(".ydk")]
    
    if not ydk_files:
        logger.warning(f"No .ydk files found in {DECKS_DIR}")
        return
        
    logger.info(f"Found {len(ydk_files)} decks. Starting import test to API...")
    
    tasks = [import_deck(f) for f in ydk_files]
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if r)
    logger.info(f"Import process finished. {success_count}/{len(ydk_files)} successful.")

if __name__ == "__main__":
    asyncio.run(main())
