from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.security import setup_cors
from api.duel_routes import router as duel_router
from api.deck_routes import router as deck_router
from ai.embeddings import embed_loader
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embeddings on startup (Fail Fast)
    if not embed_loader.is_loaded():
        import os
        filepath = os.environ.get("YGO_EMBED_PATH", "data/embed.pkl")
        embed_loader.filepath = filepath
        embed_loader.load()
        logger.info("Embeddings loaded successfully.")
        
    yield
    # Cleanup on shutdown if necessary

app = FastAPI(title="YGO Bot API", lifespan=lifespan)

setup_cors(app)

app.include_router(duel_router, prefix="/api")
app.include_router(deck_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
