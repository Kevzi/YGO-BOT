import ray
import logging
from db.models import DuelStats

@ray.remote
class DatabaseActor:
    def __init__(self):
        # Initialisation (pourrait précharger la session DB)
        pass

    def record_duel_stats(self, stats_dict: dict) -> bool:
        """
        Enregistre les stats d'un duel de manière séquentielle pour éviter
        les erreurs 'database is locked' de SQLite.
        """
        try:
            from db.session import SessionLocal
            with SessionLocal() as db:
                duel_stat = DuelStats(**stats_dict)
                db.add(duel_stat)
                db.commit()
                return True
        except Exception as e:
            logging.error(f"Erreur d'insertion DB: {e}")
            return False
