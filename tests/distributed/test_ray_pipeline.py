import pytest
import ray
from distributed.db_actor import DatabaseActor
from db.session import get_db
from sqlalchemy import text

@pytest.fixture(scope="module")
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()

def test_database_actor_insertion(ray_init):
    # Act
    db_actor = DatabaseActor.remote()
    
    # Nous simulons un enregistrement de stats
    stats = {
        "player_1_deck": "Dummy",
        "player_2_deck": "Dummy",
        "winner": 1
    }
    
    # Appel asynchrone (retourne un ObjectRef)
    future = db_actor.record_duel_stats.remote(stats)
    
    # Attente de la fin de l'exécution
    ray.get(future)
    
    # Assert
    # Vérifier que l'insertion a bien eu lieu
    # Pour ce test on vérifie que l'acteur retourne True et que la DB est modifiée
    assert ray.get(future) is True
    
    from db.session import SessionLocal
    from db.models import DuelStats
    from sqlalchemy import select
    with SessionLocal() as db:
        result = db.execute(select(DuelStats).order_by(DuelStats.id.desc()).limit(1)).scalar_one_or_none()
        assert result is not None
        assert result.player_1_deck == "Dummy"
        assert result.winner == 1

def test_rollout_worker(ray_init):
    from distributed.worker import RolloutWorker
    import jax.numpy as jnp
    
    worker = RolloutWorker.remote()
    
    # Fake params (empty dict for now since PPOAgent init_params does that, or actual params)
    # On va simuler un params bidon, le PPOAgent est mocké ou utilise des vrais params
    fake_params = {} 
    
    future = worker.collect_rollout.remote(fake_params, num_steps=5)
    trajectory = ray.get(future)
    
    assert "obs" in trajectory
    assert "actions" in trajectory
    assert "rewards" in trajectory
    assert len(trajectory["obs"]) > 0

def test_distributed_training_loop(ray_init):
    from distributed.train import LeagueTrainer
    import jax
    import jax.numpy as jnp
    
    # We test a 1-step training loop with 1 worker
    from core.ygoenv.env import YgoEnv
    env = YgoEnv()
    obs_dim = env.observation_space.shape[0]
    trainer = LeagueTrainer(num_workers=1, obs_dim=obs_dim, act_dim=200, hidden_dim=64)
    
    # Run 1 iteration
    metrics = trainer.train_step(num_steps=5)
    
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert metrics["episodes_collected"] == 1
