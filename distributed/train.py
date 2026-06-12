import time
import ray
import jax
import jax.numpy as jnp
import numpy as np

from distributed.worker import RolloutWorker
from distributed.db_actor import DatabaseActor
from ai.ppo import PPOAgent

class LeagueTrainer:
    def __init__(self, num_workers: int = 2, obs_dim: int = 100, act_dim: int = 200, hidden_dim: int = 64):
        self.num_workers = num_workers
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        # Initialisation du Learner global
        self.agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
        rng = jax.random.PRNGKey(int(time.time() * 1000) % 2147483647)
        self.params = self.agent.init_params(rng)
        
        # Acteurs Ray
        self.workers = [RolloutWorker.remote(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim) for _ in range(num_workers)]
        self.db_actor = DatabaseActor.remote()

    def train_step(self, num_steps: int = 100):
        """Effectue une itération de collecte et d'apprentissage."""
        if self.num_workers <= 0 or num_steps <= 0:
            return {}
        # 1. Demande de collecte (Asynchrone)
        futures = [w.collect_rollout.remote(self.params, num_steps=num_steps) for w in self.workers]
        
        # 2. Attente des résultats
        trajectories = ray.get(futures)
        
        # 3. Stack des batchs pour conserver la dimension temporelle (batch=num_workers, time=num_steps, ...)
        all_obs = np.stack([t["obs"] for t in trajectories], axis=0)
        all_actions = np.stack([t["actions"] for t in trajectories], axis=0)
        all_old_log_probs = np.stack([t["old_log_probs"] for t in trajectories], axis=0)
        all_advantages = np.stack([t["advantages"] for t in trajectories], axis=0)
        all_returns = np.stack([t["returns"] for t in trajectories], axis=0)
        all_dones = np.stack([t["dones"] for t in trajectories], axis=0)
        
        # Conversion JAX
        j_obs = jnp.array(all_obs)
        j_actions = jnp.array(all_actions)
        j_old_log_probs = jnp.array(all_old_log_probs)
        j_advantages = jnp.array(all_advantages)
        j_returns = jnp.array(all_returns)
        j_dones = jnp.array(all_dones)
        
        # 4. Mise à jour PPO avec BPTT
        self.params, metrics = self.agent.update_params(
            self.params, j_obs, j_actions, j_old_log_probs, j_advantages, j_returns, j_dones
        )
        
        # 5. Enregistrement asynchrone DB (Simulation de fin de duel)
        self.db_actor.record_duel_stats.remote({
            "player_1_deck": "PPO_League",
            "player_2_deck": "PPO_League_Selfplay",
            "winner": 1
        })
        
        metrics["episodes_collected"] = len(trajectories)
        return metrics

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    trainer = LeagueTrainer(num_workers=4)
    print("Démarrage de l'entraînement distribué...")
    for i in range(10):
        t0 = time.time()
        metrics = trainer.train_step(num_steps=100)
        dt = time.time() - t0
        print(f"Iteration {i+1} - Loss: {metrics['total_loss']:.4f} - Policy Loss: {metrics['policy_loss']:.4f} - Time: {dt:.2f}s")
    print("Entraînement terminé.")
