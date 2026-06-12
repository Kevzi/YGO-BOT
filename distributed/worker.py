import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import ray
import time
import numpy as np
import jax
import jax.numpy as jnp
from core.ygoenv.env import YgoEnv
from ai.ppo import PPOAgent
from ai.agent import MCTSAgent

@ray.remote
class RolloutWorker:
    def __init__(self, obs_dim: int = 100, act_dim: int = 200, hidden_dim: int = 64, use_mcts: bool = True, num_simulations: int = 8):
        # Initialise l'environnement à l'intérieur de l'acteur (CPU/Thread local)
        self.env = YgoEnv()
        self.ppo_agent = PPOAgent(obs_dim=self.env.observation_space.shape[0], act_dim=act_dim, hidden_dim=hidden_dim)
        self.use_mcts = use_mcts
        if self.use_mcts:
            self.agent = MCTSAgent(self.ppo_agent, num_simulations=num_simulations)
        else:
            self.agent = self.ppo_agent
            
        self.current_obs, _ = self.env.reset()
        self.current_hidden_state = self.ppo_agent.init_hidden_state(1)

    def collect_rollout(self, params: dict, num_steps: int = 100) -> dict:
        """
        Joue 'num_steps' étapes dans l'environnement en utilisant les poids (params) du Learner.
        Retourne la trajectoire (batch).
        """
        if num_steps <= 0:
            return {"obs": np.array([]), "actions": np.array([]), "rewards": np.array([]), "old_log_probs": np.array([]), "values": np.array([]), "returns": np.array([]), "advantages": np.array([])}

        obs_list = []
        actions_list = []
        rewards_list = []
        log_probs_list = []
        values_list = []
        term_list = []

        obs = self.current_obs
        hidden_state = self.current_hidden_state

        # Si params est vide (pour le test mock par exemple), on init dummy params
        if not params:
            rng = jax.random.PRNGKey(int(time.time() * 1000) % 2147483647)
            params = self.ppo_agent.init_params(rng)

        for _ in range(num_steps):
            obs_jax = jnp.asarray(obs, dtype=jnp.float32)
            
            if self.use_mcts:
                action, policy_improved, value, next_hidden_state = self.agent.select_action(
                    env=self.env,
                    params=params,
                    hidden_state=hidden_state,
                    obs=obs_jax
                )
                probs_np = policy_improved
            else:
                # Forward pass (Policy et Value)
                probs, value, next_hidden_state = self.agent.forward(params, hidden_state, obs_jax)
                probs_np = np.array(probs)
                if probs_np.ndim == 2 and probs_np.shape[0] == 1:
                    probs_np = probs_np[0]
                
                if np.isnan(probs_np).any():
                    raise ValueError("Model output contains NaNs.")
                    
                probs_sum = np.sum(probs_np)
                probs_np = probs_np / max(probs_sum, 1e-8)
                action = np.random.choice(len(probs_np), p=probs_np)
            
            # Enregistrement pour l'apprentissage
            obs_list.append(obs)
            actions_list.append(action)
            
            # log_prob approximé (pour PPO old_log_probs)
            log_prob = np.log(probs_np[action] + 1e-8)
            log_probs_list.append(log_prob)
            values_list.append(float(np.squeeze(value)))

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            rewards_list.append(reward)
            term_list.append(terminated or truncated)

            if terminated or truncated:
                obs, _ = self.env.reset()
                # Explicitly reset hidden state when episode ends to prevent leaking across games
                hidden_state = self.ppo_agent.init_hidden_state(1)
            else:
                obs = next_obs
                hidden_state = next_hidden_state

        self.current_obs = obs
        self.current_hidden_state = hidden_state

        # Calcul basique de 'returns' et 'advantages' pour le mock (on laisse le Learner faire le calcul précis avec GAE normalement, 
        # mais ici on simplifie pour retourner un dict complet)
        rewards_np = np.array(rewards_list, dtype=np.float32)
        values_np = np.array(values_list, dtype=np.float32)
        term_np = np.array(term_list, dtype=np.float32)
        
        # Un returns correct tenant compte des fins d'épisodes
        returns_np = np.zeros_like(rewards_np)
        
        # Value bootstrapping for truncated sequences
        last_val = 0.0
        if not term_list[-1]:
            _, next_val, _ = self.ppo_agent.forward(params, hidden_state, jnp.asarray(obs, dtype=jnp.float32))
            last_val = float(np.squeeze(next_val))
            
        for t in reversed(range(len(rewards_np))):
            returns_np[t] = rewards_np[t] + 0.99 * last_val * (1.0 - term_np[t])
            last_val = values_np[t]
            
        advantages_np = returns_np - values_np

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(actions_list, dtype=np.int32),
            "rewards": rewards_np,
            "old_log_probs": np.array(log_probs_list, dtype=np.float32),
            "values": values_np,
            "returns": returns_np,
            "advantages": advantages_np,
            "dones": term_np
        }
