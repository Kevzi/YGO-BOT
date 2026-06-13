import ray
import numpy as np
import jax
import time
import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ygoenv.env import YgoEnv
from ai.ppo import PPOAgent

@ray.remote
class RolloutWorker:
    def __init__(self):
        self.env = YgoEnv(omniscience=True)
        self.obs, self.info = self.env.reset()
        
    def collect_rollout(self, params, agent, opponent_params, steps=256, max_steps=10000):
        """Collecte un segment de trajectoire avec les poids actuels."""
        states, actions, rewards, dones, log_probs, values, masks = [], [], [], [], [], [], []
        
        hidden_state_learner = agent.init_hidden_state(1)
        hidden_state_opponent = agent.init_hidden_state(1)
        
        total_env_steps = 0
        
        while len(states) < steps and total_env_steps < max_steps:
            total_env_steps += 1
            mask = self.env.get_action_mask()
            current_player = self.info.get("current_player", 0)
            
            # Sélectionner les bons poids et état caché
            if current_player == 0:
                current_params = params
                current_hidden = hidden_state_learner
            else:
                current_params = opponent_params
                current_hidden = hidden_state_opponent
                
            probs, value, next_hidden = agent.forward(current_params, current_hidden, self.obs, mask)
            
            if current_player == 0:
                hidden_state_learner = next_hidden
            else:
                hidden_state_opponent = next_hidden
            
            masked_probs = probs * mask
            prob_sum = np.sum(masked_probs)
            if prob_sum == 0:
                masked_probs = mask / (np.sum(mask) + 1e-8)
            else:
                masked_probs = masked_probs / prob_sum
                
            action = np.random.choice(len(masked_probs), p=np.array(masked_probs))
            
            next_obs, reward, terminated, truncated, self.info = self.env.step(action)
            done = terminated or truncated
            
            # On stocke uniquement les expériences du point de vue de Player 0
            if current_player == 0:
                states.append(self.obs)
                masks.append(mask)
                actions.append(action)
                rewards.append(reward)
                dones.append(terminated)
                values.append(value)
                log_probs.append(np.log(masked_probs[action] + 1e-8))
            elif done and len(rewards) > 0:
                rewards[-1] += reward
                dones[-1] = terminated
            
            self.obs = next_obs
            if done:
                self.obs, self.info = self.env.reset()
                hidden_state_learner = agent.init_hidden_state(1)
                hidden_state_opponent = agent.init_hidden_state(1)
                
        # Calcul du bootstrap value pour l'état final s'il n'est pas terminal
        bootstrap_value = 0.0
        if len(states) > 0 and not dones[-1]:
            _, b_value, _ = agent.forward(params, hidden_state_learner, self.obs, self.env.get_action_mask())
            bootstrap_value = float(b_value)
                
        return {
            "states": np.array(states),
            "masks": np.array(masks),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "bootstrap_value": bootstrap_value
        }

def compute_gae(rollout, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]
    bootstrap_value = rollout.get("bootstrap_value", 0.0)
    
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = bootstrap_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t+1]
            
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        
    returns = advantages + values
    return advantages, returns

def main():
    print("Initialisation de Ray...")
    ray.init(ignore_reinit_error=True)
    
    from ai.self_play import SelfPlayManager
    
    # Paramètres
    num_workers = 4
    rollout_steps = 256
    epochs = 100
    
    print("Création de l'Agent PPO et initialisation des poids JAX...")
    temp_env = YgoEnv(omniscience=True)
    obs_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.n
    print(f"Dimension des observations: {obs_dim}, Dimension des actions: {act_dim}")
    
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)
    key = jax.random.PRNGKey(42)
    params = agent.init_params(key)
    
    # Initialiser le SelfPlayManager avec le premier checkpoint
    manager = SelfPlayManager.remote(max_history=50)
    ray.get(manager.add_snapshot.remote(params))
    
    print(f"Lancement de {num_workers} Rollout Workers...")
    workers = [RolloutWorker.remote() for _ in range(num_workers)]
    
    print("Début de la boucle d'apprentissage RL...")
    for epoch in range(epochs):
        t0 = time.time()
        
        # Tirer un adversaire historique pour ce lot
        opponent_params = ray.get(manager.get_opponent.remote(params))
        
        # 1. Collecter les données de tous les workers en parallèle
        futures = [w.collect_rollout.remote(params, agent, opponent_params, rollout_steps) for w in workers]
        rollouts = ray.get(futures)
        
        # 2. Concaténer et calculer GAE
        all_states, all_masks, all_actions, all_adv, all_ret, all_lp, all_dones = [], [], [], [], [], [], []
        for r in rollouts:
            adv, ret = compute_gae(r)
            all_states.append(r["states"])
            all_masks.append(r["masks"])
            all_actions.append(r["actions"])
            all_adv.append(adv)
            all_ret.append(ret)
            all_lp.append(r["log_probs"])
            all_dones.append(r["dones"])
            
        obs_batch = np.stack(all_states)
        mask_batch = np.stack(all_masks).astype(bool)
        act_batch = np.stack(all_actions)
        adv_batch = np.stack(all_adv)
        ret_batch = np.stack(all_ret)
        lp_batch = np.stack(all_lp)
        dones_batch = np.stack(all_dones)
        
        # 3. Mettre à jour le réseau PPO
        params, metrics = agent.update_params(
            params, obs_batch, mask_batch, act_batch, lp_batch, adv_batch, ret_batch, dones_batch
        )
        
        fps = (num_workers * rollout_steps) / (time.time() - t0)
        print(f"Epoch {epoch+1}/{epochs} | FPS: {fps:.1f} | Loss: {metrics['total_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f}")
        
        # Archiver périodiquement dans le SelfPlayManager
        if (epoch + 1) % 5 == 0:
            ray.get(manager.add_snapshot.remote(params))

if __name__ == "__main__":
    main()
