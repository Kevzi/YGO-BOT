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
        states, actions, prev_actions, rewards, dones, log_probs, values, masks = [], [], [], [], [], [], [], []
        
        hidden_state_learner = agent.init_hidden_state(1)
        hidden_state_opponent = agent.init_hidden_state(1)
        initial_hidden_state = hidden_state_learner
        
        prev_action_learner = np.int32(0)
        prev_action_opponent = np.int32(0)
        
        total_env_steps = 0
        
        while len(states) < steps and total_env_steps < max_steps:
            total_env_steps += 1
            mask = self.env.get_action_mask()
            current_player = self.info.get("current_player", 0)
            
            # Sélectionner les bons poids et état caché
            if current_player == 0:
                current_params = params
                current_hidden = hidden_state_learner
                current_prev_act = prev_action_learner
            else:
                current_params = opponent_params
                current_hidden = hidden_state_opponent
                current_prev_act = prev_action_opponent
                
            probs, value, next_hidden = agent.forward(current_params, current_hidden, self.obs, current_prev_act, mask)
            
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
            
            if current_player == 0:
                states.append(self.obs)
                masks.append(mask)
                actions.append(action)
                prev_actions.append(prev_action_learner)
                rewards.append(reward)
                dones.append(terminated)
                values.append(value)
                log_probs.append(np.log(masked_probs[action] + 1e-8))
                
                prev_action_learner = np.int32(action)
            else:
                prev_action_opponent = np.int32(action)
                
            if done and len(rewards) > 0:
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
            _, b_value, _ = agent.forward(params, hidden_state_learner, self.obs, prev_action_learner, self.env.get_action_mask())
            bootstrap_value = float(b_value)
                
        return {
            "states": np.array(states),
            "masks": np.array(masks),
            "actions": np.array(actions),
            "prev_actions": np.array(prev_actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "hidden_state": initial_hidden_state,
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
    manager = SelfPlayManager.remote(max_history=5)
    ray.get(manager.set_latest_params.remote(params))
    ray.get(manager.add_snapshot.remote(params))
    
    print(f"Lancement de {num_workers} Rollout Workers...")
    workers = [RolloutWorker.remote() for _ in range(num_workers)]
    
    print("Début de la boucle d'apprentissage RL...")
    for epoch in range(epochs):
        t0 = time.time()
        
        # Tirer un adversaire historique pour ce lot
        opponent_params = ray.get(manager.get_match_params.remote())
        
        # 1. Collecter les données de tous les workers en parallèle
        futures = [w.collect_rollout.remote(params, agent, opponent_params, rollout_steps) for w in workers]
        rollouts = ray.get(futures)
        
        # 2. Concaténer et calculer GAE
        all_states, all_masks, all_actions, all_prev_actions, all_adv, all_ret, all_lp, all_dones, all_hidden_h, all_hidden_c = [], [], [], [], [], [], [], [], [], []
        for r in rollouts:
            adv, ret = compute_gae(r)
            all_states.append(r["states"])
            all_masks.append(r["masks"])
            all_actions.append(r["actions"])
            all_prev_actions.append(r["prev_actions"])
            all_adv.append(adv)
            all_ret.append(ret)
            all_lp.append(r["log_probs"])
            all_dones.append(r["dones"])
            all_hidden_h.append(r["hidden_state"][0][0])
            all_hidden_c.append(r["hidden_state"][1][0])
            
        max_len = max(len(s) for s in all_states)
        for i in range(len(all_states)):
            pad_len = max_len - len(all_states[i])
            if pad_len > 0:
                all_states[i] = np.pad(all_states[i], ((0, pad_len), (0, 0)))
                all_masks[i] = np.pad(all_masks[i], ((0, pad_len), (0, 0)))
                all_actions[i] = np.pad(all_actions[i], (0, pad_len))
                all_prev_actions[i] = np.pad(all_prev_actions[i], (0, pad_len))
                all_adv[i] = np.pad(all_adv[i], (0, pad_len))
                all_ret[i] = np.pad(all_ret[i], (0, pad_len))
                all_lp[i] = np.pad(all_lp[i], (0, pad_len))
                all_dones[i] = np.pad(all_dones[i], (0, pad_len), constant_values=True)
                
        obs_batch = np.stack(all_states)
        mask_batch = np.stack(all_masks).astype(bool)
        act_batch = np.stack(all_actions)
        p_act_batch = np.stack(all_prev_actions)
        adv_batch = np.stack(all_adv)
        ret_batch = np.stack(all_ret)
        lp_batch = np.stack(all_lp)
        dones_batch = np.stack(all_dones)
        hidden_batch = (np.stack(all_hidden_h, axis=0), np.stack(all_hidden_c, axis=0))
        
        # 3. Mettre à jour le réseau PPO
        opt_state = None # Hack since train.py does not maintain opt_state properly
        params, _, metrics = agent.update_params(
            params, opt_state, hidden_batch, obs_batch, p_act_batch, mask_batch, act_batch, lp_batch, adv_batch, ret_batch, dones_batch
        )
        
        fps = (num_workers * rollout_steps) / (time.time() - t0)
        print(f"Epoch {epoch+1}/{epochs} | FPS: {fps:.1f} | Loss: {metrics['total_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f}")
        
        # Archiver périodiquement dans le SelfPlayManager
        if (epoch + 1) % 5 == 0:
            ray.get(manager.add_snapshot.remote(params))

if __name__ == "__main__":
    main()
