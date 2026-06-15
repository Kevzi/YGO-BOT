import os

import ray
import numpy as np
import jax
import time
import traceback
import gc

from core.ygoenv.env import YgoEnv
from ai.ppo import PPOAgent
from ai.mcts import MCTS

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

@ray.remote(num_gpus=0.5)
class Learner:
    """
    Acteur central qui récupère les trajectoires, met à jour le réseau PPO (JAX),
    et pousse les nouveaux poids dans le SelfPlayManager.
    Utilise le GPU pour les mises à jour de gradient (calcul lourd).
    """
    def __init__(self, parameter_server_handle, obs_dim, act_dim):
        # Détecter et utiliser le GPU si disponible
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            print(f"[Learner] GPU détecté : {gpu_devices[0].device_kind}")
            jax.config.update('jax_default_device', gpu_devices[0])
        else:
            print("[Learner] Aucun GPU détecté, fallback CPU.")
        
        self.parameter_server = parameter_server_handle
        self.agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)
        
        # Initialisation des poids — init_params retourne (params, opt_state)
        key = jax.random.PRNGKey(42)
        self.params, self.opt_state = self.agent.init_params(key)
        
        # Pousser les poids initiaux
        ray.get(self.parameter_server.set_latest_params.remote(self.params))
        ray.get(self.parameter_server.add_snapshot.remote(self.params))
        
        self.updates_count = 0
        
        # Initialisation Tensorboard
        from tensorboardX import SummaryWriter
        os.makedirs("logs/self_play", exist_ok=True)
        self.writer = SummaryWriter(log_dir="logs/self_play")

    def save_checkpoint(self):
        """Sauvegarde d'urgence ou périodique des poids sur le disque."""
        import flax
        os.makedirs("data/checkpoints/self_play", exist_ok=True)
        filepath = "data/checkpoints/self_play/model_latest.msgpack"
        try:
            with open(filepath, "wb") as f:
                f.write(flax.serialization.to_bytes(self.params))
            print(f"[Learner] Checkpoint sauvegardé sur disque : {filepath}")
        except Exception as e:
            print(f"[Learner] Erreur lors de la sauvegarde du checkpoint : {e}")

    def start_learning_loop(self, replay_queue, batch_size=4, epochs_per_snapshot=5):
        """Boucle asynchrone consommant la file d'attente."""
        print("[Learner] Démarrage de la boucle d'apprentissage...")
        while True:
            # Récupérer un lot de trajectoires (on bloque jusqu'à en avoir `batch_size`)
            rollouts = []
            for _ in range(batch_size):
                rollouts.append(replay_queue.get())
            
            # Concaténer et calculer GAE
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
            if not all_states:
                import logging
                logging.warning("Batch d'observations vide (all_states est vide). Ignoré.")
                return

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
                    
            obs_batch = np.stack(all_states, axis=0)
            mask_batch = np.stack(all_masks, axis=0).astype(bool)
            act_batch = np.stack(all_actions, axis=0)
            p_act_batch = np.stack(all_prev_actions, axis=0)
            adv_batch = np.stack(all_adv, axis=0)
            ret_batch = np.stack(all_ret, axis=0)
            lp_batch = np.stack(all_lp, axis=0)
            dones_batch = np.stack(all_dones, axis=0)
            hidden_batch = (np.stack(all_hidden_h, axis=0), np.stack(all_hidden_c, axis=0))
            
            # Mettre à jour le réseau PPO
            self.params, self.opt_state, metrics = self.agent.update_params(
                self.params, self.opt_state, hidden_batch, obs_batch, p_act_batch, mask_batch, act_batch, 
                lp_batch, adv_batch, ret_batch, dones_batch
            )
            
            self.updates_count += 1
            
            # Pousser les nouveaux poids de manière asynchrone (fire-and-forget)
            self.parameter_server.set_latest_params.remote(self.params)
            
            # Snapshot historique
            if self.updates_count % epochs_per_snapshot == 0:
                self.parameter_server.add_snapshot.remote(self.params)
                self.save_checkpoint()
                
            # Logger les perfs (on pourrait utiliser wandb ici)
            print(f"[Learner] Update {self.updates_count} | Loss: {metrics['total_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f} | Entropy: {metrics['entropy']:.4f}")
            self.writer.add_scalar("Train/Total_Loss", metrics['total_loss'], self.updates_count)
            self.writer.add_scalar("Train/Value_Loss", metrics['value_loss'], self.updates_count)
            self.writer.add_scalar("Train/Entropy", metrics['entropy'], self.updates_count)

@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    """
    Acteur qui joue des parties en utilisant l'environnement (CPU) et pousse
    les trajectoires générées dans la Queue.
    Force CPU pour les forward passes légers — le GPU est réservé au Learner.
    """
    def __init__(self, parameter_server_handle, queue_handle, obs_dim, act_dim, worker_id):
        # Forcer JAX en mode CPU pour ce Worker
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        jax.config.update('jax_default_device', jax.devices('cpu')[0])
        
        self.parameter_server = parameter_server_handle
        self.queue = queue_handle
        self.worker_id = worker_id
        
        self.env = YgoEnv(omniscience=True)
        self.agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)

    def start_collection_loop(self, rollout_steps=256, max_steps=10000):
        """Boucle asynchrone de collecte de données."""
        print(f"[Worker {self.worker_id}] Démarrage de la boucle de collecte...")
        try:
            print(f"[Worker {self.worker_id}] Appel de env.reset()...")
            obs, info = self.env.reset()
            print(f"[Worker {self.worker_id}] env.reset() terminé. obs.shape={obs.shape}")
            
            while True:
                # 1. Récupérer les poids les plus récents ET d'un adversaire (bloquant)
                learner_params = ray.get(self.parameter_server.get_latest_params.remote())
                opponent_params = ray.get(self.parameter_server.get_match_params.remote())
                
                # 2. Collecter la trajectoire
                trajectory = self._collect_rollout(learner_params, opponent_params, obs, info, rollout_steps, max_steps)
                obs = trajectory["final_obs"]
                info = trajectory["final_info"]
                
                print(f"[Worker {self.worker_id}] Rollout terminé, {len(trajectory['states'])} steps collectés. Push dans la queue...")
                
                # 3. Pousser dans la file (bloque si la queue est pleine, évitant un OOM)
                self.queue.put(trajectory)
                
                # 4. Libérer les copies de params désérialisées pour éviter l'accumulation mémoire
                del learner_params, opponent_params, trajectory
                gc.collect()
        except Exception as e:
            print(f"[Worker {self.worker_id}] ERREUR FATALE: {e}")
            traceback.print_exc()
            raise
        finally:
            print(f"[Worker {self.worker_id}] Arrêt de la boucle, fermeture de l'environnement.")
            if hasattr(self.env, "close"):
                self.env.close()

    def _collect_rollout(self, params, opponent_params, initial_obs, initial_info, steps, max_steps):
        """Méthode interne pour collecter un rollout (inspirée de train.py)."""
        states, actions, prev_actions, rewards, dones, log_probs, values, masks = [], [], [], [], [], [], [], []
        
        hidden_state_learner = self.agent.init_hidden_state(1)
        hidden_state_opponent = self.agent.init_hidden_state(1)
        initial_hidden_state = hidden_state_learner
        
        prev_action_learner = np.int32(0)
        prev_action_opponent = np.int32(0)
        
        total_env_steps = 0
        obs = initial_obs
        info = initial_info
        
        while len(states) < steps and total_env_steps < max_steps:
            total_env_steps += 1
            mask = self.env.get_action_mask()
            current_player = info.get("current_player", 0)
            
            if current_player == 0:
                current_params = params
                current_hidden = hidden_state_learner
                current_obs = obs
                current_prev_act = prev_action_learner
            else:
                current_params = opponent_params
                current_hidden = hidden_state_opponent
                current_obs = self.env._get_observation(player=current_player)
                current_prev_act = prev_action_opponent
                
            # 1. Évaluation brute via PPOAgent (pour stockage log_probs et values)
            probs, value, next_hidden = self.agent.forward(current_params, current_hidden, current_obs, current_prev_act, mask)
            
            if current_player == 0:
                hidden_state_learner = next_hidden
            else:
                hidden_state_opponent = next_hidden
            
            probs = np.asarray(probs).flatten()
            masked_probs = probs
            masked_probs = masked_probs / np.sum(masked_probs)
                
            # --- BYPASS MCTS POUR LE COLD START (MVP) ---
            use_mcts = True  # Activé pour la nouvelle méta (Epic 4/6)
            
            if use_mcts:
                # 2. Amélioration de la politique via MCTS
                mcts = MCTS(self.agent.model, params=current_params, c_puct=1.25)
                root_node = mcts.search(self.env, current_obs, current_hidden, current_prev_act, num_simulations=4)
                action_probs = mcts.get_action_probs(root_node, temperature=1.0)
            else:
                action_probs = masked_probs
            
            # Sélection de l'action selon la distribution
            action = np.random.choice(len(action_probs), p=action_probs)
                
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            if current_player == 0:
                states.append(obs)
                masks.append(mask)
                prev_actions.append(prev_action_learner)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                log_probs.append(np.log(masked_probs[action] + 1e-8))
                
                prev_action_learner = np.int32(action)
            else:
                prev_action_opponent = np.int32(action)
                
            if done and len(rewards) > 0:
                rewards[-1] += reward
                dones[-1] = done
            
            obs = next_obs
            if done:
                obs, info = self.env.reset()
                hidden_state_learner = self.agent.init_hidden_state(1)
                hidden_state_opponent = self.agent.init_hidden_state(1)
                prev_action_learner = np.int32(0)
                prev_action_opponent = np.int32(0)
                
        # Bootstrap
        bootstrap_value = 0.0
        if len(states) > 0 and not dones[-1]:
            _, b_value, _ = self.agent.forward(params, hidden_state_learner, obs, prev_action_learner, self.env.get_action_mask())
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
            "bootstrap_value": bootstrap_value,
            "final_obs": obs,
            "final_info": info,
            "hidden_state": initial_hidden_state
        }
