import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization
import logging
import random
from typing import List, Dict, Any

from ai.ppo import PPOAgent
from core.ygoenv.env import YgoEnv

logger = logging.getLogger(__name__)

class PPOInferenceAgent:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PPOInferenceAgent, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path="data/checkpoints/self_play/model_latest.msgpack"):
        if self._initialized:
            return
            
        self.ppo_agent = PPOAgent(obs_dim=60694, act_dim=250)
        self.params = None
        self.hidden_state = self.ppo_agent.init_hidden_state(1)
        self.prev_action = jnp.zeros((), dtype=jnp.int32)
        
        try:
            with open(checkpoint_path, "rb") as f:
                param_bytes = f.read()
            rng = jax.random.PRNGKey(0)
            dummy_params, _ = self.ppo_agent.init_params(rng)
            self.params = flax.serialization.msgpack_restore(param_bytes)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            
        self._initialized = True

    def select_action(self, actions: List[Dict[str, Any]], engine, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not actions:
            raise ValueError("No legal actions provided.")
            
        # Collect all valid choices across all action groups
        all_choices = []
        for a in actions:
            if a.get("msg") in ["MSG_SELECT_IDLECMD", "MSG_SELECT_BATTLECMD", "MSG_SELECT_CARD", "MSG_SELECT_CHAIN"]:
                for choice in a.get("choices", []):
                    all_choices.append(choice)

        if not all_choices:
            # Fallback to random if no choices available but engine still returned something
            return {"action_type": "UNKNOWN", "action_idx": -1}

        if self.params is None:
            choice = random.choice(all_choices)
            return {
                "action_type": choice.get("type", "UNKNOWN"),
                "action_idx": choice.get("action_idx", -1),
                "card_id": choice.get("code")
            }
            
        # Build mask
        mask = np.zeros(250, dtype=np.bool_)
        for choice in all_choices:
            idx = choice.get("action_idx")
            if idx is not None and 0 <= idx < 250:
                mask[idx] = True
                
        if "action_type" in state_dict:
            # We don't have the full ygopro-core state, mock observation for Omega
            obs = np.zeros(60694, dtype=np.float32)
        else:
            env = YgoEnv(omniscience=True)
            env.engine = engine
            env._current_state = state_dict
            
            current_player = state_dict.get("turn_player", 0)
            if actions and len(actions) > 0:
                current_player = actions[0].get("player", current_player)
                
            obs = env._get_observation(player=current_player)
        
        obs_jnp = jnp.array(obs)
        mask_jnp = jnp.array(mask)
        
        probs, value, next_hidden = self.ppo_agent.forward(
            self.params, 
            self.hidden_state, 
            obs_jnp, 
            self.prev_action, 
            mask_jnp
        )
        
        probs_np = np.array(probs)
        probs_np[~mask] = -1.0
        chosen_idx = int(np.argmax(probs_np))
        
        self.hidden_state = next_hidden
        self.prev_action = jnp.array(chosen_idx, dtype=jnp.int32)
        
        # Find chosen action
        for choice in all_choices:
            if choice.get("action_idx") == chosen_idx:
                return {
                    "action_type": choice.get("type", "UNKNOWN"),
                    "action_idx": chosen_idx,
                    "card_id": choice.get("code")
                }
                
        # Fallback if argmax fails to match any choice
        choice = random.choice(all_choices)
        return {
            "action_type": choice.get("type", "UNKNOWN"),
            "action_idx": choice.get("action_idx", -1),
            "card_id": choice.get("code")
        }
