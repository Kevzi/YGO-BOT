import math
import numpy as np
import jax.numpy as jnp

class MCTSNode:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, num_simulations: int = 16, c_puct: float = 1.25):
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def _puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        pb_c = math.log((parent.visit_count + 1.25 + 19652.0) / 19652.0) + self.c_puct
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = child.value()
        return value_score + prior_score

    def search(self, env, agent, params, hidden_state, obs, legal_actions):
        """
        Effectue la recherche MCTS (Gumbel AlphaZero style) depuis l'état actuel.
        env: l'environnement (doit supporter save_state() et restore_state())
        agent: PPOAgent contenant la fonction forward()
        params: paramètres de l'agent
        hidden_state: état caché actuel (h, c) pour un seul élément (pas de batch dim)
        obs: observation courante
        legal_actions: masque booléen des actions légales
        """
        root = MCTSNode(0.0)
        
        # 1. Expand root
        obs_jax = jnp.expand_dims(jnp.asarray(obs), axis=0) if np.asarray(obs).ndim == 1 else jnp.asarray(obs)
        probs, value, next_hidden_state = agent.forward(params, hidden_state, obs_jax)
        
        probs_np = np.asarray(probs)
        if probs_np.ndim > 1:
            probs_np = probs_np.squeeze(0)
        value = float(jnp.squeeze(value))
        # Keep next_hidden_state as JAX array!
        
        # Apply legal actions mask
        legal_actions = np.array(legal_actions, dtype=np.bool_)
        probs_np = probs_np * legal_actions
        sum_probs = np.sum(probs_np)
        if sum_probs > 0:
            probs_np /= sum_probs
        else:
            probs_np = legal_actions / max(np.sum(legal_actions), 1e-8)
            
        # Gumbel noise at root for exploration
        gumbel_noise = np.random.gumbel(loc=0.0, scale=1.0, size=probs_np.shape)
        gumbel_noise = gumbel_noise * legal_actions
        
        # Add to logits-equivalent and softmax (Simplified Gumbel AlphaZero approach)
        logits = np.log(probs_np + 1e-8) + gumbel_noise
        exp_logits = np.exp(logits) * legal_actions
        
        sum_exp_logits = np.sum(exp_logits)
        if sum_exp_logits > 0:
            probs_with_noise = exp_logits / sum_exp_logits
        else:
            probs_with_noise = probs_np

        root.is_expanded = True
        root.hidden_state = next_hidden_state
        for a in range(len(probs)):
            if legal_actions[a]:
                root.children[a] = MCTSNode(prior=float(probs_with_noise[a]))

        # Save base environment state
        base_state = env.save_state()

        # 2. Simulations
        for _ in range(self.num_simulations):
            node = root
            current_hidden = root.hidden_state
            search_path = [node]
            env.restore_state(base_state)
            
            # Selection
            done = False
            info = None
            while node.is_expanded and len(node.children) > 0:
                action, child = max(
                    node.children.items(),
                    key=lambda item: self._puct_score(node, item[1])
                )
                node = child
                search_path.append(node)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if node.is_expanded:
                    current_hidden = node.hidden_state
                
            value_for_backprop = 0.0
            
            if not done:
                # Expansion & Evaluation
                obs_jax_step = jnp.expand_dims(jnp.asarray(obs), axis=0) if np.asarray(obs).ndim == 1 else jnp.asarray(obs)
                probs_deep, value_deep, next_hidden_deep = agent.forward(params, current_hidden, obs_jax_step)
                probs_deep_np = np.asarray(probs_deep)
                if probs_deep_np.ndim > 1:
                    probs_deep_np = probs_deep_np.squeeze(0)
                value_for_backprop = float(jnp.squeeze(value_deep))
                
                # Assume all actions are legal in deeper nodes for simplicity if we don't have access to deep legal_actions
                # In a real engine, we query legal actions after env.step()
                if info is not None and "legal_actions" in info:
                    legal_actions_deep = np.asarray(info["legal_actions"], dtype=np.bool_)
                else:
                    legal_actions_deep = np.ones_like(probs_deep_np, dtype=np.bool_)
                    
                probs_deep_np = probs_deep_np * legal_actions_deep
                sum_probs = np.sum(probs_deep_np)
                if sum_probs > 0:
                    probs_deep_np /= sum_probs
                else:
                    probs_deep_np = legal_actions_deep / max(np.sum(legal_actions_deep), 1e-8)
                    
                node.is_expanded = True
                node.hidden_state = next_hidden_deep
                for a in range(len(probs_deep_np)):
                    if legal_actions_deep[a]:
                        node.children[a] = MCTSNode(prior=float(probs_deep_np[a]))
            else:
                value_for_backprop = float(reward)

            # Backpropagation
            for n in reversed(search_path):
                n.value_sum += value_for_backprop
                n.visit_count += 1
                # Flip value for opponent if it's a 2-player zero-sum game with alternating turns
                # Here we assume a single-agent perspective or symmetric zero-sum (simplification)
                value_for_backprop = -value_for_backprop 

        # Restore original env state
        env.restore_state(base_state)

        # 3. Policy Improvement
        visit_counts = np.zeros(len(probs_with_noise))
        for a, child in root.children.items():
            visit_counts[a] = child.visit_count
            
        if np.sum(visit_counts) > 0:
            policy_improved = visit_counts / np.sum(visit_counts)
        else:
            policy_improved = legal_actions / max(np.sum(legal_actions), 1e-8)
            
        action = np.argmax(policy_improved)
        
        return int(action), policy_improved
