import math
import jax
import jax.numpy as jnp
import numpy as np

class MCTSNode:
    """
    Nœud de l'arbre MCTS pour Gumbel AlphaZero.
    Stocke l'état caché du LSTM (carry_state) pour éviter de recalculer
    toute la séquence depuis le début du duel.
    """
    def __init__(self, carry_state, action_mask, is_terminal=False, reward=0.0):
        self.carry_state = carry_state
        self.action_mask = action_mask
        self.is_terminal = is_terminal
        self.reward = reward
        
        self.children = {} # action (int) -> MCTSNode
        self.N = 0 # Nombre de visites
        self.W = 0.0 # Somme des valeurs (pour le joueur 0)
        self.Q = 0.0 # Valeur moyenne
        self.P = None # Probabilités a priori (Policy issue du réseau)
        
    def expand(self, action_probs: np.ndarray):
        """Initialise les probabilités a priori (P)."""
        self.P = action_probs
        
    def add_child(self, action: int, child_node: 'MCTSNode'):
        self.children[action] = child_node

    def update(self, v: float):
        """Met à jour les statistiques du nœud (Backpropagation)."""
        self.N += 1
        self.W += v
        self.Q = self.W / self.N

class MCTS:
    """
    Recherche Arborescente Monte Carlo (MCTS) avec sélection PUCT et
    variante Gumbel AlphaZero (ajout de bruit Gumbel à la racine).
    """
    def __init__(self, actor_critic, params, c_puct=1.25):
        self.actor_critic = actor_critic
        self.params = params
        self.c_puct = c_puct
        self.action_dim = actor_critic.action_dim

    def evaluate_state(self, carry, obs, prev_action, action_mask, apply_gumbel=False):
        """
        Évalue un état avec le réseau ActorCriticLSTM.
        Si apply_gumbel est True, ajoute du bruit de Gumbel aux logits pour l'exploration.
        """
        obs_jnp = jnp.expand_dims(jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), 0), 0)
        prev_action_jnp = jnp.expand_dims(jnp.expand_dims(jnp.array(prev_action, dtype=jnp.int32), 0), 0)
        action_mask_jnp = jnp.expand_dims(jnp.expand_dims(jnp.array(action_mask, dtype=jnp.bool_), 0), 0)
        dones_jnp = jnp.zeros((1, 1), dtype=jnp.bool_)
        
        new_carry, logits, value = self.actor_critic.apply(
            {'params': self.params}, carry, obs_jnp, prev_action_jnp, action_mask_jnp, dones_jnp
        )
        
        value = float(value[0, 0])
        logits = np.array(logits[0, 0])
        
        # Masquer les actions illégales
        mask_val = -1e9
        valid_logits = np.where(action_mask, logits, mask_val)
        
        if apply_gumbel:
            gumbel_noise = np.random.gumbel(size=valid_logits.shape)
            valid_logits = np.where(action_mask, valid_logits + gumbel_noise, mask_val)
            
        # Softmax stable
        max_logit = np.max(valid_logits)
        exp_logits = np.exp(valid_logits - max_logit) * action_mask
        sum_exp = np.sum(exp_logits)
        
        if sum_exp > 0:
            probs = exp_logits / sum_exp
        else:
            probs = action_mask / max(np.sum(action_mask), 1e-8)
            
        return new_carry, probs, value

    def select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Sélectionne le meilleur nœud enfant selon la formule PUCT."""
        best_action = -1
        best_ucb = -float('inf')
        
        # First Play Urgency : on utilise le Q du parent pour les nœuds non explorés
        fpu = node.Q if node.N > 0 else 0.0
        
        for action in np.where(node.action_mask)[0]:
            if action in node.children:
                child = node.children[action]
                u = node.P[action] * self.c_puct * math.sqrt(node.N) / (1 + child.N)
                ucb = child.Q + u
            else:
                u = node.P[action] * self.c_puct * math.sqrt(node.N + 1e-8)
                ucb = fpu + u
                
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = int(action)
                
        return best_action, node.children.get(best_action, None)

    def search(self, env, obs, carry, prev_action, num_simulations=50):
        """
        Exécute la recherche MCTS à partir d'un état racine en utilisant env.clone().
        """
        root_mask = env.get_action_mask() if hasattr(env, 'get_action_mask') else env.get_legal_actions()
        root_node = MCTSNode(carry, root_mask, is_terminal=False)
        
        # Évaluation de la racine avec bruit Gumbel (Gumbel AlphaZero)
        _, root_probs, _ = self.evaluate_state(carry, obs, prev_action, root_mask, apply_gumbel=True)
        root_node.expand(root_probs)
        
        for _ in range(num_simulations):
            node = root_node
            sim_env = env.clone() # Action Replay (State Cloning)
            path = [node]
            
            # 1. Selection
            action_taken = prev_action
            while not node.is_terminal:
                action, next_node = self.select_child(node)
                action_taken = action
                if next_node is None:
                    break
                    
                # On avance dans l'environnement cloné
                step_obs, step_reward, terminated, truncated, _ = sim_env.step(action)
                node = next_node
                path.append(node)
                
            # 2. Expansion
            if not node.is_terminal:
                if action_taken == -1:
                    # Plus d'actions légales possibles, état bloqué
                    node.is_terminal = True
                    v = node.reward
                    for n in path:
                        n.update(v)
                    continue
                    
                # Appliquer l'action trouvée lors de la sélection
                step_obs, step_reward, terminated, truncated, _ = sim_env.step(action_taken)
                new_mask = sim_env.get_action_mask() if hasattr(sim_env, 'get_action_mask') else sim_env.get_legal_actions()
                
                new_carry, probs, value = self.evaluate_state(
                    node.carry_state, step_obs, action_taken, new_mask, apply_gumbel=False
                )
                
                new_node = MCTSNode(new_carry, new_mask, is_terminal=terminated, reward=step_reward)
                new_node.expand(probs)
                node.add_child(action_taken, new_node)
                
                path.append(new_node)
                v = value
            else:
                v = node.reward
                
            # 3. Backpropagation
            # Le jeu est modélisé de la perspective du Joueur 0. 
            # Les rewards de env.py sont déjà globaux (1.0 pour victoire P0, -1.0 pour P1).
            for n in path:
                n.update(v)
                
        return root_node

    def get_action_probs(self, root_node: MCTSNode, temperature=1.0) -> np.ndarray:
        """Calcule les probabilités d'action finales basées sur les visites MCTS."""
        action_visits = np.zeros(self.action_dim, dtype=np.float32)
        for action, child in root_node.children.items():
            action_visits[action] = child.N
            
        if temperature == 0:
            if np.sum(action_visits) == 0:
                probs = root_node.action_mask / max(np.sum(root_node.action_mask), 1e-8)
                return probs
            best_action = np.argmax(action_visits)
            probs = np.zeros_like(action_visits)
            probs[best_action] = 1.0
            return probs
            
        action_visits = action_visits ** (1.0 / temperature)
        sum_visits = np.sum(action_visits)
        if sum_visits > 0:
            probs = action_visits / sum_visits
        else:
            probs = root_node.action_mask / max(np.sum(root_node.action_mask), 1e-8)
            
        return probs
