import random
from typing import List, Dict, Any

class DummyAgent:
    """
    A temporary agent for testing the end-to-end loop.
    It simply chooses a random legal action.
    """
    
    def select_action(self, legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Selects an action randomly from the list of legal actions.
        """
        if not legal_actions:
            raise ValueError("No legal actions provided to DummyAgent.")
            
        return random.choice(legal_actions)

class MCTSAgent:
    """
    Agent qui utilise le Monte Carlo Tree Search (Gumbel AlphaZero) 
    pour planifier avant de choisir une action, en s'appuyant sur le PPOAgent 
    comme heuristique (Policy/Value).
    """
    def __init__(self, ppo_agent, num_simulations=16, c_puct=1.25):
        from ai.mcts import MCTS
        self.ppo_agent = ppo_agent
        self.mcts = MCTS(num_simulations=num_simulations, c_puct=c_puct)
        
    def select_action(self, env, params, hidden_state, obs):
        """
        Effectue une recherche MCTS et retourne l'action choisie, la politique améliorée,
        la valeur de l'état racine, et le nouvel état caché.
        """
        import numpy as np
        # get legal actions directly from env if supported, else assume all are legal
        try:
            legal_actions = env.get_legal_actions()
        except AttributeError:
            legal_actions = np.ones((self.ppo_agent.act_dim,), dtype=np.bool_)
            
        action, policy_improved = self.mcts.search(
            env=env,
            agent=self.ppo_agent,
            params=params,
            hidden_state=hidden_state,
            obs=obs,
            legal_actions=legal_actions
        )
        
        # We also need to return value and next_hidden to continue the rollout correctly
        # We can just do a forward pass on the selected action to get the next hidden state
        # Or, more simply, do a forward pass on the current state to get value & next hidden
        # for the worker's trajectory record.
        probs, value, next_hidden = self.ppo_agent.forward(params, hidden_state, obs)
        value = float(np.array(value))
        next_hidden = (np.array(next_hidden[0]), np.array(next_hidden[1]))
        
        return action, policy_improved, value, next_hidden
