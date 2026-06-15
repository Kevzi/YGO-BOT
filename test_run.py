import sys
import os
import numpy as np
sys.path.append(os.path.abspath('.'))

from core.ygoenv.env import YgoEnv

def test_deadlock():
    env = YgoEnv()
    obs, info = env.reset()
    
    print("Legal actions initially:")
    for i, a in enumerate(info["legal_actions"]):
        if a:
            print(f"Action {i} is legal")
            
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step < 100:
        step += 1
        mask = env.get_action_mask()
        legal = np.where(mask)[0]
        if len(legal) == 0:
            print("NO LEGAL ACTIONS!")
            break
            
        action = np.random.choice(legal)
        print(f"Step {step}: Taking action {action}...")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    print(f"Game complete! Total steps: {step}")

if __name__ == "__main__":
    test_deadlock()
