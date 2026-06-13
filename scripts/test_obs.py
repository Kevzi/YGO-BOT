import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ygoenv.env import YgoEnv

def main():
    print("Initialisation de l'environnement...")
    env = YgoEnv()
    obs, info = env.reset()
    
    cards_info = env.engine.query_field_state(0)
    print("Cards Info:", cards_info)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    
    if obs.shape == (60694,):
        print("SUCCES: La shape de l'observation est correcte !")
    else:
        print("ERREUR: Shape incorrecte.")
        
    print("Premières 5 valeurs du premier slot (statistiques) :")
    print(obs[:5])
    
if __name__ == "__main__":
    main()
