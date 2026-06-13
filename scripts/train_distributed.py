import ray
from ray.util.queue import Queue
import time
import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ygoenv.env import YgoEnv
from ai.self_play import SelfPlayManager
from ai.distributed import Learner, RolloutWorker

def main():
    print("Initialisation du cluster Ray...")
    ray.init(ignore_reinit_error=True)
    
    # Paramètres d'architecture
    num_workers = 4 # Peut être scalé pour Google TRC / TPU
    queue_maxsize = 100 # Pour éviter un OOM de trajectoires
    batch_size = 4 # Nombre de rollouts collectés avant une update PPO
    rollout_steps = 256
    
    # 1. Initialiser une instance de l'environnement pour obtenir les dimensions
    print("Analyse de l'environnement Gym...")
    temp_env = YgoEnv(omniscience=True)
    obs_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.n
    print(f"Dimension des observations: {obs_dim}, Dimension des actions: {act_dim}")
    
    # 2. Créer le Parameter Server (SelfPlayManager)
    print("Lancement du Parameter Server (SelfPlayManager)...")
    manager = SelfPlayManager.remote(max_history=50)
    
    # 3. Créer la Queue asynchrone partagée
    print(f"Création de la Queue (maxsize={queue_maxsize})...")
    replay_queue = Queue(maxsize=queue_maxsize)
    
    # 4. Lancer le Learner (va s'auto-initialiser et pousser ses poids)
    print("Lancement du Learner (GPU/Central CPU)...")
    learner = Learner.remote(parameter_server_handle=manager, obs_dim=obs_dim, act_dim=act_dim)
    
    # On attend explicitement que le Learner ait initialisé et poussé les poids
    # pour éviter que les Workers plantent en recevant 'None'
    while ray.get(manager.get_latest_params.remote()) is None:
        time.sleep(0.5)
        
    # Le Learner tourne en arrière-plan
    learner.start_learning_loop.remote(replay_queue=replay_queue, batch_size=batch_size)
    
    # 5. Lancer les RolloutWorkers
    print(f"Lancement de {num_workers} Rollout Workers (CPU)...")
    workers = []
    for i in range(num_workers):
        worker = RolloutWorker.remote(
            parameter_server_handle=manager, 
            queue_handle=replay_queue, 
            obs_dim=obs_dim, 
            act_dim=act_dim,
            worker_id=i
        )
        # Chaque worker tourne indéfiniment
        worker.start_collection_loop.remote(rollout_steps=rollout_steps)
        workers.append(worker)
        
    print("=== Architecture Distribuée Lancée avec Succès ===")
    print("Le Learner et les Workers s'exécutent de manière asynchrone.")
    
    # 6. Boucle de monitoring (Thread principal)
    t0 = time.time()
    try:
        while True:
            # Récupérer quelques métriques sans bloquer
            qsize = replay_queue.qsize()
            history_size = ray.get(manager.get_history_size.remote())
            uptime = time.time() - t0
            
            print(f"[{uptime:.0f}s] Uptime | Queue size: {qsize}/{queue_maxsize} | Snapshots: {history_size}")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nArrêt de l'entraînement distribué demandé.")
        ray.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()
