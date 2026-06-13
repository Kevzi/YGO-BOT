import os
import subprocess
import sys
from pathlib import Path

def main():
    repo_url = "https://github.com/Fluorohydride/ygopro-scripts.git"
    target_dir = Path(__file__).parent.parent / "data" / "scripts"
    
    print(f"Synchronisation des scripts Lua depuis {repo_url}...")
    
    # Créer le dossier data s'il n'existe pas
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if target_dir.exists() and (target_dir / ".git").exists():
        print(f"Le dépôt existe déjà dans {target_dir}. Mise à jour (git pull)...")
        try:
            subprocess.run(["git", "-C", str(target_dir), "pull", "origin", "master"], check=True)
            print("Mise à jour réussie.")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la mise à jour : {e}")
            sys.exit(1)
    else:
        print(f"Le dossier n'existe pas ou n'est pas un dépôt git. Clonage dans {target_dir}...")
        try:
            # S'il y a des fichiers mais pas de git, on ne veut pas écraser n'importe comment, 
            # mais généralement clone va échouer si le dossier n'est pas vide
            if target_dir.exists() and any(target_dir.iterdir()):
                print(f"ATTENTION : Le dossier {target_dir} n'est pas vide mais n'est pas un dépôt git.")
                print("Veuillez vider ou supprimer ce dossier avant de relancer le script.")
                sys.exit(1)
                
            subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)
            print("Clonage réussi.")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors du clonage : {e}")
            sys.exit(1)
            
if __name__ == "__main__":
    main()
