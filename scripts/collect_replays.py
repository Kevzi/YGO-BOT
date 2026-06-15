import os
import sys
import zipfile
import tarfile
import hashlib
import shutil
import argparse
from pathlib import Path

def is_safe_path(basedir, path, symlinks=True):
    """Vérifie si un chemin d'extraction est sécurisé (anti Zip-Slip)."""
    matchpath = os.path.abspath(os.path.join(basedir, path))
    if not symlinks:
        matchpath = os.path.realpath(matchpath)
    return matchpath.startswith(os.path.abspath(basedir))

def get_file_hash(filepath):
    """Calcule le hash SHA-256 d'un fichier pour éviter les doublons."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def extract_archive(archive_path: Path, extract_dir: Path):
    """Extrait une archive de manière sécurisée."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            for member in zf.infolist():
                if not is_safe_path(extract_dir, member.filename):
                    print(f"[Attention] Fichier suspect ignoré: {member.filename}")
                    continue
                zf.extract(member, extract_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tf:
            for member in tf.getmembers():
                if not is_safe_path(extract_dir, member.name):
                    print(f"[Attention] Fichier suspect ignoré: {member.name}")
                    continue
                tf.extract(member, extract_dir)
    else:
        raise ValueError(f"Format d'archive non supporté: {archive_path.suffix}")

def collect_replays(source_path: str, output_dir: str = "data/replays"):
    """Collecte les replays depuis un dossier ou une archive."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path("data/temp_extract")
    search_dir = source_path
    
    # Étape 1 : Extraction si c'est une archive
    if source_path.is_file():
        print(f"Extraction de l'archive: {source_path}")
        extract_archive(source_path, temp_dir)
        search_dir = temp_dir
    elif not source_path.is_dir():
        print(f"Erreur: La source {source_path} n'existe pas.")
        return

    # Étape 2 : Collecte des replays (.yrp, .yrpx, .bytes)
    yrp_files = []
    for ext in ["*.yrp", "*.yrpx", "*.bytes"]:
        yrp_files.extend(list(search_dir.rglob(ext)))
    print(f"{len(yrp_files)} fichiers de replays trouvés. Traitement en cours...")
    
    # Hash pour déduplication
    existing_hashes = set()
    for existing_file in output_dir.glob("*.yrp"):
        existing_hashes.add(existing_file.stem) # Si le nom est le hash
        
    added_count = 0
    duplicate_count = 0
    
    for yrp in yrp_files:
        file_hash = get_file_hash(yrp)
        if file_hash in existing_hashes:
            duplicate_count += 1
            continue
            
        new_filename = f"{file_hash}.yrp"
        shutil.copy2(yrp, output_dir / new_filename)
        existing_hashes.add(file_hash)
        added_count += 1

    # Nettoyage
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    print("\n=== Résultat de la Collecte ===")
    print(f"Nouveaux replays ajoutés : {added_count}")
    print(f"Doublons ignorés         : {duplicate_count}")
    print(f"Dossier de destination   : {output_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecte et nettoie des replays .yrp depuis une archive ZIP/TAR ou un dossier local.")
    parser.add_argument("source", help="Chemin vers le fichier ZIP, TAR ou le dossier contenant les replays bruts.")
    parser.add_argument("--output", default="data/replays", help="Dossier de destination (défaut: data/replays).")
    
    args = parser.parse_args()
    collect_replays(args.source, args.output)
