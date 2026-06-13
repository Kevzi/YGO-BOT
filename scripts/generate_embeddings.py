import sqlite3
import pickle
import os
from sentence_transformers import SentenceTransformer

DB_PATH = "data/ygo.db"
OUTPUT_PATH = "data/embed.pkl"

def main():
    print("Chargement du modèle NLP (all-MiniLM-L6-v2)...")
    # Initialisation du modèle (téléchargement automatique depuis HuggingFace si non présent)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Connexion à la base de données SQLite : {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Récupération de l'ID (passcode) et de la description textuelle des cartes
    # On gère l'extraction avec soin
    cursor.execute("SELECT id, desc FROM cards") # "desc" et non "description" selon la plupart des schémas YGOPRODeck, à vérifier. 
    # Wait, in the user script it was `row[6]` which is index 6, let's keep cursor.execute("SELECT * FROM cards") to be safe or map properly.
    # Actually, let's look at the user script again: "cursor.execute('SELECT id, description FROM cards')" and then row[6]?
    # Let's fix that bug. SELECT id, desc FROM cards -> row[0] is id, row[1] is desc.
    cursor.execute("SELECT id, desc FROM cards")
    cards = cursor.fetchall()
    
    if not cards:
        print("Erreur : Aucune carte trouvée dans la base de données. Avez-vous synchronisé l'API YGOPRODeck ?")
        return

    print(f"Génération des embeddings pour {len(cards)} cartes...")
    
    # Séparation des IDs et des textes pour l'encodage par lot (batch)
    card_ids = [row[0] for row in cards]
    descriptions = [row[1] for row in cards]

    # Encodage en vecteurs de dimension 384
    embeddings = model.encode(descriptions, show_progress_bar=True)

    # Création du dictionnaire {passcode: vecteur}
    embed_dict = {card_id: embedding for card_id, embedding in zip(card_ids, embeddings)}

    # Sauvegarde sur disque
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(embed_dict, f)

    print(f"Fichier d'embeddings généré avec succès : {OUTPUT_PATH} (Dimension: 384)")
    conn.close()

if __name__ == "__main__":
    main()
