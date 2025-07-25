# Importer les librairies
import pandas as pd
import numpy as np
import os
import sys
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compter_mots(texte):
    # Cette fonction compte les mots dans une chaîne de texte.
    if isinstance(texte, str):
        return len(texte.split())
    return 0

def calculate_adjustment_percentage(rate_naturel, rate_synthesized):
    # Calculer le pourcentage d'ajustement entre les taux naturel et synthétisé
    if rate_synthesized != 0:
        return ((rate_naturel - rate_synthesized) / rate_synthesized) * 100
    else:
        return 0

def calculer_metrics(df):
    # Ajouter une colonne pour indiquer si c'est une pause
    df['is_pause'] = df['syntagme'].apply(lambda x: not isinstance(x, str) or x.strip() == '')
    
    # Compter les mots seulement pour les segments non-pauses
    df['nombre_de_mots'] = df.apply(
        lambda row: compter_mots(row['syntagme']) if not row['is_pause'] else 0,
        axis=1
    )

    # Convertir les durées de secondes en minutes
    df['duree_natural_minutes'] = df['duration_syntagme_natural'] / 60
    df['duree_synthesized_minutes'] = df['duration_syntagme_synthesized'] / 60

    # Calculer le taux seulement pour les segments non-pauses
    df['rate_natural'] = df.apply(
        lambda row: row['nombre_de_mots'] / row['duree_natural_minutes'] 
        if not row['is_pause'] and row['duree_natural_minutes'] > 0 else 0, 
        axis=1
    )
    
    df['rate_synthesized'] = df.apply(
        lambda row: row['nombre_de_mots'] / row['duree_synthesized_minutes'] 
        if not row['is_pause'] and row['duree_synthesized_minutes'] > 0 else 0, 
        axis=1
    )

    # Calculer l'ajustement du taux seulement pour les segments non-pauses
    df['rate_adjustment'] = df.apply(
        lambda row: calculate_adjustment_percentage(row['rate_natural'], row['rate_synthesized'])
        if not row['is_pause'] else 0,
        axis=1
    )
    df['rate_adjustment'] = df['rate_adjustment'].replace([np.inf, -np.inf], 0)
    df['rate_adjustment'] = df['rate_adjustment'].clip(-100, 100)  # Limite les valeurs extrêmes à -100, 100 pour éviter les valeurs aberrantes
    
    return df

def calculate_rate(BDD3_dir, BDD4_dir):
    try:
        logging.info(f"Chargement du fichier BDD3: {BDD3_dir}")
        # Charger le Data Frame
        df = pd.read_csv(BDD3_dir)
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            logging.error("Le DataFrame est vide")
            sys.exit(1)
            
        logging.info(f"DataFrame chargé avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Afficher les premières lignes pour vérifier les données
        logging.info(f"Aperçu des données:\n{df.head()}")
        
        # Gestion des valeurs manquantes
        logging.info("Gestion des valeurs manquantes...")
        # Remplacer les NaN dans les colonnes pertinentes par des valeurs par défaut
        if 'syntagme' in df.columns:
            df['syntagme'] = df['syntagme'].fillna("")
            
        if 'duration_syntagme_natural' in df.columns:
            df['duration_syntagme_natural'] = df['duration_syntagme_natural'].fillna(0)
            
        if 'duration_syntagme_synthesized' in df.columns:
            df['duration_syntagme_synthesized'] = df['duration_syntagme_synthesized'].fillna(0)
        
        # Calcul des métriques
        df = calculer_metrics(df)
        
        # Ajouter la colonne rate_ajusté
        df['rate_ajusté'] = df['rate_adjustment']
        
        # Enregistrer le DataFrame final
        logging.info(f"Enregistrement du DataFrame dans: {BDD4_dir}")
        df.to_csv(BDD4_dir, index=False)
        logging.info("Traitement terminé avec succès")
        
    except Exception as e:
        logging.error(f"Erreur lors du traitement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: compute_BDD3_loudness_rate.py",
            "<BDD3_dir>",
            "<BDD4_dir>"
        )
        sys.exit(1)
    # Vérifier l'existence du fichier BDD3
    if not os.path.exists(sys.argv[1]):
        logging.error(f"Le fichier BDD3 n'existe pas: {sys.argv[1]}")
        sys.exit(1)

    # Vérifier que le répertoire de sortie existe, sinon le créer
    output_dir = os.path.dirname(sys.argv[2])
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Répertoire créé: {output_dir}")
        except Exception as e:
            logging.error(f"Impossible de créer le répertoire: {output_dir}. Erreur: {e}")
            sys.exit(1)
    calculate_rate(sys.argv[1], sys.argv[2])