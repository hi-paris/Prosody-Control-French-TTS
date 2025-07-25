# On importe les bibliothèques nécessaires
import pandas as pd
import numpy as np
import os
from pydub import AudioSegment
import sys

def _calculate_loudness(audio_file_path, start, end):
    # Vérifier si le chemin du fichier est NaN ou inexistant
    if pd.isna(audio_file_path) or not os.path.isfile(audio_file_path):
        return 0

    # Charger le fichier audio
    audio = AudioSegment.from_file(audio_file_path)[start * 1000:end * 1000]

    # Convertir en tableau de valeurs numériques
    audio_seg = audio.get_array_of_samples()
    samples = np.array(audio_seg)
    S = np.array(samples) ** 2
    # Calculer la sonie (RMS)
    rms = np.sqrt(np.abs(np.mean(S)))

    # Convertir la sonie en dB
    loudness = 20 * np.log10(rms)
    return loudness

def _calculate_coeff_adjustment(df):
    # Ajouter une colonne pour indiquer si c'est une pause
    df['is_pause'] = df['syntagme'].apply(lambda x: not isinstance(x, str) or x.strip() == '')

    # Calculer la sonie seulement pour les segments non-pauses
    df['natural_loudness'] = df.apply(
        lambda row: _calculate_loudness(row['natural_syntagme_audio_path'], 
                                        row['begin_syntagme_natural'], 
                                        row['end_syntagme_natural']) 
        if isinstance(row['syntagme'], str) and row['syntagme'].strip() != '' 
           and not pd.isna(row['begin_syntagme_natural'])
           and not pd.isna(row['end_syntagme_natural'])
        else 0,
        axis=1
    )

    df['synthesized_loudness'] = df.apply(
        lambda row: _calculate_loudness(row['synthesized_syntagme_audio_path'], 
                                        row['begin_syntagme_synthesized'], 
                                        row['end_syntagme_synthesized']) 
        if isinstance(row['syntagme'], str) and row['syntagme'].strip() != '' 
           and not pd.isna(row['begin_syntagme_synthesized'])
           and not pd.isna(row['end_syntagme_synthesized'])
        else 0,
        axis=1
    )

    EPSILON = 1e-6  # seuil minimal pour éviter les divisions par zéro

    # Calculer le coefficient d'ajustement en limitant les valeurs extrêmes
    df['loudness_adjustment'] = df.apply(
        lambda row: np.clip(
            ((row['natural_loudness'] - row['synthesized_loudness']) / row['synthesized_loudness'] * 100),
            -20, 20
        ) if isinstance(row['syntagme'], str) and row['syntagme'].strip() != ''
          and abs(row['synthesized_loudness']) > EPSILON
          else 0,
        axis=1
    )

def calculate_loudness_adjustment(BDD2_dir, BDD3_dir):
    """
    Dans cette partie, nous calculons la loudness (volume) des fichiers audio naturels et synthétisés,
    et nous calculons le coefficient d'ajustement de la loudness pour chaque segment en limitant les valeurs extrêmes.
    """
    # Charger le DataFrame
    df = pd.read_csv(BDD2_dir)

    # Calculer les coefficients d'ajustement
    _calculate_coeff_adjustment(df)

    # Remplacer les nan de la colonne syntagme par ' '
    df['syntagme'] = df['syntagme'].replace(np.nan, '')

    # Enregistrer le DataFrame modifié
    df.to_csv(BDD3_dir, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: compute_BDD2_loudness.py",
            "<BDD2_dir>",
            "<BDD3_dir>"
        )
        sys.exit(1)
    calculate_loudness_adjustment(sys.argv[1], sys.argv[2])