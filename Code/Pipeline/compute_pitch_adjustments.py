import os
import sys
import pandas as pd
import re
import parselmouth
import statistics
import numpy as np

def _calculate_pitch_means(df):
    # Filtrer les lignes où les colonnes 'average_natural_pitch_per_sentence' et 'average_synthesized_pitch_per_sentence' ne sont pas nulles
    active_natural_phrases = df[df['average_natural_pitch_per_sentence'] != 0]
    active_synthesized_phrases = df[df['average_synthesized_pitch_per_sentence'] != 0]

    # Vérifier qu'il y a suffisamment de phrases actives, sinon ajuster le nombre pour les phrases naturelles et synthétisées
    num_natural_phrases = min(50, len(active_natural_phrases))
    num_synthesized_phrases = min(50, len(active_synthesized_phrases))

    # Calculer la moyenne pour les 5 premières phrases actives (non nulles) pour le pitch naturel
    if num_natural_phrases > 0:
        mean_natural_pitch = active_natural_phrases['average_natural_pitch_per_sentence'].head(num_natural_phrases).mean()
    else:
        # Retourner None si aucune phrase active naturelle n'est disponible
        mean_natural_pitch = None

    # Calculer la moyenne pour les 5 premières phrases actives (non nulles) pour le pitch synthétisé
    if num_synthesized_phrases > 0:
        mean_synthesized_pitch = active_synthesized_phrases['average_synthesized_pitch_per_sentence'].head(num_synthesized_phrases).mean()
    else:
        # Retourner None si aucune phrase active synthétisée n'est disponible
        mean_synthesized_pitch = None

    return mean_natural_pitch, mean_synthesized_pitch

def _complete_audio_paths(df, base_path_natural, base_path_synthesized):
    # Ajouter le chemin complet pour les chemins audio naturels
    df['natural_audio_path'] = [os.path.join(base_path_natural , p) for p in df['natural_audio_path'].astype("str")]
    # Ajouter le chemin complet pour les chemins audio synthétisés
    df['synthesized_audio_path'] = [os.path.join(base_path_synthesized, p) for p in df['synthesized_audio_path'].astype("str")]
    return df

def _extract_time_info(df):
    # Définir le modèle de regex pour extraire le texte et les informations de temps
    regex_pattern = r'(.+?):\s*(.*)\s*\((\d+\.\d+)-(\d+\.\d+),\s*(\d+\.\d+)\)'

    # Extraire le texte et les informations de temps pour la première colonne (synthesized) pour le texte
    df['Text'] = df['synthesized'].str.extract(regex_pattern)[1].fillna(' ')

    # Extraire les informations de temps pour chaque colonne
    for col in ['synthesized', 'natural']:
        extracted_data = df[col].str.extract(regex_pattern)
        df[f'begin_{col}'] = extracted_data[2].astype(float)
        df[f'end_{col}'] = extracted_data[3].astype(float)
        df[f'duration_{col}'] = extracted_data[4].astype(float)

    # Fonction auxiliaire pour déterminer si un segment est une pause
    def is_pause(segment):
        # Vérifier si le texte avant les parenthèses est vide après les deux points
        return not re.search(r':\s*\w', segment)

    # Ajouter deux nouvelles colonnes pour les durées des pauses synthétisées et naturelles
    df['duree_pause_synthesized'] = df.apply(lambda x: x['duration_synthesized'] if is_pause(x['synthesized']) else 0, axis=1)
    df['duree_pause_natural'] = df.apply(lambda x: x['duration_natural'] if is_pause(x['natural']) else 0, axis=1)

    # Mettre à jour les durées des segments où il y a des pauses
    df['duration_synthesized'] = df.apply(lambda x: 0 if is_pause(x['synthesized']) else x['duration_synthesized'], axis=1)
    df['duration_natural'] = df.apply(lambda x: 0 if is_pause(x['natural']) else x['duration_natural'], axis=1)

    return df

def construct_syntagmes(df):
    # Préparation des colonnes pour le nouveau DataFrame des syntagmes
    syntagmes_data = {
        'syntagme': [],
        'begin_syntagme_synthesized': [],
        'end_syntagme_synthesized': [],
        'duration_syntagme_synthesized': [],
        'begin_syntagme_natural': [],
        'end_syntagme_natural': [],
        'duration_syntagme_natural': [],
        'duration_pause_syntagme_synthesized': [],
        'duration_pause_syntagme_natural': [],
        'natural_syntagme_audio_path': [],
        'synthesized_syntagme_audio_path': []
    }

    # Identifier les indices des pauses (où le texte est vide ou manquant)
    pause_indices = df[df['Text'].isna() | (df['Text'].str.strip() == '')].index.tolist()

    start_index = 0

    # Parcourir les intervalles entre les pauses pour former les syntagmes
    for end_index in pause_indices + [len(df)]:
        if start_index < end_index:  # S'assurer que l'intervalle n'est pas vide
            syntagme_df = df.iloc[start_index:end_index]

            # Collecter et stocker les données du syntagme
            syntagmes_data['syntagme'].append(' '.join(syntagme_df['Text'].dropna().str.strip().tolist()))
            syntagmes_data['begin_syntagme_synthesized'].append(syntagme_df.iloc[0]['begin_synthesized'])
            syntagmes_data['end_syntagme_synthesized'].append(syntagme_df.iloc[-1]['end_synthesized'])
            syntagmes_data['duration_syntagme_synthesized'].append(syntagme_df['duration_synthesized'].sum())
            syntagmes_data['begin_syntagme_natural'].append(syntagme_df.iloc[0]['begin_natural'])
            syntagmes_data['end_syntagme_natural'].append(syntagme_df.iloc[-1]['end_natural'])
            syntagmes_data['duration_syntagme_natural'].append(syntagme_df['duration_natural'].sum())
            syntagmes_data['natural_syntagme_audio_path'].append(syntagme_df.iloc[0]['natural_audio_path'])
            syntagmes_data['synthesized_syntagme_audio_path'].append(syntagme_df.iloc[0]['synthesized_audio_path'])
            syntagmes_data['duration_pause_syntagme_synthesized'].append(0)  # Pas de pause avant un syntagme avec du texte
            syntagmes_data['duration_pause_syntagme_natural'].append(0)

        # Si le syntagme suivant est une pause
        if end_index < len(df):
            pause_duration_synthesized = df.iloc[end_index]['duree_pause_synthesized']
            pause_duration_natural = df.iloc[end_index]['duree_pause_natural']

            syntagmes_data['syntagme'].append('')  # Marquer le syntagme vide
            syntagmes_data['begin_syntagme_synthesized'].append(df.iloc[end_index]['begin_synthesized'])
            syntagmes_data['end_syntagme_synthesized'].append(df.iloc[end_index]['end_synthesized'])
            syntagmes_data['duration_syntagme_synthesized'].append(0)  # Durée du syntagme vide
            syntagmes_data['begin_syntagme_natural'].append(df.iloc[end_index]['begin_natural'])
            syntagmes_data['end_syntagme_natural'].append(df.iloc[end_index]['end_natural'])
            syntagmes_data['duration_syntagme_natural'].append(0)  # Durée du syntagme vide
            syntagmes_data['natural_syntagme_audio_path'].append(df.iloc[end_index]['natural_audio_path'])
            syntagmes_data['synthesized_syntagme_audio_path'].append(df.iloc[end_index]['synthesized_audio_path'])
            syntagmes_data['duration_pause_syntagme_synthesized'].append(pause_duration_synthesized)
            syntagmes_data['duration_pause_syntagme_natural'].append(pause_duration_natural)
        # Définir le début du prochain syntagme après la pause actuelle
        start_index = end_index + 1

    return pd.DataFrame(syntagmes_data)


def compute_pitch_adjustments(BDD1_dir, audio_dir, audio_dir_microsoft, transcription_dir, transcription_dir_microsoft, BDD2_dir):
    """
    Dans le script suivant le but est de mesurer le pitch, ici je travaille au niveau des mots et des audios, plus tard, on ira au nivau des syntagmes
    buggée, a corrige
    """
    file_path = BDD1_dir
    df = pd.read_csv(file_path)

    # On ajoute les chemins des fichiers audio
    regex_pattern = r"(.*?)(?:_segment)"
    df['natural_audio_path'] = df['natural'].str.extract(regex_pattern)
    df['synthesized_audio_path'] = df['synthesized'].str.extract(regex_pattern)

    df['natural_audio_path'] += '.wav'
    df['synthesized_audio_path'] += '.wav'

    # Définir les chemins de base pour les fichiers audio naturels et synthétisés
    base_path_natural_audio = audio_dir
    base_path_synthesized_audio = audio_dir_microsoft

    # Appeler la fonction pour compléter les chemins dans le DataFrame
    df = _complete_audio_paths(df, base_path_natural_audio, base_path_synthesized_audio)

    # Application de la fonction pour extraire et traiter les informations
    df = _extract_time_info(df)

    # Je traite les nan dans duree_pause_natural et duree_pause_synthesized qui correspondent aux pauses artificielles de 0.01s
    df['duree_pause_natural'].fillna(0.01, inplace=True)
    df['duree_pause_synthesized'].fillna(0.01, inplace=True)

    # Avant de constuire le syntagme je vais supprimer les pauses successives

    # Application de la fonction pour construire les syntagmes
    df = construct_syntagmes(df)

    # Calculer le pitch suivant les syntagmes
    def calculate_pitch_segment(audio_path, start_time, end_time):
        if not os.path.exists(audio_path) or audio_path.endswith('nan'):
            print(f"Traitement en cours: {audio_path}")
            return 0  # Retourner 0 si le fichier n'existe pas ou si le chemin se termine par 'nan'
        try:
            # Charger le fichier audio
            sound = parselmouth.Sound(audio_path)

            # Extraire le segment spécifié
            
            # verifier que les temps sont valides
            if start_time>=end_time or start_time < 0 or end_time > sound.get_total_duration():
                print(f"Temps invalides pour {audio_path}: start={start_time}, end={end_time}, durée={sound.get_total_duration()}")
                return 0
            # extraire le segment specifié

            try:
                segment = sound.extract_part(from_time=start_time, to_time=end_time)
                # Calculer le pitch du segment
            except Exception as e:
                print(f"Erreur lors de l'extraction pour {audio_path}: {e}")
                return 0
        
            # essayons differents parametres de pitch ; 
            for pitch_floor in [75, 100, 150, 200]:
                try:
                    pitch = segment.to_pitch(pitch_floor=pitch_floor)
                    pitch_values = pitch.selected_array['frequency']
                    pitch_values = pitch_values[pitch_values != 0]  # exclure les valeurs de pitch nulles (non voisé)
                
                    if len(pitch_values) > 0:
                        mean_pitch = statistics.geometric_mean(pitch_values)
                        return mean_pitch
                
                except Exception as e:
                    continue
            print(f"Calcul du pitch pour {audio_path}")
            return 0
        
        except Exception as e:
            print(f"calcul des pitch pour {audio_path} avc tous les pitch_floor essayés")
            return 0

    # calculer le pitch pour chaque syntagme de la voix naturelle
    df['natural_pitch_syntagme'] = df.apply(
        lambda row: calculate_pitch_segment(
            row['natural_syntagme_audio_path'],
            row['begin_syntagme_natural'],
            row['end_syntagme_natural']
        ), axis=1
    )

    # calculer le pitch pour chaque syntagme de la voix de synthese
    df['synthesized_pitch_syntagme'] = df.apply(
        lambda row: calculate_pitch_segment(
            row['synthesized_syntagme_audio_path'],
            row['begin_syntagme_synthesized'],
            row['end_syntagme_synthesized']
        ), axis=1
    )

    # mettre les natural_pitch_syntagme et synthesized_pitch_syntagme à zero si le syntagme est vide
    df.loc[df['syntagme'].str.strip() == '', ['natural_pitch_syntagme', 'synthesized_pitch_syntagme']] = 0
    print(df[['syntagme', 'natural_pitch_syntagme', 'synthesized_pitch_syntagme']].head(50))

    def calculate_pitch_adjustment(df):
        """
        Ajoute une colonne 'pitch_adjustment' au DataFrame donné, calculée en fonction
        du pitch naturel et synthétisé. Si le pitch synthétisé est zéro, l'ajustement est fixé à zéro.

        Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'natural_pitch' et 'synthesized_pitch'.

        Returns:
        pd.DataFrame: Le DataFrame modifié avec la nouvelle colonne 'pitch_adjustment'.
        """
        
        # Ajouter une colonne pour indiquer si c'est une pause
        df['is_pause'] = df['syntagme'].apply(lambda x: not isinstance(x, str) or x.strip() == '')
        df['pitch_adjustment'] = df.apply(
        lambda row: ((row['natural_pitch_syntagme'] - row['synthesized_pitch_syntagme']) / row['synthesized_pitch_syntagme']) * 100
        if not row['is_pause'] and row['synthesized_pitch_syntagme'] != 0 else 0,
        axis=1
    )
        df['pitch_adjustment'] = df['pitch_adjustment'].replace([np.inf, -np.inf], 0)
        df['pitch_adjustment'] = df['pitch_adjustment'].clip(-100, 100)  # Limitate the change to -100% and 100% to skip extreme cases when pitch is inf or -inf, it doesnt happen normallly, but just in case
    
        return df

    calculate_pitch_adjustment(df)

    def calculate_average_pitch(df):
        # Filtrer les données pour ne considérer que les 104 premiers fichiers audio uniques dans les deux colonnes
        unique_natural_paths = df['natural_syntagme_audio_path'].dropna().unique()[:]
        unique_synthesized_paths = df['synthesized_syntagme_audio_path'].dropna().unique()[:]

        # Dictionnaire pour stocker les résultats pour les deux types de fichiers audio
        average_pitches = {
            'natural': {},
            'synthesized': {}
        }

        # Itérer sur chaque chemin audio unique pour les fichiers naturels
        for path in unique_natural_paths:
            # Filtrer le DataFrame pour obtenir seulement les entrées avec ce chemin audio naturel
            audio_data = df[df['natural_syntagme_audio_path'] == path]

            # Calculer la moyenne des valeurs de pitch naturelles qui ne sont pas nulles
            non_zero_pitches = audio_data[audio_data['natural_pitch_syntagme'] != 0]['natural_pitch_syntagme']
            average_pitch_natural = non_zero_pitches.mean() if not non_zero_pitches.empty else 0

            # Stocker le résultat dans le dictionnaire pour les fichiers naturels
            average_pitches['natural'][path] = average_pitch_natural

            # Mettre à jour le DataFrame original en attribuant la moyenne calculée à chaque segment correspondant
            df.loc[df['natural_syntagme_audio_path'] == path, ['average_natural_pitch', 'average_natural_pitch_per_sentence']] = average_pitch_natural

        # Itérer sur chaque chemin audio unique pour les fichiers synthétisés
        for path in unique_synthesized_paths:
            # Filtrer le DataFrame pour obtenir seulement les entrées avec ce chemin audio synthétisé
            audio_data = df[df['synthesized_syntagme_audio_path'] == path]

            # Calculer la moyenne des valeurs de pitch synthétisées qui ne sont pas nulles
            non_zero_pitches = audio_data[audio_data['synthesized_pitch_syntagme'] != 0]['synthesized_pitch_syntagme']
            average_pitch_synthesized = non_zero_pitches.mean() if not non_zero_pitches.empty else 0

            # Stocker le résultat dans le dictionnaire pour les fichiers synthétisés
            average_pitches['synthesized'][path] = average_pitch_synthesized

            # Mettre à jour le DataFrame pour la moyenne synthétisée par syntagme
            df.loc[df['synthesized_syntagme_audio_path'] == path, 'average_synthesized_pitch_per_sentence'] = average_pitch_synthesized

        return df, average_pitches

    # Appeler la fonction et passer le DataFrame
    df, average_pitches = calculate_average_pitch(df)
    print("Average Natural Pitches:", average_pitches['natural'])
    print("Average Synthesized Pitches:", average_pitches['synthesized'])

    df[['syntagme', 'natural_pitch_syntagme', 'synthesized_pitch_syntagme', 'average_natural_pitch_per_sentence', 'average_synthesized_pitch_per_sentence']]

    df[['syntagme', 'natural_pitch_syntagme', 'average_natural_pitch_per_sentence']]

    # Appeler la fonction et passer le DataFrame
    mean_natural_pitch, mean_synthesized_pitch = _calculate_pitch_means(df)
    print("Moyenne du pitch naturel pour les 5 premières phrases actives :", mean_natural_pitch)
    print("Moyenne du pitch synthétisé pour les 5 premières phrases actives :", mean_synthesized_pitch)
    pitch_nat_moyen = mean_natural_pitch

    # On calcule le pitch relatif de modification pour la voix naturelle et la voix de synthese
    # ? 127.01191054610585 (not directly addapted to the voice... A changer en urgence!!!)
    df['adjustment_synthesized'] = df.apply(
    lambda row: row['synthesized_pitch_syntagme'] / mean_synthesized_pitch 
    if not row['is_pause'] and row['synthesized_pitch_syntagme'] != 0 and mean_synthesized_pitch != 0 else 0,
    axis=1
)
    df['adjustment_natural'] = df.apply(
    lambda row: row['natural_pitch_syntagme'] / pitch_nat_moyen 
    if not row['is_pause'] and row['natural_pitch_syntagme'] != 0 and pitch_nat_moyen != 0 else 0,
    axis=1
)

    df['relative_pitch_modification'] = df.apply(
    lambda row: row['adjustment_synthesized'] / row['adjustment_natural'] 
    if not row['is_pause'] and row['adjustment_natural'] != 0 else 0,
    axis=1
)

    df['pourcentage_relative_pitch_modification'] = df['relative_pitch_modification'].apply(lambda x: (x - 1) * 100 if x != 0 else 0)

    df[['syntagme', 'natural_pitch_syntagme', 'synthesized_pitch_syntagme', 'pourcentage_relative_pitch_modification']]

    # Enregistrer le DataFrame avec les informations de pitch et de sonie
    df.to_csv(BDD2_dir, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python compute_BDD1.py",
            "<BDD1_dir>",
            "<audio_dir>",
            "<audio_dir_microsoft>",
            "<transcription_dir>",
            "<transcription_dir_microsoft>",
            "<BDD2_dir>"
        )
        sys.exit(1)
    compute_pitch_adjustments(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])