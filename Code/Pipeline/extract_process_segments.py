import sys
import os
import pandas as pd
import csv
import textgrid

def _extract_segments_and_phonemes(textgrid_file):
    """
    Extrait les segments et les phonèmes à partir d'un fichier TextGrid.
    """
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_file)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier `{textgrid_file}` TextGrid : {e}")
        return [], []

    segments = []
    phonemes = []

    # Suppositions : Premier tier pour les mots, second pour les phonèmes
    word_tier_index = 0

    # Extraction des segments
    for i, interval in enumerate(tg[word_tier_index]):
        segment_id = f"{os.path.basename(textgrid_file).split('.')[0]}_segment_{i + 1}"
        segments.append({
            'PhraseID': segment_id,
            'Start': interval.minTime,
            'End': interval.maxTime,
            'Duration': interval.maxTime - interval.minTime,
            'Text': interval.mark
        })

    return segments, phonemes

def _save_to_csv(data, csv_filename, fieldnames):
    """
    Sauvegarde les données dans un fichier CSV.
    """
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def _process_textgrid_files(base_dir, output_segments_dir):
    """
    Traite les fichiers TextGrid et sauvegarde les segments et phonèmes extraits.
    """
    os.makedirs(output_segments_dir, exist_ok=True)

    for filename in os.listdir(base_dir):
        if filename.endswith('.TextGrid'):
            textgrid_file = os.path.join(base_dir, filename)
            segments, phonemes = _extract_segments_and_phonemes(textgrid_file)

            segment_csv = os.path.join(output_segments_dir, f"{os.path.splitext(filename)[0]}_segments.csv")

            _save_to_csv(segments, segment_csv, ['PhraseID', 'Start', 'End', 'Duration', 'Text'])

def main(textgrid_path, output_segments_dir):
    """
    Ce code permet d'automatiser l'extraction des données des phonémes et des segments à partir des fichiers textgrid puis les sauvegarde en csv.
    ces fichiers csv sont structurés pour inclure les details tels que l identifiant, le temps de début, temps de la fin et la durée et le texte associé.
    Les résultats sont organisés dans des dossiers distincts pour chaque personne avec des sous dossiers pour les segments et les phonémes
    """
    # Création des répertoires de sortie s'ils n'existent pas
    os.makedirs(output_segments_dir, exist_ok=True)
    # Appel de la fonction de traitement
    _process_textgrid_files(textgrid_path, output_segments_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python extract_process_segments.py",
            "<textgrid_path>",
            "<output_segments_dir>",
        )
        sys.exit(1)
    textgrid_path = sys.argv[1]
    output_segments_dir = sys.argv[2]
    main(textgrid_path, output_segments_dir)