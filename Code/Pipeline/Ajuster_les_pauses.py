import pandas as pd
import os
import glob
import re
from pathlib import Path
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _numeric_sort_key(s):
    numeric_part = re.findall(r'\d+', s)
    return int(numeric_part[0]) if numeric_part else 0


def _lire_alignements_dossier(dossier):
    pattern = os.path.join(dossier, '*.txt')
    file_paths = sorted(glob.glob(pattern), key=_numeric_sort_key)

    if not file_paths:
        logger.warning(f"Aucun fichier d'alignement trouvé dans {dossier}.")
        return pd.DataFrame()

    all_data = []
    file_paths = sorted(file_paths, key=(lambda x: int(x.split("ph")[1].split("_")[0])))
    for filepath in file_paths:
        try:
            df = pd.read_csv(filepath, delimiter=r'\s*\|\|\s*', engine='python', header=None, names=['synthesized', 'natural'])
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture du fichier {filepath}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def _ajouter_pauses(df):
    gap_pattern = '-:  (0-0, 0)'
    for index, row in df.iterrows():
        if gap_pattern in row['synthesized']:
            df.at[index, 'synthesized'] = ' (0.01)'
        if gap_pattern in row['natural']:
            df.at[index, 'natural'] = ' (0.01)'
    return df


def add_breaks(needleman_wunsch_results, BDD1_dir):
    # Vérifications initiales
    if not os.path.isdir(needleman_wunsch_results):
        logger.error(f"{needleman_wunsch_results} n'est pas un dossier.")
        sys.exit(1)

    chemin_dossier = os.path.join(needleman_wunsch_results, 'Segments')
    if not os.path.exists(chemin_dossier):
        logger.error(f"Le dossier 'Segments' n'existe pas dans {needleman_wunsch_results}.")
        sys.exit(1)

    df_total = _lire_alignements_dossier(chemin_dossier)
    if df_total.empty:
        logger.warning("Aucun fichier valide n'a été trouvé dans le dossier des segments.")
        return

    df_ajuste = _ajouter_pauses(df_total)

    output_dir = Path(BDD1_dir).parent
    if not output_dir.exists():
        logger.info(f"Création du répertoire parent pour {BDD1_dir}.")
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df_ajuste.to_csv(BDD1_dir, index=False)
        logger.info(f"Les pauses ajustées ont été sauvegardées dans {BDD1_dir}.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier {BDD1_dir}: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python Ajuster_les_pauses.py <AligNeedlemanWhunch_out> <BDD1_dir>")
        sys.exit(1)

    AligNeedlemanWhunch_out = sys.argv[1]
    BDD1_dir = sys.argv[2]
    add_breaks(AligNeedlemanWhunch_out, BDD1_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue : {e}")