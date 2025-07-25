import os
import csv
from pathlib import Path
import sys


def _read_segments_from_csv2(file_path):
    segments = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            segments.append((row['PhraseID'], row['Text'], float(row['Start']), float(row['End']), float(row['Duration'])))
    return segments


# I wont touch this part to unsure I won't break anything
def needleman_wunsch_alignement(in_needleman_wunsch_microsoft, in_needleman_wunsch, AligNeedlemanWhunch_out):
    """
    Ce script automatise l alignement des segments et des phonemes entre deux ensembles de données. Il utilise l'algorithme de Needleman-Wunsch pour aligner
        les deux ensembles de données et stocke les résultats dans un fichier texte.
    Les entrées sont des fichiers csv, le script les lis avec la fonction _read_segments_from_csv et les aligne avec la fonction needleman_wunsch. Les résultats
        sont stockés dans un fichier texte avec la fonction align_and_store_results.
    avant l alignement le script filtre et traite uniquement les fichiers communs des deux ensembles de données
    Les résultats de l alignement sont écrit dans un fichier txt, les résultats de l'alignement sont stockés séparemment pour les segments et les phonemes.
    """

    def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-1):
        m, n = len(seq1), len(seq2)
        # Création de la matrice de score
        score = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialisation de la première ligne et colonne de la matrice
        for i in range(1, m + 1):
            score[i][0] = i * gap_penalty
        for j in range(1, n + 1):
            score[0][j] = j * gap_penalty

        # Remplissage de la matrice
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                w1, w2 = seq1[i - 1], seq2[j - 1]
                for to_remove in ["ß", "?", ".", ",", ";"]:
                    if len(w1[1]) and to_remove in w1[1]:
                        w1 = (w1[0], w1[1].replace(to_remove, "").lower(), w1[2])
                    if len(w2[1]) and to_remove in w2[1]:
                        w2 = (w2[0], w2[1].replace(to_remove, "").lower(), w2[2])

                if w1[1] == w2[1]:  # Comparaison basée sur la partie 'Text' des tuples
                    match = score[i - 1][j - 1] + match_score
                else:
                    match = score[i - 1][j - 1] + mismatch_score
                delete = score[i - 1][j] + gap_penalty
                insert = score[i][j - 1] + gap_penalty
                score[i][j] = max(match, delete, insert)

        # Traçage de l'alignement optimal
        align1, align2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            W1, W2 = seq1[i - 1], seq2[j - 1]
            w1, w2 = W1, W2
            for to_remove in ["ß", "?", ".", ",", ";"]:
                if len(w1[1]) and to_remove in w1[1]:
                    w1 = (w1[0], w1[1].replace(to_remove, "").lower(), w1[2])
                if len(w2[1]) and to_remove in w2[1]:
                    w2 = (w2[0], w2[1].replace(to_remove, "").lower(), w2[2])
            if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + (match_score if w1[1] == w2[1] else mismatch_score):
                align1.append(W1)
                align2.append(W2)
                i -= 1
                j -= 1
            elif i > 0 and score[i][j] == score[i - 1][j] + gap_penalty:
                align1.append(W1)
                align2.append(('-', '', 0, 0, 0))  # Ajout d'un gap
                i -= 1
            else:
                align1.append(('-', '', 0, 0, 0))  # Ajout d'un gap
                align2.append(W2)
                j -= 1

        return (align1[::-1], align2[::-1])

    # Fonction pour aligner et stocker les résultats dans des fichiers
    def align_and_store_results(folder_synthese, folder_naurelle, output_folder, datatype='Segments'):
        input_folder_synthese = os.path.join(folder_synthese, datatype)
        input_folder_naurelle = os.path.join(folder_naurelle, datatype)
        output_subfolder = os.path.join(output_folder, datatype)
        os.makedirs(output_subfolder, exist_ok=True)

        files_henri = sorted(os.listdir(input_folder_synthese))
        files_pierre = sorted(os.listdir(input_folder_naurelle))
        common_files = set(files_henri).intersection(files_pierre)

        for file_name in sorted(common_files):
            file_path_henri = os.path.join(input_folder_synthese, file_name)
            file_path_pierre = os.path.join(input_folder_naurelle, file_name)
            data1 = _read_segments_from_csv2(file_path_henri)
            data2 = _read_segments_from_csv2(file_path_pierre)
            aligned_data = needleman_wunsch(data1, data2)

            output_file = os.path.join(output_subfolder, f'aligned_{file_name[:-4]}.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                for d1, d2 in zip(aligned_data[0], aligned_data[1]):
                    f.write(f"{d1[0]}: {d1[1]} ({d1[2]}-{d1[3]}, {d1[4]}) || {d2[0]}: {d2[1]} ({d2[2]}-{d2[3]}, {d2[4]})\n")

    # Appeler la fonction pour aligner les segments et stocker les résultats
    align_and_store_results(in_needleman_wunsch_microsoft, in_needleman_wunsch, AligNeedlemanWhunch_out, datatype='Segments')


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python NeedlemanWunschAlignement.py",
            "<in_needleman_wunsch_microsoft>",
            "<in_needleman_wunsch>",
            "<AligNeedlemanWhunch_out>"
        )
        sys.exit(1)

    in_needleman_wunsch_microsoft = sys.argv[1]
    in_needleman_wunsch = sys.argv[2]
    AligNeedlemanWhunch_out = sys.argv[3]
    needleman_wunsch_alignement(in_needleman_wunsch_microsoft, in_needleman_wunsch, AligNeedlemanWhunch_out)


if __name__ == "__main__":
    main()