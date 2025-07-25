# Code/Aligners/enrichir_dictionnaire.py

import sys
import os
import pandas as pd
import re
import subprocess

# Fonction pour nettoyer les mots (sans supprimer les espaces inutiles)
def clean_word(word):
    try:
        return word.lower()
    except Exception as e:
        e
        return ""

# Convertir le dataframe en dictionnaire
def convert_lexique_to_dict(lexique_df):
    lexicon = {}
    for _, row in lexique_df.iterrows():
        word = clean_word(row['ortho'])
        transcription = row['phon']
        lexicon[word] = transcription
    return lexicon

# Utiliser espeak pour générer les transcriptions phonétiques en spécifiant la langue française
def generate_phonetic_transcription(word, espeak_path):  # Modified to accept espeak_path as parameter
    process = subprocess.Popen([espeak_path, '-q', '--ipa', '-v', 'fr', word], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    transcription = stdout.decode('utf-8').strip()
    return transcription

def _concat_text(text_file_path, transcription_dir):
    files = os.listdir(transcription_dir)
    text_concat = ""
    for n in files:
        with open(os.path.join(transcription_dir, n), "r", encoding='utf-8') as file:
            text_concat += file.read() + "\n"
    with open(text_file_path, "w", encoding="utf-8") as file:
        file.write(text_concat)

def main(transcription_dir, text_file_path, espeak_path, lexique_tsv_path, mfa_lexicon_path, dictionary_path):
    _concat_text(text_file_path, transcription_dir)

    # Lire le fichier texte et extraire les mots
    def extract_words_from_text(text_file_path):
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        words = re.findall(r'\b\w+\b', text)
        words = [clean_word(word) for word in words]
        return set(words)

    # Ajouter les mots manquants au lexique
    def add_missing_words(mfa_lexicon_path, missing_words):
        with open(mfa_lexicon_path, 'a', encoding='utf-8') as file:
            for word in missing_words:
                transcription = generate_phonetic_transcription(word, espeak_path)
                file.write(f"{word} {transcription}\n")

    # Lire les mots existants dans le lexique MFA
    mfa_lexicon = {}
    with open(mfa_lexicon_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0].lower()
            transcription = ' '.join(parts[1:])
            mfa_lexicon[word] = transcription

    # Extraire les mots du fichier texte
    text_words = extract_words_from_text(text_file_path)
    print(f"Mots extraits du fichier texte : {text_words}")

    # Trouver les mots manquants
    missing_words = [word for word in text_words if word not in mfa_lexicon]
    print(f"Mots manquants : {missing_words}")

    # Générer les transcriptions pour les mots manquants et les ajouter au lexique
    add_missing_words(mfa_lexicon_path, missing_words)

    # Relire les mots existants dans le lexique MFA pour vérifier l'ajout
    with open(mfa_lexicon_path, 'r', encoding='utf-8') as file:
        updated_lexicon = file.read()

    print(f"Nombre de nouveaux mots ajoutés : {len(missing_words)}")

    # Verify that all words in text are in the dictionary
    all_words_in_dictionary = all(word in mfa_lexicon for word in text_words)
    print(f"All words in text are in the dictionary: {all_words_in_dictionary}")

    # sauvegarde du nouveau dictionnaire enrichi
    with open(dictionary_path, 'w', encoding='utf-8') as file:
        file.write(updated_lexicon)

    print(f"Dictionnaire enrichi sauvegardé à : {dictionary_path}")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python enrichir_dictionnaire.py",
            "<transcription_dir>",
            "<text_concat_file_path>",
            "<espeak_path>",
            "<lexique_tsv_path>",
            "<mfa_lexicon_path>",
            "<dictionary_path>"
        )
        sys.exit(1)

    transcription_dir = sys.argv[1]
    text_file_path = sys.argv[2]
    espeak_path = sys.argv[3]
    lexique_tsv_path = sys.argv[4]
    mfa_lexicon_path = sys.argv[5]
    dictionary_path = sys.argv[6]

    main(
        transcription_dir,
        text_file_path,
        espeak_path,
        lexique_tsv_path,
        mfa_lexicon_path,
        dictionary_path
    )