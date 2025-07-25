import sys
import os
import pandas as pd
import re
import subprocess

# Function to clean words (without removing unnecessary spaces)
def clean_word(word):
    try:
        return word.lower()
    except Exception:
        return ""

# Convert dataframe to dictionary (unused in main, can be removed)
def convert_lexique_to_dict(lexique_df):
    lexicon = {}
    for _, row in lexique_df.iterrows():
        word = clean_word(row['ortho'])
        transcription = row['phon']
        lexicon[word] = transcription
    return lexicon

# Use espeak to generate phonetic transcriptions specifying French language
def generate_phonetic_transcription(word, espeak_path):
    process = subprocess.Popen([espeak_path, '-q', '--ipa', '-v', 'fr', word], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    transcription = stdout.decode('utf-8').strip()
    return transcription

def _concatenate_text_files(text_file_path, transcription_dir):
    """Concatenate all text files in directory into one file"""
    files = os.listdir(transcription_dir)
    concatenated_text = ""
    for filename in files:
        with open(os.path.join(transcription_dir, filename), "r", encoding='utf-8') as file:
            concatenated_text += file.read() + "\n"
    with open(text_file_path, "w", encoding="utf-8") as file:
        file.write(concatenated_text)

def main(transcription_dir, text_file_path, espeak_path, lexique_tsv_path, mfa_lexicon_path, dictionary_path):
    _concatenate_text_files(text_file_path, transcription_dir)

    # Read text file and extract words
    def extract_words_from_text(text_file_path):
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        words = re.findall(r'\b\w+\b', text)
        words = [clean_word(word) for word in words]
        return set(words)

    # Add missing words to the lexicon
    def add_missing_words(mfa_lexicon_path, missing_words):
        with open(mfa_lexicon_path, 'a', encoding='utf-8') as file:
            for word in missing_words:
                transcription = generate_phonetic_transcription(word, espeak_path)
                file.write(f"{word} {transcription}\n")

    # Read existing words in MFA lexicon
    mfa_lexicon = {}
    with open(mfa_lexicon_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if parts:  # Ensure line is not empty
                word = parts[0].lower()
                transcription = ' '.join(parts[1:])
                mfa_lexicon[word] = transcription

    # Extract words from text file
    text_words = extract_words_from_text(text_file_path)
    print(f"Words extracted from text file: {text_words}")

    # Find missing words
    missing_words = [word for word in text_words if word not in mfa_lexicon]
    print(f"Missing words: {missing_words}")

    # Generate transcriptions for missing words and add to lexicon
    add_missing_words(mfa_lexicon_path, missing_words)

    # Reread the MFA lexicon to verify additions
    with open(mfa_lexicon_path, 'r', encoding='utf-8') as file:
        updated_lexicon = file.read()

    print(f"Number of new words added: {len(missing_words)}")

    # Verify that all words in text are in the dictionary
    all_words_in_dictionary = all(word in mfa_lexicon for word in text_words)
    print(f"All words in text are in the dictionary: {all_words_in_dictionary}")

    # Save the enriched dictionary
    with open(dictionary_path, 'w', encoding='utf-8') as file:
        file.write(updated_lexicon)

    print(f"Enriched dictionary saved at: {dictionary_path}")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python enrich_dictionary.py",
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