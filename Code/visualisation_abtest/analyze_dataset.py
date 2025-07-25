import os
import glob
import re
import wave
import contextlib
import random
from collections import Counter

# --------------------------
# Chargement du tokenizer Roberta (si disponible)
# --------------------------
try:
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
except Exception as e:
    print("Impossible de charger RobertaTokenizerFast : ", e)
    tokenizer = None


def count_tokens(text):
    """
    Compte le nombre de tokens dans le texte en utilisant le tokenizer Roberta (si dispo),
    sinon split sur les espaces.
    """
    if tokenizer:
        try:
            tokens = tokenizer.tokenize(text)
            return len(tokens)
        except Exception as e:
            print(f"Erreur de tokenisation Roberta : {e}")
            return len(text.split())
    else:
        return len(text.split())


def count_sentences(text):
    """
    Découpe le texte en phrases à partir de la ponctuation terminale (point, !, ?).
    """
    sentences = re.split(r'[.!?]+', text)
    # Retirer les chaînes vides dues aux splits
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def get_wav_duration(filename):
    """
    Retourne la durée (en secondes) d'un fichier WAV.
    Ignore les autres extensions.
    """
    if not filename.lower().endswith('.wav'):
        return 0.0
    try:
        with contextlib.closing(wave.open(filename, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Erreur lors de la lecture de {filename}: {e}")
        return 0.0


def analyze_dataset(voice_dir):
    """
    Parcourt récursivement le dossier `voice_dir` en ignorant :

      - tout dossier se terminant par `_microsoft`
      - les sous-dossiers audio contenant "brute" dans leur chemin

    Règles principales :
      - Pour les fichiers audio (wav/mp3) situés dans un dossier `audio` (et pas dans `brute`) :
          * Si le parent du dossier `audio` est `audio_voxpopuli`, on extrait le speaker_id
            depuis le nom de fichier (jusqu'au premier `_`).
          * Sinon, on tente de récupérer la partie avant `_EP` (par ex. `Aznavour` dans
            `Aznavour_EP01`) et on lui assigne un ID aléatoire unique (entre 1 et 100)
            si ce n'est pas déjà fait.
        On compte ces fichiers dans `total_files` et on ajoute la durée (pour .wav).
      - On lit aussi tous les fichiers .txt (toujours en ignorant `_microsoft`) et on
        additionne `total_tokens`, `total_sentences`, ainsi que la distribution des ponctuations.

    Retourne un dict `stats` contenant :
      - total_files : nombre total de fichiers audio (wav + mp3)
      - total_speakers : nombre total de locuteurs (identifiants uniques)
      - total_tokens : nombre total de tokens dans tous les .txt
      - total_sentences : nombre total de phrases dans tous les .txt
      - punctuation_counts : distribution des ponctuations (, . ! ? ; :)
      - duration_hours : durée cumulée des .wav en heures
      - speaker_map : mapping base_name → ID aléatoire (pour debug)
    """

    stats = {
        'total_files': 0,
        'total_speakers': 0,
        'total_tokens': 0,
        'total_sentences': 0,
        'punctuation_counts': Counter(),
        'duration_hours': 0.0,
        'speaker_map': {}
    }

    # Ensemble pour stocker tous les IDs de locuteurs
    speaker_ids = set()

    # Dictionnaire pour les IDs aléatoires assignés aux noms de base (ex. "Aznavour")
    random_ids_assigned = {}

    for root, dirs, files in os.walk(voice_dir):
        # --- 1) Ignorer les dossiers se terminant par _microsoft ---
        if os.path.basename(root).endswith('_microsoft'):
            continue

        # --- 2) Analyser les fichiers texte (.txt) ---
        txt_files = [f for f in files if f.endswith('.txt')]
        for txt_file in txt_files:
            txt_path = os.path.join(root, txt_file)
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier texte {txt_path}: {e}")
                continue

            # Comptage tokens
            stats['total_tokens'] += count_tokens(text_content)

            # Comptage phrases
            stats['total_sentences'] += count_sentences(text_content)

            # Comptage ponctuations
            punctuation_list = [ch for ch in text_content if ch in ',.!?;:']
            stats['punctuation_counts'].update(punctuation_list)

        # --- 3) Analyser les fichiers audio (wav + mp3) ---
        # On ne traite que si "audio" est dans le chemin ET "brute" n'y est pas
        if "audio" in root and "brute" not in root:
            parent_dir = os.path.basename(os.path.dirname(root))

            # Récupérer tous les wav + mp3
            audio_files = glob.glob(os.path.join(root, '*.wav')) \
                + glob.glob(os.path.join(root, '*.mp3'))

            # On compte tous ces fichiers pour total_files
            stats['total_files'] += len(audio_files)

            if parent_dir == "audio_voxpopuli":
                # Cas spécial : extraire l'ID au début du nom de fichier
                for audio_file in audio_files:
                    filename = os.path.basename(audio_file)
                    # speaker_id = partie avant le premier underscore
                    spk_id = filename.split('_')[0]
                    speaker_ids.add(spk_id)

                    # Calcul de la durée si .wav
                    stats['duration_hours'] += get_wav_duration(audio_file) / 3600.0

            else:
                # Cas général : "Aznavour_EP01", "Blabla_EP02", etc.
                match = re.match(r'(.+)_EP\d+', parent_dir, re.IGNORECASE)
                if match:
                    base_name = match.group(1)
                else:
                    # Si pas de correspondance, on prend le dossier complet
                    base_name = parent_dir

                # Attribuer un ID aléatoire si pas déjà fait
                if base_name not in random_ids_assigned:
                    random_ids_assigned[base_name] = random.randint(1, 100)

                assigned_id = random_ids_assigned[base_name]
                speaker_ids.add(assigned_id)

                # Durée cumulée pour les .wav
                for audio_file in audio_files:
                    stats['duration_hours'] += get_wav_duration(audio_file) / 3600.0

    # Nombre total de locuteurs
    stats['total_speakers'] = len(speaker_ids)

    # On remplit speaker_map (utile si on veut connaître l'ID aléatoire
    # associé à chaque base_name). Pour audio_voxpopuli, on n'a pas de
    # mapping « base_name → speaker_id » fixe, car chaque fichier peut
    # avoir un speaker_id différent.
    stats['speaker_map'].update(random_ids_assigned)

    return stats


# =======================
# Exemple d’utilisation
# =======================
if __name__ == "__main__":
    voice_directory = "/tsi/hi-paris/tts/voice"
    stats = analyze_dataset(voice_directory)

    print("=== Statistiques globales ===")
    print(f"Total Files (wav+mp3)           : {stats['total_files']}")
    print(f"Total Speakers                  : {stats['total_speakers']}")
    print(f"Total Tokens (dans .txt)        : {stats['total_tokens']}")
    print(f"Total Sentences (dans .txt)     : {stats['total_sentences']}")
    print(f"Total Duration (heures .wav)    : {round(stats['duration_hours'], 2)}")

    print("\n=== Distribution des ponctuations (dans .txt) ===")
    for p, count in stats['punctuation_counts'].items():
        print(f"  '{p}' : {count}")

    print("\n=== Mapping pour les dossiers hors audio_voxpopuli ===")
    for base_name, spk_id in stats['speaker_map'].items():
        print(f"Base '{base_name}' -> ID {spk_id}")
