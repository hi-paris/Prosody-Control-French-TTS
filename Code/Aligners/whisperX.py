import os
import torch
import whisperx
from pathlib import Path
from textgrid import TextGrid, IntervalTier
import sys

if len(sys.argv) != 4:
    print(
        "Usage: python whisperX.py",
        "<audio_dir>",
        "<transcription_dir>",
        "<output_dir>"
    )
    sys.exit(1)

audio_dir = sys.argv[1]
transcription_dir = sys.argv[2]
output_dir = sys.argv[3]


def load_models():
    """
    Charge les modèles WhisperX nécessaires
    """
    # Forcer l'utilisation du CPU
    device = "cpu"
    print(f"Utilisation de : {device}")

    # Désactiver TF32 pour éviter les avertissements
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Chargement du modèle Whisper
    model = whisperx.load_model("large-v2", device, compute_type="int8")

    # Chargement du modèle d'alignement pour le français
    align_model, metadata = whisperx.load_align_model(
        language_code="fr",
        device=device
    )

    return device, model, align_model, metadata


def create_textgrid(aligned_segments, duration, output_file):
    """Crée un fichier TextGrid à partir des segments alignés"""
    tg = TextGrid(maxTime=duration)
    tier = IntervalTier(name='Mots', minTime=0.0, maxTime=duration)
    M = 0
    for segment in aligned_segments:
        # Utiliser 'start' et 'end' pour récupérer les timings
        start = segment.get('start', segment.get('start_time', None))
        end = segment.get('end', segment.get('end_time', None))
        word = segment.get('word', segment.get('text', ''))

        if start is None or end is None:
            print("Segment manquant de temps 'start' ou 'end' :")
            print(segment)
            continue

        if start >= end:
            print("Timing de segment invalide :")
            print(segment)
            continue

        if start != end:
            tier.add(max(M, start), end, word)
            M = end

    tg.append(tier)
    tg.write(output_file)


def process_file(audio_path, transcription_path, device, model, align_model, metadata):
    """Traite un fichier audio avec WhisperX, en fournissant la transcription pour l'alignement."""
    try:
        # Vérification des fichiers
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
        if not os.path.exists(transcription_path):
            raise FileNotFoundError(f"Fichier transcription non trouvé: {transcription_path}")

        print(f"Chargement de l'audio: {audio_path}")

        # Transcription (optionnel si vous avez déjà le texte)
        result = model.transcribe(
            str(Path(audio_path).absolute()),
            language="fr"
        )

        # Alignement avec la transcription
        result_aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            str(Path(audio_path).absolute()),
            device
        )

        # Afficher les clés pour débogage
        if result_aligned["word_segments"]:
            print(f"Premier segment de mot: {result_aligned['word_segments'][0]}")
            print(f"Clés du segment de mot: {result_aligned['word_segments'][0].keys()}")

        # Récupérer la durée à partir du dernier segment
        last_segment = result_aligned["segments"][-1]
        duration = last_segment.get('end', last_segment.get('end_time', None))
        if duration is None:
            duration = 0.0  # Ou gérer ce cas comme nécessaire

        return result_aligned["word_segments"], duration
    except Exception as e:
        print(f"Erreur détaillée: {str(e)}")
        raise


def main():
    os.makedirs(output_dir, exist_ok=True)
    device, model, align_model, metadata = load_models()

    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(audio_dir, audio_file)
            transcription_path = os.path.join(transcription_dir, audio_file.replace('.wav', '.txt'))
            output_path = os.path.join(output_dir, audio_file.replace('.wav', '.TextGrid'))

            print(f"Traitement de : {audio_file}")
            word_segments, duration = process_file(
                audio_path,
                transcription_path,
                device,
                model,
                align_model,
                metadata
            )
            create_textgrid(word_segments, duration, output_path)
            print(f"TextGrid créé : {output_path}")


if __name__ == "__main__":
    main()
