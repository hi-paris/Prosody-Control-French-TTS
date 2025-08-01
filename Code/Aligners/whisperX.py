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
    device = "cpu"
    print(f"Usage of : {device}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = whisperx.load_model("large-v2", device, compute_type="int8")

    align_model, metadata = whisperx.load_align_model(
        language_code="fr",
        device=device
    )

    return device, model, align_model, metadata


def create_textgrid(aligned_segments, duration, output_file):
    tg = TextGrid(maxTime=duration)
    tier = IntervalTier(name='Mots', minTime=0.0, maxTime=duration)
    M = 0
    for segment in aligned_segments:
        start = segment.get('start', segment.get('start_time', None))
        end = segment.get('end', segment.get('end_time', None))
        word = segment.get('word', segment.get('text', ''))

        if start is None or end is None:
            print("missing segment in time 'start' ou 'end' :")
            print(segment)
            continue

        if start >= end:
            print("Timing of segment is invalide :")
            print(segment)
            continue

        if start != end:
            tier.add(max(M, start), end, word)
            M = end

    tg.append(tier)
    tg.write(output_file)


def process_file(audio_path, transcription_path, device, model, align_model, metadata):
    try:
        # Verify folders and files
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
        if not os.path.exists(transcription_path):
            raise FileNotFoundError(f"Fichier transcription non trouvé: {transcription_path}")

        print(f"Chargement de l'audio: {audio_path}")

        result = model.transcribe(
            str(Path(audio_path).absolute()),
            language="fr"
        )

        result_aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            str(Path(audio_path).absolute()),
            device
        )

        if result_aligned["word_segments"]:
            print(f"Premier segment de mot: {result_aligned['word_segments'][0]}")
            print(f"Clés du segment de mot: {result_aligned['word_segments'][0].keys()}")

        last_segment = result_aligned["segments"][-1]
        duration = last_segment.get('end', last_segment.get('end_time', None))
        if duration is None:
            duration = 0.0  

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
