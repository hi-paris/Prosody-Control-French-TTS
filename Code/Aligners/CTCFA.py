# Import des librairies
import os
from pathlib import Path
import subprocess
import re
from textgrid import TextGrid, IntervalTier
import sys

base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute().__str__()
Data_dir = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute().__str__(), "Data")


# Configuration
if len(sys.argv) != 7:
    print(
        "Usage: python CTCFA.py",
        "<audio_dir>",
        "<transcription_dir>",
        "<output_dir>",
        "<language>",
        "<star_frequency>",
        "<romanize>"
    )
    sys.exit(1)
audio_dir = sys.argv[1]
transcription_dir = sys.argv[2]
output_dir = sys.argv[3]
language = sys.argv[4]
star_frequency = sys.argv[5]
romanize = (sys.argv[6] == "True")

"""
This Python script performs CTC (Connectionist Temporal Classification) forced alignment on audio files using their transcriptions.
It first preprocesses the text to remove punctuation and non-alphanumeric symbols.
Then it automatically converts the alignment results into TextGrid files, creating a precise temporal representation of spoken words in each audio file.
"""


def preprocess_text(text):
    text = re.sub(r"[.,!?;:\"\\(\)\-\_""«»]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def txt_to_textgrid(input_file, output_file):
    tg = TextGrid()
    tier = IntervalTier(name='Mots', minTime=0.0)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(':')
                if len(parts) == 2:
                    time_part, word = parts
                    start_end = time_part.strip().split('-')
                    if len(start_end) == 2:
                        try:
                            start_time = float(start_end[0].strip())
                            end_time = float(start_end[1].strip())
                            if start_time == end_time:
                                end_time += 0.005
                            label = word.strip()
                            tier.add(start_time, end_time, label)
                        except ValueError:
                            print(f"Time conversion error in line: {line}")
                    else:
                        print(f"Incorrect time format in line: {line}")
                else:
                    print(f"Incorrect line format: {line}")

    tg.append(tier)
    tg.write(output_file)


def process_files():
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            audio_path = os.path.join(audio_dir, file_name)
            transcription_path = os.path.join(transcription_dir, file_name.replace('.wav', '.txt'))

            if not os.path.exists(transcription_path):
                print(f"Missing transcription file: {transcription_path}")
                continue

            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = f.read()

            preprocessed_transcription = preprocess_text(transcription)

            temp_transcription_path = transcription_path.replace('.txt', '_clean.txt')
            with open(temp_transcription_path, 'w', encoding='utf-8') as f:
                f.write(preprocessed_transcription)

            command = (
                f'ctc-forced-aligner --audio_path "{audio_path}" '
                f'--text_path "{temp_transcription_path}" '
                f'--language "{language}" --star_frequency "{star_frequency}"'
            )

            if romanize:
                command += " --romanize"

            subprocess.run(command, shell=True)

            os.remove(temp_transcription_path)

            # Conversion en TextGrid
            ctc_output_file = audio_path.replace('.wav', '.txt')
            textgrid_output_file = os.path.join(output_dir, file_name.replace('.wav', '.TextGrid'))
            txt_to_textgrid(ctc_output_file, textgrid_output_file)
            print(f"Processed file : {audio_path} -> {textgrid_output_file}")


if __name__ == '__main__':
    process_files()
