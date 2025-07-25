import os
from pydub import AudioSegment
from pathlib import Path
import sys

def main(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if input_file.suffix.lower() == ".wav":
        print(f"The file {input_file} is already a WAV file. Skipping conversion.")
        # Copier simplement le fichier WAV vers la destination si besoin
        if input_file != output_file:
            output_file.write_bytes(input_file.read_bytes())
            print(f"Copied {input_file} to {output_file}")
    elif input_file.suffix.lower() == ".mp3":
        audio = AudioSegment.from_mp3(input_file)
        audio.export(output_file, format='wav')
        print(f"Converted {input_file} to {output_file}")
    else:
        print(f"Unsupported file format: {input_file.suffix}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_mp3_to_wav.py <input_file_path> <output_file_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    main(input_file, output_file)