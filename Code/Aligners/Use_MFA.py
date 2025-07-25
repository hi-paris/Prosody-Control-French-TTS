import subprocess
import os
from pathlib import Path
import sys
import shutil


if len(sys.argv) != 7:
    print(
        "Usage: python Use_MFA.py",
        "<dictionary_path>",
        "<acoustic_model_path>",
        "<textgrid_path>",
        "<audio_directory>",
        "<transcript_directory>",
        "<corpus_directory>"
    )
    sys.exit(1)

dictionary_path = sys.argv[1]
acoustic_model_path = sys.argv[2]
textgrid_path = sys.argv[3]
audio_directory = sys.argv[4]
transcript_directory = sys.argv[5]
corpus_directory = sys.argv[6]


def _copy_files(src, dest):
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        shutil.copy2(src_path, dest_path)


def _create_corpus(audio_directory, transcript_directory, corpus_directory):
    try:
        shutil.rmtree(corpus_directory)
    except Exception as e:
        print("got exception", e)
    os.makedirs(corpus_directory, exist_ok=True)
    _copy_files(audio_directory, corpus_directory)
    _copy_files(transcript_directory, corpus_directory)


def _run_mfa_align(audio_directory, transcript_directory, corpus_directory, dictionary_path, acoustic_model_path, output_directory):
    # Create corpus directory with audio and transcript files
    _create_corpus(audio_directory, transcript_directory, corpus_directory)

    # Execute MFA alignment command
    command = [
        "mfa", "align", corpus_directory.__str__(), dictionary_path.__str__(), acoustic_model_path.__str__(), output_directory.__str__(),
        "--beam", "100", "--retry_beam", "400", "--clean"
    ]
    str_command = " ".join(command)
    subprocess.run(str_command, shell=True)


def use_mfa(audio_directory, transcript_directory, textgrid_path, corpus_directory):
    # Ensure output directory exists and run MFA alignment
    os.makedirs(textgrid_path, exist_ok=True)
    _run_mfa_align(audio_directory, transcript_directory, corpus_directory, dictionary_path, acoustic_model_path, textgrid_path)


# Execute MFA alignment process
use_mfa(audio_directory, transcript_directory, textgrid_path, corpus_directory)