import os
from pathlib import Path
import subprocess
import textgrid
import soundfile
import sys

"""
To use this program, you need to install nemo: https://github.com/NVIDIA/NeMo/tree/main
Change path_to_nemo
Create a new virtual environment with python 3.11.10 that you name NFA
Run `setup.py`
You might need 1 or 2 pip installs that you'll see when running the code (these are pip installs on NFA, not venv-TTS):
    venv-TTS uses version 12.4 of python, but I haven't found a way to install and make NeMo work with this version
"""

path_to_nemo = "../NeMo"

base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().__str__()


def clean_text(text):
    newline_char = "\n"
    text = text.strip(newline_char)
    text = "".join([("'" if char == "\"" else char) for char in text])
    return text


def create_manifest(audio_text_filepath, manifest_filepath="Test"):
    with open(os.path.join(base_dir, manifest_filepath, "manifest.json"), "w", encoding="utf-8") as file:
        names = []
        full_path = os.path.join(base_dir, audio_text_filepath)
        [(names.append(n[:-4]) if n.endswith(".wav") else 0) for n in os.listdir(full_path)]
        s = "\n".join([f"{{\"audio_filepath\": \"{os.path.join(full_path, file_name)}.wav\", \"text\":\"{clean_text(open(f'{os.path.join(full_path, file_name)}.txt').read())}\"}}" for file_name in names])
        file.write(s)
    print("manifest created as", os.path.join(base_dir, manifest_filepath, "manifest.json"))


create_manifest(os.path.join("Data", "voice", "records"))

subprocess.run(" && ".join(
    [
        f"cd {path_to_nemo}",
        "conda run -n NFA python tools/nemo_forced_aligner/align.py pretrained_name=stt_fr_citrinet_1024_gamma_0_25 manifest_filepath=../mon_projet_TTS/Test/manifest.json output_dir=../mon_projet_TTS/Test"
    ]
), shell=True)

print('done')


def ctm_to_textgrid(ctm_file_path, ctm_file_name, textgrid_file_path):
    with open(os.path.join(ctm_file_path, f"{ctm_file_name}.ctm")) as ctm_file:
        list_to_be_textgrid = []
        Max = 0
        for line in ctm_file:
            line = line[:-1]
            line = line.split(ctm_file_name)[1].strip()
            line = line.split("NA lex NA")[0].strip()
            line = line.split(" ")[1:]
            m, M, t = float(line[0]), round(float(line[0]) + float(line[1]), 2), line[2]
            if m > Max:
                list_to_be_textgrid.append(textgrid.Interval(Max, m, ""))
            Max = M
            list_to_be_textgrid.append(textgrid.Interval(m, M, t))
        print(list_to_be_textgrid)

        name = "words"
        min_time = 0.0
        max_time = Max
        interval_tier = textgrid.IntervalTier(name=name, minTime=min_time, maxTime=max_time)
        [interval_tier.addInterval(i) for i in list_to_be_textgrid]

        tg = textgrid.TextGrid()
        tg.append(interval_tier)
        tg.write(os.path.join(textgrid_file_path, ctm_file_name + ".TextGrid"))


names = []
[(names.append(n[:-4]) if n.endswith(".ctm") else 0) for n in os.listdir(os.path.join(base_dir, "Test", "ctm", "words"))]
for names in names:
    ctm_to_textgrid(os.path.join(base_dir, "Test", "ctm", "words"), names, os.path.join(base_dir, "Test", "ctm", "words"))