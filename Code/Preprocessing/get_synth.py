import os
from pathlib import Path
import sys
import azure.cognitiveservices.speech as speechsdk
import re

def clean_text(text):
    return re.sub(r'[,;]', '', text)

def main(Input_dir, audio_dir, audio_dir_microsoft, transcription_dir, transcription_dir_microsoft, api_key, voice, style, styledegree, clean_transcription=False):
    os.makedirs(transcription_dir_microsoft, exist_ok=True)
    os.makedirs(audio_dir_microsoft, exist_ok=True)

    audio_files = os.listdir(audio_dir)
    names = [n[:-4] for n in audio_files if n.endswith(".wav")]

    region = 'francecentral'

    for n in names:
        transcription_path = os.path.join(transcription_dir, n + ".txt")
        transcription_microsoft_path = os.path.join(transcription_dir_microsoft, n + ".txt")

        with open(transcription_path, "r", encoding='utf-8') as file:
            text = file.read()

            # Option pour nettoyer la ponctuation
            if clean_transcription:
                text = clean_text(text)

            # On crée un .txt dans le dossier de la voix de synthèse
            with open(transcription_microsoft_path, "w", encoding='utf-8') as file2:
                file2.write(text)
            # print(f"Texte généré pour {n}: {text}")

            # On crée l'audio
            text_ssml = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang='fr-FR'>
                <voice name='{voice}'>
                    {f"<mstts:express-as style='{style}' styledegree='{styledegree}'>" if style else ""}
                        {text}
                    {f"</mstts:express-as>" if style else ""}
                </voice>
            </speak>
            """

            speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
            speech_config.speech_synthesis_voice_name = voice
            outpath = os.path.join(audio_dir_microsoft, n + ".wav")
            audio_config = speechsdk.AudioConfig(filename=outpath)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            speech_synthesizer.speak_ssml_async(text_ssml).get()

if __name__ == "__main__":
    if len(sys.argv) != 11:
        print(
            "Usage: get_synth.py <Input_dir> <audio_dir> <audio_dir_microsoft> <transcription_dir> "
            "<transcription_dir_microsoft> <api_key> <voice> <style> <styledegree> <clean_transcription>"
        )
        sys.exit(1)

    Input_dir = sys.argv[1]
    audio_dir = sys.argv[2]
    audio_dir_microsoft = sys.argv[3]
    transcription_dir = sys.argv[4]
    transcription_dir_microsoft = sys.argv[5]
    api_key = sys.argv[6]
    voice = sys.argv[7]
    style = sys.argv[8] if sys.argv[8] != "None" else None
    styledegree = sys.argv[9] if sys.argv[9] != "None" else None
    clean_transcription = sys.argv[10].lower() in ('true', '1', 't', 'yes')

    main(Input_dir, audio_dir, audio_dir_microsoft, transcription_dir, transcription_dir_microsoft,
         api_key, voice, style, styledegree, clean_transcription)
