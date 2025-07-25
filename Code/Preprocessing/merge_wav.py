from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import re
from pathlib import Path
import logging # Import logging

logger = logging.getLogger(__name__) # Get logger instance

def merge_wav_from_folder(input_folder, output_file):
    """
    Fusionne tous les fichiers .wav d'un dossier en respectant l'ordre numérique des noms,
    en ignorant silencieusement ceux qui ne peuvent pas être décodés.
    """
    folder_path = Path(input_folder)
    if not folder_path.exists() or not folder_path.is_dir():
        logger.warning(f"⚠️ Skipping {input_folder}: not a valid folder.") # Use logger
        return

    # Tri numérique comme avant
    wav_files = sorted(
        folder_path.glob("*.wav"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))
                    if re.search(r"(\d+)", p.stem)
                    else float("inf")
    )

    if not wav_files:
        logger.warning(f"⚠️ No .wav files in {input_folder}.") # Use logger
        return

    logger.info(f"🔊 Attempting to merge {len(wav_files)} files in {input_folder}") # Use logger

    combined = None
    skipped = []

    for wav in wav_files:
        try:
            seg = AudioSegment.from_wav(wav)
        except CouldntDecodeError as e:
            skipped.append(wav.name)
            logger.warning(f"   ⚠️ Skipping corrupt file `{wav.name}`: {e}") # Use logger
            continue

        if combined is None:
            combined = seg
        else:
            combined += seg

    if combined is None:
        logger.error(f"❌ Aucun fichier valide à fusionner dans {input_folder}.") # Use logger
        return

    # Export final
    combined.export(output_file, format="wav")
    logger.info(f"✅ Créé : {output_file}") # Use logger
    if skipped:
        logger.info(f"   🗒️ {len(skipped)} fichiers ignorés : {skipped}") # Use logger


def merge_all_microsoft_audio(voice_root):
    """
    Pour chaque dossier se terminant par _microsoft sous `voice_root`,
    fusionne tous les .wav dans son sous-dossier audio/ en un fichier
    nommé <folder_name>_merged.wav à la racine du dossier Microsoft.
    """
    root = Path(voice_root)
    count = 0
    for ms_dir in sorted(root.glob("*_microsoft")):
        audio_dir = ms_dir / "audio"
        if not audio_dir.is_dir():
            logger.warning(f"⚠️ {audio_dir} n'existe pas, on passe.") # Use logger
            continue

        # build output name from the folder name
        folder_name = ms_dir.name
        output_wav = ms_dir / f"{folder_name}_merged.wav"

        merge_wav_from_folder(str(audio_dir), str(output_wav))
        count += 1
    logger.info(f"✅ {count} dossiers traités sous {voice_root}.") # Use logger


# Remove or modify the __main__ block if it exists to use logger
# if __name__ == "__main__":
#     voice_base = "/home/infres/horstmann-24/mon_projet_TTS/Data/voice"
#     merge_all_microsoft_audio(voice_base)