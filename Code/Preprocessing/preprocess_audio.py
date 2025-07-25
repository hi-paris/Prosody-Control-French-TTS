import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from pathlib import Path
import logging
import sys

"""
We cut the audio brut into small audio files
"""

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def segment_audio_file(input_file, min_silence_len, silence_thresh, keep_silence):
    """
    Segmente un fichier audio en utilisant les silences comme points de découpe.

    Parameters:
    - input_file: chemin vers le fichier audio (mp3 ou wav)
    - min_silence_len: durée minimale du silence en ms pour être considéré comme une pause
    - silence_thresh: seuil de volume en dB pour détecter le silence
    - keep_silence: durée de silence à conserver avant/après chaque segment en ms

    Returns:
    - Liste des segments audio
    """
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Le fichier {input_file} n'existe pas")

        logger.info(f"Chargement du fichier audio: {input_file}")
        audio = AudioSegment.from_file(input_file)
        logger.info("Segmentation de l'audio...")
        segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        logger.info(f"Segmentation terminée. {len(segments)} segments créés.")
        return segments
    except Exception as e:
        logger.error(f"Erreur lors de la segmentation: {str(e)}")
        raise

def save_segments(segments, output_dir, format='wav'):
    """
    Sauvegarde les segments dans des fichiers séparés.

    Parameters:
    - segments: liste des segments audio
    - output_dir: répertoire de sortie
    - format: format de sortie (wav ou mp3)
    """
    try:
        # Création du répertoire de sortie avec parents si nécessaire
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Sauvegarde des segments dans: {output_dir}")
        for i, segment in enumerate(segments):
            output_file = os.path.join(output_dir, f'segment_ph{i+1}.{format}')
            segment.export(output_file, format=format)
            logger.debug(f"Segment sauvegardé: {output_file}")
        logger.info("Tous les segments ont été sauvegardés avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des segments: {str(e)}")
        raise

def analyze_segment_lengths(segments):
    """
    Analyse la durée des segments et des silences.

    Parameters:
    - segments: liste des segments audio

    Returns:
    - Dict avec statistiques sur les segments
    """
    try:
        lengths = [len(segment) for segment in segments]
        stats = {
            'nombre_segments': len(segments),
            'duree_moyenne': np.mean(lengths) / 1000,  # conversion en secondes
            'duree_min': min(lengths) / 1000,
            'duree_max': max(lengths) / 1000,
            'duree_totale': sum(lengths) / 1000
        }
        logger.info("Analyse des segments terminée")
        return stats
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des segments: {str(e)}")
        raise

def main(input_file, output_dir, min_silence_len=1000, silence_thresh=-50, keep_silence=300):
    try:
        # Segmenter l'audio
        segments = segment_audio_file(input_file, min_silence_len, silence_thresh, keep_silence)

        # Analyser les segments
        stats = analyze_segment_lengths(segments)
        logger.info("Statistiques de segmentation:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        # Sauvegarder les segments
        save_segments(segments, output_dir, format='wav')

    except Exception as e:
        logger.error(f"Erreur dans le programme principal: {str(e)}")
        raise

"""
if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.absolute()
    input_file = base_dir / "Data" / "voice" / "BERTRAND_PERIER_EP03" / "brute"/ "segment.wav"
    output_dir = base_dir / "Data" / "voice" / "BERTRAND_PERIER_EP03" / "audio"
    main(input_file, output_dir)
"""
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_audio.py <file_input_path> <directory_output_path>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_file, output_dir)