"""
Whisper Timestamped Implementation
Configuration: Medium Model + Auditok VAD
With extended debugging and special handling for empty/noise files
"""

import whisper_timestamped as whisper
import torch
import json
import logging
from pathlib import Path
import re
import os
import warnings
import spacy
import logging
from typing import Dict, Any, Optional
import sys
import textgrid
import traceback
import numpy as np
from scipy.io import wavfile

# Logging configuration (Keep commented out or remove)
# logging.basicConfig(...)
# logger = logging.getLogger(__name__) 

base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().__str__()
_nlp = spacy.load("fr_core_news_sm", disable=["ner"])
_FORBIDDEN = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON"}
_logger = logging.getLogger(__name__)

def remove_spurious_commas(text: str) -> str:
    
    _PAUSE_MARKERS = {"[*]"}  # extend with "[pause]" or others if needed
    doc = _nlp(text)
    _logger.debug(">>> remove_spurious_commas input: %r", text)
    _logger.debug(">>> tokens+POS: %s", [(tok.text, tok.pos_) for tok in doc])
    out_tokens = []
    for tok in doc:
        _logger.debug("    checking tok %r (pos=%s)", tok.text, tok.pos_)
        if tok.text in {",", "."} | _PAUSE_MARKERS and out_tokens:
            prev = out_tokens[-1]
            _logger.debug("      prev tok %r (pos=%s)", prev.text, prev.pos_)
            if prev.pos_ in _FORBIDDEN:
                _logger.debug("      → dropping %r after forbidden %r", tok.text, prev.text)
                continue
        out_tokens.append(tok)

    result = "".join(t.text_with_ws for t in out_tokens)
    _logger.debug(">>> remove_spurious_commas output: %r", result)
    return result



class WhisperTranscriber:
    """Expert class for transcription with Whisper Timestamped"""

    def __init__(self,
                 model_size: str = "medium",
                 device: Optional[str] = None,
                 language: str = "fr",
                 logger: Optional[logging.Logger] = None
                ) -> None:
        """
        Initialisation of the transcriber

        Args:
            model_size: whisper model size
            device: Device for inference (cuda/cpu)
            language: Target language for transcription
            logger: Logger instance to use
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.model = None
        
        # Store the passed logger, provide a fallback if None (e.g., for direct instantiation)
        self.logger = logger if logger else logging.getLogger(__name__)

        warnings.filterwarnings("ignore", category=FutureWarning)

        # Use self.logger for logging within the class
        self.logger.info(f"Initialisation with model={model_size}, device={self.device}")
        self.logger.debug(f"Infos système - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.debug(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.debug(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    def load_model(self) -> None:
        try:
            self.logger.info(f"Loading the model  {self.model_size}")
            self.logger.debug(f"Available memory before loading: {self._get_gpu_memory_info()}")
            
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            self.logger.info("Model successfully loaded")
            self.logger.debug(f"Available memory after loading: {self._get_gpu_memory_info()}")
        except Exception as e:
            self.logger.error(f"Error during model loading: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _get_gpu_memory_info(self):
        if not torch.cuda.is_available():
            return "CUDA non available"
        
        try:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = t - (r + a)  # free inside reserved
            return f"Total: {t/1e9:.1f}GB, Réservé: {r/1e9:.1f}GB, Alloué: {a/1e9:.1f}GB, Libre: {f/1e9:.1f}GB"
        except Exception as e:
            return f"Erreur: {str(e)}"

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        try:
            audio_file = Path(audio_path)
            if not audio_file.exists():
                self.logger.error(f"Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            self.logger.info(f"Processing File: {audio_path}")
            self.logger.debug(f"Size file: {os.path.getsize(audio_path)} octets")

            # Vérification de l'audio (si c'est du bruit ou du silence)
            is_valid, message = self._check_audio_content(audio_path)
            if not is_valid:
                self.logger.warning(f"The audio file seems problematic: {message}")
                # On retourne un "faux résultat" pour un fichier sans contenu
                return self._create_empty_result()

            # Chargement audio
            self.logger.debug("Starting audio loading")
            try:
                audio = whisper.load_audio(audio_path)
                self.logger.debug(f"Audio loaded successfully: {len(audio)} échantillons")
                self.logger.debug(f"Audio min/max values: {audio.min():.2f}/{audio.max():.2f}")
            except Exception as e:
                self.logger.error(f"Specific error during audio loading: {e}")
                self.logger.error(traceback.format_exc())
                raise
            
            if self.model is None:
                self.load_model()

            transcription_config = {
                "language": self.language,
                "vad": "auditok",  
                "compute_word_confidence": True,
                "detect_disfluencies": True,
                "trust_whisper_timestamps": True,
            }
            self.logger.debug(f"Configuration transcription: {transcription_config}")

            # Transcription
            self.logger.info("Début de la transcription")
            self.logger.debug(f"Mémoire avant transcription: {self._get_gpu_memory_info()}")
            try:
                result = whisper.transcribe(self.model, audio, **transcription_config)
            except ValueError as e:
                if "max_silence" in str(e):
                    self.logger.warning(
                        "Auditok VAD failed (short audio); retrying transcription without VAD splitting"
                    )
                    transcription_config["vad"] = None
                    result = whisper.transcribe(self.model, audio, **transcription_config)
                else:
                    self.logger.error(f"Error during transcription: {e}")
                    raise
            self.logger.info("Transcription ended")
            self.logger.debug(f"Memomry after transcription: {self._get_gpu_memory_info()}")
            
            # Debug des résultats
            self.logger.debug(f"Number of segments: {len(result['segments'])}")
            if result['segments']:
                total_words = sum(len(seg['words']) for seg in result['segments'])
                self.logger.debug(f"Total number of: {total_words}")
                self.logger.debug(f"First segment: {result['segments'][0]['text'][:50]}...")
                self.logger.debug(f"Last segment: {result['segments'][-1]['text'][:50]}...")

            # Si très peu de mots détectés, on considère que c'est un fichier de bruit/silence
            if self._is_empty_result(result):
                self.logger.warning(f"Very little content detected in the audio file: {audio_path}")
                return self._create_empty_result()

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement: {e}")
            self.logger.error(traceback.format_exc())
            raise
            
    def _check_audio_content(self, audio_path):
        try:
            sample_rate, data = wavfile.read(audio_path)
            
            if len(data.shape) > 1:
                data = data[:, 0]
            
            # Compute RMS
            rms = np.sqrt(np.mean(np.square(data.astype(np.float32))))
            
            # Compute ratio signal/silence
            silence_threshold = 500  # Asjuste as needed
            non_silence_samples = np.sum(np.abs(data) > silence_threshold)
            silence_ratio = 1.0 - (non_silence_samples / len(data))
            
            self.logger.debug(f"Audio check: RMS={rms}, silence_ratio={silence_ratio:.2f}")
            
            file_size = os.path.getsize(audio_path)
            if file_size < 1000:  # less than 1 ko
                return False, f"File too small ({file_size} octets)"
                
            # Si plus de 95% de silence ou niveau RMS très bas
            if silence_ratio > 0.95:
                return False, f"File mainly contains silence ({silence_ratio:.2f})"
                
            if rms < 100:  # Ajuster selon les besoins
                return False, f"Very low audio level (RMS={rms})"
                
            return True, "Audio valide"
            
        except Exception as e:
            self.logger.error(f"Error during audio check: {e}")
            return True, f"Unable to check the audio: {str(e)}"
            
    def _is_empty_result(self, result):
        if not result['segments']:
            return True
        total_words = sum(len(seg['words']) for seg in result['segments'])
        if total_words < 3:
            return True
            
        total_text = " ".join([seg['text'] for seg in result['segments']])
        if len(total_text.strip()) < 10:
            return True
            
        return False
        
    def _create_empty_result(self):
        self.logger.debug("Creating an empty result for file with no content")
        return {
            "text": "...",
            "segments": [{
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "...",
                "words": [{
                    "start": 0.0,
                    "end": 1.0,
                    "text": "...",
                    "confidence": 0.0
                }]
            }],
            "language": self.language
        }
    
    def clean_text(self, text: str) -> str:
        original = text
        # 1) whitespace normalization
        text = re.sub(r"\s+", " ", text).strip()

        # 2) drop commas/periods after forbidden POS classes
        text = remove_spurious_commas(text)

        # 3) strip commas/periods AND [*] after any of your function words
        _fw = r"\b(?:que|et|ou|mais|donc|car|ni|où|dont|à|de|du|au|aux|en|par|pour|avec|sans|sur|sous)\b"
        # 3a) commas or periods
        text = re.sub(
            rf"({_fw})\s*[,\.]+",
            lambda m: m.group(1),
            text,
            flags=re.IGNORECASE,
        )
        # 3b) the literal pause marker "[*]"
        text = re.sub(
            rf"({_fw})\s*\[\*\]\s*",
            lambda m: m.group(1),
            text,
            flags=re.IGNORECASE,
        )

        # 4) global semicolon removal
        text = text.replace(";", "")

        if original != text:
            self.logger.debug(f"Texte nettoyé: '{original}' -> '{text}'")
        return text

    def save_results(self, result: Dict[str, Any], output_path: str) -> None:
        try:
            self.logger.debug(f"clean transcription and save")
            for segment in result["segments"]:
                segment["text"] = self.clean_text(segment["text"])
                for word in segment["words"]:
                    word["text"] = self.clean_text(word["text"])

            # save in JSON
            self.logger.debug(f"Save in JSON: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved: {output_path}")
            self.logger.debug(f"Size of JSON file: {os.path.getsize(output_path)} octets")
        except Exception as e:
            self.logger.error(f"Error During saving: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def display_results(self, result: Dict[str, Any]) -> None:
        try:
            self.logger.debug("Displaying the results")
            for segment in result["segments"]:
                print(f"\nSegment {segment['start']:.2f}s -> {segment['end']:.2f}s:")
                print(f"Texte: {segment['text']}")
                print("Mots détaillés:")
                for word in segment['words']:
                    print(f"  {word['start']:.2f}s -> {word['end']:.2f}s : "
                          f"{word['text']} (conf: {word['confidence']:.2f})")
        except Exception as e:
            self.logger.error(f"Errors lors de l'affichage: {e}")
            self.logger.error(traceback.format_exc())
            raise


def json_to_textgrid(json_file, logger): # Add logger parameter
    try:
        logger.debug(f"Conversion JSON -> TextGrid: {json_file}")

        if not os.path.exists(json_file):
            logger.error(f"Fichier JSON non trouvé: {json_file}")
            raise FileNotFoundError(f"Fichier JSON non trouvé: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                logger.debug(f"JSON chargé avec succès: {len(data['segments'])} segments")
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de décodage JSON: {e}")
                logger.error(f"Contenu du fichier problématique (premiers 100 caractères): {open(json_file, 'r').read(100)}")
                raise

        tg = textgrid.TextGrid()
        tier_words = textgrid.IntervalTier(name='words')
        current_time = 0.0

        # Debug variables
        total_words = 0
        corrected_intervals = 0

        for segment in data['segments']:
            logger.debug(f"Traitement segment: {segment['start']:.2f}s -> {segment['end']:.2f}s")

            for word in segment['words']:
                total_words += 1

                # Correction pour éviter les intervalles invalides
                if word['start'] >= word['end']:
                    corrected_intervals += 1
                    original_start = word['start']
                    original_end = word['end']
                    word['end'] = word['start'] + 0.01
                    logger.debug(f"Correction intervalle invalide: '{word['text']}', start={original_start:.3f} -> {word['start']:.3f}, end={original_end:.3f} -> {word['end']:.3f}")

                if word['start'] > current_time:
                    # Ajout d'un silence
                    logger.debug(f"Ajout silence: {current_time:.3f} -> {word['start']:.3f}")
                    tier_words.add(current_time, word['start'], " ")

                logger.debug(f"Ajout mot: '{word['text']}' ({word['start']:.3f} -> {word['end']:.3f})")
                tier_words.add(word['start'], word['end'], word['text'].replace("[*]", " "))
                current_time = word['end']

        # Si aucun mot n'a été ajouté, ajouter un intervalle factice
        if total_words == 0:
            logger.warning("Aucun mot trouvé dans le JSON, création d'un TextGrid avec un seul intervalle")
            xmax = 1.0
            if data['segments'] and len(data['segments']) > 0 and 'end' in data['segments'][-1]:
                xmax = data['segments'][-1]['end']
            tier_words.add(0.0, xmax, "...")
            current_time = xmax

        tg.append(tier_words)
        tg.maxTime = current_time

        logger.debug(f"TextGrid créé: {total_words} mots, {corrected_intervals} intervalles corrigés, durée totale: {current_time:.3f}s")
        return tg
    except Exception as e:
        logger.error(f"Erreur lors de la conversion JSON → TextGrid: {e}")
        logger.error(traceback.format_exc())
        raise


def check_dependencies(logger): # Add logger parameter
    """Vérifie les dépendances requises pour le script"""
    logger.debug("Vérification des dépendances...")

    # Vérifier torch
    logger.debug(f"PyTorch version: {torch.__version__}")

    # Vérifier whisper_timestamped
    try:
        whisper_version = whisper.__version__ if hasattr(whisper, "__version__") else "version inconnue"
        logger.debug(f"whisper_timestamped version: {whisper_version}")
    except Exception as e:
        logger.warning(f"Impossible de déterminer la version de whisper_timestamped: {e}")

    # Vérifier textgrid
    try:
        textgrid_version = textgrid.__version__ if hasattr(textgrid, "__version__") else "version inconnue"
        logger.debug(f"textgrid version: {textgrid_version}")
    except Exception as e:
        logger.warning(f"Impossible de déterminer la version de textgrid: {e}")

    # Vérifier l'environnement Python
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Python executable: {sys.executable}")


def create_matching_textgrids(natural_dir, synthetic_dir, logger): # Add logger parameter
    """
    S'assure que les mêmes fichiers TextGrid existent dans les deux répertoires.
    Si un fichier existe dans natural_dir mais pas dans synthetic_dir, crée
    un fichier TextGrid vide dans synthetic_dir.
    """
    logger.info(f"Vérification de la correspondance des TextGrids entre répertoires")
    logger.debug(f"Répertoire naturel: {natural_dir}")
    logger.debug(f"Répertoire synthétique: {synthetic_dir}")

    # S'assurer que les deux répertoires existent
    os.makedirs(natural_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)

    # Lister les fichiers dans les deux répertoires
    natural_files = {f for f in os.listdir(natural_dir) if f.endswith('.TextGrid')}
    synthetic_files = {f for f in os.listdir(synthetic_dir) if f.endswith('.TextGrid')}

    logger.debug(f"Trouvé {len(natural_files)} TextGrids dans le répertoire naturel")
    logger.debug(f"Trouvé {len(synthetic_files)} TextGrids dans le répertoire synthétique")

    # Fichiers manquants dans le répertoire synthétique
    missing_in_synthetic = natural_files - synthetic_files
    # Fichiers manquants dans le répertoire naturel
    missing_in_natural = synthetic_files - natural_files

    if missing_in_synthetic:
        logger.info(f"{len(missing_in_synthetic)} fichiers manquent dans le répertoire synthétique")
        for filename in missing_in_synthetic:
            natural_file_path = os.path.join(natural_dir, filename)
            synthetic_file_path = os.path.join(synthetic_dir, filename)

            logger.debug(f"Création d'un TextGrid vide dans: {synthetic_file_path}")

            # Obtenir la durée maximale du fichier naturel
            try:
                tg = textgrid.TextGrid.fromFile(natural_file_path)
                max_time = tg.maxTime
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du TextGrid {natural_file_path}: {e}")
                max_time = 1.0  # Valeur par défaut

            # Créer un TextGrid vide avec un intervalle factice
            empty_tg = textgrid.TextGrid()
            tier = textgrid.IntervalTier(name='words', minTime=0.0, maxTime=max_time)
            tier.add(0.0, max_time, "...")
            empty_tg.append(tier)
            empty_tg.write(synthetic_file_path)

            logger.info(f"TextGrid vide créé: {synthetic_file_path}")

    if missing_in_natural:
        logger.info(f"{len(missing_in_natural)} fichiers manquent dans le répertoire naturel")
        for filename in missing_in_natural:
            synthetic_file_path = os.path.join(synthetic_dir, filename)
            natural_file_path = os.path.join(natural_dir, filename)

            logger.debug(f"Création d'un TextGrid vide dans: {natural_file_path}")

            # Obtenir la durée maximale du fichier synthétique
            try:
                tg = textgrid.TextGrid.fromFile(synthetic_file_path)
                max_time = tg.maxTime
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du TextGrid {synthetic_file_path}: {e}")
                max_time = 1.0  # Valeur par défaut

            # Créer un TextGrid vide avec un intervalle factice
            empty_tg = textgrid.TextGrid()
            tier = textgrid.IntervalTier(name='words', minTime=0.0, maxTime=max_time)
            tier.add(0.0, max_time, "...")
            empty_tg.append(tier)
            empty_tg.write(natural_file_path)

            logger.info(f"TextGrid vide créé: {natural_file_path}")


def main(audio_path, out_path, whisper_model="medium", device=None, logger=None):
    """Point d'entrée principal"""
    # If no logger is provided (e.g., running standalone), get a default logger
    if logger is None:
        logger = logging.getLogger(__name__)
        # Basic configuration if run standalone and no handler is configured upstream
        if not logger.hasHandlers():
             logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
             logger.info("No logger provided, using basic configuration.")

    logger.debug(f"Démarrage du script avec audio_path={audio_path}, out_path={out_path}")
    check_dependencies(logger) # Pass logger

    try:
        # Vérifier si le dossier audio existe
        if not os.path.exists(audio_path):
            logger.error(f"Le dossier audio n'existe pas: {audio_path}")
            sys.exit(1)

        logger.debug(f"Contenu du dossier audio: {os.listdir(audio_path)}")

        # Initialisation - Pass the logger instance
        transcriber = WhisperTranscriber(
            model_size=whisper_model,
            device="cuda" if device is None else device,
            language="fr",
            logger=logger # Pass logger here
        )

        names = []
        try:
            audio_files = os.listdir(audio_path)
            logger.debug(f"Fichiers trouvés dans le dossier: {len(audio_files)}")
            for n in audio_files:
                if n.endswith(".wav"):
                    names.append(n[:-4])
                    logger.debug(f"Fichier WAV détecté: {n}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du dossier audio: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)

        N = len(names)
        logger.info(f"Nombre de fichiers .wav trouvés: {N}")

        if N == 0:
            logger.warning(f"Aucun fichier .wav trouvé dans {audio_path}")
            sys.exit(0)

        # Créer les dossiers de sortie
        logger.debug(f"Création des dossiers de sortie")
        OP = os.path.join(out_path + "_transcription")
        os.makedirs(OP, exist_ok=True)
        logger.debug(f"Dossier de transcription créé: {OP}")

        textgrid_dir = out_path
        os.makedirs(textgrid_dir, exist_ok=True)
        logger.debug(f"Dossier TextGrid créé: {textgrid_dir}")

        count = 0
        processed_files = []

        for n in names:
            logger.info(f"Traitement du fichier {count+1}/{N}: {n}")
            # Configuration
            file_path = os.path.join(audio_path, f"{n}.wav")

            # Vérifier si le fichier existe
            if not os.path.exists(file_path):
                logger.warning(f"Le fichier {file_path} n'existe pas, passage au suivant.")
                continue

            json_file = os.path.join(OP, f"{n}.json")
            txt_file = os.path.join(OP, f"{n}.txt")
            logger.debug(f"Chemins de sortie: JSON={json_file}, TXT={txt_file}")

            try:
                # Vérifier si le fichier audio est du bruit
                is_noise = False

                try:
                    # Lire le fichier audio et vérifier si c'est principalement du bruit/silence
                    sample_rate, data = wavfile.read(file_path)
                    if len(data.shape) > 1:  # Stéréo à mono
                        data = data[:, 0]

                    # Calculer le RMS (Root Mean Square) pour estimer le niveau sonore
                    rms = np.sqrt(np.mean(np.square(data.astype(np.float32))))

                    # Calculer le ratio silence
                    silence_threshold = 500  # Seuil pour considérer comme silence
                    silence_ratio = 1.0 - (np.sum(np.abs(data) > silence_threshold) / len(data))

                    logger.debug(f"Fichier {n}: RMS={rms}, silence_ratio={silence_ratio:.2f}")

                    # Si le fichier est principalement du silence ou a un niveau très bas
                    if silence_ratio > 0.95 or rms < 100:
                        logger.warning(f"Fichier {n} détecté comme bruit/silence (RMS={rms}, silence_ratio={silence_ratio:.2f})")
                        is_noise = True
                except Exception as e:
                    logger.error(f"Erreur lors de l'analyse audio de {n}: {e}")
                    # En cas d'erreur, on continue normalement

                if is_noise:
                    # Créer un fichier de transcription minimal pour les fichiers de bruit
                    with open(txt_file, "w", encoding="utf-8") as f:
                        f.write("...")
                    logger.info(f"Fichier de transcription minimaliste créé pour fichier de bruit: {txt_file}")

                    # Créer un TextGrid minimal
                    empty_tg = textgrid.TextGrid()
                    tier = textgrid.IntervalTier(name='words', minTime=0.0, maxTime=1.0)
                    tier.add(0.0, 1.0, "...")
                    empty_tg.append(tier)

                    output_file = os.path.join(textgrid_dir, f"{n}.TextGrid")
                    empty_tg.write(output_file)
                    logger.info(f"Minimal TextGrid created for noise file: {output_file}")

                    processed_files.append(n)
                    count += 1
                    continue

                result = transcriber.process_audio(file_path) # Transcriber uses its own logger

                # ── DUMP RAW JSON ──────────────────────────────────
                raw_json_dir = Path(out_path + "_raw_json")
                raw_json_dir.mkdir(parents=True, exist_ok=True)
                # one file per segment
                raw_json_path = raw_json_dir / f"{n}.raw.json"
                raw_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            
                # ── BUILD RAW TextGrids ────────────────────
                raw_tg_dir = Path(out_path + "_textgrid_raw")
                raw_tg_dir.mkdir(parents=True, exist_ok=True)
                tg = json_to_textgrid(str(raw_json_path), logger)
                # call the TextGrid‘s write(), not Path.write()
                tg.write(str(raw_tg_dir / f"{n}.TextGrid"))
                # ───────────────────────────────────────────────────

                for segment in result["segments"]:
                    segment["text"] = transcriber.clean_text(segment["text"])
                    for word in segment["words"]:
                        word["text"] = transcriber.clean_text(word["text"])

                transcriber.save_results(result, json_file)

                clean_text = " ".join([seg["text"] for seg in result["segments"]])
                clean_text = transcriber.clean_text(clean_text) 
                logger.debug(f"Writing the text file: {txt_file} ({len(clean_text)} caractères)")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(clean_text)
                logger.info(f"Cleaned transcription saved: {txt_file}")

                
                logger.debug("Conversion of JSON into TextGrid")

                tg = json_to_textgrid(json_file, logger) 
                output_file = os.path.join(textgrid_dir, f"{n}.TextGrid")
                logger.debug(f"Writing the TextGrid file: {output_file}")
                tg.write(output_file)
                logger.info(f"TextGrid created: {output_file}")

                processed_files.append(n)

            except Exception as e:
                logger.error(f"Error during file processing {n}: {e}")
                logger.error(traceback.format_exc())
                logger.warning(f"Moving to the next file...")
                continue

            count += 1
            logger.info(f"Progression: {count}/{N} Files processed successfully")

        logger.info(f"Processing completed: {count}/{N} ")

        # Détection des fichiers sans contenu pertinent
        problematic_files = [n for n in names if n not in processed_files]
        if problematic_files:
            logger.warning(f"{len(problematic_files)}  Problematic files identified:")
            for n in problematic_files:
                logger.warning(f"  - {n}.wav")

            # Création de TextGrids vides pour les fichiers problématiques
            for n in problematic_files:
                output_file = os.path.join(textgrid_dir, f"{n}.TextGrid")

                # Création d'un TextGrid vide avec un seul intervalle
                empty_tg = textgrid.TextGrid()
                tier = textgrid.IntervalTier(name='words', minTime=0.0, maxTime=1.0)
                tier.add(0.0, 1.0, "...")
                empty_tg.append(tier)
                empty_tg.write(output_file)

                logger.info(f"Empty TextGrid created for problematic file: {output_file}")

                # Création d'un fichier de transcription vide
                txt_file = os.path.join(OP, f"{n}.txt")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write("...")
                logger.info(f"Empty transcription file created: {txt_file}")

        base_path = os.path.dirname(audio_path)
        if "_microsoft" in base_path:
            # Synthetic voice directory
            natural_dir_name = os.path.basename(base_path).replace("_microsoft", "")
            natural_base_path = os.path.dirname(base_path)
            natural_textgrid_dir = os.path.join(natural_base_path, natural_dir_name, "WhisperTS_textgrid_files") # Adjust based on your structure

            if os.path.exists(natural_textgrid_dir):
                logger.info(f"Checking the correspondence of TextGrids between {natural_textgrid_dir} and {textgrid_dir}")
                create_matching_textgrids(natural_textgrid_dir, textgrid_dir, logger) # Pass logger

        elif "_microsoft" not in base_path:
			# Natural voice directory
            synthetic_dir_name = os.path.basename(base_path) + "_microsoft"
            synthetic_base_path = os.path.dirname(base_path) # Assuming they are siblings
            synthetic_textgrid_dir = os.path.join(synthetic_base_path, synthetic_dir_name, "WhisperTS_textgrid_files") # Adjust based on your structure

            if os.path.exists(synthetic_textgrid_dir):
                logger.info(f"Checking the correspondence of TextGrids between {textgrid_dir} and {synthetic_textgrid_dir}")
                create_matching_textgrids(textgrid_dir, synthetic_textgrid_dir, logger) # Pass logger


    except Exception as e:
        logger.error(f"Error in principal code: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def cli_main():
    log_level = logging.INFO 
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__) 

    if len(sys.argv) != 3:
        logger.error(f"Arguments incorrects: {sys.argv}")
        print("Usage: python use_whisper_timestamped.py <audio_path> <out_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    out_path = sys.argv[2]

    logger.info(f"Arguments: audio_path={audio_path}, out_path={out_path}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Call main, passing the logger explicitly
    main(audio_path, out_path, logger=logger)
    logger.info("Script end of execution.")


if __name__ == "__main__":
    cli_main()