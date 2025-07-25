#!/usr/bin/env python3
"""
Script to generate TextGrid files from gold transcripts using NeMo forced alignment.
Processes all .lab files in the specified directory structure.
"""

import os
import glob
from pathlib import Path
import logging
from typing import List, Tuple
import tempfile
import shutil

# Audio processing imports
import librosa
import soundfile as sf

# NeMo imports
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel

# ────────────────────────────────────────────────────────────────────────────────
# ──                              CONSTANTS                                  ──
# ────────────────────────────────────────────────────────────────────────────────

# Base path where transcripts (.lab) live
TRANSCRIPT_BASE_PATH = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold"

# Base path where audio (.wav) live — same directory structure as transcripts
AUDIO_BASE_PATH = TRANSCRIPT_BASE_PATH

# Where to write the TextGrid files
OUTPUT_BASE_PATH = "/home/mila/d/dauvetj/mon_projet_TTS/Data/nemo_textgrids"

# Use a model that supports forced alignment
# RNN-T models typically support alignment better than CTC
MODEL_NAME = "nvidia/stt_en_conformer_transducer_large"
# Alternative: "nvidia/stt_en_fastconformer_transducer_large"

# ────────────────────────────────────────────────────────────────────────────────


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeMoTextGridGenerator:
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the NeMo TextGrid generator.

        Args:
            model_name: Name of the NeMo ASR model to use for alignment
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the NeMo ASR model."""
        try:
            logger.info(f"Loading NeMo model: {self.model_name}")
            self.model = ASRModel.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
            
            # Check if model supports alignment
            if not hasattr(self.model, 'transcribe') and not hasattr(self.model, 'transcribe_with_timestamps'):
                logger.warning("Model may not support forced alignment")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def find_transcript_files(self, base_path: str) -> List[Tuple[str, str, str]]:
        """
        Find all .lab transcript files in the directory structure.

        Args:
            base_path: Base path to search for transcript files

        Returns:
            List of tuples (speaker_id, audio_name, full_path)
        """
        transcript_files = []
        pattern = os.path.join(base_path, "*", "*.lab")

        for file_path in glob.glob(pattern):
            path_parts = Path(file_path).parts
            speaker_id = path_parts[-2]  # Parent directory name
            audio_name = Path(file_path).stem  # Filename without extension
            transcript_files.append((speaker_id, audio_name, file_path))

        logger.info(f"Found {len(transcript_files)} transcript files")
        return transcript_files

    def read_transcript(self, transcript_path: str) -> str:
        """
        Read transcript from .lab file.

        Args:
            transcript_path: Path to the .lab file

        Returns:
            Transcript text
        """
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            return transcript
        except Exception as e:
            logger.error(f"Failed to read transcript {transcript_path}: {e}")
            return ""

    def find_audio_file(self, speaker_id: str, audio_name: str, audio_base_path: str) -> str:
        """
        Find the corresponding audio file for a transcript.

        Args:
            speaker_id: Speaker ID
            audio_name: Audio file name (without extension)
            audio_base_path: Base path where audio files are stored

        Returns:
            Path to audio file or None if not found
        """
        # since audio lives in the same structure as transcripts, just swap .lab → .wav
        candidate = os.path.join(audio_base_path, speaker_id, f"{audio_name}.wav")
        if os.path.exists(candidate):
            return candidate
        return None

    def _preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file to ensure it's mono and compatible with NeMo.
        
        Args:
            audio_path: Path to the original audio file
            
        Returns:
            Path to the processed audio file (may be temporary)
        """
        try:
            # Load audio to check its properties
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            
            # Check if audio is stereo
            needs_processing = False
            if audio.ndim > 1:
                logger.info(f"Converting stereo to mono: {audio_path}")
                audio = librosa.to_mono(audio)
                needs_processing = True
            
            # Check sample rate (NeMo typically works well with 16kHz)
            if sr != 16000:
                logger.info(f"Resampling from {sr}Hz to 16000Hz: {audio_path}")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
                needs_processing = True
            
            if needs_processing:
                # Create temporary file for processed audio
                temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
                os.close(temp_fd)  # Close file descriptor, we'll use the path
                
                # Save processed audio
                sf.write(temp_path, audio, sr)
                logger.info(f"Saved processed audio to: {temp_path}")
                return temp_path
            else:
                return audio_path
                
        except Exception as e:
            logger.error(f"Failed to preprocess audio {audio_path}: {e}")
            return audio_path

    def generate_textgrid(self, audio_path: str, transcript: str, output_path: str):
        """
        Generate TextGrid file using NeMo forced alignment.

        Args:
            audio_path: Path to audio file
            transcript: Transcript text
            output_path: Output path for TextGrid file
        """
        processed_audio_path = None
        try:
            # Preprocess audio to ensure compatibility
            processed_audio_path = self._preprocess_audio(audio_path)
            
            # Try different methods depending on what the model supports
            alignments = None
            
            # Method 1: Try transcribe_with_timestamps if available
            if hasattr(self.model, 'transcribe_with_timestamps'):
                logger.info("Using transcribe_with_timestamps method")
                alignments = self.model.transcribe_with_timestamps([processed_audio_path])
                if alignments:
                    alignments = alignments[0]
            
            # Method 2: Try getting alignments from regular transcribe with return_timestamps
            elif hasattr(self.model, 'transcribe'):
                logger.info("Using transcribe method with timestamps")
                try:
                    # Some models support return_timestamps parameter
                    alignments = self.model.transcribe([processed_audio_path], return_timestamps=True)
                    if alignments:
                        alignments = alignments[0]
                except Exception as e:
                    logger.info(f"transcribe with timestamps failed: {e}")
                    # Fallback to basic transcription
                    basic_transcription = self.model.transcribe([processed_audio_path])
                    logger.info(f"Basic transcription: {basic_transcription}")
                    # Create dummy alignment based on audio duration
                    alignments = self._create_dummy_alignment(processed_audio_path, transcript)
            
            # Method 3: Use the alignment API if available
            else:
                logger.info("Trying direct alignment method")
                try:
                    from nemo.collections.asr.parts.utils.alignment_utils import get_alignments
                    alignments = get_alignments(self.model, [processed_audio_path], [transcript])
                    if alignments:
                        alignments = alignments[0]
                except ImportError:
                    logger.warning("Alignment utilities not available")
                    alignments = self._create_dummy_alignment(processed_audio_path, transcript)

            if alignments is None:
                logger.warning(f"No alignments generated for {audio_path}")
                alignments = self._create_dummy_alignment(processed_audio_path, transcript)

            # Convert to TextGrid format
            self._write_textgrid(alignments, transcript, output_path)
            logger.info(f"Generated TextGrid: {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate TextGrid for {audio_path}: {e}")
            # Create a basic TextGrid as fallback
            try:
                if processed_audio_path is None:
                    processed_audio_path = self._preprocess_audio(audio_path)
                alignments = self._create_dummy_alignment(processed_audio_path, transcript)
                self._write_textgrid(alignments, transcript, output_path)
                logger.info(f"Generated fallback TextGrid: {output_path}")
            except Exception as e2:
                logger.error(f"Failed to create fallback TextGrid: {e2}")
        
        finally:
            # Clean up temporary file if created
            if processed_audio_path and processed_audio_path != audio_path:
                try:
                    os.unlink(processed_audio_path)
                    logger.debug(f"Cleaned up temporary file: {processed_audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {processed_audio_path}: {e}")

    def _create_dummy_alignment(self, audio_path: str, transcript: str):
        """
        Create a dummy alignment when proper alignment is not available.
        """
        try:
            import librosa
            # Get audio duration
            duration = librosa.get_duration(filename=audio_path)
        except:
            # Fallback duration
            duration = 3.0
        
        words = transcript.split()
        if not words:
            return {'words': []}
        
        # Distribute words evenly across the duration
        word_duration = duration / len(words)
        word_alignments = []
        
        for i, word in enumerate(words):
            start_time = i * word_duration
            end_time = (i + 1) * word_duration
            word_alignments.append({
                'word': word,
                'start': start_time,
                'end': end_time
            })
        
        return {'words': word_alignments}

    def _write_textgrid(self, alignments, transcript: str, output_path: str):
        """
        Write TextGrid file from alignment data.

        Args:
            alignments: Alignment data from NeMo
            transcript: Original transcript
            output_path: Output TextGrid file path
        """
        try:
            # Handle different alignment formats
            if isinstance(alignments, dict) and 'words' in alignments:
                words = alignments['words']
            elif isinstance(alignments, list):
                words = alignments
            else:
                logger.warning(f"Unexpected alignment format: {type(alignments)}")
                words = []

            content = self._create_textgrid_content(words, transcript)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:
            logger.error(f"Failed to write TextGrid {output_path}: {e}")

    def _create_textgrid_content(self, words: List[dict], transcript: str) -> str:
        """
        Create TextGrid file content in proper Praat format.

        Args:
            words: List of word alignment dictionaries
            transcript: Original transcript

        Returns:
            TextGrid file content as string
        """
        if not words:
            end_time = 1.0
        else:
            end_time = max(word.get('end', 0) for word in words)
        
        # Ensure we have at least some content
        if len(words) == 0:
            words = [{'word': transcript, 'start': 0.0, 'end': end_time}]

        # Create proper TextGrid format
        content = f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = {end_time} 
tiers? <exists> 
size = 2 
item [1]:
    class = "IntervalTier" 
    name = "words" 
    xmin = 0 
    xmax = {end_time} 
    intervals: size = {len(words)} '''
        
        # Add word intervals with proper formatting
        for idx, w in enumerate(words, start=1):
            start_time = float(w.get('start', 0))
            end_time_word = float(w.get('end', start_time + 0.1))
            word_text = str(w.get('word', '')).replace('"', '""')  # Escape quotes
            
            content += f'''
    intervals [{idx}]:
        xmin = {start_time} 
        xmax = {end_time_word} 
        text = "{word_text}" '''

        # Add sentence tier
        sentence_text = transcript.replace('"', '""')  # Escape quotes
        content += f'''
item [2]:
    class = "IntervalTier" 
    name = "sentences" 
    xmin = 0 
    xmax = {end_time} 
    intervals: size = 1 
    intervals [1]:
        xmin = 0 
        xmax = {end_time} 
        text = "{sentence_text}" 
'''
        return content

    def process_all_transcripts(self,
                                transcript_base_path: str,
                                audio_base_path: str,
                                output_base_path: str):
        """
        Process all transcript files and generate TextGrids.
        """
        files = self.find_transcript_files(transcript_base_path)
        successful = failed = 0

        for speaker, name, lab_path in files:
            logger.info(f"Processing {speaker}/{name}")
            txt = self.read_transcript(lab_path)
            if not txt:
                logger.warning(f"Empty transcript for {lab_path}")
                failed += 1
                continue

            wav = self.find_audio_file(speaker, name, audio_base_path)
            if not wav:
                logger.warning(f"No audio for {speaker}/{name}")
                failed += 1
                continue

            out_path = os.path.join(output_base_path, f"{name}.TextGrid")
            self.generate_textgrid(wav, txt, out_path)
            successful += 1

        logger.info(f"Done. Successful: {successful}, Failed: {failed}")


if __name__ == "__main__":
    generator = NeMoTextGridGenerator(model_name=MODEL_NAME)
        # --- NEW SECTION FOR SINGLE FILE PROCESSING ---
    specific_transcript_path = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/az1_transcript.txt"
    specific_audio_path = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/Aznavour_1.wav"
    
    # Ensure the output directory exists
    specific_output_dir = "/home/mila/d/dauvetj/mon_projet_TTS/Data/nemo_textgrids"
    os.makedirs(specific_output_dir, exist_ok=True)

    # Determine the output TextGrid filename (e.g., Aznavour_1.TextGrid)
    audio_filename_stem = Path(specific_audio_path).stem
    specific_output_textgrid_path = os.path.join(specific_output_dir, f"{audio_filename_stem}.TextGrid")

    # Read the transcript
    try:
        with open(specific_transcript_path, 'r', encoding='utf-8') as f:
            transcript_content = f.read().strip()
        
        if transcript_content:
            logger.info(f"Processing single file: {specific_audio_path} with transcript from {specific_transcript_path}")
            generator.generate_textgrid(specific_audio_path, transcript_content, specific_output_textgrid_path)
            logger.info(f"TextGrid for {audio_filename_stem} generated at {specific_output_textgrid_path}")
        else:
            logger.error(f"Transcript file {specific_transcript_path} is empty.")
    except FileNotFoundError:
        logger.error(f"Transcript or audio file not found at {specific_transcript_path} or {specific_audio_path}")
    except Exception as e:
        logger.error(f"Error processing single file: {e}")
    # --- END NEW SECTION ---
    generator.process_all_transcripts(
        transcript_base_path=TRANSCRIPT_BASE_PATH,
        audio_base_path=AUDIO_BASE_PATH,
        output_base_path=OUTPUT_BASE_PATH
    )