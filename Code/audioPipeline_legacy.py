import os
from pathlib import Path
from pydub import AudioSegment
import spacy
import logging
import glob
import re
import sys
import shutil
import subprocess

from Preprocessing.preprocess_audio import main as preprocess_main
from Preprocessing.demucs_process import main as demucs_main
from Preprocessing.convert_mp3_to_wav import main as convert_main

from Aligners import use_whisper_timestamped
from Preprocessing import gen_break_ssml
from Preprocessing import synthesize_ssml_voice
from Preprocessing.get_synth import main as get_synth_main
from Pipeline.utils import save_clean_transcriptions_from_textgrids
import Pipeline.create_training_data as create_training_data

# Set base directory to the parent directory of the current file
BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR_STR = str(BASE_DIR)

# Logging configuration
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all log levels

# Console handler (INFO and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler (DEBUG and above)
logs_dir = BASE_DIR / "Code" / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "pipeline_debug.log"
file_handler = logging.FileHandler(str(log_file_path))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Tee class: also write all terminal output to a file
log_output_path = logs_dir / "pipeline_output.log"
class Tee:
    def __init__(self, filename, stream):
        self.file = open(filename, 'a')
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

sys.stdout = Tee(str(log_output_path), sys.stdout)
sys.stderr = Tee(str(log_output_path), sys.stderr)

# Define output and data directories
OUT_DIR = Path(BASE_DIR) / "Out"
DATA_DIR = Path(BASE_DIR) / "Data"

# Append the 'Codes' directory to sys.path and import the pipeline module
sys.path.append(str(Path(BASE_DIR) / "Code"))
import Code.pipeline_class_legacy as module_pipeline

# Path to NeMo directory
PATH_TO_NEMO = str(Path(BASE_DIR) / "NeMo")

class VoicePipeline:
    def execute_levenstein_correction(self) -> None:

        from Aligners.levenshtein_dist_align_txtgrids import main as levenshtein_main

        print("Using levenshtein_distance to align textgrids...")
        textgrid_path_nat = self.textgrid_dir
        textgrid_path_synth = self.textgrid_microsoft_dir
        if not os.path.exists(textgrid_path_nat) or not os.path.exists(textgrid_path_synth):
            raise FileNotFoundError(f"TextGrid directories missing: {textgrid_path_nat} or {textgrid_path_synth}")
        nat_files = os.listdir(textgrid_path_nat)
        if not nat_files:
            raise FileNotFoundError(f"No TextGrid files found in {textgrid_path_nat}")
        for file in nat_files:
            if not os.path.exists(os.path.join(textgrid_path_synth, file)):
                print(f"Warning: No matching synthetic TextGrid for {file}")
                continue
            print("Working on", file)
            levenshtein_main(
            textgrid1_input_path=os.path.join(textgrid_path_nat, file),
            textgrid2_input_path=os.path.join(textgrid_path_synth, file),
            transcription1_dir=self.transcription_dir,
            transcription2_dir=self.transcription_dir_microsoft
        )

    def preprocess_audio(self):
        mp3_file_path = os.path.join(self.Input_dir, "brute", "segment.mp3")
        wav_file_path = os.path.join(self.Input_dir, "brute", "segment.wav")
        demucs_file_path = os.path.join(self.Input_dir, "brute", "segment_demucs.wav")
        output_dir = os.path.join(self.Input_dir, "audio")

        try:
            if self.convert_to_wav:
                if os.path.exists(mp3_file_path):
                    logging.info(f"Conversion MP3 vers WAV : {mp3_file_path} -> {wav_file_path}")
                    convert_main(mp3_file_path, wav_file_path)
            elif not os.path.exists(wav_file_path):
                raise FileNotFoundError(f"Fichier audio introuvable dans : {os.path.dirname(mp3_file_path)}")
            
            if os.path.exists(demucs_file_path):
                os.remove(demucs_file_path)

            logging.info(f"Denoising (Demucs) : {wav_file_path} -> {demucs_file_path}")
            demucs_main(wav_file_path, demucs_file_path)

            logging.info(f"Découpage en segments : {demucs_file_path} -> {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            preprocess_main(demucs_file_path, output_dir)

            logging.info("✅ Préprocessing terminé.")

        except Exception as e:
            logging.error(f"Erreur lors du préprocessing : {e}")
            raise

    def verify_wav_file(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            print(f"✅ The file {file_path} is valid")
            return True
        except Exception as e:
            print(f"❌ Error with the file {file_path}: {e}")

    def __init__(self,
             dir_name,
             preprocess,
             clean_transcriptions,
             do_alignement,
             getSynth,
             levenshtein_correction=True,
             aligner="WhisperTS",
             convert_to_wav=True,
             gen_voice=True,
             voice="fr-FR-HenriNeural",
            gen_raw_synth=True,
             style=None,
             styledegree=2):
        """
        Initializes the Voc instance, setting up parameters, loading resources, and defining
        directory paths needed for processing audio and transcription data for voice synthesis.

        Parameters
        ----------
        dir_name : str
            The folder name containing the .wav and .txt files.
        preprocess : bool
            Whether to preprocess the data (expects a "brute" folder with a single audio file).
        clean_transcriptions : bool
            If True, clean and regnerate the transcription; otherwise, use existing transcriptions.
        do_alignement : bool
            Whether to perform audio-text alignment.
        getSynth : bool
            Whether to create/recreate the synthesized voice folder.
        levenshtein_correction : bool
            Whether to use Levenshtein correction when aligning textgrids.
        aligner : str
            The aligner to use; must be one of the implemented aligners.
        convert_to_wav : bool
            Relevant only if `preprocess` is True; indicates if conversion from .mp3 to .wav is needed.
        gen_voice : bool
            True if the final synthetic voice is to be generated.
        voice : str
            The name of the Azure synthetic voice to use.
        style, styledegree : Any
            Currently unused parameters.
        """

        self.Implemented_aligners = ["MFA", "NeMo", "WhisperTS", "CTCFA", "whisperX"]

        # Parameters
        self.levenshtein_correction = levenshtein_correction
        self.gen_voice = gen_voice
        self.preprocess = preprocess
        self.clean_transcriptions = clean_transcriptions
        self.style = style
        self.styledegree = styledegree
        self.aligner = aligner if aligner in self.Implemented_aligners else "WhisperTS"
        self.voice = voice
        self.getSynth = getSynth
        self.do_alignement = do_alignement
        self.dir_name = dir_name
        self.convert_to_wav = convert_to_wav
        self.gen_raw_synth = gen_raw_synth

        # Variables for later use
        self.last_was_break = False
        self.pitch_nat_moyen = ...  # Placeholder for average natural pitch

        # Load SpaCy French model for parsing trees
        self.nlp = spacy.load("fr_core_news_sm")
        self.api_key = 

        # Directories for data usage
        self.Input_dir = os.path.join(DATA_DIR, 'voice', self.dir_name)
        # Define two separate Microsoft directories: one for raw synthesis and one for SSML processing.
        self.dir_microsoft_raw = self.Input_dir + "_microsoft"
        self.dir_microsoft_ssml = self.Input_dir + "_microsoft_ssml"

        self.audio_dir = os.path.join(self.Input_dir, "audio")
        self.audio_dir_microsoft_raw = os.path.join(self.dir_microsoft_raw, "audio")
        
        # For the processed (SSML) synthesis we use the _microsoft_ssml folder:
        self.audio_dir_microsoft = os.path.join(self.dir_microsoft_ssml, "audio")
        self.transcription_dir = os.path.join(self.Input_dir, "transcription")
        self.transcription_dir_microsoft_raw = os.path.join(self.dir_microsoft_raw, "transcription")
        self.transcription_dir_microsoft = os.path.join(self.dir_microsoft_ssml, "transcription")
        self.textgrid_dir = os.path.join(self.Input_dir, f"{self.aligner}_textgrid_files")
        self.textgrid_microsoft_dir = os.path.join(self.dir_microsoft_ssml, f"{self.aligner}_textgrid_files")


        # Directories for processed data
        self.Temp_dir = os.path.join(OUT_DIR, "Temp")
        self.extracted_segments_info = os.path.join(self.Temp_dir, self.dir_name, 'alignement_result', 'Results_Voix_naturelle', 'Segments')
        self.extracted_segments_info_microsoft = os.path.join(self.Temp_dir, self.dir_name, 'alignement_result', 'Results_Voix_de_synthese', 'Segments')
        self.in_needleman_wunsch = os.path.join(OUT_DIR, 'Temp', self.dir_name, 'alignement_result', 'Results_Voix_de_synthese')
        self.in_needleman_wunsch_microsoft = os.path.join(OUT_DIR, 'Temp', self.dir_name, 'alignement_result', 'Results_Voix_naturelle')
        self.AligNeedlemanWhunch_out = os.path.join(OUT_DIR, 'Temp', self.dir_name, 'alignement_result', 'AligNeedlemanWhunch_with_start_end_durations')

        # CSV file paths
        self.BDD1_dir = os.path.join(self.Temp_dir, self.dir_name, 'BDD1.csv')
        self.BDD2_dir = os.path.join(OUT_DIR, 'Temp', self.dir_name, 'BDD2.csv')
        self.BDD3_dir = os.path.join(OUT_DIR, 'Temp', self.dir_name, 'BDD3.csv')
        self.BDD4_dir = os.path.join(OUT_DIR, 'Temp', self.dir_name, 'BDD4.csv')
        self.BDD5_dir = os.path.join(OUT_DIR, 'results', self.dir_name, 'BDD_ssml.csv')

        # Paths for files in the Tools folder
        self._tooldir = os.path.join(BASE_DIR, 'Tools')
        self.dictionary_path = os.path.join(self._tooldir, 'MFA', 'pretrained_models', 'dictionary', 'enriched_french_mfa.dict')
        self.acoustic_model_path = os.path.join(self._tooldir, 'MFA', 'pretrained_models', 'acoustic', 'french_mfa.zip')
        self.lexique_tsv_path = os.path.join(self._tooldir, 'MFA', 'lexique383', 'Lexique383.tsv')
        self.mfa_lexicon_path = os.path.join(self._tooldir, 'MFA', 'pretrained_models', 'dictionary', 'french_mfa.dict')
        self.espeak_path = os.path.join(self._tooldir, 'eSpeak', 'command_line', 'espeak.exe')

    def use_whisperTS(self, audio_dir: str) -> None:
        """
        Executes the Whisper Timestamped aligner on the specified audio directory.

        Creates a dedicated directory for TextGrid files and calls the aligner's main function
        using the audio subdirectory.

        Args:
            audio_dir (str): Path to the folder containing the 'audio' subdirectory with WAV files.
        """
        textgrid_dir = os.path.join(audio_dir, "WhisperTS_textgrid_files")
        audio_directory = os.path.join(audio_dir, "audio")
        os.makedirs(textgrid_dir, exist_ok=True)
        print(f"Calling use_whisper_timestamped.main with audio_directory={audio_directory} and textgrid_dir={textgrid_dir}")
        use_whisper_timestamped.main(audio_directory, textgrid_dir, whisper_model="turbo", logger=logger)

    def save_clean_textgrid_transcriptions(self) -> None:
        """
        Extracts and cleans transcriptions from TextGrid files and saves them as plain text files.

        Utilizes the utility function 'save_clean_transcriptions_from_textgrids' to convert
        TextGrid alignment files into raw transcription text stored in the designated transcription folder.
        """
        try:
            textgrid_dir = self.textgrid_dir
            transcription_dir = self.transcription_dir
            os.makedirs(transcription_dir, exist_ok=True)

            if not os.path.exists(textgrid_dir):
                raise FileNotFoundError(f"TextGrid folder not found: {textgrid_dir}")

            save_clean_transcriptions_from_textgrids(textgrid_dir, Path(transcription_dir))
        except Exception as e:
            logging.error(f"Error in regenerate_transcription: {e}")
            raise

    def gen_break_ssml(self):
        TEXTGRID_FOLDER = self.textgrid_dir  # TextGrid corrigé par Levenshtein
        TRANSCRIPTION_FOLDER = self.transcription_dir  # Transcriptions corrigées par Levenshtein
        SSML_OUTPUT_FOLDER = self.dir_microsoft_ssml
        XML_FILES_FOLDER = os.path.join(SSML_OUTPUT_FOLDER, "xml_files")
    
        os.makedirs(XML_FILES_FOLDER, exist_ok=True)
        os.makedirs(SSML_OUTPUT_FOLDER, exist_ok=True)
    
        print(f"Using corrected transcriptions for SSML generation.")
        print(f"Calling gen_break_ssml.main with TEXTGRID_FOLDER={TEXTGRID_FOLDER}, TRANSCRIPTION_FOLDER={TRANSCRIPTION_FOLDER}, SSML_OUTPUT_FOLDER={XML_FILES_FOLDER}")
    
        gen_break_ssml.main(TEXTGRID_FOLDER, TRANSCRIPTION_FOLDER, XML_FILES_FOLDER)


    def generate_raw_synth(self):
        print("Generating raw synthetic voice (without pause insertion) in _microsoft folder...")
        # Create a temporary pipeline object dedicated to raw synthesis:
        os.makedirs(os.path.dirname(self.audio_dir_microsoft_raw), exist_ok=True)
        os.makedirs(self.audio_dir_microsoft_raw, exist_ok=True)
        os.makedirs(self.transcription_dir_microsoft_raw, exist_ok=True)
        get_synth_main(
            self.Input_dir,
            self.audio_dir,
            self.audio_dir_microsoft_raw,
            self.transcription_dir,
            self.transcription_dir_microsoft_raw,
            self.api_key,
            self.voice,
            str(self.style),
            str(self.styledegree),
            clean_transcription= True
        )


    def synthesize_ssml_audio(self):
        """
        Processes SSML files to synthesize audio and prepares transcription files for the synthetic voice.

        Moves existing XML files to a dedicated subfolder, calls the Azure TTS synthesis via the 
        'read_ssml_break.main' function using the provided API key and region, and then copies over 
        natural voice transcription files to the corresponding synthetic voice folder.

        This method also outputs status messages indicating the number of generated WAV files.
        """
        ssml_folder = self.dir_microsoft_ssml
        xml_files_folder = os.path.join(ssml_folder, "xml_files")
        audio_folder = os.path.join(ssml_folder, "audio")
        transcription_folder = os.path.join(ssml_folder, "transcription")
     
        # Créer tous les dossiers nécessaires
        os.makedirs(ssml_folder, exist_ok=True)
        os.makedirs(xml_files_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        os.makedirs(transcription_folder, exist_ok=True)
 
        # Déplacer les fichiers XML existants vers le sous-dossier xml_files
        for xml_file in glob.glob(os.path.join(ssml_folder, "*.xml")):
            target_path = os.path.join(xml_files_folder, os.path.basename(xml_file))
            shutil.move(xml_file, target_path)
            print(f"Moved {os.path.basename(xml_file)} to {xml_files_folder}")

        AZURE_SPEECH_KEY = self.api_key
        AZURE_SPEECH_REGION = "francecentral"
        AZURE_VOICE = self.voice

        # Display folders for verification
        print(f"📁 SSML source folder: {xml_files_folder}")
        print(f"📁 Audio output folder: {audio_folder}")

        # Count and display how many SSML files are present
        total_files = len(glob.glob(os.path.join(xml_files_folder, '*.[Xx][Mm][Ll]')))
        print(f"📊 Found {total_files} SSML files in {xml_files_folder}")

        try:
            print(f"Calling read_ssml_break.main with AZURE_SPEECH_KEY={AZURE_SPEECH_KEY}, AZURE_SPEECH_REGION={AZURE_SPEECH_REGION}, ssml_folder={xml_files_folder}, output_folder={audio_folder}")
            synthesize_ssml_voice.main(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, xml_files_folder, audio_folder, AZURE_VOICE)
        except Exception as e:
            print(f"❌ Exception occurred: {e}")

        # After processing, list the generated WAV files for verification
        wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))
        print(f"📊 Found {len(wav_files)} WAV files in output folder")
        if wav_files:
            print(f"📊 Sample files: {[os.path.basename(f) for f in wav_files[:5]]}")
        
        # Copy transcription files from natural voice
        if os.path.exists(self.transcription_dir):
            for txt_file in glob.glob(os.path.join(self.transcription_dir, "*.txt")):
                filename = os.path.basename(txt_file)
                target_path = os.path.join(transcription_folder, filename)
                shutil.copy2(txt_file, target_path)
                print(f"Copied {filename} to {transcription_folder}")

    def get_synthetic_voice(self):
        # 0. Prétraitement automatique
        if self.preprocess:
            self.preprocess_audio()

        # 1. Run the aligner on the natural voice side
        self.use_whisperTS(self.Input_dir)
        if self.clean_transcriptions:
            self.save_clean_textgrid_transcriptions()

        # 2. Generate the raw synthetic voice and save it in the _microsoft folder
        if self.gen_raw_synth:
            self.generate_raw_synth()

        # 3. Run levensthein correction to correct the textgrids
        if self.levenshtein_correction:
            self.execute_levenstein_correction()

        # 3. Generate SSML files (which will use the natural transcriptions and textgrids)
        # to insert the correct pauses and create the processed synthetic voice.
        self.gen_break_ssml()
        self.synthesize_ssml_audio()

        # 4. Run whisperTS on the ssml generated audio files
        self.use_whisperTS(self.dir_microsoft_ssml)

        # 5. create the pipeline object for further processing
        pipeline_obj = module_pipeline.Voc(
            self.dir_name,
            preprocess=False,
            regen_txt=False,
            do_alignement=False,
            getSynth=False,
            levenshtein_correction=self.levenshtein_correction,
            aligner=self.aligner,
            convert_to_wav=False, 
            gen_voice=self.gen_voice,
            voice=self.voice,
            check_checkfiles=False
        )

        # 6. Update the paths to point to ssml directories
        pipeline_obj.audio_dir_microsoft = self.audio_dir_microsoft
        pipeline_obj.textgrid_microsoft_dir = self.textgrid_microsoft_dir
        pipeline_obj.transcription_dir_microsoft = self.transcription_dir_microsoft     

        # 7. Run the remaining steps:
        if self.do_alignement:
            pipeline_obj.log_textgrid_info(self.textgrid_dir, "Textgrids for natural voice")   
            pipeline_obj.log_textgrid_info(self.textgrid_microsoft_dir, "Textgrids for synthetic voice")

        # Excute manually the step needed without running run_pipeline()
        if self.do_alignement:
            pipeline_obj.log_textgrid_info(self.textgrid_dir, "Textgrids for natural voice")
            pipeline_obj.log_textgrid_info(self.textgrid_microsoft_dir, "Textgrids for synthetic voice")

        # Execute steps without generating voice
        pipeline_obj.extract_process_segments()
        pipeline_obj.needleman_wunsch_alignement()
        pipeline_obj.add_breaks()
        pipeline_obj.complete_audio_paths()
        pipeline_obj.calculate_loudness_adjustement()
        pipeline_obj.calculate_rate()
        pipeline_obj.get_wav()
        
        print("\n" + "-"*60)
        print("ÉTAPE FINALE: GÉNÉRATION DES FICHIERS AUDIO")
        print("-"*60)
        if self.gen_voice:
            os.makedirs(os.path.join(OUT_DIR, 'results', self.dir_name), exist_ok=True)

            # lancer la generation de la voix avec les parametres explicites
            from Pipeline.TTS_df import main as tts_main
            api_key= self.api_key
            BDD4_dir= self.BDD4_dir
            out_dir = OUT_DIR


            print(f"Lancement de la génération audio avec {BDD4_dir}")
            tts_main(self.dir_name, api_key, BDD4_dir, out_dir, 630.0, 1430.0)
            print("✅ Génération des fichiers audio terminée")

        print("\n" + "-"*60)
        print("Create Training Data")
        print("-"*60)
        bdd_ssml_path = os.path.join(OUT_DIR, 'results', self.dir_name, 'BDD_ssml.csv')
        output_path = os.path.join(OUT_DIR, 'results', self.dir_name, f'training_data_{self.dir_name}.json')
        create_training_data.create_training_data(bdd_ssml_path, output_path)
        results_folder = os.path.join(OUT_DIR, "results")
        combined_json_path = os.path.join(results_folder, "bdd.json")
        create_training_data.combine_training_jsons(results_folder, combined_json_path)
        print(f"Training data created at {output_path}")
 
        print("\n" + "=" * 60)
        print("VOICE CREATION PROCESS COMPLETED")
        print("="*60)

if __name__ == '__main__':
    # Example voices: fr-FR-HenriNeural; fr-FR-VivienneMultilingualNeural; fr-FR-DeniseNeural
    for VOICE in ["Aznavour_EP04"]:
        obj = VoicePipeline(
            VOICE,
            preprocess=False, # let it true if you need to cut your original .wav
            clean_transcriptions=True, # let it true if you dont have transcriptions
            do_alignement=True, # let it true to do the audio text alignement 
            getSynth=False, # Let it true to generate the synthetic voice with pauses 
            levenshtein_correction=False, # Let it true to correct differences
            aligner="WhisperTS",
            convert_to_wav=False, # Let it True if your original segment is in mp3
            gen_voice=True, # let it true to generate finale voice 
            voice="fr-FR-HenriNeural",
            gen_raw_synth=False) # let it true to generate raw voice synthesis
        obj.get_synthetic_voice()
        