import os
from pathlib import Path
import subprocess
from pydub import AudioSegment
import spacy
import fr_core_news_sm
import logging
import pandas as pd
import numpy as np
import glob
import re
import textgrid
import shutil
import logging

from Aligners import use_whisper_timestamped
from Aligners.levenshtein_dist_align_txtgrids import main as levenshtein_main
from Aligners.enrichir_dictionnaire import main as enrichir_dict_main

from Preprocessing.get_synth import main as get_synth_main
from Preprocessing.convert_mp3_to_wav import main as convert_main
from Preprocessing.demucs_process import main as demucs_main
from Preprocessing.preprocess_audio import main as preprocess_main

from Pipeline.utils import save_clean_transcriptions_from_textgrids
from Pipeline.extract_process_segments import main as extract_process_segments_main
from Pipeline.NeedlemanWunschAlignement import needleman_wunsch_alignement
from Pipeline.Ajuster_les_pauses import add_breaks
from Pipeline.Get_Wav import get_wav
from Pipeline.TTS_df import main as tts_main
import Pipeline.compute_pitch_adjustments as compute_pitch_adjustments
import Pipeline.compute_loudness_adjustments as compute_loudness_adjustments
import Pipeline.compute_rate_adjustments as compute_BDD3_loudness_rate
import Pipeline.create_training_data as create_training_data

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# We won't write anything here
# base_dir = Path("/home/infres/ext-3478/mon_projet_TTS")
base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().__str__()

# autre_base = Path("/tsi/hi-paris/tts/")

# We write output here
# Out_dir = autre_base/ "Out"
Out_dir = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().__str__(), "Out")

# We might write stuff here also (textgrids, and preprocessed audio)
# Data_dir = autre_base
Data_dir = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().__str__(), "Data")


path_to_nemo = os.path.join(base_dir, "NeMo")

class Voc:
    def verify_wav_file(self, chemin_fichier):
        try:
            audio = AudioSegment.from_file(chemin_fichier)
            print((f"✅ The file {chemin_fichier} is valide"))
            return True
        except Exception as e:
            print(f"❌ Error with the file {chemin_fichier}: {str(e)}")


    def __init__(
            
        self,
            dir_name,
            preprocess,
            regen_txt,
            do_alignement,
            getSynth,
            levenshtein_correction=True,
            aligner="WhisperTS",
            convert_to_wav=True,
            gen_voice=True,
            voice="fr-FR-HenriNeural",
            style=None,
            styledegree=2,
            check_checkfiles=False,
            data_dir = None,
            out_dir = None,
            microsoft_folder_suffix = "_microsoft_ssml" # nouveau parametre pour choisir la voix de synthese
    ):
        """Initialisation of the Voc class
        The Voc class will be used in order to do the whole process of creating an improved sythetic voice using natural one

        Parameters
        ----------
        dir_name : str
            The name of the folder containing the .wav & .txt files

        preprocess : bool
            Do we preprocess the data: requier a "brute" folder instead of an "audio" one,
            this folder must contain the whole audio in just one file.
            This audio file is either a .mp3 if `convert_to_wav == True` otherwise .wav

        regen_txt : bool
            If we regenerate the transcription if True or assume those are here otherwise

        do_alignement : bool
            Do we use audio-text alignement

        getSynth : bool
            Do we create/recreate the `dir_name`_microsoft folder containing the synthetised voice

        levenshtein_correction : bool
            Whether we use the code trying to align the textgrids (that we called levenstein_correction)

        aligner : str
            The aligner we will use are one of `self.Implemented_aligners`

        convert_to_wav : bool
            Only matters if `preprocess==True`
            Whether we assumed the file in `Data/voice/brute` to be a .mp3 or not (thus wav)

        gen_voice : bool
            True if we want to generate the final voice (with azure)

        voice: str
            The name of azure's synthetic voice we will use

        style, styledegree are both unused
        """
        self.Data_dir = data_dir if data_dir else os.path.join(base_dir, "Data")
        self.Out_dir = out_dir if out_dir else os.path.join(base_dir, "Out")
      
        self.Implemented_aligners = ["MFA", "NeMo", "WhisperTS", "CTCFA", "whisperX"]

        # -------------------
        # Parametters:
        # -------------------
        self.levenshtein_correction = levenshtein_correction
        self.gen_voice = gen_voice
        self.preprocess = preprocess
        self.regen_txt = regen_txt
        self.style = style
        self.styledegree = styledegree
        self.aligner = (
            aligner if aligner in self.Implemented_aligners else "WhisperTS"
        )
        self.voice = voice
        self.getSynth = getSynth
        self.do_alignement = do_alignement
        self.dir_name = dir_name
        self.convert_to_wav = convert_to_wav
        self.check_checkfiles = check_checkfiles

        # -------------------
        # Variables used later:
        # -------------------
        self.last_was_break = False
        self.pitch_nat_moyen = ...
        # Load SpaCy French model for sparsedtrees
        self.nlp = fr_core_news_sm.load()
        self.api_key = 

        # -------------------
        # Directories:
        # -------------------
        # Path of to the used data:
#
        self.microsoft_folder_suffix = microsoft_folder_suffix

        self.Input_dir = os.path.join(self.Data_dir, 'voice', self.dir_name)

        self.audio_dir = os.path.join(self.Input_dir, "audio")
        self.audio_dir_microsoft = os.path.join(self.Input_dir + self.microsoft_folder_suffix, "audio")
        self.transcription_dir = os.path.join(self.Input_dir, "transcription")
        self.transcription_dir_microsoft =  os.path.join(self.Input_dir + self.microsoft_folder_suffix, "transcription")
        self.textgrid_dir = os.path.join(self.Input_dir, f"{self.aligner}_textgrid_files")
        self.textgrid_microsoft_dir = os.path.join(self.Input_dir + self.microsoft_folder_suffix, f"{self.aligner}_textgrid_files")

        # Path of to some processed data:
        self.Temp_dir = os.path.join(Out_dir, "Temp")

        self.extracted_segments_info = os.path.join(self.Temp_dir, self.dir_name, 'alignement_result', 'Results_Voix_naturelle', 'Segments')
        self.extracted_segments_info_microsoft = os.path.join(self.Temp_dir, self.dir_name, 'alignement_result', 'Results_Voix_de_synthese', 'Segments')
        self.in_needleman_wunsch = os.path.join(Out_dir, 'Temp', self.dir_name, 'alignement_result', 'Results_Voix_de_synthese')
        self.in_needleman_wunsch_microsoft = os.path.join(Out_dir, 'Temp', self.dir_name, 'alignement_result', 'Results_Voix_naturelle')
        self.needleman_wunsch_results = os.path.join(Out_dir, 'Temp', self.dir_name, 'alignement_result', 'AligNeedlemanWhunch_with_start_end_durations')

        # BDD file path
        self.BDD1_dir = os.path.join(self.Temp_dir, self.dir_name, 'BDD1.csv')
        self.BDD2_dir = os.path.join(Out_dir, 'Temp', self.dir_name, 'BDD2.csv')
        self.BDD3_dir = os.path.join(Out_dir, 'Temp', self.dir_name, 'BDD3.csv')
        self.BDD4_dir = os.path.join(Out_dir, 'Temp', self.dir_name, 'BDD4.csv')
        self.BDD5_dir = os.path.join(Out_dir, 'results', self.dir_name, 'BDD_ssml.csv')

        # Path to files from the `Tool` folder:
        self._tooldir = os.path.join(base_dir, 'Tools')
        self.dictionary_path = os.path.join(self._tooldir, 'MFA', 'pretrained_models', 'dictionary', 'enriched_french_mfa.dict')
        self.acoustic_model_path = os.path.join(self._tooldir, 'MFA', 'pretrained_models', 'acoustic', 'french_mfa.zip')
        self.lexique_tsv_path = os.path.join(self._tooldir, 'MFA', 'lexique383', 'Lexique383.tsv')
        self.mfa_lexicon_path = os.path.join(self._tooldir, 'MFA', 'pretrained_models', 'dictionary', 'french_mfa.dict')
        self.espeak_path = os.path.join(self._tooldir, 'eSpeak', 'command_line', 'espeak.exe')

    ########################################################
    #  This part concerns the use of MFA to get Textgrids  #
    ########################################################

    def log_textgrid_info(self, textgrid_path, message=""):
        """Display information about TextGrid files in a directory"""
        if os.path.exists(textgrid_path):
            files = glob.glob(os.path.join(textgrid_path, "*.TextGrid"))
            print(f"\n{'='*50}")
            print(f"{message} - Directory: {textgrid_path}")
            print(f"Number of TextGrid files: {len(files)}")
            if files:
                print(f"Sample files: {[os.path.basename(f) for f in files[:3]]}")
                # Show preview of the first file content
                if len(files) > 0:
                    try:
                        with open(files[0], 'r', encoding='utf-8') as f:
                            content = f.read(500)  # Read first 500 characters
                        print(f"Content preview of {os.path.basename(files[0])}:")
                        print(f"{content}...")
                    except Exception as e:
                        print(f"Error reading file: {e}")
            print(f"{'='*50}\n")
        else:
            print(f"\n{'='*50}")
            print(f"{message} - Directory: {textgrid_path}")
            print(f"WARNING: Directory does not exist!")
            print(f"{'='*50}\n")

    def use_CTCFA(self):
        # Process each directory for 'Voix naturelle' and 'Voix de synthese'
        for name, audio_dir in [("Voix_naturelle", self.dir_name), ("Voix_de_synthese", self.dir_name + "_microsoft")]:
            textgrid_path = (self.textgrid_dir if name == "Voix_naturelle" else self.textgrid_microsoft_dir)
            audio_directory = os.path.join(self.Data_dir, 'voice', audio_dir, "audio")
            transcript_directory = os.path.join(self.Data_dir, 'voice', audio_dir, "transcription")
            os.makedirs(textgrid_path, exist_ok=True)
            try:
                shutil.rmtree(textgrid_path)
            except Exception as e:
                print("got exception", e)
            Command = (
                f"conda run -n CTCFA python {os.path.join(base_dir, 'Code', 'Aligners', 'CTCFA.py')}"
                f" {audio_directory}"
                f" {transcript_directory}"
                f" {textgrid_path}"
                f" fra"
                f" segment"
                f" True"
            )
            print(Command)
            res = subprocess.run(" && ".join(
                [
                    Command
                ]
            ), shell=True)
            assert res.returncode == 0, "programm crashed"

    def use_whisperX(self):
        # Process each directory for 'Voix naturelle' and 'Voix de synthese'
        for name, audio_dir in [("Voix_naturelle", self.dir_name), ("Voix_de_synthese", self.dir_name + "_microsoft")]:
            textgrid_path = (self.textgrid_dir if name == "Voix_naturelle" else self.textgrid_microsoft_dir)
            audio_directory = os.path.join(self.Data_dir, 'voice', audio_dir, "audio")
            transcript_directory = os.path.join(self.Data_dir, 'voice', audio_dir, "transcription")
            os.makedirs(textgrid_path, exist_ok=True)
            try:
                shutil.rmtree(textgrid_path)
            except Exception as e:
                print("got exception", e)
            Command = (
                f"conda run -n whisperX python {os.path.join(base_dir, 'Code', 'Aligners', 'whisperX.py')}"
                f" {audio_directory}"
                f" {transcript_directory}"
                f" {textgrid_path}"
            )
            print(Command)
            res = subprocess.run(" && ".join(
                [
                    Command
                ]
            ), shell=True)
            assert res.returncode == 0, "programm crashed"

    def use_mfa(self):
        # Process each directory for 'Voix naturelle' and 'Voix de synthese'
        for name, audio_dir in [("Voix_naturelle", self.dir_name), ("Voix_de_synthese", self.dir_name + "_microsoft")]:
            textgrid_path = (self.textgrid_dir if name == "Voix_naturelle" else self.textgrid_microsoft_dir)
            audio_directory = os.path.join(self.Data_dir, 'voice', audio_dir, "audio")
            transcript_directory = os.path.join(self.Data_dir, 'voice', audio_dir, "transcription")
            os.makedirs(textgrid_path, exist_ok=True)
            try:
                shutil.rmtree(textgrid_path)
            except Exception as e:
                print("got exception", e)
            corpus_dir = os.path.join(Out_dir, "Temp", self.dir_name, "MFA_corpus")
            Command = (
                f"conda run -n base python {os.path.join(base_dir, 'Code', 'Aligners', 'Use_MFA.py')}"
                f" {self.dictionary_path}"
                f" {self.acoustic_model_path}"
                f" {textgrid_path}"
                f" {audio_directory}"
                f" {transcript_directory}"
                f" {corpus_dir}"
            )
            print(Command)
            res = subprocess.run(" && ".join(
                [
                    Command
                ]
            ), shell=True)
            assert res.returncode == 0, "programm crashed"

    def _clean_text_NeMo(self, text):
        accent_dict = {
            'â': 'a', 'à': 'a', 'ä': 'a', 'á': 'a', 'ã': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'î': 'i', 'ï': 'i', 'ì': 'i', 'í': 'i',
            'ô': 'o', 'ö': 'o', 'ò': 'o', 'ó': 'o', 'õ': 'o',
            'û': 'u', 'ù': 'u', 'ü': 'u', 'ú': 'u',
            'ç': 'c', 'ÿ': 'y', 'ñ': 'n'
        }

        newline_char = "\n"
        text = text.strip(newline_char)
        text = "".join([("'" if char == "\"" else char) for char in text])
        punctuation_signs = [
            '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}',
        ]
        separation_signs = [
            '-', '_',
        ]
        for p in punctuation_signs:
            text = text.replace(p, "")
        for s in separation_signs:
            text = text.replace(s, " ")

        Words = text.split(" ")
        pairs = []
        New_words = []
        for w in Words:
            if len(w) == 0:
                continue
            w1 = w
            w = w.lower()
            for k in accent_dict:
                w = w.replace(k, accent_dict[k])
            New_words.append(w)
            pairs.append([w, w1])

        return " ".join(New_words), pairs

    def _create_manifest(self, audio_text_filepath, manifest_filepath):
        with open(os.path.join(base_dir, manifest_filepath, "manifest.json"), "w", encoding="utf-8") as file:
            names = []
            full_path = os.path.join(base_dir, audio_text_filepath)
            [(names.append(n[:-4]) if n.endswith(".wav") else 0) for n in os.listdir(full_path)]

            lines = []
            pairs = []
            for file_name in names:
                text, new_pairs = self._clean_text_NeMo(open(f'{os.path.join(full_path, file_name)}.txt').read())
                pairs.append(new_pairs)
                lines.append(f"{{\"audio_filepath\": \"{os.path.join(full_path, file_name)}.wav\", \"text\":\"{text}\"}}")

            s = "\n".join(lines)
            file.write(s)
        print("manifest created as", os.path.join(base_dir, manifest_filepath, "manifest.json"))
        return pairs

    def _ctm_to_textgrid(self, ctm_file_path, ctm_file_name, textgrid_file_path, pairs):
        to_replace = {
            "<space>": " ",
        }
        with open(os.path.join(ctm_file_path, f"{ctm_file_name}.ctm")) as ctm_file:
            list_to_be_textgrid = []
            Max = 0
            gap = 0.0
            for line in ctm_file:
                for k in to_replace:
                    line = line.replace(k, to_replace[k])
                line = line[:-1]
                line = line.split(ctm_file_name)[1].strip()
                line = line.split("NA lex NA")[0].strip()
                line = line.split(" ")[1:]
                m, M, t = round(float(line[0]) - gap, 3), round(float(line[0]) + float(line[1]) - gap, 3), line[2]
                p = pairs.pop(0)
                if t == p[0]:
                    t = p[1]
                else:
                    print("might be a problem here:", t, p)
                    1 / 0
                dt = m - Max
                if dt > 0.11:
                    list_to_be_textgrid.append(textgrid.Interval(Max, Max + dt, ""))
                else:
                    gap += dt
                    m = round(m - dt, 3)
                    M = round(M - dt, 3)
                Max = M
                list_to_be_textgrid.append(textgrid.Interval(m, M, t))
            list_to_be_textgrid.append(textgrid.Interval(Max, Max + 0.5, ""))
            print(list_to_be_textgrid)

            name = "words"
            min_time = 0.0
            max_time = Max + 0.5
            interval_tier = textgrid.IntervalTier(name=name, minTime=min_time, maxTime=max_time)
            [interval_tier.addInterval(i) for i in list_to_be_textgrid]

            tg = textgrid.TextGrid()
            tg.append(interval_tier)
            tg.write(os.path.join(textgrid_file_path, ctm_file_name + ".TextGrid"))

    def use_NeMo(self):
        for name, audio_dir in [("Voix_naturelle", self.Input_dir), ("Voix_de_synthese", self.Input_dir + "_microsoft")]:
            print("Using NeMo for", name)
            audio_directory = (self.audio_dir if name == "Voix_naturelle" else self.audio_dir_microsoft)
            textgrid_path = (self.textgrid_dir if name == "Voix_naturelle" else self.textgrid_microsoft_dir)

            # Ensure output base directory exists
            os.makedirs(textgrid_path, exist_ok=True)

            pairs = self._create_manifest(audio_dir, audio_directory)
            # This is a path taken from `path_to_nemo`
            aligner_py_path = os.path.join("tools", "nemo_forced_aligner", "align.py")
            model_name = "stt_fr_citrinet_1024_gamma_0_25"
            manifest_filepath = ...
            res = subprocess.run(" && ".join(
                [
                    f"cd {path_to_nemo}",
                    f"conda run -n NFA python {aligner_py_path} pretrained_name={model_name} manifest_filepath={os.path.join(manifest_filepath, 'manifest.json')} output_dir={textgrid_path}"
                ]
            ), shell=True)
            assert res.returncode == 0, "programm crashed"
            names = []
            [(names.append(n[:-4]) if n.endswith(".ctm") else 0) for n in os.listdir(os.path.join(textgrid_path, "ctm", "words"))]
            for n in names:
                print("\n" * 100, n)
                print(pairs[0])
                self._ctm_to_textgrid(os.path.join(textgrid_path, "ctm", "words"), n, textgrid_path, pairs.pop(0))

    def use_Whisper_timestamped(self, voices="NS"):
        N_audio_dir = []
        if "N" in voices:
            N_audio_dir.append(("Voix_naturelle", self.Input_dir))
        if "S" in voices:
            N_audio_dir.append(("Voix_de_synthese", self.Input_dir + "_microsoft_ssml"))
        for name, audio_dir in N_audio_dir:
            textgrid_path = os.path.join(audio_dir, "WhisperTS_textgrid_files")
            print(f"🔊 Using Whisper-timestamped for {name}")

            audio_directory = os.path.join(audio_dir, "audio")
            print(f"🎯 Source audio: {audio_directory}")
            print(f"🎯 Destination textgrid: {textgrid_path}")

            # Ensure output base directory exists
            os.makedirs(textgrid_path, exist_ok=True)
            use_whisper_timestamped.main(audio_directory, textgrid_path)

            # Display information about generated textgrid
            self.log_textgrid_info(textgrid_path, f"TextGrids generated by Whisper-timestamped for {name}")

    def execute_levenstein_correction(self) -> None:
        """
        This function uses the levenshtein_distance algorithm to align the text in the TextGrid files
        """
        print("Using levenshtein_distance to make textgrids to use the same words...")
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
            print("\r Working on", file)
            levenshtein_main(
                textgrid1_input_path=os.path.join(textgrid_path_nat, file)
                , textgrid2_input_path=os.path.join(textgrid_path_synth, file)
                , transcription1_dir=self.transcription_dir
                , transcription2_dir=self.transcription_dir_microsoft
            )
            
    ########################################################
    #  Nous allons enrichir le dictionnaire french_mfa en  #
    #  lui ajoutant les mots de notre corpus de texte      #
    #  qu'il ne connait pas, nous utilisons espeak pour    #
    #  la transcription phonétique                         #
    ########################################################

    def enrichir_dict(self):
        enrichir_dict_main(
            self.transcription_dir,
            os.path.join(self.Temp_dir, self.dir_name, 'text_concat.txt'),
            self.espeak_path,
            self.lexique_tsv_path,
            self.mfa_lexicon_path,
            self.dictionary_path
        )

    ########################################################
    #         Here we extract and process segments         #
    ########################################################

    def extract_process_segments(self):
        """
        This code automatically extracts data from phonemes and segments from textgrid files and saves them as csv.
        These csv files are structured to include details such as ID, start time, end time, duration and associated text.
        Results are organized in separate folders for each person with sub-folders for segments and phonemes.
        """
        print("\n" + "=" * 60)
        print("Extracting and processing segments")
        print("=" * 60)

        Names = ["Voix_de_synthese", "Voix_naturelle"]

        for NAME in Names:
            print(f"\n🔍 Processing for {NAME}")
            # Base path where TextGrid files are located
            textgrid_path = (self.textgrid_dir if NAME == "Voix_naturelle" else self.textgrid_microsoft_dir)
            output_segments_dir = (self.extracted_segments_info if NAME == "Voix_naturelle" else self.extracted_segments_info_microsoft)

            # Display information about source textgrid files
            self.log_textgrid_info(textgrid_path, f"TextGrids for {NAME}")
            print(f"📊 Extracting segments to: {output_segments_dir}")

            extract_process_segments_main(textgrid_path, output_segments_dir)

            # check and display results
            if os.path.exists(output_segments_dir):
                csv_files = glob.glob(os.path.join(output_segments_dir, "*.csv"))
                print(f"✅ {len(csv_files)} CSV files generated in {output_segments_dir}")
                if csv_files:
                    print(f"Examples: {[os.path.basename(f) for f in csv_files[:3]]}")
            else:
                print(f"❌ No CSV files generated in {output_segments_dir}")

    ########################################################
    #  Here we use Needleman Wunsch alignement algorithm   #
    ########################################################

    def needleman_wunsch_alignement(self):
        checkpoint_file = os.path.join(self.Temp_dir, self.dir_name, 'checkpoint_needleman_wunsch_done.txt')

        if self.check_checkfiles and os.path.exists(checkpoint_file):
            logging.info("Alignement Needleman-Wunsch déjà effectué. On passe à l'étape suivante.")
            return

        try:
            # Vérifie si les dossiers d'entrée existent
            if not os.path.exists(self.in_needleman_wunsch) or not os.path.exists(self.in_needleman_wunsch_microsoft):
                raise FileNotFoundError(f"Un des dossiers de segments est manquant : {self.in_needleman_wunsch} ou {self.in_needleman_wunsch_microsoft}")

            logging.info("Launching de l'alignement Needleman-Wunsch")
            needleman_wunsch_alignement(self.in_needleman_wunsch, self.in_needleman_wunsch_microsoft, self.needleman_wunsch_results)

            # ✅ Si tout est bon, on crée le checkpoint
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint_file).touch()
            logging.info(f"✅ Alignement Needleman-Wunsch is done. Checkpoint created : {checkpoint_file}")

        except Exception as e:
            logging.error(f"Erreur dans needleman_wunsch_alignement : {e}")
            raise





    ########################################################
    #          Here we add pauses to the dataset           #
    ########################################################

    def add_breaks(self):
        try:
            logging.info("Adding breaks...")
            add_breaks(self.needleman_wunsch_results, self.BDD1_dir)
            logging.info("Breaks added successfully.")
        except Exception as e:
            logging.error(f"Erreur dans add_breaks : {e}")
            raise

    ########################################################
    #         Now we compute the path to the audio         #
    ########################################################

    def complete_audio_paths(self):
        print(f"🔄 Running compute_BDD1.complete_audio_paths with BDD1_dir: {self.BDD1_dir}, BDD2_dir: {self.BDD2_dir}")
        compute_pitch_adjustments.compute_pitch_adjustments(
            self.BDD1_dir,
            self.audio_dir,
            self.audio_dir_microsoft,
            self.transcription_dir,
            self.transcription_dir_microsoft,
            self.BDD2_dir
        )

    ########################################################
    #    Here we compute the adjustements for loudness     #
    ########################################################

    def _calculate_loudness(audio_file_path, start, end):
        # Check if the file path is NaN or contains "nan" or doesn't exist
        if pd.isna(audio_file_path) or "nan" in str(audio_file_path) or not os.path.isfile(audio_file_path):
            return 0

        # Load the audio file
        audio = AudioSegment.from_file(audio_file_path)[start * 1000:end * 1000]

        # Convert to numerical values
        audio_seg = audio.get_array_of_samples()
        samples = np.array(audio_seg)
        S = np.array(samples) ** 2
        # Calculate loudness (RMS)
        rms = np.sqrt(np.abs(np.mean(S)))

        # Convert loudness to dB
        loudness = 20 * np.log10(rms)
        return loudness

    ########################################################
    #    Here we compute the adjustements for loudness     #
    ########################################################

    def calculate_rate(self):
        print(f"🔄 Running compute_BDD3_loudness_rate.calculate_rate with BDD3_dir: {self.BDD3_dir}, BDD4_dir: {self.BDD4_dir}")
        compute_BDD3_loudness_rate.calculate_rate(self.BDD3_dir, self.BDD4_dir)

    ########################################################
    #            Here we compute the wav file              #
    ########################################################

    def get_wav(self):
        audio_output = os.path.join(Out_dir, 'results', self.dir_name)
        try:
            logging.info("Creating SSML...")
            get_wav(self.BDD4_dir, audio_output, self.voice, self.style, self.styledegree, self.BDD5_dir)
            logging.info("SSML created successfully.")
        except Exception as e:
            logging.error(f"Erreur dans get_wav : {e}")
            raise

    ########################################################
    #     Here we take care of the voice generation        #
    ########################################################
    def Text_to_speech_df(self):
        tts_main(self.dir_name, self.api_key, self.BDD4_dir, Out_dir, 630.0, 1430.0)

    def get_synth(self):
        # add security
        if os.path.exists(self.audio_dir_microsoft) and len(os.listdir(self.audio_dir_microsoft)) > 0:
            logging.info("✅ Microsoft voice synthesised already exists. Moving on the next step")

        # we  create necessary directories
        os.makedirs(os.path.dirname(self.audio_dir_microsoft), exist_ok=True)
        os.makedirs(self.audio_dir_microsoft, exist_ok=True)
        os.makedirs(self.transcription_dir_microsoft, exist_ok=True)

        checkpoint_file = os.path.join(self.Temp_dir, self.dir_name, 'checkpoint_getSynth_done.txt')

        if self.check_checkfiles and os.path.exists(checkpoint_file):
            logging.info("✅ Voice synthesis already completed. Moving on the next step")
            return

        try:
            get_synth_main(
                self.Input_dir,
                self.audio_dir,
                self.audio_dir_microsoft,
                self.transcription_dir,
                self.transcription_dir_microsoft,
                self.api_key,
                self.voice,
                str(self.style),
                str(self.styledegree), 
                clean_transcription=True
            )

            # ✅ Si tout est bon, on crée le checkpoint
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint_file).touch()
            logging.info(f"✅ Microsoft voice synthesised is done. Checkpoint created : {checkpoint_file}")

        except Exception as e:
            logging.error(f"General Error in get_synth : {e}")
            raise

    def _classify_pauses(self, words, tresholds):
        """
        Here we classify pauses,
        for a pause lower than treshold i, the integer associated with it will be i+1
        """
        new_words = []
        for i in words:
            t, m, M = self._correct_word(i.mark), i.minTime, i.maxTime
            if t == "":
                dt = M - m
                if dt < tresholds[0] / 5:
                    # We use this case to avoid having pauses at weird positions (due to "liaisons" as an example)
                    new_words += [("::0", m, M)]
                else:
                    treshold = len(tresholds)
                    for i in range(len(tresholds)):
                        if treshold == len(tresholds) and dt < tresholds[i]:
                            treshold = i
                    new_words += [(f"::{treshold + 1}", m, M)]
            else:
                new_words += [(t, m, M)]

        # We now need to convert those triplets we sored in new_words into intervals:
        new_words = [textgrid.Interval(m, M, t) for t, m, M in new_words]
        return new_words

    def _alignement(self):
        checkpoint_file = os.path.join(self.Temp_dir, self.dir_name, f'checkpoint_alignement_{self.aligner}_done.txt')

        if self.check_checkfiles and os.path.exists(checkpoint_file):
            logging.info(f"Alignement avec {self.aligner} déjà effectué. On passe à l'étape suivante.")
            return

        try:
            if self.aligner == "MFA":
                logging.info("Using MFA...")
                self.use_mfa()
                logging.info("Improving dict...")
                self.enrichir_dict()
                logging.info("Using MFA again...")
                self.use_mfa()

            elif self.aligner == "NeMo":
                logging.info("Using NeMo...")
                self.use_NeMo()

            elif self.aligner == "WhisperTS":
                logging.info("Using Whisper-timestamped...")
                self.use_Whisper_timestamped()

            elif self.aligner == "CTCFA":
                logging.info("Using CTC Forced Aligner...")
                self.use_CTCFA()

            elif self.aligner == "whisperX":
                logging.info("Using whisperX...")
                self.use_whisperX()

            else:
                raise ValueError(f"Aligner {self.aligner} non implémenté.")

            # ✅ Création du checkpoint après réussite
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint_file).touch()
            logging.info(f"Alignement avec {self.aligner} terminé. Checkpoint créé : {checkpoint_file}")

        except Exception as e:
            logging.error(f"Erreur dans _alignement avec {self.aligner} : {e}")
            raise


    def regenerate_transcription(self):
        checkpoint_file = os.path.join(self.Temp_dir, self.dir_name, 'checkpoint_regen_txt_done.txt')

        if self.check_checkfiles and os.path.exists(checkpoint_file):
            logging.info("✅ Transcription regeneration already completed (checkpoint found)")
            return

        try:
            logging.info("Using Whisper-timestamped to regenerate transcriptions...")
            self.use_Whisper_timestamped(voices="NS")

            # Copier les transcriptions naturelles générées par Whisper
            whisper_nat_transcription_dir = os.path.join(self.Data_dir, 'voice', self.dir_name, "WhisperTS_textgrid_files_transcription")
            target_nat_transcription_dir = os.path.join(self.Data_dir, 'voice', self.dir_name, "transcription")

            os.makedirs(target_nat_transcription_dir, exist_ok=True)
            for txt_file in glob.glob(os.path.join(whisper_nat_transcription_dir, '*.txt')):
                shutil.copy(txt_file, target_nat_transcription_dir)
                logging.info(f"Transcription copied: {txt_file} to {target_nat_transcription_dir}")

            # Copier les transcriptions synthétiques si elles existent
            whisper_synth_transcription_dir = os.path.join(self.Input_dir + self.microsoft_folder_suffix, "WhisperTS_textgrid_files_transcription")
            target_synth_transcription_dir = self.transcription_dir_microsoft

            os.makedirs(target_synth_transcription_dir, exist_ok=True)
            if os.path.exists(whisper_synth_transcription_dir):
                for txt_file in glob.glob(os.path.join(whisper_synth_transcription_dir, '*.txt')):
                    shutil.copy(txt_file, target_synth_transcription_dir)
                    logging.info(f"Transcription copied: {txt_file} to {target_synth_transcription_dir}")

            # ✅ Création du checkpoint après réussite
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint_file).touch()
            logging.info(f"✅ Transcription regeneration is done. Checkpoint created: {checkpoint_file}")

        except Exception as e:
            logging.error(f"Error in regenerate_transcription : {e}")
            raise

    def preprocess_audio(self):
        checkpoint_file = os.path.join(self.Temp_dir, self.dir_name, 'checkpoint_preprocess_audio_done.txt')

        if self.check_checkfiles and os.path.exists(checkpoint_file):
            logging.info("✅ Preprocessing audio is already done. We go to next step.")
            return

        mp3_file_path = os.path.join(self.Data_dir, 'voice', self.Input_dir, "brute", "segment.mp3")
        wav_file_path = os.path.join(self.Data_dir, 'voice', self.Input_dir, "brute", "segment.wav")
        demucs_file_path = os.path.join(self.Data_dir, 'voice', self.Input_dir, "brute", "segment_demucs.wav")
        output_dir = os.path.join(self.Data_dir, 'voice', self.Input_dir, "audio")

        try:
            if self.convert_to_wav:
                if os.path.exists(mp3_file_path):
                    logging.info(f"MP3 Found. Conversion into WAV : {mp3_file_path} -> {wav_file_path}")
                    convert_main(mp3_file_path, wav_file_path)
                elif os.path.exists(wav_file_path):
                    logging.info(f"✅ WAV already exits : {wav_file_path}")
                else:
                    raise FileNotFoundError(f"Neither MP3 nor Wav were found in : {os.path.dirname(mp3_file_path)}")

            # Suppression de l'ancien fichier Demucs s'il existe
            if os.path.exists(demucs_file_path):
                os.remove(demucs_file_path)

            logging.info(f"Using Demucs to remove noise : {wav_file_path} -> {demucs_file_path}")
            demucs_main(wav_file_path, demucs_file_path)

            logging.info(f"Splitting file into segments : {demucs_file_path} -> {output_dir}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            preprocess_main(demucs_file_path, output_dir)

            # ✅ Si tout est bon, on crée le checkpoint
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint_file).touch()
            logging.info(f"✅ Audio Preprocessing is done. Checkpoint created : {checkpoint_file}")

        except subprocess.CalledProcessError as e:
            logging.error(f"Error during the commad execution : {e}")
            raise
        except Exception as e:
            logging.error(f"General Error in preprocess_audio : {e}")
            raise

    def calculate_loudness_adjustement(self):
        print(f"🔄 Running compute_BDD2_loudness.calculate_loudness_adjustement with BDD2_dir: {self.BDD2_dir}, BDD3_dir: {self.BDD3_dir}")
        compute_loudness_adjustments.calculate_loudness_adjustment(self.BDD2_dir, self.BDD3_dir)


    def run_pipeline(self):
        print("\n" + "=" * 60)
        print("Starting VOICE CREATION PROCESS")
        print("=" * 60)

        if self.preprocess:
            print("\n" + "=" * 60)
            print("STEP 1 : AUDIO PREPROCESSING")
            print("=" * 60)
            self.preprocess_audio()

        ######################
        # Are any of these steps below really needed?
        # --> all of this is already been done in new_pipeline.py, no?

        if self.regen_txt:
            print("\n" + "=" * 60)
            print("Step 2 : REGENERATING TRANSCRIPTION")
            print("=" * 60)
            self.regenerate_transcription()

        if self.get_synth:
            print("\n" + "=" * 60)
            print("Step 3 : GENERATING SYNTHETIC VOICE")
            print("=" * 60)
            self.get_synth()
        
        ######################

        print("\n" + "-"*60)
        print("STEP 4: VERIFYING WAV FILES")
        print("-"*60)
        # We preprocess wav files:
        voice_dir = self.Input_dir
        files = os.listdir(voice_dir)
        names = []
        [(names.append(n[:-4]) if n[-4:] == ".wav" else 0) for n in files]

        print(f"Normalizing {len(names)} WAV files...")
        for i, n in enumerate(names):
            print(f"Processing file {i+1}/{len(names)}: {n}.wav")
            audio = AudioSegment.from_file(os.path.join(voice_dir, n + ".wav"))
            audio = audio.set_channels(1)
            audio.export(os.path.join(voice_dir, n + ".wav"), format="wav")

        if self.do_alignement:
            print("\n" + "-"*60)
            print(f"STEP 5: ALIGNMENT WITH {self.aligner}")
            print("-"*60)
            self._alignement()

            # Display information about generated textgrid
            self.log_textgrid_info(self.textgrid_dir, "Textgrids for natural voice")
            self.log_textgrid_info(self.textgrid_microsoft_dir, "Textgrids for synthetic voice")

        if self.levenshtein_correction:
            print("\n" + "=" * 60)
            print("Step 6 : LEVENSHTEIN CORRECTION")
            print("=" * 60)
            self.execute_levenstein_correction()

        print("\n" * 60)
        print("STEP 7: EXTRACTING DATA FROM SEGMENTS")
        print("-"*60)
        self.extract_process_segments()

        print("\n" + "-"*60)
        print("STEP 8: USING NEEDLEMAN-WUNSCH ALIGNMENT METHOD")
        print("-" * 60)
        self.needleman_wunsch_alignement()

        print("\n" + "-"*60)
        print("STEP 9: ADDING BREAKS")
        print("-"*60)
        self.add_breaks()
        
        print("\n" + "-"*60)
        print("STEP 10: COMPLETING AUDIO PATHS")
        print("-"*60)
        self.complete_audio_paths()
        
        print("\n" + "-"*60)
        print("STEP 11: COMPUTING LOUDNESS AND ADJUSTMENT COEFFICIENT")
        print("-"*60)
        self.calculate_loudness_adjustement()

        if not os.path.exists(self.BDD3_dir):
            raise FileNotFoundError(f"❌ ERROR: The file {self.BDD3_dir} is not created.verify compute_BDD2_loudness.py.")

        print("\n" + "-"*60)
        print("STEP 11: COMPUTING RATE CHANGES")
        print("-"*60)
        self.calculate_rate()

        print("\n" + "-"*60)
        print("STEP 12: CREATING SSML")
        print("-"*60)
        self.get_wav()

        if self.gen_voice:
            print("\n" + "-"*60)
            print("STEP 13: GENERATING WAV FILES")
            print("-"*60)
            self.Text_to_speech_df()

        #### Generate final XML for LLM fine-tuning
        # text / break / attributes

        print("\n" + "-"*60)
        print("STEP 14: CREATING TRAINING DATA FOR LLM FINE-TUNING")
        print("-"*60)
        bdd_ssml_path = os.path.join(Out_dir, 'results', self.dir_name, 'BDD_ssml.csv')
        output_path = os.path.join(Out_dir, 'results', self.dir_name, f'training_data_{self.dir_name}.json')
        create_training_data.create_training_data(bdd_ssml_path, output_path)
        results_folder = os.path.join(Out_dir, "results")
        combined_json_path = os.path.join(results_folder, "bdd.json")
        create_training_data.combine_training_jsons(results_folder, combined_json_path)
        print(f"Training data created at {output_path}")
 
        print("\n" + "=" * 60)
        print("VOICE CREATION PROCESS COMPLETED")
        print("="*60)

if __name__ == '__main__':
    # fr-FR-HenriNeural; fr-FR-VivienneMultilingualNeural fr-FR-DeniseNeura
    for VOICE in ["Aznavour_EP04"]:
        Object = Voc(
            VOICE,
            preprocess=True, # put it true if you want to cut original audio
            regen_txt=True,
            do_alignement=True,
            getSynth=False,
            levenshtein_correction=True,
            aligner="WhisperTS",
            convert_to_wav=True,
            gen_voice=False,
            voice="fr-FR-HenriNeural",
            check_checkfiles=False)
        Object.run_pipeline()