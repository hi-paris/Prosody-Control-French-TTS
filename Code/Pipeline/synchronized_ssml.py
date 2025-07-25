#!/usr/bin/env python3
"""
Module de synchronisation des pauses entre voix naturelle et voix de synthèse.

Ce module implémente une nouvelle pipeline permettant de générer des fichiers SSML
avec des pauses parfaitement synchronisées entre la voix naturelle et la voix de synthèse.
Il suit un processus en plusieurs étapes comme détaillé sur le plan de conception.
"""

import os
import re
import csv
import sys
import glob
import logging
import pandas as pd
import numpy as np
import textgrid
import xml.etree.ElementTree as ET
import xml.dom.minidom
import azure.cognitiveservices.speech as speechsdk
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import make_chunks

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SynchronizedSSMLPipeline:
    """
    Classe pour la génération de voix de synthèse avec synchronisation précise des pauses.
    
    Cette classe implémente un workflow en plusieurs étapes :
    1. Génération de SSML initial avec les pauses extraites des voix naturelles
    2. Synthèse TTS intermédiaire pour calibration
    3. Génération de SSML final avec les pauses ajustées et les facteurs de modification
    4. Synthèse TTS finale avec durées optimisées
    
    Attributes:
        dir_name (str): Nom du répertoire contenant les fichiers source
        base_dir (Path): Chemin de base du projet
        out_dir (Path): Chemin de sortie pour les résultats
        data_dir (Path): Chemin des données
        api_key (str): Clé API Azure pour la synthèse vocale
        region (str): Région Azure pour la synthèse vocale
        voice (str): Voix Azure à utiliser (ex: fr-FR-HenriNeural)
        style (str, optional): Style de voix à utiliser
        style_degree (int, optional): Degré d'intensité du style
        textgrid_folder (Path): Dossier contenant les fichiers TextGrid
        transcription_folder (Path): Dossier contenant les transcriptions
        ssml_output_folder (Path): Dossier de sortie pour les fichiers SSML
        initial_pause_threshold (int): Seuil pour ignorer les très courtes pauses (en ms)
    """
    
    def __init__(
        self,
        dir_name,
        base_dir=None,
        out_dir=None,
        data_dir=None,
        api_key=None,
        region="francecentral",
        voice="fr-FR-HenriNeural",
        style=None,
        style_degree=2,
        aligner="WhisperTS",
        initial_pause_threshold=50
    ):
        """
        Initialise la pipeline de synchronisation SSML.
        
        Args:
            dir_name (str): Nom du répertoire contenant les fichiers source
            base_dir (Path, optional): Chemin de base du projet
            out_dir (Path, optional): Chemin de sortie pour les résultats
            data_dir (Path, optional): Chemin des données
            api_key (str, optional): Clé API Azure pour la synthèse vocale
            region (str, optional): Région Azure pour la synthèse vocale
            voice (str, optional): Voix Azure à utiliser
            style (str, optional): Style de voix à utiliser
            style_degree (int, optional): Degré d'intensité du style
            aligner (str, optional): Type d'aligneur utilisé pour les TextGrids
            initial_pause_threshold (int, optional): Seuil pour ignorer les pauses courtes
        """
        # Configuration des chemins
        self.dir_name = dir_name
        
        # Si les chemins ne sont pas fournis, utiliser les valeurs par défaut
        if base_dir is None:
            self.base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().__str__()
        else:
            self.base_dir = Path(base_dir)
            
        if out_dir is None:
            self.out_dir = os.path.join(self.base_dir, "Out")
        else:
            self.out_dir = Path(out_dir)
            
        if data_dir is None:
            self.data_dir = os.path.join(self.base_dir, "Data")
        else:
            self.data_dir = Path(data_dir)
            
        # Configuration Azure
        self.api_key = api_key if api_key else self._load_api_key()
        self.region = region
        self.voice = voice
        self.style = style
        self.style_degree = style_degree
        
        # Configuration des dossiers
        self.input_dir = os.path.join(self.data_dir, 'voice', self.dir_name)
        self.input_dir_microsoft = os.path.join(self.data_dir, 'voice', f"{self.dir_name}_microsoft")
        
        # Chemins des dossiers TextGrid et transcription
        self.aligner = aligner
        self.textgrid_folder = os.path.join(self.input_dir, f"{self.aligner}_textgrid_files")
        self.transcription_folder = os.path.join(self.input_dir, "transcription")
        self.ssml_output_folder = os.path.join(self.input_dir_microsoft, "ssml")
        
        # Chemins des dossiers audio
        self.audio_dir = os.path.join(self.input_dir, "audio")
        self.audio_dir_microsoft = os.path.join(self.input_dir_microsoft, "audio")
        
        # Configuration des paramètres
        self.initial_pause_threshold = initial_pause_threshold
        
        # Dossiers de sortie pour les étapes intermédiaires
        self.temp_dir = os.path.join(self.out_dir, "Temp", self.dir_name)
        self.results_dir = os.path.join(self.out_dir, "results", self.dir_name)
        
        # Créer les dossiers nécessaires
        self._create_directories()
        
        # Journalisation
        logging.info(f"Initialisation de SynchronizedSSMLPipeline pour {self.dir_name}")
        logging.info(f"Dossier TextGrid: {self.textgrid_folder}")
        logging.info(f"Dossier Transcription: {self.transcription_folder}")
        logging.info(f"Dossier de sortie SSML: {self.ssml_output_folder}")
    
    def _load_api_key(self):
        """Charge la clé API Azure depuis un fichier."""
        try:
            api_key_path = os.path.join(self.base_dir, "Azure_API_key.txt")
            with open(api_key_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            logging.error("Fichier de clé API Azure non trouvé.")
            return None
    
    def _create_directories(self):
        """Crée les répertoires nécessaires pour le fonctionnement de la pipeline."""
        directories = [
            self.ssml_output_folder,
            self.temp_dir,
            self.results_dir,
            os.path.join(self.results_dir, "Temp")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Dossier créé/vérifié: {directory}")
    
    def extract_words_and_pauses(self, textgrid_file):
        """
        Extrait la séquence exacte de mots et pauses du TextGrid.
        
        Args:
            textgrid_file (str): Chemin vers le fichier TextGrid
            
        Returns:
            list: Séquence d'éléments [(type, contenu, durée)]
        """
        logging.info(f"Extraction des mots et pauses de {textgrid_file}")
        
        tg = textgrid.TextGrid()
        tg.read(textgrid_file)
        
        sequence = []
        ignore_initial_pause = True
        
        # On suppose que la première tier contient les mots
        tier = tg.tiers[0]
        
        for interval in tier.intervals:
            text = interval.mark.strip()
            start_ms = round(interval.minTime * 1000)
            end_ms = round(interval.maxTime * 1000)
            duration_ms = end_ms - start_ms
            
            if not text or text == " ":  # C'est une pause
                if not ignore_initial_pause or duration_ms >= self.initial_pause_threshold:
                    sequence.append(("pause", None, duration_ms))
            else:  # C'est un mot
                sequence.append(("word", text, 0))
                ignore_initial_pause = False
        
        logging.info(f"Séquence extraite: {len(sequence)} éléments")
        return sequence
    
    def normalize_word(self, word):
        """
        Normalise un mot pour l'alignement.
        
        Args:
            word (str): Mot à normaliser
            
        Returns:
            str: Mot normalisé
        """
        if not word:
            return ""
        # Convertir en minuscules
        word = word.lower()
        # Enlever la ponctuation
        word = re.sub(r'[^\w\s]', '', word)
        # Enlever les accents (simplification)
        accents = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'î': 'i', 'ï': 'i',
            'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ÿ': 'y', 'ç': 'c'
        }
        for accent, sans_accent in accents.items():
            word = word.replace(accent, sans_accent)
        return word
    
    def align_sequences(self, natural_sequence, synth_words):
        """
        Aligne précisément les mots synthétiques avec la séquence naturelle.
        
        Args:
            natural_sequence: Liste [(type, contenu, durée)]
            synth_words: Liste des mots de la voix synthétique
        
        Returns:
            list: Séquence alignée pour le SSML [(type, contenu)]
        """
        logging.info("Alignement des séquences naturelle et synthétique")
        
        # Extraire juste les mots de la séquence naturelle
        natural_words = [item[1] for item in natural_sequence if item[0] == "word"]
        
        # Normaliser les mots pour une meilleure correspondance
        normalized_natural = [self.normalize_word(word) for word in natural_words]
        normalized_synth = [self.normalize_word(word) for word in synth_words]
        
        # Trouver les correspondances entre mots
        word_mappings = {}  # synth_idx -> natural_sequence_idx
        
        # Pour chaque mot synthétique, chercher sa correspondance
        for syn_idx, syn_word in enumerate(normalized_synth):
            best_match_idx = -1
            best_match_score = 0
            
            # Rechercher dans les mots naturels
            for nat_word_idx, nat_word in enumerate(normalized_natural):
                # Calcul simple de correspondance
                if syn_word == nat_word:
                    best_match_idx = nat_word_idx
                    break
                elif syn_word in nat_word or nat_word in syn_word:
                    # Calculer un score de correspondance basé sur la longueur de chevauchement
                    overlap = min(len(syn_word), len(nat_word))
                    score = overlap / max(len(syn_word), len(nat_word))
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = nat_word_idx
            
            if best_match_idx >= 0:
                word_mappings[syn_idx] = best_match_idx
        
        # Maintenant, créer la séquence SSML avec les pauses correctement placées
        ssml_sequence = []
        
        # Convertir les mappings de mots en mappings d'indices dans la séquence naturelle complète
        nat_word_to_seq_idx = {}
        nat_word_idx = 0
        
        for seq_idx, item in enumerate(natural_sequence):
            if item[0] == "word":
                nat_word_to_seq_idx[nat_word_idx] = seq_idx
                nat_word_idx += 1
        
        # Créer la séquence SSML
        for syn_idx, word in enumerate(synth_words):
            ssml_sequence.append(("word", word))
            
            # Si ce mot a une correspondance et qu'il existe une pause après
            if syn_idx in word_mappings:
                nat_word_idx = word_mappings[syn_idx]
                nat_seq_idx = nat_word_to_seq_idx[nat_word_idx]
                
                # Vérifier s'il y a une pause après ce mot
                if nat_seq_idx + 1 < len(natural_sequence) and natural_sequence[nat_seq_idx + 1][0] == "pause":
                    pause_duration = natural_sequence[nat_seq_idx + 1][2]
                    ssml_sequence.append(("pause", pause_duration))
        
        logging.info(f"Séquence alignée: {len(ssml_sequence)} éléments")
        return ssml_sequence
    
    def generate_ssml(self, aligned_sequence, pitch_adj=0, rate_adj=0, volume_adj=0):
        """
        Génère le SSML à partir de la séquence alignée, dans un format compatible avec Azure.
        
        Args:
            aligned_sequence: Liste [(type, contenu)]
            pitch_adj (float): Ajustement de la hauteur (en %)
            rate_adj (float): Ajustement du débit (en %)
            volume_adj (float): Ajustement du volume (en %)
        
        Returns:
            str: Contenu SSML
        """
        logging.info("Génération du SSML avec ajustements prosodiques")
        
        # Formatage des ajustements prosodiques
        pitch_mod = f"{pitch_adj:+.2f}%" if pitch_adj != 0 else "+0%"
        rate_mod = f"{rate_adj:+.2f}%" if rate_adj != 0 else "+0%"
        volume_mod = f"{volume_adj:+.2f}%" if volume_adj != 0 else "+0%"
        
        # Construire le texte avec des balises de pause intégrées
        text_parts = []
        
        for element_type, content in aligned_sequence:
            if element_type == "word":
                text_parts.append(content)
            elif element_type == "pause":
                # Ajouter la balise de pause
                text_parts.append(f'<break time="{content}ms"/>')
        
        # Joindre le texte avec des espaces
        full_text = " ".join(text_parts)
        
        # Ajouter la balise prosody si des ajustements sont spécifiés
        if pitch_adj != 0 or rate_adj != 0 or volume_adj != 0:
            full_text = f'<prosody pitch="{pitch_mod}" rate="{rate_mod}" volume="{volume_mod}">{full_text}</prosody>'
        
        # Ajouter le style si spécifié
        if self.style:
            full_text = f'<mstts:express-as style="{self.style}" styledegree="{self.style_degree}">{full_text}</mstts:express-as>'
        
        # Créer le SSML complet au format attendu par Azure
        ssml = f"""<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" version="1.0" xml:lang="fr-FR">
    <voice name="{self.voice}">
        {full_text}
    </voice>
</speak>"""
        
        # Formater proprement le XML
        try:
            pretty_xml = xml.dom.minidom.parseString(ssml).toprettyxml(indent="  ")
            return pretty_xml
        except:
            logging.warning("Impossible de formater le XML proprement, retour du SSML brut")
            return ssml
    
    def save_ssml(self, ssml_content, output_file):
        """
        Sauvegarde le fichier SSML.
        
        Args:
            ssml_content (str): Contenu SSML à sauvegarder
            output_file (str): Chemin du fichier de sortie
        """
        # Créer le dossier parent si nécessaire
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ssml_content)
        logging.info(f"SSML généré et sauvegardé: {output_file}")
    
    def synthesize_ssml(self, ssml_text, output_file):
        """
        Synthétise un texte SSML avec la voix configurée et sauvegarde le résultat.
        
        Args:
            ssml_text (str): Le texte SSML à synthétiser
            output_file (str): Chemin du fichier audio de sortie
            
        Returns:
            bool: True si la synthèse a réussi, False sinon
        """
        logging.info(f"Synthèse vocale vers {output_file}")
        
        if not self.api_key:
            logging.error("Clé API Azure non configurée!")
            return False
        
        try:
            # Configurer le service de synthèse vocale
            speech_config = speechsdk.SpeechConfig(subscription=self.api_key, region=self.region)
            speech_config.speech_synthesis_voice_name = self.voice
            
            # Configurer la sortie audio
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
            
            # Créer le synthétiseur
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            result = speech_synthesizer.speak_ssml_async(ssml_text).get()
            
            # Vérifier le résultat
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logging.info(f"Synthèse audio terminée: {output_file}")
                return True
            else:
                logging.error(f"Échec de la synthèse: {result.cancellation_details.reason}")
                if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logging.error(f"Détails: {result.cancellation_details.error_details}")
                return False
        
        except Exception as e:
            logging.error(f"Erreur lors de la synthèse: {e}")
            return False
    
    def process_files_v1(self):
        """
        Première étape: Génère les fichiers SSML avec les pauses naturelles.
        
        Cette fonction:
        1. Trouve les paires de fichiers TextGrid et Transcription correspondants
        2. Extrait les séquences de mots et pauses
        3. Aligne les séquences
        4. Génère et sauvegarde les SSML initiaux
        
        Returns:
            list: Liste des fichiers SSML générés
        """
        logging.info("=== ÉTAPE 1: GÉNÉRATION DES FICHIERS SSML INITIAUX ===")
        
        # Trouver les paires de fichiers
        textgrid_files = glob.glob(os.path.join(self.textgrid_folder, "*.TextGrid"))
        transcription_files = glob.glob(os.path.join(self.transcription_folder, "*.txt"))
        
        # Extraire les noms de base pour faciliter la correspondance
        textgrid_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in textgrid_files}
        transcription_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in transcription_files}
        
        ssml_files = []
        
        # Traiter chaque paire
        for basename in sorted(set(textgrid_basenames) & set(transcription_basenames)):
            logging.info(f"Traitement de {basename}")
            
            textgrid_file = textgrid_basenames[basename]
            transcription_file = transcription_basenames[basename]
            output_file = os.path.join(self.ssml_output_folder, f"SSML_V1_{basename}.xml")
            
            try:
                # 1. Extraire les mots et pauses du TextGrid
                natural_sequence = self.extract_words_and_pauses(textgrid_file)
                
                # 2. Lire le texte de la transcription
                with open(transcription_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                synth_words = text.split()
                
                # 3. Aligner les séquences
                aligned_sequence = self.align_sequences(natural_sequence, synth_words)
                
                # 4. Générer le SSML
                ssml_content = self.generate_ssml(aligned_sequence)
                
                # 5. Sauvegarder le SSML
                self.save_ssml(ssml_content, output_file)
                
                ssml_files.append(output_file)
                
            except Exception as e:
                logging.error(f"Erreur lors du traitement de {basename}: {e}")
        
        logging.info(f"Génération de {len(ssml_files)} fichiers SSML V1 terminée")
        return ssml_files
    
    def synthesize_calibration(self, ssml_files):
        """
        Deuxième étape: Synthétise les fichiers SSML pour calibration.
        
        Args:
            ssml_files (list): Liste des fichiers SSML à synthétiser
            
        Returns:
            list: Liste des fichiers audio générés
        """
        logging.info("=== ÉTAPE 2: SYNTHÈSE TTS DE CALIBRATION ===")
        
        audio_files = []
        
        for ssml_file in ssml_files:
            try:
                basename = os.path.basename(ssml_file).replace("SSML_V1_", "")
                output_file = os.path.join(self.temp_dir, f"TTS_V1_{basename}.wav")
                
                # Lire le contenu SSML
                with open(ssml_file, "r", encoding="utf-8") as f:
                    ssml_content = f.read()
                
                # Synthétiser le SSML
                if self.synthesize_ssml(ssml_content, output_file):
                    audio_files.append(output_file)
            
            except Exception as e:
                logging.error(f"Erreur lors de la synthèse de {ssml_file}: {e}")
        
        logging.info(f"Synthèse de {len(audio_files)} fichiers audio de calibration terminée")
        return audio_files
    
    def analyze_durations(self, audio_files):
        """
        Analyse les durées des fichiers audio de calibration pour ajuster les paramètres.
        
        Args:
            audio_files (list): Liste des fichiers audio à analyser
            
        Returns:
            dict: Dictionnaire des ajustements recommandés
        """
        logging.info("=== ÉTAPE 3: ANALYSE DES DURÉES ET CALCUL DES AJUSTEMENTS ===")
        
        adjustments = {}
        
        for audio_file in audio_files:
            try:
                basename = os.path.basename(audio_file).replace("TTS_V1_", "").replace(".wav", "")
                
                # Charger le fichier audio
                audio = AudioSegment.from_file(audio_file)
                duration_ms = len(audio)
                
                # Trouver les fichiers correspondants (natural et synthetic)
                natural_audio_file = os.path.join(self.audio_dir, f"{basename}.wav")
                
                if os.path.exists(natural_audio_file):
                    natural_audio = AudioSegment.from_file(natural_audio_file)
                    natural_duration_ms = len(natural_audio)
                    
                    # Calculer l'ajustement du débit nécessaire
                    if duration_ms > 0:
                        rate_ratio = natural_duration_ms / duration_ms
                        rate_adjustment = (rate_ratio - 1) * 100  # en pourcentage
                        
                        # Limites raisonnables pour l'ajustement du débit
                        rate_adjustment = max(-50, min(100, rate_adjustment))
                        
                        adjustments[basename] = {
                            'rate_adjustment': rate_adjustment,
                            'natural_duration': natural_duration_ms,
                            'synthetic_duration': duration_ms
                        }
                        
                        logging.info(f"{basename}: Durée naturelle={natural_duration_ms}ms, Durée synthétique={duration_ms}ms, Ajustement={rate_adjustment:.2f}%")
            
            except Exception as e:
                logging.error(f"Erreur lors de l'analyse de {audio_file}: {e}")
        
        logging.info(f"Analyse de {len(adjustments)} fichiers audio terminée")
        return adjustments
    
    def generate_optimized_ssml(self, adjustments):
        """
        Quatrième étape: Génère des fichiers SSML optimisés avec les ajustements.
        
        Args:
            adjustments (dict): Dictionnaire des ajustements recommandés
            
        Returns:
            list: Liste des fichiers SSML optimisés générés
        """
        logging.info("=== ÉTAPE 4: GÉNÉRATION DES FICHIERS SSML OPTIMISÉS ===")
        
        ssml_files = []
        
        for basename, values in adjustments.items():
            try:
                # Fichier SSML V1 source
                ssml_v1_file = os.path.join(self.ssml_output_folder, f"SSML_V1_{basename}.xml")
                output_file = os.path.join(self.ssml_output_folder, f"SSML_V2_{basename}.xml")
                
                if not os.path.exists(ssml_v1_file):
                    logging.warning(f"Fichier SSML V1 non trouvé pour {basename}")
                    continue
                
                # Lire le contenu SSML V1
                with open(ssml_v1_file, "r", encoding="utf-8") as f:
                    ssml_v1_content = f.read()
                
                # Extraire les séquences alignées du SSML V1
                root = ET.fromstring(ssml_v1_content)
                
                # Trouver les éléments de texte et de pause
                voice_element = root.find(".//voice")
                
                if voice_element is None:
                    logging.warning(f"Balise voice non trouvée dans {ssml_v1_file}")
                    continue
                
                # Reconstruire la séquence alignée
                aligned_sequence = []
                
                # Traiter tous les éléments dans la balise voice
                for elem in voice_element.iter():
                    if elem.tag == "break":
                        pause_duration = int(elem.attrib.get("time", "0ms").replace("ms", ""))
                        aligned_sequence.append(("pause", pause_duration))
                    elif elem.text and elem.text.strip():
                        aligned_sequence.append(("word", elem.text.strip()))
                
                # Générer le SSML optimisé avec l'ajustement du débit
                rate_adjustment = values.get('rate_adjustment', 0)
                ssml_content = self.generate_ssml(aligned_sequence, rate_adj=rate_adjustment)
                
                # Sauvegarder le SSML optimisé
                self.save_ssml(ssml_content, output_file)
                
                ssml_files.append(output_file)
            
            except Exception as e:
                logging.error(f"Erreur lors de la génération du SSML optimisé pour {basename}: {e}")
        
        logging.info(f"Génération de {len(ssml_files)} fichiers SSML V2 terminée")
        return ssml_files
    
    def synthesize_final(self, ssml_files):
        """
        Cinquième étape: Synthétise les fichiers SSML finaux.
        
        Args:
            ssml_files (list): Liste des fichiers SSML à synthétiser
            
        Returns:
            list: Liste des fichiers audio générés
        """
        logging.info("=== ÉTAPE 5: SYNTHÈSE TTS FINALE ===")
        
        audio_files = []
        
        for ssml_file in ssml_files:
            try:
                basename = os.path.basename(ssml_file).replace("SSML_V2_", "")
                output_file = os.path.join(self.results_dir, f"{basename}")
                
                # Lire le contenu SSML
                with open(ssml_file, "r", encoding="utf-8") as f:
                    ssml_content = f.read()
                
                # Synthétiser le SSML
                if self.synthesize_ssml(ssml_content, output_file):
                    audio_files.append(output_file)
            
            except Exception as e:
                logging.error(f"Erreur lors de la synthèse de {ssml_file}: {e}")

        logging.info(f"Synthèse de {len(audio_files)} fichiers audio finaux terminée")
        return audio_files
    
    def concatenate_audio_files(self, audio_files, output_file=None):
        """
        Sixième étape: Concatène tous les fichiers audio en un seul.
        
        Args:
            audio_files (list): Liste des fichiers audio à concaténer
            output_file (str, optional): Chemin du fichier audio de sortie
            
        Returns:
            str: Chemin du fichier audio concaténé
        """
        logging.info("=== ÉTAPE 6: CONCATÉNATION DES FICHIERS AUDIO ===")
        
        if not audio_files:
            logging.warning("Aucun fichier audio à concaténer")
            return None
        
        if output_file is None:
            output_file = os.path.join(self.results_dir, f"{self.dir_name}_final.wav")
        
        try:
            # Trier les fichiers par numéro pour maintenir l'ordre correct
            sorted_files = sorted(audio_files, key=lambda x: int(re.search(r'segment_ph(\d+)', x).group(1)) if re.search(r'segment_ph(\d+)', x) else 0)
            
            # Concaténer les fichiers audio
            combined = AudioSegment.empty()
            for audio_file in sorted_files:
                if os.path.exists(audio_file):
                    segment = AudioSegment.from_file(audio_file)
                    combined += segment
            
            # Sauvegarder le fichier concaténé
            combined.export(output_file, format="wav")
            logging.info(f"Fichier audio final sauvegardé: {output_file}")
            
            return output_file
        
        except Exception as e:
            logging.error(f"Erreur lors de la concaténation des fichiers audio: {e}")
            return None
    
    
    def clean_temp_files(self, keep_ssml=True):
        """
        Nettoie les fichiers temporaires générés pendant le processus.
        
        Args:
            keep_ssml (bool): Conserver les fichiers SSML
        """
        logging.info("=== NETTOYAGE DES FICHIERS TEMPORAIRES ===")
        
        # Fichiers à supprimer
        patterns = [
            os.path.join(self.temp_dir, "TTS_V1_*.wav"),
            os.path.join(self.results_dir, "Temp", "*.wav")
        ]
        
        if not keep_ssml:
            patterns.append(os.path.join(self.ssml_output_folder, "SSML_V1_*.xml"))
        
        # Supprimer les fichiers correspondant aux patterns
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    os.remove(file)
                    logging.debug(f"Fichier supprimé: {file}")
                except Exception as e:
                    logging.warning(f"Erreur lors de la suppression de {file}: {e}")
        
        logging.info("Nettoyage terminé")
    
    def run_pipeline(self):
        """
        Exécute la pipeline complète de synchronisation SSML.
        
        Cette méthode enchaîne toutes les étapes du processus:
        1. Génération des fichiers SSML initiaux
        2. Synthèse TTS de calibration
        3. Analyse des durées
        4. Génération des fichiers SSML optimisés
        5. Synthèse TTS finale
        6. Concaténation des fichiers audio
        
        Returns:
            str: Chemin du fichier audio final
        """
        logging.info(f"=== DÉMARRAGE DE LA PIPELINE POUR {self.dir_name} ===")
        
        # Étape 1: Génération des fichiers SSML initiaux
        ssml_files_v1 = self.process_files_v1()
        
        if not ssml_files_v1:
            logging.error("Aucun fichier SSML initial généré, arrêt de la pipeline")
            return None
        
        # Étape 2: Synthèse TTS de calibration
        audio_files_v1 = self.synthesize_calibration(ssml_files_v1)
        
        if not audio_files_v1:
            logging.error("Aucun fichier audio de calibration généré, arrêt de la pipeline")
            return None
        
        # Étape 3: Analyse des durées
        adjustments = self.analyze_durations(audio_files_v1)
        
        if not adjustments:
            logging.warning("Aucun ajustement calculé, utilisation des SSML initiaux pour la suite")
            ssml_files_v2 = ssml_files_v1
        else:
            # Étape 4: Génération des fichiers SSML optimisés
            ssml_files_v2 = self.generate_optimized_ssml(adjustments)
        
        if not ssml_files_v2:
            logging.error("Aucun fichier SSML optimisé généré, arrêt de la pipeline")
            return None
        
        # Étape 5: Synthèse TTS finale
        audio_files_v2 = self.synthesize_final(ssml_files_v2)
        
        if not audio_files_v2:
            logging.error("Aucun fichier audio final généré, arrêt de la pipeline")
            return None
        
        # Étape 6: Concaténation des fichiers audio
        final_audio = self.concatenate_audio_files(audio_files_v2)
        
        # Nettoyage des fichiers temporaires
        self.clean_temp_files()
        
        logging.info(f"=== PIPELINE TERMINÉE AVEC SUCCÈS: {final_audio} ===")
        
        return final_audio
    

if __name__ == "__main__":
    # Paramètres définis en dur
    pipeline = SynchronizedSSMLPipeline(
        dir_name="/root/mon_projet_TTS/mon_projet_TTS/Data/voice/Aznavour_EP01",
        base_dir="os.path.join(os.path.dirname(os.path.abspath(__file__))",
        out_dir=None,  # Laisse à None si tu veux les chemins par défaut
        data_dir=None,  # Pareil ici
        api_key= "Be3j8kmplJvJNVgBnwrnXMNSABCQKD2kPyiDd9U4HR9U1l0xm43LJQQJ99BAAC5T7U2XJ3w3AAAAACOGlKVG",  # Ou colle ta clé directement ici
        region="francecentral",
        voice="fr-FR-HenriNeural",
        style=None,
        style_degree=2,
        aligner="WhisperTS",
        initial_pause_threshold=50
    )

    final_audio = pipeline.run_pipeline()

    if final_audio:
        print(f"\nAudio final généré avec succès: {final_audio}")
    else:
        print("\nÉchec de la génération de l'audio final")
