"""
Batch process SSML files to generate synthesized voice audio using Azure Speech Services.

Usage: python synthesize_ssml_voice.py <AZURE_SPEECH_KEY> <AZURE_SPEECH_REGION> <SSML_FOLDER>
"""


import os
import azure.cognitiveservices.speech as speechsdk
import tempfile
import sys
import glob
import re
import logging

# Configuration du logging
# REMOVE or COMMENT OUT the basicConfig call to let the main script handle configuration
# if not logging.getLogger().hasHandlers():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
logger = logging.getLogger(__name__)


def read_ssml_file(ssml_file_path):
    """
    Lit un fichier SSML et retourne son contenu.

    Args:
        ssml_file_path (str): Chemin vers le fichier SSML à lire.

    Returns:
        str or None: Contenu du fichier SSML ou None en cas d'erreur.
    """
    try:
        with open(ssml_file_path, 'r', encoding='utf-8') as f:
            ssml_content = f.read()
        return ssml_content
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier SSML: {e}")
        return None


def clean_ssml_for_azure(ssml_text, voice="fr-FR-HenriNeural"):
    """
    Nettoie et prépare le SSML pour le rendre compatible avec Azure Speech Services.
    
    Cette fonction:
    - Supprime la déclaration XML si présente
    - S'assure que le SSML contient une balise <voice>
    - Vérifie que l'attribut xml:lang est présent
    
    Args:
        ssml_text (str): Texte SSML à nettoyer.
    
    Returns:
        str: SSML nettoyé et préparé pour Azure.
    """
    # Supprimer la déclaration XML si présente
    if "<?xml" in ssml_text:
        ssml_text = ssml_text[ssml_text.find("<speak"):]
    
    # Vérifier si le SSML contient déjà une balise voice
    if "<voice" not in ssml_text:
        # Trouver la position après la balise speak d'ouverture
        start_tag_end = ssml_text.find(">", ssml_text.find("<speak"))
        if start_tag_end > 0:
            # Insérer la balise voice après la balise speak
            voice_tag = f'<voice name="{voice}">'
            voice_end_tag = '</voice>'
            
            # Diviser le SSML
            start_part = ssml_text[:start_tag_end+1]
            end_part = ssml_text[start_tag_end+1:]
            
            # Ajouter la balise voice et end_voice
            closing_speak_pos = end_part.rfind("</speak>")
            if closing_speak_pos > 0:
                ssml_text = start_part + voice_tag + end_part[:closing_speak_pos] + voice_end_tag + end_part[closing_speak_pos:]
    
    # S'assurer que xml:lang est présent
    if 'xml:lang="fr-FR"' not in ssml_text and 'lang="fr-FR"' in ssml_text:
        ssml_text = ssml_text.replace('lang="fr-FR"', 'xml:lang="fr-FR"')
    
    return ssml_text


def extract_text_from_ssml(ssml_text):
    """
    Extrait le texte brut d'un SSML en supprimant toutes les balises XML.
    
    Utilisé comme solution de secours en cas d'échec de la synthèse avec SSML complexe.
    
    Args:
        ssml_text (str): Texte SSML contenant des balises.
    
    Returns:
        str: Texte brut sans balises XML.
    """
    # Supprimer toutes les balises XML
    text = re.sub(r'<[^>]+>', ' ', ssml_text)
    # Normaliser les espaces
    text = ' '.join(text.split())
    return text


def create_simplified_ssml_from_text(text, voice="fr-FR-HenriNeural"):
    """
    Crée un SSML minimal à partir d'un texte brut.
    
    Cette fonction est utilisée comme solution de secours lorsque le SSML complexe
    cause des erreurs avec l'API Azure Speech Services.
    
    Args:
        text (str): Texte brut à encapsuler dans un SSML minimal.
    
    Returns:
        str: SSML minimal contenant uniquement le texte brut.
    """
    return (
        '<speak xmlns="http://www.w3.org/2001/10/synthesis" version="1.0" xml:lang="fr-FR">\n'
        f'<voice name="{voice}">\n'
        f'{text}\n'
        '</voice>\n'
        '</speak>'
    )


def synthesize_with_simplified_ssml(ssml_text, output_file, speech_key, speech_region, voice="fr-FR-HenriNeural"):
    """
    Tente une synthèse vocale avec un SSML simplifié en cas d'échec du SSML complexe.
    
    Args:
        ssml_text (str): SSML simplifié à utiliser pour la synthèse.
        output_file (str): Chemin du fichier audio de sortie.
        speech_key (str): Clé d'API Azure Speech Services.
        speech_region (str): Région Azure.
    
    Returns:
        str or None: Chemin du fichier audio généré ou None en cas d'échec.
    """
    # Configurer le service de synthèse vocale
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = voice
    
    # Configurer la sortie audio
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
    
    # Créer le synthétiseur
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, 
        audio_config=audio_config
    )
    
    logger.info(f"Tentative de synthèse avec SSML simplifié...")
    result = speech_synthesizer.speak_ssml_async(ssml_text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        logger.info(f"Synthèse audio terminée avec SSML simplifié. Fichier sauvegardé à: {output_file}")
        return output_file
    else:
        logger.error("Échec de la synthèse même avec SSML simplifié.")
        return None


def synthesize_ssml(ssml_text, output_file, speech_key, speech_region, voice):
    """
    Synthétise un texte SSML en utilisant Microsoft Azure.
    
    Args:
        ssml_text (str): Contenu SSML à synthétiser.
        output_file (str): Chemin du fichier audio de sortie.
        speech_key (str): Clé d'API Azure Speech Services.
        speech_region (str): Région Azure.
    
    Returns:
        str or None: Chemin du fichier audio généré ou None en cas d'échec.
    """
    # Nettoyer et préparer le SSML
    ssml_text = clean_ssml_for_azure(ssml_text, voice)
    
    # Afficher un aperçu du SSML préparé (limité pour la lisibilité)
    preview = ssml_text[:200] + "..." if len(ssml_text) > 200 else ssml_text
    logger.info(f"SSML préparé pour Azure:\n{preview}")
    
    # Configurer le service de synthèse vocale
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = voice #"fr-FR-HenriNeural"
    
    # Configurer la sortie audio
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
    
    # Créer le synthétiseur
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, 
        audio_config=audio_config
    )
    
    logger.info(f"Synthétisation avec la voix {voice} (Microsoft) vers {output_file}...")
    result = speech_synthesizer.speak_ssml_async(ssml_text).get()
    
    # Vérifier le résultat
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        logger.info(f"Synthèse audio terminée. Fichier sauvegardé à: {output_file}")
        return output_file
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        logger.error(f"Synthèse annulée: {cancellation_details.reason}")
        
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            logger.error(f"Erreur: {cancellation_details.error_details}")
            
            if "401" in str(cancellation_details.error_details):
                logger.error("Erreur d'authentification (401): Votre clé API ou région est incorrecte.")
            elif "1007" in str(cancellation_details.error_details):
                logger.error("Erreur de format SSML (1007): Le format SSML n'est pas valide.")
                # ADDED: Log the full invalid SSML
                logger.error(f"Invalid SSML content that caused the error:\n{ssml_text}")
                logger.info("Tentative de correction avec un format SSML simplifié...")
                
                # Essayer avec un format SSML très simple
                simplified_ssml = create_simplified_ssml_from_text(extract_text_from_ssml(ssml_text), voice)
                preview = simplified_ssml[:200] + "..." if len(simplified_ssml) > 200 else simplified_ssml
                logger.info(f"SSML simplifié:\n{preview}")
                
                return synthesize_with_simplified_ssml(simplified_ssml, output_file, speech_key, speech_region, voice)
        return None


def process_ssml_folder(ssml_folder, speech_key, speech_region, output_folder=None, voice="fr-FR-HenriNeural"):
    """
    Traite tous les fichiers SSML d'un dossier et génère les fichiers audio correspondants.
    
    Les fichiers audio sont générés avec exactement le même nom que les fichiers SSML,
    mais avec l'extension .wav au lieu de .xml.
    
    Args:
        ssml_folder (str): Chemin vers le dossier contenant les fichiers SSML (.xml)
        speech_key (str): Clé d'API Azure Speech Services
        speech_region (str): Région Azure
        output_folder (str, optional): Dossier où placer les fichiers audio. Si None, utilise le même dossier que les fichiers XML.
    
    Returns:
        tuple: (nombre de succès, nombre total de fichiers)
    """
    # Trouver tous les fichiers XML dans le dossier SSML
    ssml_files = glob.glob(os.path.join(ssml_folder, "*.xml"))
    
    logger.info(f"\n🔍 {len(ssml_files)} fichiers SSML trouvés dans {ssml_folder}")
    
    # Créer le dossier de sortie s'il n'existe pas
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Les fichiers audio seront placés dans: {output_folder}")
    
    # Traiter chaque fichier SSML
    success_count = 0
    for i, ssml_file in enumerate(ssml_files, 1):
        try:
            # Obtenir le nom complet du fichier (sans l'extension)
            filename_without_ext = os.path.splitext(ssml_file)[0]
            basename = os.path.basename(filename_without_ext)
            
            # Créer le chemin de sortie pour le fichier audio
            if output_folder:
                output_file = os.path.join(output_folder, f"{basename}.wav")
            else:
                output_file = f"{filename_without_ext}.wav"
            
            logger.info(f"\n[{i}/{len(ssml_files)}] Traitement de {basename}...")
            
            # Lire le fichier SSML
            ssml_content = read_ssml_file(ssml_file)
            
            if ssml_content:
                # Synthétiser l'audio
                if synthesize_ssml(ssml_content, output_file, speech_key, speech_region, voice):
                    success_count += 1
            else:
                logger.error(f"❌ Impossible de lire le fichier SSML: {ssml_file}")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de {ssml_file}: {str(e)}")
    
    logger.info(f"\n✅ Traitement terminé: {success_count}/{len(ssml_files)} fichiers traités avec succès")
    return success_count, len(ssml_files)


def main(speech_key, speech_region, ssml_folder, output_folder=None, voice="fr-FR-HenriNeural"):
    # Change print to logger calls
    if speech_key == "votre_clé_azure_ici":
        logger.warning("ATTENTION: Vous devez remplacer 'votre_clé_azure_ici' par votre clé API Azure.")
        logger.warning("Modifiez le fichier pour définir AZURE_SPEECH_KEY et AZURE_SPEECH_REGION.")
        return 0, 0
    
    # S'assurer que le dossier SSML existe
    if not os.path.isdir(ssml_folder):
        logger.error(f"Le dossier SSML spécifié n'existe pas: {ssml_folder}")
        return 0, 0
    
    logger.info("🚀 Démarrage du traitement par lots...")
    logger.info(f"  - Dossier SSML: {ssml_folder}")
    logger.info(f"  - Région Azure: {speech_region}")
    if output_folder:
        logger.info(f"  - Dossier de sortie: {output_folder}")
    
    # Traiter tous les fichiers SSML
    success_count, total_count = process_ssml_folder(ssml_folder, speech_key, speech_region, output_folder, voice)
    
    # Afficher un résumé
    if success_count == total_count:
        logger.info("✨ Tous les fichiers ont été traités avec succès!")
    else:
        logger.warning(f"⚠️ {total_count - success_count} fichiers n'ont pas pu être traités correctement.")
    
    return success_count, total_count