import os
import textgrid
import re
import sys
import glob 
import logger

# ✅ Seuil pour ignorer les très courtes pauses en début de phrase
INITIAL_PAUSE_THRESHOLD = 150
MIN_PAUSE_THRESHOLD = 150

def extract_words_and_pauses(textgrid_file):
    """
    Extrait la séquence exacte de mots et pauses du TextGrid.
    
    Returns:
        list: Séquence d'éléments [(type, contenu, durée)]
    """
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
            if not ignore_initial_pause or duration_ms >= INITIAL_PAUSE_THRESHOLD:
                sequence.append(("pause", None, duration_ms))
        else:  # C'est un mot
            sequence.append(("word", text, duration_ms))
            # sequence.append(("word", text, 0))
            ignore_initial_pause = False
    
    return sequence

def normalize_word(word):
    """Normalise un mot pour l'alignement."""
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

def align_sequences(natural_sequence, synth_words):
    """
    Aligne précisément les mots synthétiques avec la séquence naturelle.
    
    Args:
        natural_sequence: Liste [(type, contenu, durée)]
        synth_words: Liste des mots de la voix synthétique
    
    Returns:
        list: Séquence alignée pour le SSML [(type, contenu)]
    """
    # Extraire juste les mots de la séquence naturelle
    natural_words = [item[1] for item in natural_sequence if item[0] == "word"]
    
    # Normaliser les mots pour une meilleure correspondance
    normalized_natural = [normalize_word(word) for word in natural_words]
    normalized_synth = [normalize_word(word) for word in synth_words]
    
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

    # ✅ Ajouter explicitement la pause finale si elle existe
    if natural_sequence and natural_sequence[-1][0] == "pause":
        final_pause_duration = natural_sequence[-1][2]
        ssml_sequence.append(("pause", final_pause_duration))

    return ssml_sequence

def generate_ssml(aligned_sequence):
    """
    Génère le SSML à partir de la séquence alignée, dans un format compatible avec Azure.
    
    Args:
        aligned_sequence: Liste [(type, contenu)]
    
    Returns:
        str: Contenu SSML
    """
    # Construire le texte avec des balises de pause intégrées
    text_parts = []
    
    for element_type, content in aligned_sequence:
        if element_type == "word":
            text_parts.append(content)
        elif element_type == "pause" and content >= MIN_PAUSE_THRESHOLD: # skip very short pauses
            # Ajouter la balise de pause
            text_parts.append(f'<break time="{content}ms"/>')
    
    # Joindre le texte avec des espaces
    full_text = " ".join(text_parts)
    
    # Créer le SSML complet au format attendu par Azure
    ssml = f"""<speak xmlns="http://www.w3.org/2001/10/synthesis" version="1.0" xml:lang="fr-FR">
    <voice name="fr-FR-HenriNeural">
        {full_text}
    </voice>
</speak>"""
    
    # Formater proprement le XML
    try:
        import xml.dom.minidom
        pretty_xml = xml.dom.minidom.parseString(ssml).toprettyxml(indent="  ")
        return pretty_xml
    except:
        return ssml

def save_ssml(ssml_content, output_file):
    """Sauvegarde le fichier SSML."""
    # Créer le dossier parent si nécessaire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(ssml_content)
    print(f"✅ SSML généré : {output_file}")

def process_file_pair(textgrid_file, transcription_file, output_file):
    try:
        print(f"\n🔄 Traitement : {os.path.basename(textgrid_file)} + {os.path.basename(transcription_file)}")

        # 1. Extraction séquence mots + pauses du TextGrid (naturel)
        natural_sequence = extract_words_and_pauses(textgrid_file)
        print(f"  - Séquence naturelle: {len(natural_sequence)} éléments")

        # 2. Lire le texte synthétique (CORRIGÉ par Levenshtein)
        with open(transcription_file, "r", encoding="utf-8") as f:
            corrected_text = f.read().strip()

        # 👉 Remplacement explicite des points de suspension par un simple point :
        corrected_text = corrected_text.replace('...', '.')

        synth_words = corrected_text.split()
        print(f"  - Mots synthétiques (corrigés): {len(synth_words)} mots")

        # 3. Aligner précisément les séquences
        aligned_sequence = align_sequences(natural_sequence, synth_words)
        print(f"  - Séquence alignée: {len(aligned_sequence)} éléments")

        # 4. Générer le SSML avec les pauses
        ssml_content = generate_ssml(aligned_sequence)

        # 5. Sauvegarder SSML
        save_ssml(ssml_content, output_file)

        return True

    except Exception as e:
        print(f"❌ Erreur lors du traitement de {textgrid_file} + {transcription_file}: {str(e)}")
        return False



def find_matching_files(textgrid_folder, transcription_folder):
    """
    Trouve les paires de fichiers TextGrid et Transcription qui correspondent.
    
    Returns:
        list: Liste de tuples (textgrid_file, transcription_file, basename)
    """
    # Obtenir tous les fichiers TextGrid et Transcription
    textgrid_files = glob.glob(os.path.join(textgrid_folder, "*.TextGrid"))
    transcription_files = glob.glob(os.path.join(transcription_folder, "*.txt"))
    
    # Extraire les noms de base pour faciliter la correspondance
    textgrid_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in textgrid_files}
    transcription_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in transcription_files}
    
    # Trouver les correspondances
    matches = []
    for basename in textgrid_basenames:
        if basename in transcription_basenames:
            matches.append((
                textgrid_basenames[basename],
                transcription_basenames[basename],
                basename
            ))
    
    return matches

def process_all_files(textgrid_folder, transcription_folder, output_folder):
    """
    Traite tous les fichiers TextGrid et Transcription pour générer des SSML.
    
    Args:
        textgrid_folder: Dossier contenant les fichiers TextGrid
        transcription_folder: Dossier contenant les fichiers de transcription
        output_folder: Dossier de sortie pour les fichiers SSML
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Trouver les paires de fichiers correspondants
    file_pairs = find_matching_files(textgrid_folder, transcription_folder)
    
    print(f"🔍 {len(file_pairs)} paires de fichiers trouvées")
    
    # Traiter chaque paire
    success_count = 0
    for textgrid_file, transcription_file, basename in file_pairs:
        output_file = os.path.join(output_folder, f"{basename}.xml")
        if process_file_pair(textgrid_file, transcription_file, output_file):
            success_count += 1
    
    print(f"\n✅ Traitement terminé: {success_count}/{len(file_pairs)} fichiers traités avec succès")

def main(TEXTGRID_FOLDER, TRANSCRIPTION_FOLDER, SSML_OUTPUT_FOLDER):
    print("🚀 Démarrage du traitement par lots...")
    print(f"  - Dossier TextGrid: {TEXTGRID_FOLDER}")
    print(f"  - Dossier Transcription: {TRANSCRIPTION_FOLDER}")
    print(f"  - Dossier de sortie SSML: {SSML_OUTPUT_FOLDER}")
    process_all_files(TEXTGRID_FOLDER, TRANSCRIPTION_FOLDER, SSML_OUTPUT_FOLDER)

def cli_main():
    if len(sys.argv) != 4:
        logger.error(f"Arguments incorrects: {sys.argv}")
        print("Usage: python gen_break_ssml.py <TEXTGRID_FOLDER> <TRANSCRIPTION_FOLDER> <SSML_OUTPUT_FOLDER>")
        sys.exit(1)
    TEXTGRID_FOLDER = sys.argv[1]
    TRANSCRIPTION_FOLDER = sys.argv[2]
    SSML_OUTPUT_FOLDER = sys.argv[3]
    main(TEXTGRID_FOLDER, TRANSCRIPTION_FOLDER, SSML_OUTPUT_FOLDER)

if __name__ == "__main__":
    cli_main()