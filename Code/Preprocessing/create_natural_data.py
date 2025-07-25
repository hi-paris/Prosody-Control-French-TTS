import os
import shutil

def generer_natural_voice(root_path: str):
    """
    Parcourt le répertoire root_path à la recherche de sous-dossiers
    (ignorer ceux qui se terminent par _microsoft ou qui s'appellent audio_voxpopuli).
    Pour chaque sous-dossier retenu, on récupère les fichiers segment_ph*.wav dans audio/
    et segment_ph*.txt dans transcription/ puis on les copie dans un dossier 'natural_voice'
    en les renommant <nom_du_dossier>_segment_ph*.wav/.txt
    """

    # Crée le dossier 'natural_voice' s'il n'existe pas déjà
    natural_voice_dir = os.path.join(root_path, "natural_voice")
    os.makedirs(natural_voice_dir, exist_ok=True)

    # Parcourt tous les éléments du répertoire racine
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        # On ne traite que les répertoires
        if not os.path.isdir(folder_path):
            continue

        # Ignore les dossiers se terminant par _microsoft ou s'appelant audio_voxpopuli
        if folder_name.endswith("_microsoft") or folder_name == "audio_voxpopuli":
            continue

        # On suppose que le dossier audio est à <folder_path>/audio
        audio_path = os.path.join(folder_path, "audio")
        # On suppose que le dossier transcription est à <folder_path>/transcription
        transcription_path = os.path.join(folder_path, "transcription")

        # Vérifie que les dossiers audio et transcription existent
        if not os.path.isdir(audio_path) or not os.path.isdir(transcription_path):
            continue

        # Récupère tous les fichiers .wav commençant par "segment_ph"
        for audio_file in os.listdir(audio_path):
            if audio_file.endswith(".wav") and audio_file.startswith("segment_ph"):
                # Construction du chemin complet vers le fichier audio
                audio_file_path = os.path.join(audio_path, audio_file)

                # Nom de base : segment_phX (sans l'extension .wav)
                base_name = audio_file[:-4]  # enlève l'extension .wav
                txt_file = base_name + ".txt"
                txt_file_path = os.path.join(transcription_path, txt_file)

                # Vérifie que le fichier texte correspondant existe
                if not os.path.isfile(txt_file_path):
                    continue  # on ignore s'il n'y a pas de correspondance .txt

                # Nouveau nom pour le fichier audio et texte
                new_audio_filename = f"{folder_name}_{audio_file}"
                new_txt_filename = f"{folder_name}_{txt_file}"

                # Copie du fichier audio vers natural_voice/
                shutil.copy2(audio_file_path,
                             os.path.join(natural_voice_dir, new_audio_filename))

                # Copie du fichier texte vers natural_voice/
                shutil.copy2(txt_file_path,
                             os.path.join(natural_voice_dir, new_txt_filename))

                # Affichage en console
                print(f"Copié : {audio_file_path} --> "
                      f"{os.path.join(natural_voice_dir, new_audio_filename)}")
                print(f"Copié : {txt_file_path} --> "
                      f"{os.path.join(natural_voice_dir, new_txt_filename)}")

    print(f"\nDossier 'natural_voice' créé ou mis à jour dans : {root_path}\n")


generer_natural_voice('mon_projet_TTS/Data')