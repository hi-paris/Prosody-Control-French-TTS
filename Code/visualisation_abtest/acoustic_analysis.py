
"""
Description:
    Compare natural and TTS audio pour la même phrase au niveau des mots,
    en affichant waveforms, spectrogrammes et contours de pitch côte à côte.
    - Barre d'échelle dB supprimée sous le spectrogramme.
    - Étiquettes de mots (non vides) placées au milieu de leur intervalle,
      avec des pointillés blancs délimitant l'intervalle sur les 3 graphiques.

Usage Example:
    plot_comparison(
        natural_wav="path/to/natural.wav",
        natural_textgrid="path/to/natural.TextGrid",
        tts_wav="path/to/tts.wav",
        tts_textgrid="path/to/tts.TextGrid"
    )

"""

import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import textgrid


def parse_textgrid(textgrid_path, tier_name="words"):
    """
    Parse le fichier TextGrid donné pour extraire les intervalles d’un tier
    spécifié. Retourne les intervalles et leurs labels.

    Args:
        textgrid_path (str): Chemin vers le fichier TextGrid.
        tier_name (str): Nom du tier contenant les annotations au niveau des mots.

    Returns:
        List[Tuple[float, float, str]]: Liste de (start_time, end_time, label).
    """
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    # Recherche du tier spécifié
    tier = None
    for t in tg.tiers:
        if t.name.lower() == tier_name.lower():
            tier = t
            break

    if tier is None:
        raise ValueError(f"Tier '{tier_name}' non trouvé dans {textgrid_path}.")

    intervals = []
    for interval in tier.intervals:
        start = interval.minTime
        end = interval.maxTime
        label = interval.mark.strip()  # Le champ 'text' de Praat est stocké dans 'mark'
        intervals.append((start, end, label))

    return intervals


def is_pause(label):
    """
    Détermine si un label donné indique une pause (vide ou seulement des espaces).

    Args:
        label (str): Label textuel de l'intervalle dans le TextGrid.

    Returns:
        bool: True si cet intervalle est considéré comme une pause, sinon False.
    """
    return len(label) == 0


def compute_pitch(audio, sr, fmin=60.0, fmax=2000.0, hop_length=256):
    """
    Calcule le contour de pitch (F0) à l'aide de la méthode PYIN de librosa.

    Args:
        audio (np.ndarray): Échantillons audio.
        sr (int): Taux d'échantillonnage.
        fmin (float): Fréquence minimale pour la détection.
        fmax (float): Fréquence maximale pour la détection.
        hop_length (int): Nombre d'échantillons entre les calculs successifs.

    Returns:
        (np.ndarray, np.ndarray):
            time_f0: Axe temporel pour le contour de pitch.
            f0: Contour de pitch estimé (Hz), avec np.nan pour les frames non voisées.
    """
    f0, _, _ = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
    # Création d'un array de temps pour chaque frame
    time_f0 = np.arange(len(f0)) * hop_length / sr
    return time_f0, f0


def compute_spectrogram(audio, sr, n_fft=1024, hop_length=256):
    """
    Calcule le spectrogramme en échelle de décibels pour le signal audio donné.

    Args:
        audio (np.ndarray): Échantillons audio.
        sr (int): Taux d'échantillonnage.
        n_fft (int): Taille de la fenêtre FFT.
        hop_length (int): Nombre d'échantillons entre deux FFT successives.

    Returns:
        np.ndarray: Spectrogramme en dB.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return D


def plot_one_audio(
    fig, axs, col_index, wav_path, textgrid_path, title_prefix
):
    """
    Affiche waveforms, spectrogram et contour de pitch pour un seul fichier audio.
    Supprime la colorbar et ajoute des lignes pointillées pour chaque mot
    (début/fin) sur les trois graphiques, ainsi qu'une étiquette de mot sur
    le spectrogramme.

    Args:
        fig (matplotlib.figure.Figure): Instance de la figure.
        axs (np.ndarray): Tableau d'axes pour les subplots.
        col_index (int): Index de colonne (0 ou 1) pour les subplots.
        wav_path (str): Chemin vers le fichier WAV.
        textgrid_path (str): Chemin vers le fichier TextGrid.
        title_prefix (str): Préfixe du titre pour distinguer naturel vs. TTS.
    """
    # ----------------------
    # Chargement de l'audio et des intervalles
    # ----------------------
    audio, sr = librosa.load(wav_path, sr=None)
    intervals = parse_textgrid(textgrid_path, tier_name="words")

    # Axe temporel pour le waveform
    time = np.linspace(0, len(audio) / sr, num=len(audio))

    # ----------------------
    # 1. Waveform
    # ----------------------
    axs[0, col_index].plot(time, audio, color="royalblue", linewidth=1)
    axs[0, col_index].set_title(f"{title_prefix} Waveform", fontsize=10)
    axs[0, col_index].set_ylabel("Amplitude")
    axs[0, col_index].set_xlim([0, time[-1]])

    # ----------------------
    # 2. Spectrogram
    # ----------------------
    D = compute_spectrogram(audio, sr)
    librosa.display.specshow(
        D,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        ax=axs[1, col_index],
        cmap="viridis"
    )
    # >>> On supprime la colorbar :
    # fig.colorbar(img, ax=axs[1, col_index], format="%+2.f dB", orientation="horizontal")

    axs[1, col_index].set_title(f"{title_prefix} Spectrogram (dB)", fontsize=10)
    axs[1, col_index].set_xlim([0, time[-1]])

    # ----------------------
    # 3. Pitch contour (F0)
    # ----------------------
    time_f0, f0 = compute_pitch(audio, sr, fmin=60, fmax=2000, hop_length=256)
    axs[2, col_index].plot(time_f0, f0, color="blue", linewidth=1)
    axs[2, col_index].set_title(f"{title_prefix} Pitch (F0)", fontsize=10)
    axs[2, col_index].set_xlabel("Time (s)")
    axs[2, col_index].set_ylabel("Frequency (Hz)")
    axs[2, col_index].set_xlim([0, time[-1]])

    # ----------------------
    # 4. Ajout des pointillés et étiquettes de mots
    # ----------------------
    # On récupère la limite supérieure du spectrogramme pour y placer l'étiquette
    y_min, y_max = axs[1, col_index].get_ylim()

    for (start, end, label) in intervals:
        if not is_pause(label):
            # Lignes verticales pointillées blanches sur les 3 subplots (wave, spec, pitch)
            for row in range(3):
                axs[row, col_index].axvline(
                    start, color="white", linestyle="--", linewidth=1
                )
                axs[row, col_index].axvline(
                    end, color="white", linestyle="--", linewidth=1
                )

            # Placement de l'étiquette au milieu de l'intervalle, sur le spectrogramme
            mid_time = (start + end) / 2.0
            axs[1, col_index].text(
                mid_time,
                0.9 * y_max,      # 90% de la hauteur du spectrogramme
                label,
                color="white",
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
                fontsize=9
            )


def plot_comparison(natural_wav, natural_textgrid, tts_wav, tts_textgrid):
    """
    Compare l’audio naturel et TTS pour la même phrase.
    Pour chacun, on affiche le waveform, le spectrogramme, et le contour de pitch
    dans une grille de 3 lignes x 2 colonnes.

    - Pas de colorbar pour le spectrogramme.
    - Lignes pointillées blanches début/fin de mot sur chaque subplot.
    - Étiquette du mot affichée sur le spectrogramme à 90% de la hauteur Y.

    Args:
        natural_wav (str): Chemin vers le fichier WAV de la parole naturelle.
        natural_textgrid (str): Chemin vers le fichier TextGrid de la parole naturelle.
        tts_wav (str): Chemin vers le fichier WAV de la parole TTS.
        tts_textgrid (str): Chemin vers le fichier TextGrid de la parole TTS.
    """
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=False)

    # Audio naturel en colonne de gauche (col_index = 0)
    plot_one_audio(
        fig=fig,
        axs=axs,
        col_index=0,
        wav_path=natural_wav,
        textgrid_path=natural_textgrid,
        title_prefix="Natural",
    )

    # Audio TTS en colonne de droite (col_index = 1)
    plot_one_audio(
        fig=fig,
        axs=axs,
        col_index=1,
        wav_path=tts_wav,
        textgrid_path=tts_textgrid,
        title_prefix="TTS",
    )

    # Ajustement de la mise en page
    fig.suptitle("Natural vs. TTS Audio Comparison", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Exemple d'utilisation
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()

    # Mettez à jour ces chemins selon vos données
    natural_wav_path = os.path.join(
        BASE_DIR,
        "Data",
        "voice",
        "records_natural",
        "transcription",
        "segment_ph1.wav"
    )
    natural_textgrid_path = os.path.join(
        BASE_DIR,
        "Data",
        "voice",
        "records_natural",
        "transcription",
        "segment_ph1.TextGrid"
    )
    tts_wav_path = os.path.join(
        BASE_DIR,
        "Data",
        "voice",
        "records_microsoft",
        "transcription",
        "segment_ph1.wav"
    )
    tts_textgrid_path = os.path.join(
        BASE_DIR,
        "Data",
        "voice",
        "records_microsoft",
        "transcription",
        "segment_ph1.TextGrid"
    )

    plot_comparison(
        natural_wav="/root/mon_projet_TTS/mon_projet_TTS/Data/voice/records/audio/segment_ph1.wav",
        natural_textgrid="/root/mon_projet_TTS/mon_projet_TTS/Data/voice/records/audio/segment_ph1.TextGrid",
        tts_wav="/root/mon_projet_TTS/mon_projet_TTS/Data/voice/records_microsoft/audio/segment_ph1.wav",
        tts_textgrid="/root/mon_projet_TTS/mon_projet_TTS/Data/voice/records_microsoft/audio/segment_ph1.TextGrid"
    )
