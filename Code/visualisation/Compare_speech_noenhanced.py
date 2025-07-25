import os
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.font_manager as fm



def extract_pitch_mean(audio_path, time_step=0.01):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch(time_step=time_step)
    values = pitch.selected_array['frequency']
    voiced = values[values > 0]
    if len(voiced) == 0:
        return np.nan
    return np.nanmean(voiced)

def extract_mean_volume(audio_path):
    snd = parselmouth.Sound(audio_path)
    intensity = snd.to_intensity()
    values = intensity.values[0]
    voiced = values[values > 0]
    if len(voiced) == 0:
        return np.nan
    return np.nanmean(voiced)

def extract_duration(audio_path):
    snd = parselmouth.Sound(audio_path)
    return snd.get_total_duration()

def compare_pitch(naturelle_dir, synthese_dir):
    files_naturelle = set(f for f in os.listdir(naturelle_dir) if f.lower().endswith('.wav'))
    files_synthese = set(f for f in os.listdir(synthese_dir) if f.lower().endswith('.wav'))
    common_files = sorted(files_naturelle & files_synthese)
    naturelle_pitches = []
    synthese_pitches = []
    labels = []
    for fname in common_files:
        nat_path = os.path.join(naturelle_dir, fname)
        syn_path = os.path.join(synthese_dir, fname)
        try:
            nat_pitch = extract_pitch_mean(nat_path)
            syn_pitch = extract_pitch_mean(syn_path)
            if not np.isnan(nat_pitch) and not np.isnan(syn_pitch):
                naturelle_pitches.append(nat_pitch)
                synthese_pitches.append(syn_pitch)
                labels.append(fname)
        except Exception as e:
            print(f"Erreur avec {fname}: {e}")
    return naturelle_pitches, synthese_pitches, labels

def plot_feature_comparison(naturelle, synthese, sample_labels, anonymized_speakers, feature_name, save_path=None):
    # Attribution d'une couleur par speaker anonymisé
    unique_speakers = list(dict.fromkeys(anonymized_speakers))
    cmap = plt.get_cmap('tab20', len(unique_speakers))
    font_path = "/your/path/times_new_roman/times.ttf"
    try:
        title_font = fm.FontProperties(fname=font_path, size=20)
        label_font = fm.FontProperties(fname=font_path, size=21)
    except Exception:
        title_font = {'family': 'Times', 'size': 20}
        label_font = {'family': 'Times', 'size': 21}
    color_map = {spk: cmap(i) for i, spk in enumerate(unique_speakers)}
    colors = [color_map[spk] for spk in anonymized_speakers]
    # Si c'est le plot volume, légende à droite, sinon en dessous
    is_volume = 'volume' in feature_name.lower()
    is_pitch  = 'pitch'  in feature_name.lower()
    plt.figure(figsize=(6,10))
    scatter = plt.scatter(synthese, naturelle, c=colors, s=80)
    min_val = min(naturelle + synthese)
    max_val = max(naturelle + synthese)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.xlabel(f'Mean {feature_name} synthesis', fontsize=20, fontproperties=label_font)
    plt.ylabel(f'Mean {feature_name} natural', fontsize=20, fontproperties=label_font)
    # Pas de titre
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[spk], markersize=10) for spk in unique_speakers]
    ncol = min(4, len(unique_speakers))
    if is_volume:
        # Légende sur le plot, en bas à gauche
        plt.legend(handles, unique_speakers, title="Speaker", loc='lower left', bbox_to_anchor=(0.01, 0.01), ncol=1, frameon=True, prop=label_font, title_fontproperties=title_font)
        plt.tight_layout(rect=[0,0,1,1])
    elif is_pitch:
        # Légende sur le plot, en bas à droite
        plt.legend(handles, unique_speakers, title="Speaker", loc='lower right', bbox_to_anchor=(0.99, 0.01), ncol=1, frameon=True, prop=label_font, title_fontproperties=title_font)
        plt.tight_layout(rect=[0,0,1,1])
    else:
        # Légende SOUS le plot
        plt.legend(handles, unique_speakers, title="Speaker", loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=ncol, frameon=False, prop=label_font, title_fontproperties=title_font)
        plt.tight_layout(rect=[0,0.08,1,1])
    plt.tick_params(axis='both', labelsize=16)
    if not save_path:
        save_path = f"{feature_name.replace(' ', '_')}_scatterplot.pdf"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Scatterplot sauvegardé dans {save_path}")
    plt.show()

def plot_feature_histogram(naturelle, synthese, sample_labels, feature_name, save_path=None):
    import numpy as np
    # Limite à 50 samples pour l'affichage
    max_samples = 50
    naturelle = naturelle[:max_samples]
    synthese = synthese[:max_samples]
    sample_labels = [f"sample_{i+1}" for i in range(len(naturelle))]
    x = np.arange(len(sample_labels))
    width = 0.35
    plt.figure(figsize=(max(8, len(sample_labels)*0.7), 6))
    plt.bar(x - width/2, naturelle, width, label='Naturelle')
    plt.bar(x + width/2, synthese, width, label='Synthèse')
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel(f'Mean {feature_name}', fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    # Pas de titre
    plt.xticks(x, sample_labels, rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, prop={'family': 'Times New Roman', 'size': 12})
    plt.tight_layout(rect=[0,0.08,1,1])
    if not save_path:
        save_path = f"{feature_name.replace(' ', '_')}_histogram.pdf"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Histogramme sauvegardé dans {save_path} (50 premiers samples)")
    plt.close()

def plot_feature_boxplot(naturelle, synthese, feature_name, save_path=None):
    plt.figure(figsize=(6,6))
    plt.boxplot([naturelle, synthese], labels=['Naturelle', 'Synthèse'])
    plt.ylabel(f'Mean {feature_name}', fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    # Pas de titre
    plt.tight_layout()
    if not save_path:
        save_path = f"{feature_name.replace(' ', '_')}_boxplot.pdf"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Boxplot sauvegardé dans {save_path}")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0.08,1,1])
    plt.show()

def plot_zscore_pitch(nat_path, syn_path):
    snd_nat = parselmouth.Sound(nat_path)
    snd_syn = parselmouth.Sound(syn_path)
    pitch_nat = snd_nat.to_pitch().selected_array['frequency']
    pitch_syn = snd_syn.to_pitch().selected_array['frequency']
    # On ne garde que les frames voisées
    pitch_nat = pitch_nat[pitch_nat > 0]
    pitch_syn = pitch_syn[pitch_syn > 0]
    # Z-score
    z_nat = (pitch_nat - np.mean(pitch_nat)) / np.std(pitch_nat) if len(pitch_nat) > 1 else pitch_nat
    z_syn = (pitch_syn - np.mean(pitch_syn)) / np.std(pitch_syn) if len(pitch_syn) > 1 else pitch_syn
    plt.plot(z_nat, label='Naturelle')
    plt.plot(z_syn, label='Synthèse')
    plt.title('Comparaison des formes d\'intonation (z-score)')
    plt.xlabel('Frame')
    plt.ylabel('Pitch (z-score)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_zscore_feature(nat_path, syn_path, feature, save_path=None):
    snd_nat = parselmouth.Sound(nat_path)
    snd_syn = parselmouth.Sound(syn_path)
    if feature == 'pitch':
        arr_nat = snd_nat.to_pitch().selected_array['frequency']
        arr_syn = snd_syn.to_pitch().selected_array['frequency']
        ylabel = 'Pitch (z-score)'
        title = "Comparaison des formes d'intonation (z-score)"
    elif feature == 'volume':
        arr_nat = snd_nat.to_intensity().values[0]
        arr_syn = snd_syn.to_intensity().values[0]
        ylabel = 'Volume (z-score)'
        title = "Comparaison des formes d'intensité (z-score)"
    elif feature == 'rate':
        arr_nat = np.ones_like(snd_nat.to_pitch().selected_array['frequency']) * (1.0 / snd_nat.get_total_duration())
        arr_syn = np.ones_like(snd_syn.to_pitch().selected_array['frequency']) * (1.0 / snd_syn.get_total_duration())
        ylabel = 'Rate (z-score)'
        title = "Comparaison des formes de débit (z-score)"
    else:
        print("Feature inconnue pour la variabilité. Utilisez pitch, volume ou rate.")
        return
    arr_nat = arr_nat[arr_nat > 0] if feature != 'rate' else arr_nat
    arr_syn = arr_syn[arr_syn > 0] if feature != 'rate' else arr_syn
    z_nat = (arr_nat - np.mean(arr_nat)) / np.std(arr_nat) if len(arr_nat) > 1 else arr_nat
    z_syn = (arr_syn - np.mean(arr_syn)) / np.std(arr_syn) if len(arr_syn) > 1 else arr_syn
    plt.figure()
    plt.plot(z_nat, label='Naturelle')
    plt.plot(z_syn, label='Synthèse')
    # Pas de titre
    plt.xlabel('Frame', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0.08,1,1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def print_quartiles(naturelle_pitches, synthese_pitches):
    naturelle_pitches = np.array(naturelle_pitches)
    synthese_pitches = np.array(synthese_pitches)
    print("Voix naturelle :")
    print(f"  Q1 (25%)   : {np.percentile(naturelle_pitches, 25):.2f} Hz")
    print(f"  Médiane    : {np.percentile(naturelle_pitches, 50):.2f} Hz")
    print(f"  Q3 (75%)   : {np.percentile(naturelle_pitches, 75):.2f} Hz")
    print("Voix synthèse :")
    print(f"  Q1 (25%)   : {np.percentile(synthese_pitches, 25):.2f} Hz")
    print(f"  Médiane    : {np.percentile(synthese_pitches, 50):.2f} Hz")
    print(f"  Q3 (75%)   : {np.percentile(synthese_pitches, 75):.2f} Hz")

# --- Fonctions spécifiques à chaque feature ---
def save_feature_only(filepath, naturelle, synthese, speakers, samples):
    np.savez(filepath,
             naturelle=naturelle,
             synthese=synthese,
             speakers=speakers,
             samples=samples)

def load_feature_only(filepath):
    data = np.load(filepath, allow_pickle=True)
    return (data['naturelle'].tolist(),
            data['synthese'].tolist(),
            data['speakers'].tolist(),
            data['samples'].tolist())

def extract_and_cache_feature(root_dir, feature):
    naturelle = []
    synthese = []
    speakers = []
    samples = []
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # Accept both with and without _EPXX
        m_nat = re.match(r"(.+?)(_EP\d+)?$", entry)
        if not m_nat:
            continue
        speaker_name = m_nat.group(1)
        episode = entry
        synth_dirname = episode + "_microsoft"
        nat_audio_dir = os.path.join(root_dir, episode, "audio")
        synth_audio_dir = os.path.join(root_dir, synth_dirname, "audio")
        if not (os.path.isdir(nat_audio_dir) and os.path.isdir(synth_audio_dir)):
            continue
        nat_segments = set(f for f in os.listdir(nat_audio_dir) if f.startswith("segment_ph") and f.endswith(".wav"))
        synth_segments = set(f for f in os.listdir(synth_audio_dir) if f.startswith("segment_ph") and f.endswith(".wav"))
        for seg in sorted(nat_segments):
            seg_num = re.match(r"segment_ph(\d+)\.wav", seg)
            if not seg_num:
                continue
            seg_id = seg_num.group(1)
            synth_seg = f"segment_ph{seg_id}.wav"
            if synth_seg not in synth_segments:
                continue
            nat_path = os.path.join(nat_audio_dir, seg)
            synth_path = os.path.join(synth_audio_dir, synth_seg)
            try:
                if feature == 'pitch':
                    nat_val = extract_pitch_mean(nat_path)
                    syn_val = extract_pitch_mean(synth_path)
                elif feature == 'volume':
                    nat_val = extract_mean_volume(nat_path)
                    syn_val = extract_mean_volume(synth_path)
                elif feature == 'rate':
                    nat_dur = extract_duration(nat_path)
                    syn_dur = extract_duration(synth_path)
                    nat_val = 1.0 / nat_dur if nat_dur > 0 else np.nan
                    syn_val = 1.0 / syn_dur if syn_dur > 0 else np.nan
                else:
                    continue
                if not (np.isnan(nat_val) or np.isnan(syn_val)):
                    # Use _EPXX if present, else just speaker name
                    if m_nat.group(2):
                        sample_id = f"{speaker_name}{m_nat.group(2)}_ph{seg_id}"
                    else:
                        sample_id = f"{speaker_name}_ph{seg_id}"
                    naturelle.append(nat_val)
                    synthese.append(syn_val)
                    speakers.append(speaker_name)
                    samples.append(sample_id)
            except Exception as e:
                print(f"Erreur avec {nat_path} ou {synth_path}: {e}")
    return naturelle, synthese, speakers, samples

def plot_raw_feature(nat_path, syn_path, feature, save_path=None):
    import matplotlib.font_manager as fm
    font_path = "/your/path/times_new_roman/times.ttf"
    try:
        title_font = fm.FontProperties(fname=font_path, size=13)
        label_font = fm.FontProperties(fname=font_path, size=20)
        plt.rcParams['font.family'] = title_font.get_name()
    except Exception:
        title_font = {'family': 'Times', 'size': 13}
        label_font = {'family': 'Times', 'size': 20}
        plt.rcParams['font.family'] = 'Times'
    snd_nat = parselmouth.Sound(nat_path)
    snd_syn = parselmouth.Sound(syn_path)
    if feature == 'pitch':
        arr_nat = snd_nat.to_pitch().selected_array['frequency']
        arr_syn = snd_syn.to_pitch().selected_array['frequency']
        ylabel = 'Pitch (Hz)'
    elif feature == 'volume':
        arr_nat = snd_nat.to_intensity().values[0]
        arr_syn = snd_syn.to_intensity().values[0]
        ylabel = 'Volume (dB)'
    elif feature == 'rate':
        arr_nat = np.ones_like(snd_nat.to_pitch().selected_array['frequency']) * (1.0 / snd_nat.get_total_duration())
        arr_syn = np.ones_like(snd_syn.to_pitch().selected_array['frequency']) * (1.0 / snd_syn.get_total_duration())
        ylabel = 'Rate (1/s)'
    else:
        print("Feature inconnue pour la courbe brute. Utilisez pitch, volume ou rate.")
        return
    arr_nat = arr_nat[arr_nat > 0] if feature != 'rate' else arr_nat
    arr_syn = arr_syn[arr_syn > 0] if feature != 'rate' else arr_syn
    plt.figure()
    plt.plot(arr_nat, label='Raw Synthesis')
    plt.plot(arr_syn, label='Enhanced Synthesis')
    # Pas de titre
    plt.xlabel('Frame', fontsize=20, fontproperties=label_font)
    plt.ylabel(ylabel, fontsize=20, fontproperties=label_font)
    # Légende SOUS le plot, alignée à droite
    plt.legend(prop=label_font, loc='upper right', bbox_to_anchor=(1, -0.18), ncol=2, frameon=False, borderaxespad=0, handlelength=2, columnspacing=0.7, alignment='right')
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout(rect=[0,0.18,1,1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

# --- Main refactor ---
if __name__ == "__main__":
    import sys
    font_path = "/your/path/times_new_roman/times.ttf"
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

    plots_dir = "/your/path/mon_projet_TTS/plots"
    os.makedirs(plots_dir, exist_ok=True)

    if len(sys.argv) >= 2 and sys.argv[1] in ["zscore", "raw"]:
        mode = sys.argv[1]
        if len(sys.argv) == 5:
            feature = sys.argv[2]
            nat_path = sys.argv[3]
            syn_path = sys.argv[4]
            pdf_path = os.path.join(plots_dir, f"{feature}_{mode}_variability.pdf")
            if mode == "zscore":
                plot_zscore_feature(nat_path, syn_path, feature, save_path=pdf_path)
            else:
                plot_raw_feature(nat_path, syn_path, feature, save_path=pdf_path)
            print(f"Figure sauvegardée dans {pdf_path}")
            sys.exit(0)
        elif len(sys.argv) == 4:
            nat_path = sys.argv[2]
            syn_path = sys.argv[3]
            pdf_path = os.path.join(plots_dir, f"pitch_{mode}_variability.pdf")
            if mode == "zscore":
                plot_zscore_feature(nat_path, syn_path, 'pitch', save_path=pdf_path)
            else:
                plot_raw_feature(nat_path, syn_path, 'pitch', save_path=pdf_path)
            print(f"Figure sauvegardée dans {pdf_path}")
            sys.exit(0)
        else:
            print("Usage: python compare_pitch_naturelle_synthese.py zscore|raw [feature] <path_naturelle> <path_synthese>")
            sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: python compare_pitch_naturelle_synthese.py <root_dir> [plot_type] [feature]")
        print("plot_type: scatter (défaut), hist, box")
        print("feature: pitch (défaut), volume, rate")
        sys.exit(1)
    root_dir = sys.argv[1]
    plot_type = sys.argv[2] if len(sys.argv) > 2 else 'scatter'
    feature = sys.argv[3] if len(sys.argv) > 3 else 'pitch'
    feature_file = os.path.join(root_dir, f"{feature}_data.npz")
    if os.path.exists(feature_file):
        print(f"Chargement des valeurs de {feature} depuis {feature_file}")
        naturelle, synthese, speakers, samples = load_feature_only(feature_file)
    else:
        print(f"Extraction des valeurs de {feature} depuis la base de données...")
        naturelle, synthese, speakers, samples = extract_and_cache_feature(root_dir, feature)
        save_feature_only(feature_file, naturelle, synthese, speakers, samples)
        print(f"Valeurs de {feature} sauvegardées dans {feature_file}")
    if speakers:
        unique_speakers = list(dict.fromkeys(speakers))
        speaker_map = {spk: f"speaker_{i+1}" for i, spk in enumerate(unique_speakers)}
        anonymized_speakers = [speaker_map[spk] for spk in speakers]
        sample_labels = [f"sample_{i+1}" for i in range(len(samples))]
        if feature == 'pitch':
            feature_label = 'pitch (Hz)'
        elif feature == 'volume':
            feature_label = 'volume (dB)'
        elif feature == 'rate':
            feature_label = 'rate (1/s)'
        else:
            feature_label = feature
        pdf_path = os.path.join(plots_dir, f"{feature}_{plot_type}.pdf")
        if plot_type == 'scatter':
            plot_feature_comparison(naturelle, synthese, sample_labels, anonymized_speakers, feature_label, save_path=pdf_path)
        elif plot_type == 'hist':
            plot_feature_histogram(naturelle, synthese, sample_labels, feature_label, save_path=pdf_path)
        elif plot_type == 'box':
            plot_feature_boxplot(naturelle, synthese, feature_label, save_path=pdf_path)
        else:
            print("plot_type inconnu. Utilisez scatter, hist ou box.")
    else:
        print("Aucune paire de segments audio commune trouvée ou features non calculables.")


