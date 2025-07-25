import sys
import os
import pandas as pd
import numpy as np
import re


def create_ssml_fragment(text, pitch_adj, rate_adj, loudness_adj, duration_pause_syntagme_natural, voice, style, styledegree):
    # Gestion des modifications prosodiques
    if str(text).strip() == "":
        pitch_mod = "+0%"
        rate_mod = "+0%"
        loudness_mod = "+0%"
    else:
        # Apply adjustments (keep existing logic)
        rate_sign = np.sign(rate_adj)
        rate_norm = (np.abs(rate_adj) ** 0.80)
        rate_adj = rate_sign * rate_norm
        rate_adj = min(2, rate_adj)

        pitch_sign = np.sign(pitch_adj)
        pitch_norm = (np.abs(pitch_adj) ** 0.5)
        pitch_adj = pitch_sign * pitch_norm

        pitch_mod = f"{pitch_adj:+.2f}%" if pitch_adj not in [0, -float("inf")] else "+0%"
        rate_mod = f"{rate_adj:+.2f}%" if rate_adj not in [0, -float("inf")] else "+0%"
        loudness_mod = f"{loudness_adj:+.2f}%" if loudness_adj not in [0, -float("inf")] else "+0%"

    # Pause calculation (keep existing logic)
    duration_pause_syntagme_natural = duration_pause_syntagme_natural * 1000
    duration_pause_syntagme_natural /= 3
    if pd.isna(duration_pause_syntagme_natural) or duration_pause_syntagme_natural == 0:
        duration_pause_syntagme_natural = max_pause
    else:
        duration_pause_syntagme_natural *= pause_coef
        if duration_pause_syntagme_natural > max_pause:
            duration_pause_syntagme_natural = max_pause
        if duration_pause_syntagme_natural < min_pause:
            duration_pause_syntagme_natural = min_pause
        duration_pause_syntagme_natural = int(duration_pause_syntagme_natural)

    # Return only the inner fragment
    if str(text).strip() == "":
        # This is a pause segment
        return f"<break time='{duration_pause_syntagme_natural}ms'/>"
    else:
        # Clean text for SSML - remove control characters, handle special XML chars if necessary
        clean_text = re.sub(r'[\x00-\x1F\x7F]', '', str(text)).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Handle punctuation hints (keep existing logic, but apply to clean_text)
        punctuation = ""
        if clean_text.endswith( (",", "ß") ):
             clean_text = clean_text[:-1] + ", h" # Keep the breath hint
        elif clean_text.endswith("!"):
             clean_text = clean_text[:-1] + "! h"
        elif clean_text.endswith("?"):
             clean_text = clean_text[:-1] + "? h"
        # No change needed for "." or ";" or other cases based on original logic

        prosody_content = f"<prosody pitch='{pitch_mod}' rate='{rate_mod}' volume='{loudness_mod}'>{clean_text}</prosody>"

        # Optional: Add style if provided
        if style:
             return f"<mstts:express-as style='{style}' styledegree='{styledegree}'>{prosody_content}</mstts:express-as>"
        else:
             return prosody_content


def get_wav(BDD4_dir, audio_output_root, voice, style, styledegree, BDD5_dir): # Renamed audio_output to audio_output_root
    df = pd.read_csv(BDD4_dir)
    n = len(df["synthesized_syntagme_audio_path"])
    # ... (segment calculation remains the same) ...
    S_N = df["natural_syntagme_audio_path"].apply(lambda row: (int(row.split("segment_ph")[1].split(".")[0]) if isinstance(row, str) and "segment_ph" in row else -10))
    S_S = df["synthesized_syntagme_audio_path"].apply(lambda row: (int(row.split("segment_ph")[1].split(".")[0]) if isinstance(row, str) and "segment_ph" in row else -10))
    df["segment"] = [(S_N[i] if S_N[i] != -10 else S_S[i]) for i in range(n)]
    # Fill remaining -10 (handle cases where both might be NaN initially)
    last_valid_segment = -1
    for i in range(n):
        if df["segment"][i] != -10:
            last_valid_segment = df["segment"][i]
        else:
            df.loc[i, "segment"] = last_valid_segment # Use loc for assignment
    # Ensure audio_output_root and Temp exist (adjust path if needed)
    # os.makedirs(audio_output_root, exist_ok=True) # This is handled by the pipeline now
    # os.makedirs(os.path.join(audio_output_root, "Temp"), exist_ok=True) # This is handled by the pipeline now

    # ... (fillna and replace logic remains the same) ...
    df['syntagme'] = df['syntagme'].fillna('')
    df['syntagme'] = df['syntagme'].astype(str).replace('0', '')

    global pause_coef, max_pause, min_pause # Make params global if create_ssml_fragment needs them
    pause_coef = 1.0
    max_pause = 500
    min_pause = 1

    pd.set_option('display.max_colwidth', None)

    # Apply the fragment function
    df['ssml_fragment'] = df.apply(lambda row: create_ssml_fragment(
        row['syntagme'],
        row['pourcentage_relative_pitch_modification'],
        row['rate_ajusté'],
        row['loudness_adjustment'],
        row['duration_pause_syntagme_natural'],
        voice, style, styledegree # Pass voice/style here
    ), axis=1)

    # Group by segment and join fragments
    df_ = df.loc[:, ["syntagme", "segment", "ssml_fragment"]]
    df_["syntagme"] = df_["syntagme"].apply(lambda row: (" " if row == "" else row)) # Keep space for joining text
    
    merged_df_ = df_.groupby('segment', as_index=False).agg(
        # Join syntagmes for reference text
        syntagme=('syntagme', lambda x: ''.join(x).strip()),
        # Join SSML fragments
        ssml_content=('ssml_fragment', ' '.join) # Join with space
    )

    # Construct the final, valid SSML string for each segment
    merged_df_['ssml'] = merged_df_['ssml_content'].apply(
        lambda content: f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='fr-FR'><voice name='{voice}'>{content}</voice></speak>"
    )

    # Clean up and save
    merged_df_["ssml"] = merged_df_["ssml"].apply(lambda row: row.replace("\n", "").replace("    ", " ")) # Clean whitespace
    merged_df_ = merged_df_[merged_df_['segment'] != -1] # Filter invalid segments
    
    # Select final columns and save
    final_df = merged_df_[['segment', 'syntagme', 'ssml']]
    final_df.to_csv(BDD5_dir, index=False) # Save without the default pandas index


def main():
    if len(sys.argv) != 7:
        print(
            "Usage: python Get_Wav.py",
            "<BDD4_dir>",
            "<audio_output_root>", # Changed name
            "<voice>",
            "<style>",
            "<styledegree>",
            "<BDD5_dir>"
        )
        sys.exit(1)

    BDD4_dir = sys.argv[1]
    audio_output_root = sys.argv[2] # Changed name
    voice = sys.argv[3]
    # Handle None strings for style/styledegree
    style = sys.argv[4] if sys.argv[4].lower() != 'none' else None
    styledegree = sys.argv[5] if sys.argv[5].lower() != 'none' else None
    BDD5_dir = sys.argv[6]
    get_wav(BDD4_dir, audio_output_root, voice, style, styledegree, BDD5_dir)


if __name__ == "__main__":
    main()