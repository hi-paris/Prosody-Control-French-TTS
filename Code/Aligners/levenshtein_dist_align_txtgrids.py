from unidecode import unidecode
import os
from pathlib import Path
import textgrid
import sys
import logging
import re

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def list_to_textgrid(L, name="words"):
    """Convert a list of intervals to a TextGrid object"""
    L = sorted(L, key=lambda x: x[1])
    adjusted_L = []
    last_max = 0.0
    for text, min_t, max_t in L:
        if min_t < last_max:
            min_t = last_max
        if max_t <= min_t:
            max_t = min_t + 0.01
        adjusted_L.append((text, min_t, max_t))
        last_max = max_t

    intervals = [textgrid.Interval(m, M, t) for t, m, M in adjusted_L]
    min_time = 0.0
    max_time = adjusted_L[-1][2] if adjusted_L else 0.0
    interval_tier = textgrid.IntervalTier(name=name, minTime=min_time, maxTime=max_time)
    for i in intervals:
        interval_tier.addInterval(i)
    tg = textgrid.TextGrid()
    tg.append(interval_tier)
    return tg

def normalize_word(word):
    """Normalize word by removing accents, punctuation, and converting to lowercase"""
    normalized = unidecode(word)
    normalized = normalized.replace(" ", "")
    for symbol in [".", ", ", "!", "?", ";", ":", "-"]:
        normalized = normalized.replace(symbol, "")
    normalized = normalized.lower()
    return normalized

def levenshtein_distance(s1, s2):
    """
    Efficiently calculate the Levenshtein distance between two strings.
    Args:
        s1 (str): first string
        s2 (str): second string

    Returns:
        int: Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def extract_transcription_from_textgrid(tg_path, output_txt_path):
    """Extract transcription from a TextGrid file and save to text file"""
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
        words = [interval.mark for interval in tg[0] if interval.mark.strip()]
        transcription = re.sub(r'\s+', ' ', " ".join(words)).strip()
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logging.info(f"Transcription extracted to {output_txt_path}")
        return True
    except Exception as e:
        logging.error(f"Error extracting transcription: {e}")
        return False

def update_transcription(textgrid_path, transcription_dir):
    """Update transcription file based on aligned TextGrid"""
    try:
        base_name = os.path.basename(textgrid_path).replace(".TextGrid", "")
        txt_path = os.path.join(transcription_dir, f"{base_name}.txt")
        if extract_transcription_from_textgrid(textgrid_path, txt_path):
            logging.info(f"Transcription updated: {base_name}")
        else:
            logging.warning(f"Update failed: {base_name}")
    except Exception as e:
        logging.error(f"Error updating transcription: {e}")

def main(textgrid1_input_path, textgrid2_input_path, transcription1_dir=None, transcription2_dir=None):
    """Main function to align two TextGrid files using Levenshtein distance"""
    logging.info(f"Processing {textgrid1_input_path}, {textgrid2_input_path}")

    tg1 = textgrid.TextGrid.fromFile(textgrid1_input_path)
    tg2 = textgrid.TextGrid.fromFile(textgrid2_input_path)
    I1, I2 = list(tg1[0]), list(tg2[0])
    n1, n2 = len(I1), len(I2)
    words1, words2 = [i.mark for i in I1], [i.mark for i in I2]

    New_tg1, New_tg2 = [], []
    last1 = last2 = -1
    i = j = 0
    w1, w2 = words1[i], words2[j]

    while i < n1 and j < n2:
        d = levenshtein_distance(w1, w2)
        i_, j_ = min(i + 1, n1 - 1), min(j + 1, n2 - 1)

        if w1.strip() == "":
            New_tg1.append((" ", I1[last1].maxTime if last1 != -1 else I1[0].minTime, I1[i].maxTime))
            last1, i, w1 = i, i_, words1[i_]
            continue
        if w2.strip() == "":
            New_tg2.append((" ", I2[last2].maxTime if last2 != -1 else I2[0].minTime, I2[j].maxTime))
            last2, j, w2 = j, j_, words2[j_]
            continue

        di = levenshtein_distance(w1 + words1[i_], w2)
        dj = levenshtein_distance(w1, w2 + words2[j_])

        if d <= di and d <= dj:
            chosen_word = w2 if len(w2) > len(w1) else w1
            New_tg1.append((chosen_word, I1[last1].maxTime if last1 != -1 else I1[0].minTime, I1[i].maxTime))
            New_tg2.append((chosen_word, I2[last2].maxTime if last2 != -1 else I2[0].minTime, I2[j].maxTime))
            last1, last2, i, j = i, j, i_, j_
            w1, w2 = words1[i], words2[j]
        elif di <= dj:
            i, w1 = i_, w1 + " " + words1[i_]
        else:
            j, w2 = j_, w2 + " " + words2[j_]

    while i < n1:
        min_t = I1[last1].maxTime if last1 != -1 else I1[0].minTime
        New_tg1.append((words1[i], min_t, I1[i].maxTime))
        i, last1 = i + 1, i

    while j < n2:
        min_t = I2[last2].maxTime if last2 != -1 else I2[0].minTime
        New_tg2.append((words2[j], min_t, I2[j].maxTime))
        j, last2 = j + 1, j

    list_to_textgrid(New_tg1).write(textgrid1_input_path)
    list_to_textgrid(New_tg2).write(textgrid2_input_path)

    if transcription1_dir:
        update_transcription(textgrid1_input_path, transcription1_dir)
    if transcription2_dir:
        update_transcription(textgrid2_input_path, transcription2_dir)

    logging.info("Alignment completed successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python levenshtein_dist_align_txtgrids.py <TextGrid1> <TextGrid2> [<Transcription1> <Transcription2>]")
        sys.exit(1)
    main(*sys.argv[1:5])