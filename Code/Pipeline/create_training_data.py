import csv
import json
import os
import re
import xml.etree.ElementTree as ET
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SSML_NS = "http://www.w3.org/2001/10/synthesis"

def clean_ssml_str(ssml_string: str) -> str:
    """
    Strip out xmlns declarations and any namespace prefix
    (e.g. ns0:prosody → prosody, ns0:break → break).
    """
    # remove xmlns="…" or xmlns:ns0="…"
    ssml_string = re.sub(r'\sxmlns(:\w+)?="[^"]+"', '', ssml_string)
    # remove any prefix like ns0:prosody or abc:break
    return re.sub(r'\w+:(prosody|break)', r'\1', ssml_string)

def create_training_data(bdd_ssml_path, output_path):
    if not os.path.exists(bdd_ssml_path):
        raise FileNotFoundError(f"CSV not found: {bdd_ssml_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Reading CSV from: {bdd_ssml_path}")

    # we'll pull out each <speak>…</speak> block
    speak_block = re.compile(r'(<speak.*?</speak>)', re.DOTALL)

    combined_texts  = []
    parsed_sequence = []
    raw_ssml        = {}
    stripped_ssml   = {}

    with open(bdd_ssml_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg       = row['segment'].strip()
            syntagme  = row['syntagme'].strip()
            ssml_full = row['ssml'].strip()

            # collect for x / raw_ssml scaffolding
            if syntagme:
                combined_texts.append(syntagme)
            raw_ssml.setdefault(seg, []).append(ssml_full)
            stripped_ssml.setdefault(seg, [])

            # now parse out each <speak> block in that cell
            for block in speak_block.findall(ssml_full):
                root = ET.fromstring(block)
                voice = root.find(f".//{{{SSML_NS}}}voice")
                if voice is None:
                    continue

                prosody = voice.find(f".//{{{SSML_NS}}}prosody")
                if prosody is None:
                    continue

                pitch  = prosody.get("pitch",  "")
                rate   = prosody.get("rate",   "")
                volume = prosody.get("volume", "")

                # 1) any text *before* the first <break>
                if prosody.text and prosody.text.strip():
                    txt = prosody.text.strip()
                    parsed_sequence.append({
                        "segment": seg,
                        "type":    "text",
                        "text":    txt,
                        "prosody": {"pitch": pitch, "rate": rate, "volume": volume}
                    })
                    # capture the full prosody element once
                    raw = ET.tostring(prosody, encoding="unicode", method="xml")
                    stripped_ssml[seg].append(clean_ssml_str(raw))

                # 2) now interleave each <break> + the tail text that follows
                for child in prosody:
                    tag = child.tag.split("}")[-1]
                    if tag == "break":
                        t = child.get("time", "")
                        parsed_sequence.append({
                            "segment": seg,
                            "type":    "break",
                            "time":    t
                        })
                        raw = ET.tostring(child, encoding="unicode", method="xml")
                        stripped_ssml[seg].append(clean_ssml_str(raw))

                    # tail text (text immediately after that break)
                    if child.tail and child.tail.strip():
                        tail = child.tail.strip()
                        parsed_sequence.append({
                            "segment": seg,
                            "type":    "text",
                            "text":    tail,
                            "prosody": {"pitch": pitch, "rate": rate, "volume": volume}
                        })
                        # (we don’t re-append stripped SSML for each tail chunk)

    if not parsed_sequence:
        raise ValueError("No SSML elements found in CSV.")

    combined_x = " ".join(combined_texts).strip()
    logging.info(f"Collected {len(combined_texts)} segments, {len(parsed_sequence)} SSML entries")

    out = {
        "x": combined_x,
        "y": {
            "parsed_sequence": parsed_sequence,
            "stripped_ssml":   stripped_ssml,
            "raw_ssml":        raw_ssml
        }
    }

    with open(output_path, 'w', encoding='utf-8') as jf:
        json.dump(out, jf, ensure_ascii=False, indent=2)
    logging.info(f"✅ Training data written to {output_path}")

def combine_training_jsons(results_folder, combined_json_path):
    combined = {}
    if not os.path.isdir(results_folder):
        logging.warning(f"No results folder at: {results_folder}")
        return

    for name in os.listdir(results_folder):
        folder = os.path.join(results_folder, name)
        if not os.path.isdir(folder):
            continue

        merged = {"x":"", "y":{"parsed_sequence":[], "stripped_ssml":{}, "raw_ssml":{}}}
        for fn in os.listdir(folder):
            if fn.startswith("training_data_") and fn.endswith(".json") and fn!="bdd.json":
                path = os.path.join(folder, fn)
                with open(path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                merged["x"] += data.get("x","") + " "
                for e in data["y"].get("parsed_sequence",[]):
                    merged["y"]["parsed_sequence"].append(e)
                for seg, lst in data["y"].get("stripped_ssml",{}).items():
                    merged["y"]["stripped_ssml"].setdefault(seg,[]).extend(lst)
                for seg, lst in data["y"].get("raw_ssml",{}).items():
                    merged["y"]["raw_ssml"].setdefault(seg,[]).extend(lst)

        merged["x"] = merged["x"].strip()
        combined[name] = merged
        logging.info(f"Combined folder {name}: {len(merged['y']['parsed_sequence'])} entries")

    with open(combined_json_path, 'w', encoding='utf-8') as jf:
        json.dump(combined, jf, ensure_ascii=False, indent=2)
    logging.info(f"✅ All combined at {combined_json_path}")

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("Usage: create_training_data.py <BDD_ssml.csv> <output.json>")
        sys.exit(1)
    try:
        create_training_data(sys.argv[1], sys.argv[2])
        combine_training_jsons(
            os.path.dirname(sys.argv[2]),
            os.path.join(os.path.dirname(sys.argv[2]), "bdd.json")
        )
    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)