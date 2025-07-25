import json
import re
from typing import Dict, List, Any


# Original functions (extract_pure_ssml_from_parsed_sequence, extract_ssml_with_structure, extract_breaks_only_ssml)
# are kept here for completeness if other parts of your script use them.
# You can remove them if they are no longer needed.

def extract_pure_ssml_from_parsed_sequence(parsed_sequence: List[Dict]) -> str:
    """
    Convert parsed sequence to pure SSML without parameter values.
    (Original function - kept for compatibility if used elsewhere)
    """
    ssml_parts = []
    for segment in parsed_sequence:
        stype = segment.get("type", "")
        text = segment.get("text", "")
        if stype == "text":
            ssml_parts.append(text)
        elif stype == "break":
            ssml_parts.append("<break/>")
        elif stype == "emphasis":
            ssml_parts.append(f"<emphasis>{text}</emphasis>")
        elif stype == "say-as":
            ssml_parts.append(f"<say-as>{text}</say-as>")
        elif stype == "phoneme":
            ssml_parts.append(f"<phoneme>{text}</phoneme>")
        elif stype == "sub":
            ssml_parts.append(f"<sub>{text}</sub>")
        elif stype == "voice":
            ssml_parts.append(f"<voice>{text}</voice>")
        elif stype == "audio":
            ssml_parts.append("<audio/>")
        else:
            if text:
                ssml_parts.append(text)
    return "".join(ssml_parts)


def extract_ssml_with_structure(parsed_sequence: List[Dict]) -> str:
    """
    Extract SSML with structural tags annotated but without actual values.
    (Original function - kept for compatibility if used elsewhere)
    """
    ssml_parts = []
    for segment in parsed_sequence:
        stype = segment.get("type", "")
        text = segment.get("text", "")
        if stype == "text":
            prosody = segment.get("prosody", {})
            if prosody:
                attrs = []
                for key in ("pitch", "rate", "volume"):
                    if key in prosody:
                        attrs.append(key)
                attr_str = " ".join(attrs)
                if attr_str:
                    ssml_parts.append(f"<prosody {attr_str}>{text}</prosody>")
                else:
                    ssml_parts.append(f"<prosody>{text}</prosody>")
            else:
                ssml_parts.append(text)
        elif stype == "break":
            if segment.get("time") is not None:
                ssml_parts.append("<break time/>")
            else:
                ssml_parts.append("<break/>")
        elif stype == "emphasis":
            if segment.get("level"):
                ssml_parts.append(f"<emphasis level>{text}</emphasis>")
            else:
                ssml_parts.append(f"<emphasis>{text}</emphasis>")
        else: # Simplified from original for brevity, other tags would be text
            if text:
                ssml_parts.append(text)
    return "".join(ssml_parts)

def extract_breaks_only_ssml(parsed_sequence: List[Dict]) -> str:
    """
    Extract only the break segments and convert to SSML.
    (Original function - kept for compatibility if used elsewhere,
    though convert_breaks_to_full_ssml now uses a different logic for filtering)
    """
    ssml_parts = []
    for segment in parsed_sequence:
        stype = segment.get("type", "")
        if stype == "break":
            if segment.get("time") is not None:
                time_val = segment.get("time", "")
                ssml_parts.append(f"<break time=\"{time_val}\"/>")
            else:
                ssml_parts.append("<break/>")
    return "".join(ssml_parts)

# --- New helper functions for the desired output format ---

def extract_text_and_simple_breaks_for_x(parsed_sequence: List[Dict]) -> str:
    """
    Generates the 'x' field: text content interspersed with simple <break/> tags.
    Example: "text1<break/>text2<break/><break/>text3"
    """
    parts = []
    for segment in parsed_sequence:
        stype = segment.get("type", "")
        text = segment.get("text", "")
        if stype == "text":
            parts.append(text)
        elif stype == "break":
            parts.append("<break/>")
        # Other SSML types are ignored for 'x' as per the target format
    return "".join(parts)


def format_y_ssml_with_values_and_structure(parsed_sequence: List[Dict]) -> str:
    """
    Generates the 'y' field: complete SSML with attributes, values, and specific formatting.
    """
    ssml_elements = [] # Stores strings for each SSML line/block

    idx = 0
    while idx < len(parsed_sequence):
        segment = parsed_sequence[idx]
        stype = segment.get("type", "")
        # It's good practice to escape XML special characters in text, but not implemented here.
        text = segment.get("text", "")

        if stype == "text":
            prosody = segment.get("prosody", {})
            attr_parts = []
            # Only add attributes if they exist in the prosody dict
            if "pitch" in prosody: attr_parts.append(f'pitch="{prosody["pitch"]}"')
            if "rate" in prosody: attr_parts.append(f'rate="{prosody["rate"]}"')
            if "volume" in prosody: attr_parts.append(f'volume="{prosody["volume"]}"')

            attr_str = (" " + " ".join(attr_parts)) if attr_parts else ""
            ssml_elements.append(f'  <prosody{attr_str}>\n    {text}\n  </prosody>')
            idx += 1
        elif stype == "break":
            current_breaks_tags = []
            temp_idx = idx # Use a temporary index for lookahead within breaks
            while temp_idx < len(parsed_sequence) and parsed_sequence[temp_idx].get("type") == "break":
                break_segment = parsed_sequence[temp_idx]
                time_val = break_segment.get("time")
                # Use <break/> if time is None or an empty string, otherwise include time attribute
                tag = f'<break time="{time_val}"/>' if time_val is not None and time_val != "" else "<break/>"
                current_breaks_tags.append(tag)
                temp_idx += 1

            if current_breaks_tags:
                ssml_elements.append("  " + "".join(current_breaks_tags))
            idx = temp_idx # Advance main index past all processed breaks

        elif stype == "emphasis":
            level = segment.get("level")
            attr_s = f' level="{level}"' if level else ""
            ssml_elements.append(f'  <emphasis{attr_s}>{text}</emphasis>')
            idx += 1
        elif stype == "say-as":
            attrs_list = []
            if segment.get("interpret-as"): attrs_list.append(f'interpret-as="{segment["interpret-as"]}"')
            if segment.get("format"): attrs_list.append(f'format="{segment["format"]}"')
            if segment.get("detail"): attrs_list.append(f'detail="{segment["detail"]}"') # Added detail
            attr_s = (" " + " ".join(attrs_list)) if attrs_list else ""
            ssml_elements.append(f'  <say-as{attr_s}>{text}</say-as>')
            idx += 1
        elif stype == "phoneme":
            attrs_list = []
            orthography = text # The text content of the tag
            phonemic_str = segment.get("ph") # The 'ph' attribute value
            if segment.get("alphabet"): attrs_list.append(f'alphabet="{segment["alphabet"]}"')
            if phonemic_str: attrs_list.append(f'ph="{phonemic_str}"')
            attr_s = (" " + " ".join(attrs_list)) if attrs_list else ""
            ssml_elements.append(f'  <phoneme{attr_s}>{orthography}</phoneme>')
            idx += 1
        elif stype == "sub":
            alias = segment.get("alias")
            attr_s = f' alias="{alias}"' if alias else ""
            # Text is the original content to be substituted.
            ssml_elements.append(f'  <sub{attr_s}>{text}</sub>')
            idx += 1
        elif stype == "voice":
            attrs_list = []
            if segment.get("name"): attrs_list.append(f'name="{segment["name"]}"')
            if segment.get("gender"): attrs_list.append(f'gender="{segment["gender"]}"')
            # xml:lang might be stored differently, e.g., segment.get("lang") or segment.get("xml:lang")
            # Adjust based on your bdd.json structure. Assuming "xml:lang" for now.
            if segment.get("xml:lang"): attrs_list.append(f'xml:lang="{segment["xml:lang"]}"')
            attr_s = (" " + " ".join(attrs_list)) if attrs_list else ""
            ssml_elements.append(f'  <voice{attr_s}>{text}</voice>')
            idx += 1
        elif stype == "audio":
            src = segment.get("src")
            fallback_text = text # Text content acts as fallback
            if src:
                ssml_elements.append(f'  <audio src="{src}">{fallback_text if fallback_text else ""}</audio>')
            else:
                ssml_elements.append(f'  <audio>{fallback_text}</audio>') # No src, text is primary
            idx += 1
        else: # Fallback for unknown types or segments that are just text (should ideally not happen if schema is strict)
            if text:
                ssml_elements.append(f"  {text}") # Simple indented text
            idx += 1

    # Join elements with newlines, adding an extra newline between break groups and subsequent prosody.
    final_y_output_parts = []
    num_elements = len(ssml_elements)
    for i, current_element_str in enumerate(ssml_elements):
        final_y_output_parts.append(current_element_str)
        is_break_element = "  <break" in current_element_str # More robust check
        is_prosody_element = lambda s: s.startswith("  <prosody")

        if is_break_element:
            if (i + 1) < num_elements and is_prosody_element(ssml_elements[i+1]):
                final_y_output_parts.append("") # Add an empty string for an extra blank line

    if not final_y_output_parts: return ""
    return " " + "\n".join(final_y_output_parts)


def format_z_ssml_template_from_parsed_sequence(parsed_sequence: List[Dict]) -> str:
    """
    Generates the 'z' field: an SSML template based on the structure of 'y',
    with placeholder attribute values.
    """
    ssml_elements = []
    idx = 0
    while idx < len(parsed_sequence):
        segment = parsed_sequence[idx]
        stype = segment.get("type", "")
        text = segment.get("text", "") # Text content or orthography

        if stype == "text":
            # For 'z', always include the prosody tag with placeholder attributes
            ssml_elements.append(f'  <prosody pitch="_%" rate="_%" volume="_%">\n    {text}\n  </prosody>')
            idx += 1
        elif stype == "break":
            current_breaks_tags = []
            temp_idx = idx
            while temp_idx < len(parsed_sequence) and parsed_sequence[temp_idx].get("type") == "break":
                # For 'z', always use the placeholder for time if it's a break tag.
                # The original example for 'z' uses 'ms' without underscore.
                # The user's transform_x_to_z example used '_ms'. Let's stick to '_ms' for consistency.
                current_breaks_tags.append('<break time="_ms"/>')
                temp_idx += 1
            
            if current_breaks_tags:
                ssml_elements.append("  " + "".join(current_breaks_tags))
            idx = temp_idx
        
        elif stype == "emphasis":
            # For 'z', always show the 'level' attribute as a placeholder
            ssml_elements.append(f'  <emphasis level="_">{text}</emphasis>')
            idx += 1
        elif stype == "say-as":
            # For 'z', show common attributes as placeholders
            ssml_elements.append(f'  <say-as interpret-as="_" format="_" detail="_">{text}</say-as>')
            idx += 1
        elif stype == "phoneme":
            orthography = text
            # For 'z', show common attributes as placeholders
            ssml_elements.append(f'  <phoneme alphabet="_" ph="_">{orthography}</phoneme>')
            idx += 1
        elif stype == "sub":
            # For 'z', show 'alias' as a placeholder
            ssml_elements.append(f'  <sub alias="_">{text}</sub>')
            idx += 1
        elif stype == "voice":
            # For 'z', show common voice attributes as placeholders
            ssml_elements.append(f'  <voice name="_" gender="_" xml:lang="_">{text}</voice>')
            idx += 1
        elif stype == "audio":
            fallback_text = text
            # For 'z', show 'src' as a placeholder
            ssml_elements.append(f'  <audio src="_">{fallback_text if fallback_text else ""}</audio>')
            idx += 1
        else: # Fallback
            if text:
                ssml_elements.append(f"  {text}")
            idx += 1

    # Same joining logic as for 'y' to maintain structural parity
    final_z_output_parts = []
    num_elements = len(ssml_elements)
    for i, current_element_str in enumerate(ssml_elements):
        final_z_output_parts.append(current_element_str)
        is_break_element = "  <break" in current_element_str
        is_prosody_element = lambda s: s.startswith("  <prosody")
        
        if is_break_element:
            if (i + 1) < num_elements and is_prosody_element(ssml_elements[i+1]):
                final_z_output_parts.append("") 
    
    if not final_z_output_parts: return ""
    return " " + "\n".join(final_z_output_parts)


# --- Modified convert_breaks_to_full_ssml ---

def convert_breaks_to_full_ssml(input_file: str, output_file: str) -> int:
    """
    Create dataset where:
    - Entries are included only if their parsed_sequence contains at least one break.
    - 'x' contains text segments interspersed with simple <break/> tags, truncated to ~100 words
      (splitting at nearest period if possible).
    - 'y' is the corresponding full SSML.
    - 'z' is the SSML template corresponding to 'y'.
    Returns number of entries written.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    MAX_WORDS = 100 # Max words for 'x' content before attempting to chunk

    for eid, edata in data.items():
        parsed_original_full = edata.get('y', {}).get('parsed_sequence')
        if not (isinstance(parsed_original_full, list) and parsed_original_full):
            continue

        # Filter: ensure the original full sequence for this EID contains at least one break
        has_breaks_in_original_sequence = any(segment.get("type") == "break" for segment in parsed_original_full)
        if not has_breaks_in_original_sequence:
            continue

        # Step 1: Segment the original `parsed_original_full` sequence into chunks
        parsed_sequence_source_chunks = []  # List of lists of tokens (List[List[Dict]])
        current_chunk_tokens = []
        current_word_count = 0

        for token_data in parsed_original_full:
            token_text_content = ""
            num_new_words = 0
            is_text_token_with_content = token_data.get("type") == "text" and token_data.get("text")
            if is_text_token_with_content:
                token_text_content = token_data.get("text", "")
                num_new_words = len(token_text_content.split())

            if current_chunk_tokens and (current_word_count + num_new_words > MAX_WORDS):
                split_after_idx_in_current_chunk = -1
                # Try to find a period in text segments towards the end of the current chunk
                for i in range(len(current_chunk_tokens) - 1, -1, -1):
                    ct = current_chunk_tokens[i]
                    if ct.get("type") == "text" and ct.get("text", "").strip().endswith("."):
                        split_after_idx_in_current_chunk = i
                        break
                
                if split_after_idx_in_current_chunk != -1:
                    chunk_to_finalize = current_chunk_tokens[:split_after_idx_in_current_chunk + 1]
                    parsed_sequence_source_chunks.append(list(chunk_to_finalize)) # Add a copy
                    current_chunk_tokens = current_chunk_tokens[split_after_idx_in_current_chunk + 1:]
                else:
                    parsed_sequence_source_chunks.append(list(current_chunk_tokens)) # Add a copy
                    current_chunk_tokens = []

                current_word_count = 0
                for t in current_chunk_tokens: # Recalculate for the new (potentially partial) current_chunk_tokens
                    if t.get("type") == "text" and t.get("text"):
                        current_word_count += len(t.get("text", "").split())
            
            current_chunk_tokens.append(token_data)
            if is_text_token_with_content:
                current_word_count += num_new_words

        if current_chunk_tokens:
            parsed_sequence_source_chunks.append(list(current_chunk_tokens)) # Add the final chunk

        # Step 2: Process each chunk to generate x, y, z
        temp_results_for_this_eid = []
        for chunk_seq in parsed_sequence_source_chunks:
            if not chunk_seq: continue

            # Crucially, the chunk itself must contain at least one break to be included
            # as per the spirit of 'xy_pairs_breaks_to_full.json'.
            # If not, this chunk might not be representative for break-focused modeling.
            # However, the initial filter is at EID level. If an EID has breaks, all its
            # chunks are processed. If a specific chunk has no breaks, its x will be just text,
            # and its y/z might also lack breaks. This seems to be the current logic.
            # Let's ensure it still makes sense. The problem statement is to make "z" part of the generation.
            # The existing code filters EIDs if *their original* sequence has no breaks.
            # This means a *chunk* might not have breaks but is still processed.
            
            x_content = extract_text_and_simple_breaks_for_x(chunk_seq)
            
            x_text_only_for_validation = re.sub(r'<break\s*/>', ' ', x_content)
            x_text_only_for_validation = re.sub(r'\s+', ' ', x_text_only_for_validation).strip()
            
            if not x_text_only_for_validation: # Skip if chunk has no actual text for X
                continue

            y_content = format_y_ssml_with_values_and_structure(chunk_seq)
            z_content = format_z_ssml_template_from_parsed_sequence(chunk_seq) # Generate 'z'
            
            temp_results_for_this_eid.append({"x": x_content, "y": y_content, "z": z_content})

        # Step 3: Assign IDs and add to final results list
        if not temp_results_for_this_eid:
            continue 
        
        if len(temp_results_for_this_eid) == 1:
            single_entry = temp_results_for_this_eid[0]
            results.append({
                "id": eid,
                "x": single_entry["x"],
                "y": single_entry["y"],
                "z": single_entry["z"] # Add 'z'
            })
        else:
            for i, part_entry in enumerate(temp_results_for_this_eid):
                results.append({
                    "id": f"{eid}_part{i + 1}",
                    "x": part_entry["x"],
                    "y": part_entry["y"],
                    "z": part_entry["z"] # Add 'z'
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return len(results)


# --- Other functions from your original script (unchanged) ---

def convert_full_paragraphs(input_file: str,
                            output_file: str,
                            include_structure_info: bool = False) -> int:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for eid, edata in data.items():
        parsed = edata.get('y', {}).get('parsed_sequence')
        if isinstance(parsed, list) and parsed:
            segments = []
            current_seg = []
            word_count = 0
            for token in parsed:
                token_text = token.get('text', '')
                token_words = token_text.split()
                if word_count + len(token_words) > 100 and current_seg: # Max 100 words per segment
                    segments.append(current_seg)
                    current_seg = []
                    word_count = 0
                current_seg.append(token)
                word_count += len(token_words)
            if current_seg:
                segments.append(current_seg)

            for i, seg in enumerate(segments, start=1):
                seg_id = eid if len(segments) == 1 else f"{eid}_part{i}"
                seg_texts = [tok.get('text', '') for tok in seg]
                x_text = " ".join(s.strip() for s in seg_texts if s.strip())
                
                if include_structure_info:
                    y_ssml = extract_ssml_with_structure(seg)
                else:
                    y_ssml = extract_pure_ssml_from_parsed_sequence(seg)
                results.append({"id": seg_id, "x": x_text, "y": y_ssml})
        else:
            text = edata.get('x', '')
            seg_id = eid
            if parsed: 
                 y_ssml = (extract_ssml_with_structure(parsed)
                          if include_structure_info else extract_pure_ssml_from_parsed_sequence(parsed))
            else: 
                y_ssml = f"<speak>{text}</speak>"
            results.append({"id": seg_id, "x": text, "y": y_ssml})
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return len(results)


def convert_segments(input_file: str, output_file: str) -> int:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments_data = []
    for edata in data.values():
        parsed = edata.get('y', {}).get('parsed_sequence')
        if isinstance(parsed, list):
            for seg_dict in parsed:
                if seg_dict.get('type') == 'text' and seg_dict.get('text'):
                    text = seg_dict['text']
                    ssml = extract_pure_ssml_from_parsed_sequence([seg_dict]) 
                    segments_data.append({"x": text, "y": ssml})
        else:
            text = edata.get('x', '')
            ssml = f"<speak>{text}</speak>"
            segments_data.append({"x": text, "y": ssml})
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)
    return len(segments_data)


def train_val_test_split(json_path: str,
                         train_split_ratio: float, 
                         val_split_ratio: float,   
                         test_split_ratio: float) -> None:
    if not (0 <= train_split_ratio <= 1 and 0 <= val_split_ratio <= 1 and 0 <= test_split_ratio <= 1):
        raise ValueError("Split ratios must be between 0 and 1.")
    if round(train_split_ratio + val_split_ratio + test_split_ratio, 5) != 1.0:
        raise ValueError("Sum of split ratios must be 1.")
    with open(json_path, 'r', encoding='utf-8') as f:
        arr = json.load(f)
    total = len(arr)
    t_end = int(total * train_split_ratio)
    v_end = t_end + int(total * val_split_ratio)
    splits = {
        'train': arr[:t_end],
        'val': arr[t_end:v_end],
        'test': arr[v_end:]
    }
    output_dir = os.path.dirname(json_path) # Save splits in the same directory as the input json_path
    if not output_dir: # If json_path is just a filename, output to current dir
        output_dir = "."

    for name, subset in splits.items():
        output_filename = os.path.join(output_dir, f"{name}.json")
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            json.dump(subset, f_out, ensure_ascii=False, indent=2)
        print(f"Wrote {len(subset)} {name} entries to {output_filename}")
    # Original print, a bit confusing with 'name.json'
    # print(f"Wrote {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test entries to {name}.json files.")


def verify_ssml_format_empty(input_file: str) -> bool:
    pattern = re.compile(r'^<prosody pitch="___" rate="___" volume="___">.+</prosody>$')
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    is_valid = True
    for idx, entry in enumerate(data):
        y_ssml = entry.get('y', '')
        if not pattern.match(y_ssml):
            print(f"Invalid SSML (according to specific test pattern) at index {idx}: {y_ssml}")
            is_valid = False
    if is_valid:
        print("All entries valid (according to specific test pattern).")
    return is_valid

if __name__ == "__main__":
    import os # Added for path operations

    # --- Configuration ---
    # IMPORTANT: Adjust these paths to your environment
    BASE_PROJECT_DIR = "/home/mila/d/dauvetj/mon_projet_TTS/Code/ssml_models/jonah"
    INPUT_BDD_JSON = os.path.join(BASE_PROJECT_DIR, "bdd.json") 
    OUTPUT_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "full_data_xyz") # New directory for outputs with Z

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    # Output file for the breaks dataset (will contain x, y, and z)
    breaks_out_xyz = os.path.join(OUTPUT_DATA_DIR, "xy_pairs_breaks_to_full_xyz.json")
    
    # --- File Existence Check ---
    if not os.path.exists(INPUT_BDD_JSON):
        print(f"ERROR: Input file not found at {INPUT_BDD_JSON}")
        print("Please ensure 'bdd.json' is at the correct path and BASE_PROJECT_DIR is set correctly.")
        exit()

    # --- Generating the xy_pairs_breaks_to_full_xyz.json with x, y, and z ---
    print(f"Converting entries from {INPUT_BDD_JSON} to {breaks_out_xyz} with x, y, and z format...")
    count_breaks_xyz = convert_breaks_to_full_ssml(INPUT_BDD_JSON, breaks_out_xyz)
    print(f"Converted {count_breaks_xyz} entries (containing breaks and chunked) to {breaks_out_xyz}")
    
    # Create train/val/test splits for the new xyz breaks dataset
    if count_breaks_xyz > 0:
        print(f"Splitting {breaks_out_xyz} into train/val/test sets (in {OUTPUT_DATA_DIR})...")
        # train_val_test_split will save train.json, val.json, test.json in the same dir as breaks_out_xyz
        train_val_test_split(breaks_out_xyz, 0.8, 0.1, 0.1) 
    else:
        print(f"Skipping split for {breaks_out_xyz} as no entries were generated.")

    # --- Original calls (can be uncommented and paths adjusted if needed) ---
    # full_out_original = os.path.join(BASE_PROJECT_DIR, "full_data", "xy_pairs_full.json") # Example adjusted path
    # os.makedirs(os.path.join(BASE_PROJECT_DIR, "full_data"), exist_ok=True) # Ensure dir exists
    # print(f"\nConverting to paragraph entries (full_out_original)...")
    # count_full = convert_full_paragraphs(INPUT_BDD_JSON, full_out_original)
    # print(f"Converted {count_full} paragraph entries to {full_out_original}")
    # if count_full > 0:
    #     print(f"Splitting {full_out_original} into train/val/test sets...")
    #     train_val_test_split(full_out_original, 0.8, 0.1, 0.1)

    # seg_out_original = os.path.join(BASE_PROJECT_DIR, "full_data", "xy_pairs_segments.json") # Example adjusted path
    # print(f"\nConverting to segment entries (seg_out_original)...")
    # count_seg = convert_segments(INPUT_BDD_JSON, seg_out_original)
    # print(f"Converted {count_seg} segment entries to {seg_out_original}")
    # if count_seg > 0:
    #    print(f"Splitting {seg_out_original} into train/val/test sets...")
    #    train_val_test_split(seg_out_original, 0.8, 0.1, 0.1)

    print("\nProcessing complete.")