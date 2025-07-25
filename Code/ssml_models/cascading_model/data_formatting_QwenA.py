import json
import os
import random

def chunk_parsed_sequence(parsed_sequence, max_words=100):
    """
    Split a parsed_sequence (list of tokens) into chunks of at most max_words of text.
    Each chunk is a sub-list of tokens. Splitting occurs at the nearest period if possible.
    """
    chunks = []
    current_chunk = []
    current_word_count = 0

    for token in parsed_sequence:
        is_text = token.get("type") == "text" and token.get("text", "").strip() != ""
        new_words = len(token.get("text", "").split()) if is_text else 0

        # If adding this token would exceed max_words, attempt to split
        if current_chunk and (current_word_count + new_words > max_words):
            # Look backwards for a text token ending with a period
            split_idx = -1
            for i in range(len(current_chunk) - 1, -1, -1):
                ct = current_chunk[i]
                if ct.get("type") == "text" and ct.get("text", "").strip().endswith("."):
                    split_idx = i
                    break

            if split_idx != -1:
                # Split after that period‐ending token
                chunks.append(current_chunk[: split_idx + 1])
                remaining = current_chunk[split_idx + 1 :]
                current_chunk = remaining.copy()
            else:
                # No period found, just cut at current point
                chunks.append(current_chunk.copy())
                current_chunk = []

            # Recompute word count for any tokens carried over
            current_word_count = 0
            for t in current_chunk:
                if t.get("type") == "text" and t.get("text", "").strip():
                    current_word_count += len(t.get("text", "").split())

        # Now add the new token
        current_chunk.append(token)
        if is_text:
            current_word_count += new_words

    # Append the last chunk if nonempty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_plain_and_breaky_simple(parsed_sequence_chunk):
    """
    Given a chunk (sub‐list of tokens), return:
      - x: plain transcript (all “text” tokens, joined with spaces)
      - y: transcript + simple <break/> tags in‐line (no time attribute)
    """
    texts = []
    breaky_parts = []

    for segment in parsed_sequence_chunk:
        stype = segment.get("type")
        if stype == "text":
            txt = segment.get("text", "").strip()
            if txt:
                texts.append(txt)
                breaky_parts.append(txt)
        elif stype == "break":
            # Insert a plain <break/> tag
            breaky_parts.append("<break/>")
        # Ignore other types

    x = " ".join(texts)
    y = " ".join(breaky_parts)
    return x, y


def convert_and_split_with_chunking(
    input_bdd_path,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    max_words=100
):
    """
    1) Load bdd.json
    2) For each entry, split its parsed_sequence into chunks of ≤ max_words (cut at nearest period when possible)
    3) For each chunk, produce {"id": ..., "x": ..., "y": ...}, where:
         - x = plain transcript of that chunk
         - y = transcript + <break/> tags for that chunk
    4) Aggregate all chunked entries, shuffle, and split into train/val/test by the given ratios
    5) Write train.json, val.json, test.json into output_dir
    """
    with open(input_bdd_path, "r", encoding="utf-8") as f:
        bdd = json.load(f)

    all_entries = []
    for eid, edata in bdd.items():
        parsed = edata.get("y", {}).get("parsed_sequence", [])
        if not isinstance(parsed, list) or not parsed:
            continue

        # 2) Chunk the parsed_sequence
        chunks = chunk_parsed_sequence(parsed, max_words=max_words)
        if not chunks:
            continue

        # 3) Build x/y for each chunk
        if len(chunks) == 1:
            chunk_id = eid
            x_text, y_breaks = build_plain_and_breaky_simple(chunks[0])
            if x_text.strip():
                all_entries.append({"id": chunk_id, "x": x_text, "y": y_breaks})
        else:
            for idx, chunk_seq in enumerate(chunks, start=1):
                x_text, y_breaks = build_plain_and_breaky_simple(chunk_seq)
                if not x_text.strip():
                    continue
                chunk_id = f"{eid}_part{idx}"
                all_entries.append({"id": chunk_id, "x": x_text, "y": y_breaks})

    # 4) Shuffle and split
    random.seed(seed)
    random.shuffle(all_entries)

    total = len(all_entries)
    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)
    n_test  = total - n_train - n_val

    train_set = all_entries[:n_train]
    val_set   = all_entries[n_train : n_train + n_val]
    test_set  = all_entries[n_train + n_val : ]

    # 5) Write out JSON files
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    return {
        "total": total,
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
    }


if __name__ == "__main__":
    # ─── Adjust this to point at your project directory ─────────────────────
    BASE_DIR = "/home/mila/d/dauvetj/mon_projet_TTS/Code/ssml_models/jonah"
    INPUT_BDD_JSON = os.path.join(BASE_DIR, "bdd.json")
    OUTPUT_SUBDIR   = os.path.join(BASE_DIR, "simple_xy_split")

    if not os.path.exists(INPUT_BDD_JSON):
        print(f"ERROR: could not find input file at {INPUT_BDD_JSON}")
        exit(1)

    stats = convert_and_split_with_chunking(
        input_bdd_path=INPUT_BDD_JSON,
        output_dir=OUTPUT_SUBDIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        max_words=100
    )

    print(f"Processed {stats['total']} chunked entries total.")
    print(f"  → train: {stats['train']}  |  val: {stats['val']}  |  test: {stats['test']}")
    print(f"Files written into: {OUTPUT_SUBDIR}")
