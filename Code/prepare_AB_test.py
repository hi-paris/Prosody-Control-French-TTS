import yaml
import logging
import random
import re
from pathlib import Path
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Helper to extract numeric index from a stem name
def idx_key(stem: str) -> int:
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else -1


def load_config():
    base = Path(__file__).resolve().parent.parent
    cfg_file = base / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing config.yaml at {cfg_file}")
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty config.yaml")
    return cfg, base


def find_voices(cfg, base):
    out_dir = base / cfg['out_dir']
    results_root = out_dir / 'results'
    voices_cfg = cfg.get('ab_test', {}).get('voices')
    if voices_cfg is None:
        voices = [p.name for p in results_root.iterdir() if p.is_dir()]
    else:
        voices = voices_cfg
    return voices


def collect_segments(raw_dir, imp_dir):
    raw_files = {p.stem: p for p in raw_dir.glob('*.wav')}
    imp_files = {p.stem: p for p in imp_dir.glob('*.wav')}
    stems = set(raw_files) & set(imp_files)
    # sort by numeric suffix
    common = sorted(stems, key=idx_key)
    return common, raw_files, imp_files


def compute_durations(stems, raw_files):
    dur_map = {}
    for stem in stems:
        path = raw_files.get(stem)
        if not path:
            continue
        try:
            audio = AudioSegment.from_file(path)
            dur_map[stem] = audio.duration_seconds
        except CouldntDecodeError:
            logging.warning(f"⚠️ Couldn’t decode {path.name}; skipping this segment")
        except Exception as e:
            logging.warning(f"Unexpected error decoding {path.name}: {e}; skipping")
    return dur_map


def build_chunks(segments, dur_map, target, margin):
    lower = target - margin
    upper = target + margin
    avail = [s for s in segments if s in dur_map]
    chunks = []
    # STEP1: singles within bounds
    for stem in list(avail):
        dur = dur_map[stem]
        if lower <= dur <= upper:
            chunks.append({'segments': [stem], 'trim_last': False, 'trim_duration_s': None})
            avail.remove(stem)
    # STEP2: sequential grouping
    idx = 0
    while idx < len(avail):
        total = 0.0
        group = []
        j = idx
        last_idx = None
        # build group only along contiguous stems
        while j < len(avail) and total < lower:
            seg = avail[j]
            seg_idx = idx_key(seg)
            if last_idx is not None and seg_idx != last_idx + 1:
                break
            group.append(seg)
            total += dur_map[seg]
            last_idx = seg_idx
            j += 1
        if total < lower:
            break
        if total <= upper:
            chunks.append({'segments': group.copy(), 'trim_last': False, 'trim_duration_s': None})
            idx = j
        else:
            # overshoot; try dropping last
            last = group[-1]
            prev_total = total - dur_map[last]
            if prev_total >= lower:
                good = group[:-1]
                chunks.append({'segments': good.copy(), 'trim_last': False, 'trim_duration_s': None})
                idx = idx + len(good)
            else:
                # trim last
                needed = target - prev_total
                chunks.append({'segments': group.copy(), 'trim_last': True, 'trim_duration_s': needed})
                idx = j
    return chunks


def export_pairs(pairs, raw_map, imp_map, cfg, base):
    ab = cfg['ab_test']
    out_dir = base / ab.get('output_dir', cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, pair in enumerate(pairs):
        voice = pair['voice']
        segs = pair['segments']
        trim = pair.get('trim_last', False)
        trim_sec = pair.get('trim_duration_s')
        folder = f"{idx}-{voice}_{'-'.join(segs)}"
        pair_dir = out_dir / folder
        pair_dir.mkdir(parents=True, exist_ok=True)
        # raw
        combined = AudioSegment.empty()
        for i, stem in enumerate(segs):
            seg_audio = AudioSegment.from_file(raw_map[voice][stem])
            if trim and i == len(segs) - 1 and trim_sec:
                seg_audio = seg_audio[:int(trim_sec * 1000)]
            combined += seg_audio
        combined.export(pair_dir / 'raw.wav', format='wav')
        # improved
        combined = AudioSegment.empty()
        for i, stem in enumerate(segs):
            seg_audio = AudioSegment.from_file(imp_map[voice][stem])
            if trim and i == len(segs) - 1 and trim_sec:
                seg_audio = seg_audio[:int(trim_sec * 1000)]
            combined += seg_audio
        combined.export(pair_dir / 'improved.wav', format='wav')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg, base = load_config()
    ab = cfg.get('ab_test')
    if not ab:
        logging.error("No 'ab_test' section in config.yaml")
        return
    target = ab.get('target_duration_s', 60)
    margin = ab.get('margin_s', 20)
    num_pairs = ab.get('num_pairs', 0)
    if num_pairs <= 0:
        logging.error("'num_pairs' must be >0 in ab_test config")
        return

    voices = find_voices(cfg, base)
    all_chunks = []
    raw_map = {}
    imp_map = {}

    for voice in voices:
        raw_dir = base / cfg['data_dir'] / f"{voice}_raw" / 'audio'
        imp_dir = base / cfg['out_dir'] / 'results' / voice / 'segmented_audio'
        if not raw_dir.exists() or not imp_dir.exists():
            logging.warning(f"Skipping {voice}: missing directories")
            continue
        stems, raw_files, imp_files = collect_segments(raw_dir, imp_dir)
        if not stems:
            logging.warning(f"No matching segments for {voice}")
            continue
        dur_map = compute_durations(stems, raw_files)
        valid = sorted(dur_map.keys(), key=idx_key)
        if not valid:
            logging.warning(f"No readable segments for {voice}")
            continue
        raw_map[voice] = raw_files
        imp_map[voice] = imp_files
        # split into runs of consecutive segments
        runs = []
        run = [valid[0]]
        for stem in valid[1:]:
            if idx_key(stem) == idx_key(run[-1]) + 1:
                run.append(stem)
            else:
                runs.append(run)
                run = [stem]
        runs.append(run)
        for r in runs:
            chunks = build_chunks(r, dur_map, target, margin)
            for c in chunks:
                c['voice'] = voice
                all_chunks.append(c)

    if len(all_chunks) < num_pairs:
        logging.error(f"Not enough candidate pairs ({len(all_chunks)}) for requested {num_pairs}")
        return
    selected = random.sample(all_chunks, num_pairs)
    export_pairs(selected, raw_map, imp_map, cfg, base)
    logging.info(f"Exported {num_pairs} pairs to {ab.get('output_dir')}")

if __name__ == '__main__':
    main()
