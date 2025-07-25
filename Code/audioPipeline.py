import os
import sys
import logging
import torch
import yaml
import numpy as np
import pandas as pd
import parselmouth
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pathlib import Path
from tqdm import tqdm
from xml.sax.saxutils import escape as xml_escape
import pyloudnorm as pyln
from Preprocessing.gen_break_ssml import extract_words_and_pauses
import multiprocessing
import re
import shutil
import pandas as pd
import textgrid
from difflib import SequenceMatcher
import spacy
import json


_nlp = spacy.load("fr_core_news_sm", disable=["ner"])
_FORBIDDEN = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON"}

# 1) Load configuration
def load_config():
    base = Path(__file__).resolve().parent.parent
    cfg_file = base / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing config.yaml at {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty config.yaml")
    return cfg

# 2) Setup unified logger
def setup_logging(out_dir: Path):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(str(logs / "pipeline_debug.log"), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return root

def remove_spurious_commas(text: str) -> str:
    _PAUSE_MARKERS = {"[*]"}  # extend with "[pause]" or others if needed
    _PUNCT_TO_STRIP = {",", "."} | _PAUSE_MARKERS
    """
    Strip any comma immediately following a DET/ADP/CCONJ/SCONJ/PART token.
    """
    doc = _nlp(text)
    out = []
    for tok in doc:
        # is this a comma or a recognized pause token?
        if (tok.text == "," or tok.text in _PAUSE_MARKERS) and out:
            prev = out[-1]
            if prev.pos_ in _FORBIDDEN:
                # skip this comma/pause
                continue
        out.append(tok)
    # rebuild with original spacing
    return "".join(t.text_with_ws for t in out)

class AudioPipeline:
    def __init__(self, name, cfg):
        self.name = name
        base = Path(__file__).resolve().parent.parent

        # Configured paths
        self.cfg = cfg
        self.data_dir = base / cfg["data_dir"]
        self.out_dir = base / cfg["out_dir"]
        self.voice_dir = self.data_dir / name
        self.raw_synth_dir = self.data_dir / f"{name}_raw"
        self.ssml_dir = self.data_dir / f"{name}_ssml"
        self.xml_dir = self.ssml_dir / "xml_files"
        self.audio_out = self.ssml_dir / "audio"
        self.results_dir = self.out_dir / "results" / name
        self.audio_ssml_dir = self.results_dir / "segmented_audio"
        self.azure_key_file = base / cfg["azure_key_file"]

        # Paths derived from steps (define them here for robustness when skipping steps)
        self.textgrid_dir = self.voice_dir / "WhisperTS_textgrid_files"
        self.transcription_dir = self.voice_dir / "transcription"
        self.raw_audio_dir = self.raw_synth_dir / "audio"
        self.bdd_ssml_csv = self.results_dir / "BDD_ssml.csv"
        self.bdd_syntagme_ssml_csv = self.results_dir / "BDD_syntagme_ssml.csv" 
        self.bdd_syntagme_synth_csv = self.results_dir / "BDD_syntagme_for_synth.csv"


        # Azure & whisper settings
        self.azure_voice = cfg["azure_voice_name"]
        self.whisper_device = cfg.get("whisper_device", "cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model = cfg.get("whisper_model", "turbo")
        self.azure_region = cfg.get("azure_region", "francecentral")

        # Silence-split
        sil = cfg["silence"]
        self.min_silence_len = sil["min_silence_len"]
        self.silence_thresh = sil["silence_thresh"]
        self.keep_silence = sil["keep_silence"]

        # Prosody settings
        # prosody_cfg = cfg.get("prosody_settings", {})
        # self.p_st = prosody_cfg.get("pitch_semitones", 2.0)
        # self.pitch_lower_clip_factor = prosody_cfg.get("pitch_lower_clip_factor", 0.7)
        # self.v_db = prosody_cfg.get("volume_db", 7.0)
        prosody_cfg = cfg.get("prosody_settings", {})
        self.p_st = prosody_cfg.get("pitch_semitones", 2.0)
        self.pitch_lower_clip_factor = prosody_cfg.get("pitch_lower_clip_factor", 0.7)
        self.v_pct = prosody_cfg.get("volume_pct", 7.0)

        self.r_pct_clamp = prosody_cfg.get("rate_percent", 15.0)
        self.alpha = prosody_cfg.get("smoothing_alpha", 0.4)
        self.max_jump = prosody_cfg.get("max_jump_percent", 5.0)
        self.end_pause_ms = prosody_cfg.get("end_punctuation_pause_ms", 150)
        self.baseline_window = prosody_cfg.get("baseline_window", None)
        self.inter_syntagme_pause_factor = prosody_cfg.get("inter_syntagme_pause_factor", 1)
        self.threshold_duration_before_slowing_down = prosody_cfg.get("threshold_duration_before_slowing_down", 1.0)
        self.slow_floor_per_sec = prosody_cfg.get("slow_floor_per_sec", 2.0)

        # Create all needed dirs
        for d in [
            self.raw_synth_dir,
            self.ssml_dir,
            self.xml_dir,
            self.audio_out,
            self.audio_ssml_dir,
            self.results_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Read Azure key
        self.api_key = self.azure_key_file.read_text(encoding="utf-8").strip()

    def preprocess(self):
        logging.info(">>> Preprocess: Demucs + Silence-Split")
        from Preprocessing.demucs_process import main as demucs_main
        from Preprocessing.preprocess_audio import main as preprocess_main

        brute_mp3 = self.voice_dir / "brute" / "segment.mp3"
        brute_wav = self.voice_dir / "brute" / "segment.wav"
        demucs_out = self.voice_dir / "brute" / "segment_demucs.wav"

        if brute_mp3.exists():
            demucs_main(str(brute_mp3), str(demucs_out))
        elif brute_wav.exists():
            demucs_main(str(brute_wav), str(demucs_out))
        else:
            raise FileNotFoundError("No brute audio found for Demucs")

        preprocess_main(
            str(demucs_out),
            str(self.voice_dir / "audio"),
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=self.keep_silence
        )

    def align_and_transcribe(self):
        logging.info(">>> Align & Transcribe: WhisperTS")
        from Aligners.use_whisper_timestamped import main as whisper_main
        from Pipeline.utils import (
            save_clean_transcriptions_from_textgrids,
        )

        audio_folder = str(self.voice_dir / "audio")
        tg_folder = self.voice_dir / "WhisperTS_textgrid_files"
        txt_folder = self.voice_dir / "transcription"

        txt_raw_folder     = self.voice_dir / "transcription_raw"
        raw_json_dir   = Path(str(tg_folder) + "_raw_json") 

        # Delete existing folders, if they already exist
        shutil.rmtree(tg_folder, ignore_errors=True)
        shutil.rmtree(txt_folder, ignore_errors=True)
        shutil.rmtree(self.voice_dir / "WhisperTS_textgrid_files_transcription", ignore_errors=True) # this folder will be created by whisper_main
        shutil.rmtree(txt_raw_folder, ignore_errors=True)
        shutil.rmtree(raw_json_dir, ignore_errors=True)

        # Now, make sure the folders exist
        tg_folder.mkdir(parents=True, exist_ok=True)
        txt_folder.mkdir(parents=True, exist_ok=True)
        txt_raw_folder.mkdir(parents=True, exist_ok=True)
        raw_json_dir.mkdir(parents=True, exist_ok=True)


        whisper_main(
            audio_folder,
            str(tg_folder),
            whisper_model=self.whisper_model,
            device=self.whisper_device,
            logger=logging.getLogger(),
        )

    
        # -------- Raw Transcrpts & JSONs --------    
        # — save the raw JSON files for later use —
        for js in raw_json_dir.glob("*.raw.json"):
            data = json.loads(js.read_text(encoding="utf-8"))
            raw_text = " ".join(seg["text"] for seg in data["segments"])
            (txt_raw_folder / js.name.replace(".raw.json", ".txt")) \
                .write_text(raw_text, encoding="utf-8")

        # — add a placeholder transcription for any missing files in the raw folder —
        for wav in Path(audio_folder).glob("*.wav"):
            stem = wav.stem
            raw_txt = txt_raw_folder / f"{stem}.txt"
            if not raw_txt.exists():
                raw_txt.write_text("...", encoding="utf-8")
        # ------- End of Raw Transcrpts & JSONs -------

        # Proceed with the regular transcription
        save_clean_transcriptions_from_textgrids(
            tg_folder, txt_folder
        )

        # clean punctuation _only_ for SSML-driven steps
        for txt in Path(txt_folder).glob("*.txt"):
            txt_str = txt.read_text(encoding="utf-8")
            cleaned = remove_spurious_commas(txt_str)
            txt.write_text(cleaned, encoding="utf-8")

    def raw_synthesis(self):
        logging.info(">>> Raw Synthetic Synthesis (archive)")
        from Preprocessing.get_synth import main as get_synth_main

        get_synth_main(
            str(self.voice_dir),
            str(self.voice_dir / "audio"),
            str(self.raw_synth_dir / "audio"),
            str(self.voice_dir / "transcription_raw"), # previously self.transcription_dir
            str(self.raw_synth_dir / "transcription"),
            self.api_key,
            self.azure_voice,
            None,
            None,
            False,
        )


    def measure_prosody_and_build_ssml(self):
        logging.info(">>> Measure Prosody & Build SSML")
        # print(self.inter_syntagme_pause_factor)

        def construct_syntagmes_seq(seq: list) -> list:
            """
            Turn a sequence [(type, token, pause_ms)] into a list of syntagmes:
            each dict has keys: words, start_ms, end_ms, pause_ms
            """
            synts = []
            time_cursor = 0
            current = []
            start_time = 0

            for kind, tok, dur in seq:
                if kind == "word":
                    # start a new word‐syntagme if needed
                    if not current:
                        start_time = time_cursor
                    current.append(tok.strip())
                    # **advance** the cursor by the actual word duration
                    time_cursor += dur
                else:  # pause
                    # close off any running words‐syntagme
                    if current:
                        synts.append({
                            "words": " ".join(current),
                            "start_ms": start_time,
                            "end_ms": time_cursor,
                            "pause_ms": 0
                        })
                        current = []
                    # record the pause as its own syntagme
                    synts.append({
                        "words": "",
                        "start_ms": time_cursor,
                        "end_ms": time_cursor + dur,
                        "pause_ms": dur
                    })
                    time_cursor += dur

            # final words‐only syntagme
            if current:
                synts.append({
                    "words": " ".join(current),
                    "start_ms": start_time,
                    "end_ms": time_cursor,
                    "pause_ms": 0
                })

            return synts
        
        # ── new helper: get the actual duration of a slice of an audio file ─────────
        def get_part_duration(wav_path, t0=0.0, t1=None):
            """
            Return the duration (in seconds) of the audio between t0 and t1.
            Falls back to a tiny epsilon if the slice is empty.
            """
            audio = AudioSegment.from_file(str(wav_path))
            if t1 is not None:
                seg = audio[int(t0*1000):int(t1*1000)]
                return seg.duration_seconds or 1e-4
            return audio.duration_seconds or 1e-4
    
        # Helpers to extract median pitch over a time window
        def get_median_pitch(wav_path, t0=0.0, t1=None):
            snd_full = parselmouth.Sound(str(wav_path))
            if t1 is None:
                pitch = snd_full.to_pitch(pitch_floor=150, pitch_ceiling=600)
            else:
                part = snd_full.extract_part(from_time=t0, to_time=t1, preserve_times=True)
                pitch = part.to_pitch(pitch_floor=150, pitch_ceiling=600)
            freqs = pitch.selected_array["frequency"]
            voiced = freqs[freqs > 0]
            return float(np.median(voiced)) if voiced.size>0 else 0.0

        # Helpers to compute LUFS over a time window
        def get_lufs(wav_path, meter, t0=0.0, t1=None):
            # load full audio, then slice if needed
            audio = AudioSegment.from_file(str(wav_path))
            if t1 is not None:
                audio = audio[int(t0*1000):int(t1*1000)]
            samples = np.array(audio.get_array_of_samples(), dtype=float)
            # if we sliced out zero samples, fall back to the full segment
            if samples.size == 0:
                full = AudioSegment.from_file(str(wav_path))
                fsamples = np.array(full.get_array_of_samples(), dtype=float)
                samples = fsamples
            peak = np.abs(samples).max() or 1.0
            norm = samples / peak
            try:
                return meter.integrated_loudness(norm)
            except ValueError:
                # fallback to full segment-level loudness
                full = AudioSegment.from_file(str(wav_path))
                fsamples = np.array(full.get_array_of_samples(), dtype=float)
                fpeak = np.abs(fsamples).max() or 1.0
                return meter.integrated_loudness(fsamples / fpeak)

        def get_duration(wav_path):
            return AudioSegment.from_file(str(wav_path)).duration_seconds or 1e-4

        # ── 1) Segment-level prosody stats & sliding-window baselines ─────────
        seg_files = sorted(
            self.voice_dir.joinpath("audio").glob("*.wav"),
            key=lambda p: int(re.search(r"segment_ph(\d+)", p.stem).group(1))
        )
        if not seg_files:
            logging.error("No audio segments found!")
            return

        # segment stats
        meter = pyln.Meter(AudioSegment.from_file(str(seg_files[0])).frame_rate)
        seg_stats = []
        for wav in seg_files:
            seg = wav.stem
            raw = self.raw_audio_dir / f"{seg}.wav"
            seq = extract_words_and_pauses(str(self.textgrid_dir/f"{seg}.TextGrid"))
            wc  = sum(1 for k,t,m in seq if k=="word" and t.strip())
            p_nat = get_median_pitch(wav)
            l_nat = get_lufs(wav, meter)
            try:
                l_syn = get_lufs(raw, meter)
                d_syn = get_duration(raw)
            except CouldntDecodeError:
                logging.warning(f"Couldn’t decode raw audio {raw.name}; falling back to natural metrics")
                l_syn = l_nat
                d_syn = get_duration(wav)
            d_nat = get_duration(wav)
            rate_ratio = (wc/d_nat)/(wc/d_syn) if wc>0 and d_syn>0 else 1.0
            seg_stats.append({
                "segment": seg,
                "p_nat": p_nat,
                "l_nat": l_nat,
                "l_syn": l_syn,
                "d_nat": d_nat,
                "d_syn": d_syn,
                "wc": wc,
                "rate_ratio": rate_ratio
            })
        n_seg = len(seg_stats)
        win = self.baseline_window
        if win is None or win >= n_seg:
            # global baseline
            f0_all = float(np.median([s["p_nat"] for s in seg_stats if s["p_nat"]>0])) or 1.0
            loud_all = float(np.median([s["l_nat"] for s in seg_stats]))
            rate_all = float(np.median([s["rate_ratio"] for s in seg_stats]))
            dynamic = False
        else:
            half = win//2
            dynamic = True
        # precompute baseline per segment
        baselines = []
        for i in range(n_seg):
            if dynamic:
                start = max(0, i-half)
                end   = min(n_seg, i+half+1)
                window = seg_stats[start:end]
                f0_base = float(np.median([w["p_nat"] for w in window if w["p_nat"]>0])) or 1.0
                loud_base = float(np.median([w["l_nat"] for w in window]))
                rate_base = float(np.median([w["rate_ratio"] for w in window]))
            else:
                f0_base, loud_base, rate_base = f0_all, loud_all, rate_all
            baselines.append({"f0":f0_base, "loud":loud_base, "rate":rate_base})

        # prosody clamp & smoothing params
        P_ST  = self.p_st
        # V_DB  = self.v_db
        V_PCT = self.v_pct
        R_PCT = self.r_pct_clamp
        α     = self.alpha
        max_jump = self.max_jump
        end_ms   = self.end_pause_ms

        # ── 2) Build raw adjustments per syntagme ───────────────────────────────
        raw_rows = []
        for idx, wav in enumerate(seg_files):
            seg = wav.stem
            raw = self.raw_audio_dir / f"{seg}.wav"
            # 1) pull out the raw (word,token,pause) list
            raw_seq = extract_words_and_pauses(str(self.textgrid_dir/f"{seg}.TextGrid"))

            # clean up any spurious commas in each word token
            raw_seq = [
                (kind,
                 remove_spurious_commas(tok) if kind == "word" else tok,
                 dur)
                for kind, tok, dur in raw_seq
            ]

            filtered_seq = []
            prev_item = None
            for item in raw_seq:
                kind, tok, dur = item
                if kind == "pause" and prev_item is not None:
                    pkind, ptok, _ = prev_item
                    if pkind == "word":
                        pos = _nlp(ptok.strip())[0].pos_
                        if pos in _FORBIDDEN:
                            # skip this pause entirely
                            prev_item = item
                            continue
                filtered_seq.append(item)
                prev_item = item
            raw_seq = filtered_seq

            # 2) build a new seq where:
            #    – any real pause after . ? ! is bumped up to at least end_pause_ms
            #    – if there was no pause at all after a sentence‐ending word, we inject one
            punctuated_seq = []
            i = 0
            while i < len(raw_seq):
                kind, tok, dur = raw_seq[i]

                # if this is a pause that follows a sentence‐ending word, clamp it
                if kind == "pause" and i > 0:
                    prev_kind, prev_tok, _ = raw_seq[i-1]
                    if prev_kind == "word" and prev_tok.strip().endswith((".", "?", "!")):
                        dur = max(dur, self.end_pause_ms)

                punctuated_seq.append((kind, tok, dur))

                # if this is a sentence‐ending word *and* the next item isn't a pause,
                # inject a minimum one
                if kind == "word" and tok.strip().endswith((".", "?", "!")):
                    if not (i+1 < len(raw_seq) and raw_seq[i+1][0] == "pause"):
                        punctuated_seq.append(("pause", "", self.end_pause_ms))

                i += 1

            # 3) now build syntagmes out of that
            synts = construct_syntagmes_seq(punctuated_seq)
            meter_seg = pyln.Meter(AudioSegment.from_file(str(wav)).frame_rate)
            base = baselines[idx]
            for syn in synts:
                t0 = syn.get("start_ms",0)/1000
                t1 = syn.get("end_ms",0)/1000
                wc_syn = len(syn.get("words","").split())
                p_nat = get_median_pitch(wav, t0, t1)
                l_nat = get_lufs(wav, meter_seg, t0, t1)
                l_syn = None
                syn_total = None
                try:
                    l_syn      = get_lufs(raw, meter_seg, t0, t1)
                    syn_total  = get_part_duration(raw, t0, t1)
                except CouldntDecodeError:
                    logging.warning(f"⚠️ Couldn’t decode raw piece {raw.name}; using natural slice instead")
                    l_syn     = get_lufs(wav, meter_seg, t0, t1)
                    syn_total = get_part_duration(wav, t0, t1)

                # d_nat = max(t1-t0, 1e-4)
                # d_syn = max(t1-t0, 1e-4)
                 
                # remove the pause time so we only measure “speaking” seconds
                pause_s = syn.get("pause_ms", 0) / 1000.0


                # total segment length minus the pause you inserted
                nat_total   = get_part_duration(wav, t0, t1)
                d_nat       = max(nat_total - pause_s, 1e-4)
                d_syn       = max(syn_total - pause_s, 1e-4)


                # -------------- I. PITCH ADJUSTMENT --------------
                if p_nat>0:
                    st = 12*np.log2(p_nat/base["f0"])
                    st = np.clip(st, -P_ST*self.pitch_lower_clip_factor, P_ST)
                    p_pct = (2**(st/12)-1)*100
                else:
                    p_pct = 0.0
                
                # -------------- II. LOUDNESS ADJUSTMENT --------------
                # g_db = np.clip(base["loud"] - l_syn, -V_DB, V_DB)
                # v_pct = (10**(g_db/20)-1)*100
                db_diff = base["loud"] - l_syn
                v_pct = (10**(db_diff/20) - 1.0) * 100.0
                v_pct = np.clip(v_pct, -self.v_pct, +self.v_pct)


                # -------------- III. RATE ADJUSTMENT --------------
                if wc_syn > 0:
                    nat_r = wc_syn / d_nat
                    syn_r = wc_syn / d_syn
                    rp    = (nat_r - syn_r) / syn_r * 100
                else:
                    rp = 0.0

                # asymmetrically scale rate adjustments by segment length:
                length_s = d_nat
                if length_s <= 1.0:
                    slow_factor = 1.0
                    fast_factor = 1.0
                else:
                    slow_factor = length_s ** 1.5
                    fast_factor = np.sqrt(length_s)
                
                if rp < 0:
                    # stronger slow-down
                    rp = rp * slow_factor
                else:
                    # much weaker speed-up
                    rp = rp / fast_factor

                # “Floor” every chunk longer than 1 s by subtracting
                # an extra slowdown that grows with length
                extra_slow = max(0.0, length_s - self.threshold_duration_before_slowing_down) * self.slow_floor_per_sec
                rp = rp - extra_slow

                # then your existing clamp:
                if length_s > 5.0:
                    max_slowdown = R_PCT * 1.5
                    max_speedup  = R_PCT * 0.5
                else:
                    max_slowdown = R_PCT
                    max_speedup  = R_PCT

                rp = np.clip(rp, -max_slowdown, +max_speedup)


                # ------------------------------------------------

                raw_rows.append({
                    "segment": seg,
                    "syntagme": syn.get("words",""),
                    "pause": syn.get("pause_ms",0),
                    "raw_pitch": float(p_pct),
                    "raw_volume": float(v_pct),
                    "raw_rate": float(rp)
                })
        df = pd.DataFrame(raw_rows)

        # ── 3) Smooth pitch & rate across syntagmes ─────────────────────────────
        sm_p = [df.loc[0, "raw_pitch"]]
        sm_r = [df.loc[0, "raw_rate"]]
        for i in range(1, len(df)):
            sm_p.append(α*df.loc[i,"raw_pitch"] + (1-α)*sm_p[-1])
            sm_r.append(α*df.loc[i,"raw_rate"]  + (1-α)*sm_r[-1])
        for i in range(1, len(sm_p)):
            if abs(sm_p[i]-sm_p[i-1])>max_jump:
                sm_p[i] = sm_p[i-1] + np.sign(sm_p[i]-sm_p[i-1])*max_jump
            if abs(sm_r[i]-sm_r[i-1])>max_jump:
                sm_r[i] = sm_r[i-1] + np.sign(sm_r[i]-sm_r[i-1])*max_jump

        # ── 4a) Segment‐level SSML CSV (for synthesis)
        pieces = []
        for (row, p_adj, r_adj) in zip(raw_rows, sm_p, sm_r):
            text = xml_escape(row["syntagme"])
            # build prosody start
            pros = (
                f'<prosody pitch="{p_adj:+.2f}%" '
                f'rate="{r_adj:+.2f}%" '
                f'volume="{row["raw_volume"]:+.2f}%">'
                f'{text}'
            )
            # if there's a pause, append it _inside_ the prosody
            if row["pause"] >= 50:
                # if the syntagme *ends* in punctuation, treat as segment-break → keep full
                last_char = row["syntagme"][-1] if row["syntagme"] else None
                if last_char is not None and last_char in ".?!":
                    dur = row["pause"]
                else:
                    dur = int(row["pause"] * self.inter_syntagme_pause_factor)
                pros += f'<break time="{dur}ms"/>'
            # close the prosody tag
            pros += '</prosody>'

            pieces.append({"segment": row["segment"], "ssml_piece": pros})

        final = []
        by_seg = {}
        for p in pieces:
            by_seg.setdefault(p["segment"], []).append(p["ssml_piece"])
        for seg, seg_pieces in by_seg.items():
            ssml = (
                '<speak xmlns="http://www.w3.org/2001/10/synthesis" '
                'xmlns:mstts="http://www.w3.org/2001/mstts" '
                'version="1.0" xml:lang="fr-FR">'
                f'<voice name="{self.azure_voice}">'
                '<mstts:silence type="Leading-exact" value="0"/>'
                + "".join(seg_pieces) +
                '<mstts:silence type="Tailing-exact" value="0"/>'
                '</voice>'
                '</speak>'
            )
            final.append({"segment": seg, "ssml": ssml})

        pd.DataFrame(final).to_csv(self.bdd_ssml_csv, index=False)

        # ── 4b) Syntagme‐level SSML CSV (for training)
        syn_rows = []
        for (row, p_adj, r_adj) in zip(raw_rows, sm_p, sm_r):
            text = xml_escape(row["syntagme"])
            pros = (
                f'<prosody pitch="{p_adj:+.2f}%" '
                f'rate="{r_adj:+.2f}%" '
                f'volume="{row["raw_volume"]:+.2f}%">'
                f'{text}'
            )
            if row["pause"] >= 50:
                # if the syntagme *ends* in punctuation, treat as segment-break → keep full
                last_char = row["syntagme"][-1] if row["syntagme"] else None
                if last_char is not None and last_char in ".?!":
                    dur = row["pause"]
                else:
                    dur = int(row["pause"] * self.inter_syntagme_pause_factor)
                pros += f'<break time="{dur}ms"/>'
            pros += '</prosody>'

            ssml = (
                '<speak xmlns="http://www.w3.org/2001/10/synthesis" '
                'version="1.0" xml:lang="fr-FR">'
                f'<voice name="{self.azure_voice}">'
                + pros +
                '</voice></speak>'
            )
            syn_rows.append({
                "segment":  row["segment"],
                "syntagme": row["syntagme"],
                "pause":    row["pause"],
                "ssml":      ssml
            })
        pd.DataFrame(syn_rows).to_csv(self.bdd_syntagme_ssml_csv, index=False)

        # ── 4c) Syntagme‐level SSML CSV (for synthesis, no <break/> tags) ────
        synth_rows = []
        for (row, p_adj, r_adj) in zip(raw_rows, sm_p, sm_r):
            text = xml_escape(row["syntagme"])
            pros_no_break = (
                f'<prosody pitch="{p_adj:+.2f}%" '
                f'rate="{r_adj:+.2f}%" '
                f'volume="{row["raw_volume"]:+.2f}%">'
                f'{text}</prosody>'
            )
            ssml = (
                '<speak xmlns="http://www.w3.org/2001/10/synthesis" '
                'xmlns:mstts="http://www.w3.org/2001/mstts" '
                'version="1.0" xml:lang="fr-FR">'
                f'<voice name="{self.azure_voice}">'
                  '<mstts:silence type="Leading-exact" value="0"/>'
                  + pros_no_break +
                  '<mstts:silence type="Tailing-exact" value="0"/>'
                '</voice>'
                '</speak>'
            )
            synth_rows.append({
                "segment":  row["segment"],
                "syntagme": row["syntagme"],
                "pause":    row["pause"],
                "ssml":     ssml
            })
        pd.DataFrame(synth_rows).to_csv(self.bdd_syntagme_synth_csv, index=False)



    def synthesize_and_merge(self):
        logging.info(">>> Synthesize SSML & Merge (exact SSML breaks)")
        from Preprocessing.synthesize_ssml_voice import main as synth_main
        from pydub import AudioSegment
        from pydub.exceptions import CouldntDecodeError
        from pydub.silence import detect_nonsilent
        import shutil
        import re

        # — clean last run —
        shutil.rmtree(self.xml_dir, ignore_errors=True)
        self.xml_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.audio_out, ignore_errors=True)
        self.audio_out.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.audio_ssml_dir, ignore_errors=True)
        self.audio_ssml_dir.mkdir(parents=True, exist_ok=True)

        # — write XMLs for only the text rows —
        df = pd.read_csv(self.bdd_syntagme_synth_csv, dtype={"syntagme": str})
        df["syntagme"] = df["syntagme"].fillna("")
        has_text = df["syntagme"].str.contains(r"\w", regex=True, na=False)
        content_df = df[has_text].reset_index(drop=True)
        for idx, row in content_df.iterrows():
            (self.xml_dir / f"{idx:04d}.xml").write_text(row["ssml"], encoding="utf-8")

        # — batch-synth all XMLs →
        synth_main(
            self.api_key,
            self.azure_region,
            str(self.xml_dir),
            str(self.audio_out),
            self.azure_voice,
        )

        # — now stitch them back together, *correctly segmented* —
        combined = AudioSegment.empty()
        segment_combined = AudioSegment.empty()
        current_seg = None
        content_idx = 0
        prev_text = None

        # create debug folder
        debug_dir = self.audio_ssml_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            seg_id = row["segment"]

            # —— 1) if we’ve moved into a new segment, flush the old one ——
            if seg_id != current_seg:
                if current_seg is not None:
                    segment_combined.export(
                        self.audio_ssml_dir / f"{current_seg}.wav",
                        format="wav"
                    )
                segment_combined = AudioSegment.empty()
                current_seg = seg_id



            txt = row["syntagme"].strip()
            if txt:
                if txt == "...":
                    # skip ellipses
                    continue

                # —— a) TTS chunk — use the next WAV file ——
                wav_path = self.audio_out / f"{content_idx:04d}.wav"
                try:
                    seg = AudioSegment.from_file(str(wav_path))
                    # # strip Azure padding (we set leading and trailing silence to 0 in azure instructions instead
                    # thresh = seg.dBFS - 30  # -20 before
                    # nonsilent = detect_nonsilent(
                    #     seg,
                    #     min_silence_len=40, # only count silence > 40 ms, so that little breaths inside TTS are kept
                    #     silence_thresh=thresh,
                    # )
                    # if nonsilent:
                    #     # s,e = nonsilent[0][0], nonsilent[-1][1]
                    #     # seg = seg[s:e]
                    #     start = nonsilent[0][0]
                    #     end = nonsilent[-1][1]
                    #     seg = seg[start:end]
                except (CouldntDecodeError, FileNotFoundError):
                    logging.warning(f"⚠️ Couldn’t load TTS wav for “{txt}”; inserting silence")
                    seg = AudioSegment.silent(duration=0)

                # append to the global and to the per-segment buffer
                seg = seg.fade_in(5).fade_out(5) # fade in/out to avoid clicks

                # debug: export this voice seg as a separate file
                # seg.export(debug_dir / f"{content_idx:04d}_voice_{'_'.join(txt.split())}.wav", format="wav")

                combined += seg
                segment_combined += seg
                content_idx += 1
                prev_text = txt

            else:
                # —— b) pure-pause ——
                pause_ms = int(row["pause"])
                # enforce min end-of-sentence pause
                if prev_text and prev_text.endswith((".", "?", "!")):
                    pause_ms = max(pause_ms, self.end_pause_ms)
                sil = AudioSegment.silent(duration=pause_ms)

                # debug: export this silence as a separate file
                sil.export(self.audio_ssml_dir / "debug" / f"{content_idx:04d}_silence.wav", format="wav")

                combined += sil
                segment_combined += sil

        # —— flush the very last segment —
        if current_seg is not None and len(segment_combined) > 0:
            segment_combined.export(
                self.audio_ssml_dir / f"{current_seg}.wav",
                format="wav"
            )


        # — finally write OUT.wav —
        out = self.results_dir / "OUT.wav"
        combined.export(out, format="wav")
        logging.info(f"✅ Final merged with exact SSML breaks → {out}")

    def export_training_json(self):
        logging.info(">>> Export Training JSON")
        import Pipeline.create_training_data as ctd

        # Use the syntagme‐level CSV so 'syntagme' column exists
        j1 = self.results_dir / f"training_data_{self.name}.json"
        ctd.create_training_data(
            str(self.bdd_syntagme_ssml_csv),
            str(j1)
        )
        # combine across *all* voices
        ctd.combine_training_jsons(
            str(self.out_dir / "results"),
            str(self.out_dir / "results" / "bdd.json")
        )

    def final_transcribe(self):
        logging.info(">>> Final Transcribe: WhisperTS on OUT.wav")
        from Aligners.use_whisper_timestamped import main as whisper_main
        from Pipeline.utils import save_clean_transcriptions_from_textgrids

        # Path to the merged file
        out_wav = self.results_dir / "OUT.wav"
        if not out_wav.exists():
            logging.error(f"No OUT.wav found at {out_wav}")
            return

        # Temp dirs under results_dir
        temp_dir = self.results_dir / "final_whisper"
        tg_dir = temp_dir / "WhisperTS_textgrid_files"
        txt_dir = temp_dir / "transcription_final"
        tg_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        # Run Whisper Timestamped on OUT.wav
        whisper_main(
            str(out_wav.parent),           # audio folder (will pick up OUT.wav)
            str(tg_dir),                   # output TextGrid folder
            whisper_model=self.whisper_model,
            device=self.whisper_device,
            logger=logging.getLogger()
        )

        # Clean .TextGrid → .txt
        save_clean_transcriptions_from_textgrids(tg_dir, txt_dir)

        # Move results into top‐level results folder
        for tg in tg_dir.glob("*.TextGrid"):
            tg.rename(self.results_dir / tg.name)
        for txt in txt_dir.glob("*.txt"):
            txt.rename(self.results_dir / txt.name)

        logging.info(f"✅ Final transcription files saved in {self.results_dir}")


    def compare_breaks(self, tol_ms: int = 5) -> pd.DataFrame:
        """
        Compare the expected SSML breaks from the CSV against the actual silences
        in OUT.TextGrid by aligning each CSV speech chunk to its best TextGrid
        speech block and pulling in the following silence (or 0 ms if missing).
        """
        import logging
        import pandas as pd
        import re
        from difflib import SequenceMatcher
        import textgrid
        from collections import defaultdict

        # 1) Load & group TextGrid intervals into speech chunks + following silence
        tg = textgrid.TextGrid.fromFile(str(self.results_dir / "OUT.TextGrid"))
        tier = tg.tiers[0]  # assume first tier is your word IntervalTier
        intervals = [(iv.minTime, iv.maxTime, iv.mark.strip()) for iv in tier.intervals]

        tg_speech = []
        silence_after = []
        idx = 0
        while idx < len(intervals):
            start, end, mark = intervals[idx]
            if mark:
                # collect a run of words
                words = []
                while idx < len(intervals) and intervals[idx][2].strip():
                    words.append(intervals[idx][2])
                    idx += 1
                tg_speech.append(" ".join(words))
                # then the very next silence (or 0 ms)
                if idx < len(intervals) and not intervals[idx][2].strip():
                    s0, s1, _ = intervals[idx]
                    silence_after.append(int(round((s1 - s0) * 1000)))
                    idx += 1
                else:
                    silence_after.append(0)
            else:
                idx += 1

        # 2) Load your CSV, extract speech rows & collect the pure‐pause events
        df = pd.read_csv(self.bdd_syntagme_synth_csv)
        df["syntagme"] = df["syntagme"].fillna("").astype(str)

        csv_speech = []                     # list of {csv_idx, text, segment}
        seq_to_speech_idx = {}              # map row → index in csv_speech
        for i, row in df.iterrows():
            txt = row["syntagme"].strip()
            if re.search(r"\w", txt):
                seq_to_speech_idx[i] = len(csv_speech)
                csv_speech.append({
                    "csv_idx": i,
                    "text": txt,
                    "segment": row["segment"]
                })

        break_events = []  # one per “pure‐pause” row, pointing to the preceding speech_idx
        for i, row in df.iterrows():
            if not row["syntagme"].strip() and i > 0 and re.search(r"\w", df.at[i - 1, "syntagme"]):
                prev_i = i - 1
                sp = seq_to_speech_idx.get(prev_i)
                if sp is not None:
                    break_events.append({
                        "speech_idx": sp,
                        "expected_ms": int(round(float(row["pause"]))),
                        "segment": row["segment"],
                        "text": df.at[prev_i, "syntagme"].strip()
                    })

        # 3) Align csv_speech -> tg_speech via DP (max total fuzzy match)
        def normalize(s: str) -> str:
            s = s.lower()
            s = re.sub(r"[^\w\s]", "", s)
            return re.sub(r"\s+", " ", s).strip()

        def sim(a: str, b: str) -> float:
            return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

        n, m = len(csv_speech), len(tg_speech)
        dp = [[0.0] * (m+1) for _ in range(n+1)]
        prev = [[None] * (m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                match_score = dp[i-1][j-1] + sim(csv_speech[i-1]["text"], tg_speech[j-1])
                # three choices: skip CSV, skip TG, or match
                if dp[i-1][j] >= dp[i][j-1] and dp[i-1][j] >= match_score:
                    dp[i][j] = dp[i-1][j]
                    prev[i][j] = (i-1, j)
                elif dp[i][j-1] >= match_score:
                    dp[i][j] = dp[i][j-1]
                    prev[i][j] = (i, j-1)
                else:
                    dp[i][j] = match_score
                    prev[i][j] = (i-1, j-1)

        # backtrack to collect one-to-one matches
        matches = []
        i, j = n, m
        while i > 0 and j > 0:
            pi, pj = prev[i][j]
            if pi == i-1 and pj == j-1:
                matches.append((i-1, j-1))
            i, j = pi, pj
        matches.reverse()
        speech_to_tg = {csv_i: tg_i for csv_i, tg_i in matches}

        # 3a) build extended-span map: each CSV speech covers tg_speech[tg_i:next_tg_i)
        match_list = sorted(speech_to_tg.items(), key=lambda x: x[0])
        match_list.append((len(csv_speech), len(tg_speech)))
        ext_span = {}
        for k in range(len(match_list)-1):
            csv_i, tg_i = match_list[k]
            next_csv, next_tg = match_list[k+1]
            for ci in range(csv_i, next_csv):
                ext_span[ci] = list(range(tg_i, next_tg))

        # 4) assign each break_event to the last TG index in its CSV span
        event_tg = []
        for ev in break_events:
            span = ext_span.get(ev["speech_idx"], [])
            if span:
                event_tg.append(span[-1])
            else:
                # fallback to direct 1:1 match if available
                single = speech_to_tg.get(ev["speech_idx"])
                event_tg.append(single if single is not None else None)

        # group break_events by TG index so only the last in each group gets the real silence
        tg_to_events = defaultdict(list)
        for idx, tg_idx in enumerate(event_tg):
            if tg_idx is not None:
                tg_to_events[tg_idx].append(idx)

        # 5) build results rows
        rows = []
        for idx, ev in enumerate(break_events):
            tg_idx = event_tg[idx]
            exp_ms = ev["expected_ms"]
            seg = ev["segment"]
            txt = ev["text"]

            # only last event in each TG span gets the measured silence
            if tg_idx is not None and idx == tg_to_events[tg_idx][-1]:
                synth_ms = silence_after[tg_idx]
            else:
                synth_ms = 0

            diff = synth_ms - exp_ms
            mq = sim(txt, tg_speech[tg_idx]) if (tg_idx is not None) else 0.0
            if tg_idx is not None and mq < 0.5:
                logging.warning(f"Low match quality for “{txt}” → “{tg_speech[tg_idx]}”: {mq:.2f}")

            rows.append({
                "segment":        seg,
                "syntagme":       txt,
                "nat_voice_ms":   exp_ms,
                "synth_voice_ms": synth_ms,
                "diff_ms":        diff,
                "ok":             abs(diff) <= tol_ms,
                "match_quality":  round(mq, 2)
            })

        df_res = pd.DataFrame(rows)
        # summary logging
        total = len(df_res)
        if total:
            within = df_res["ok"].sum()
            avg_diff = int(round(df_res["diff_ms"].abs().mean()))
            avg_mq   = round(df_res["match_quality"].mean(), 2)
            logging.info(f"Breaks compared: {total}")
            logging.info(f"Within ±{tol_ms} ms: {within}/{total} ({within/total*100:.1f}%)")
            logging.info(f"Avg |diff|: {avg_diff} ms")
            logging.info(f"Avg match_quality: {avg_mq}")
        else:
            logging.warning("No breaks found to compare.")

        # overwrite CSV and return
        out_path = self.results_dir / "pause_comparison_full.csv"
        df_res.to_csv(out_path, index=False)
        return df_res

    def run(self):
        steps = [
            ("Preprocess", self.preprocess),
            ("Align+Transcribe", self.align_and_transcribe),
            ("Raw Synthesis", self.raw_synthesis),
            ("Measure & Build SSML", self.measure_prosody_and_build_ssml),
            ("Synthesize+Merge", self.synthesize_and_merge),
            ("Export JSON", self.export_training_json),
            ("Final Transcribe", self.final_transcribe),
            ("Compare Breaks", self.compare_breaks),
        ]
        to_run = self.cfg.get("steps_to_run") or [n for n,_ in steps]
        run_list = [(n,f) for n,f in steps if n in to_run]

        pbar = tqdm(run_list, total=len(run_list))
        for name, function in pbar:
            pbar.set_description(f"{self.name} - {name}")
            try:
                function()
            except Exception:
                logging.exception(f"Failed step {name}")
                sys.exit(1)
        
        # save the config used for this run in the results dir
        config_path = self.results_dir / "used_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, default_flow_style=False, allow_unicode=True)
        logging.info(f"Config saved to {config_path}")

def run_pipeline_for_voice(args):
    """Helper function to run the pipeline for a single voice."""
    name, cfg = args
    logger = setup_logging(Path(__file__).resolve().parent.parent / cfg["out_dir"])
    logger.info(f"--- Starting pipeline for: {name} ---")

    try:
        pipeline = AudioPipeline(name, cfg)
        pipeline.run()
        logger.info(f"--- Finished pipeline for: {name} ---")
        return True, name
    except Exception as e:
        logger.error(f"--- Pipeline failed for: {name} ---")
        logger.exception(e)
        return False, name

if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logging(Path(__file__).resolve().parent.parent / cfg["out_dir"])

    voice_names_config = cfg.get("voice_names")
    if not voice_names_config:
        logger.error("Missing 'voice_names' in config.yaml")
        sys.exit(1)

    # Ensure voice_names is a list
    if isinstance(voice_names_config, str):
        voice_names_list = [voice_names_config]
    elif isinstance(voice_names_config, list):
        voice_names_list = voice_names_config
    else:
        logger.error("'voice_names' in config.yaml must be a string or a list")
        sys.exit(1)

    logger.info(f"Processing voices: {', '.join(voice_names_list)}")

    use_multiprocessing = cfg.get("multiprocessing", False)

    if use_multiprocessing and len(voice_names_list) > 1:
        num_processes = cfg.get("num_processes", multiprocessing.cpu_count())
        logger.info(f"Using {num_processes} processes for parallel")
        pool_args = [(name, cfg) for name in voice_names_list]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_processes) as pool:
            results = pool.map(run_pipeline_for_voice, pool_args)

        failed_voices = [name for success, name in results if not success]
        if failed_voices:
            logger.error(f"Some pipelines failed: {', '.join(failed_voices)}")
    else:
        if use_multiprocessing and len(voice_names_list) <= 1:
            logger.info("Multiprocessing enabled, but only one voice to process. Running sequentially.")
        else:
            logger.info("Running sequentially.")

        for name in voice_names_list:
            success, _ = run_pipeline_for_voice((name, cfg))
            if not success:
                continue

    logger.info("All pipelines finished.")
