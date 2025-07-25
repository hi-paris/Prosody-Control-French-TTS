# app.py
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import librosa
import textgrid
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ── Directories ────────────────────────────────────────────────────────────────
ABTEST_DIR         = Path("/path/to/your/abtest/directory")  # Change this to your actual pat
MICRO_AUDIO_DIR    = ABTEST_DIR / "Audios_microsoft"
IMPROVED_AUDIO_DIR = ABTEST_DIR / "Audios_improved"
RESULTS_DIR        = IMPROVED_AUDIO_DIR.parent.parent / "Out" / "results"

# Where your TextGrids live:
TTS_TEXTGRID_DIR   = Path("/path/to/your/textgrids")  # Change this to your actual path

# Static files for the front‑end
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── In‑memory cache ─────────────────────────────────────────────────────────────
# Keys: (audio_type, segment) -> JSON‑serializable plot data
plot_data_cache = {}

# ── Helper to find merged TTS audio ─────────────────────────────────────────────
def find_tts_file(segment: str) -> Path:
    candidate = MICRO_AUDIO_DIR / f"{segment}_microsoft_merged.wav"
    if candidate.exists():
        return candidate
    matches = list(MICRO_AUDIO_DIR.glob(f"{segment}*_merged.wav"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"TTS audio not found for '{segment}'")

# ── API: List segment IDs ───────────────────────────────────────────────────────
@app.get("/segments")
def list_segments():
    if not RESULTS_DIR.exists():
        raise HTTPException(404, "Results directory not found")
    segs = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())
    if not segs:
        raise HTTPException(404, "No segments found")
    return {"segments": segs}

# ── API: Serve raw audio ────────────────────────────────────────────────────────
@app.get("/audio/{audio_type}/{segment}")
def get_audio(audio_type: str, segment: str):
    try:
        if audio_type == "improved":
            path = IMPROVED_AUDIO_DIR / f"{segment}_OUT.wav"
        elif audio_type == "tts":
            path = find_tts_file(segment)
        else:
            raise HTTPException(404, "Unknown audio type")
        if not path.exists():
            raise HTTPException(404, f"Audio file not found: {path}")
        return FileResponse(str(path), media_type="audio/wav")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

# ── Synchronous worker for plot data ────────────────────────────────────────────
def compute_plot_data_sync(audio_type: str, segment: str):
    """
    Compute waveform, spectrogram, pitch, intervals for one pair.
    Returns a dict ready to be JSON‑returned.
    """
    # 1) Decide wav path & collect TextGrid intervals
    if audio_type == "improved":
        wav_path = IMPROVED_AUDIO_DIR / f"{segment}_OUT.wav"
        intervals = []
    else:  # tts
        wav_path = find_tts_file(segment)
        intervals = []
        # Grab every ph chunk TextGrid for this segment
        for tg_file in sorted(TTS_TEXTGRID_DIR.glob(f"{segment}_ph*.TextGrid")):
            tg = textgrid.TextGrid.fromFile(str(tg_file))
            tier = next((t for t in tg.tiers if t.name.lower()=="words"), None)
            if tier:
                for iv in tier.intervals:
                    intervals.append((iv.minTime, iv.maxTime, iv.mark))

    if not wav_path.exists():
        raise FileNotFoundError(f"Audio not found: {wav_path}")

    # 2) Load & downsample waveform
    audio, sr = librosa.load(str(wav_path), sr=None)
    t = np.linspace(0, len(audio)/sr, num=len(audio))[::100]
    audio_ds = audio[::100]

    # 3) Spectrogram
    D = librosa.stft(audio, n_fft=1024, hop_length=256)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)[:, ::2]
    times = librosa.times_like(D, sr=sr, hop_length=256)[::2]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    # 4) Pitch (PYIN returns 3 arrays)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, sr=sr, fmin=60.0, fmax=2000.0, hop_length=256
    )
    f0_t = librosa.times_like(f0, sr=sr, hop_length=256)

    # 5) Sanitize for JSON
    def clean(x): return np.where(np.isfinite(x), x, None).tolist()

    return {
        "waveform":    {"time": clean(t),       "audio": clean(audio_ds)},
        "spectrogram": {"D": [clean(r) for r in D_db], "time": clean(times), "freqs": clean(freqs)},
        "pitch":       {"time": clean(f0_t),    "f0": clean(f0)},
        "intervals":   intervals
    }

# ── API: Serve cached plot data ────────────────────────────────────────────────
@app.get("/plot_data/{audio_type}/{segment}")
def get_plot_data(audio_type: str, segment: str):
    key = (audio_type, segment)
    if key not in plot_data_cache:
        raise HTTPException(404, f"Plot data not found for {audio_type}/{segment}")
    return plot_data_cache[key]

# ── Preload everything in parallel on startup ──────────────────────────────────
@app.on_event("startup")
def preload_all():
    segs = list_segments()["segments"]
    tasks = [(atype, seg) for seg in segs for atype in ("improved","tts")]

    logger.info(f"Preloading {len(tasks)} plot_data tasks across CPU cores...")
    with ProcessPoolExecutor() as executor:
        future_to_key = {
            executor.submit(compute_plot_data_sync, atype, seg): (atype, seg)
            for atype, seg in tasks
        }
        for future in as_completed(future_to_key):
            atype, seg = future_to_key[future]
            try:
                plot_data_cache[(atype, seg)] = future.result()
            except Exception as e:
                logger.error(f"Failed to preload {atype}/{seg}: {e}")

    logger.info(f"Preloaded plot data for {len(plot_data_cache)} entries.")

# ── Root + favicon ────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)