from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import os
import numpy as np
import librosa
import textgrid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger.info(f"Static directory mounted at: {STATIC_DIR}")

# Define paths to audio and TextGrid files
BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "voice"))
NATURAL_AUDIO_DIR = os.path.join(BASE_DATA_DIR, "records", "audio")
TTS_AUDIO_DIR = os.path.join(BASE_DATA_DIR, "records_microsoft", "audio")
NATURAL_TEXTGRID_DIR = os.path.join(BASE_DATA_DIR, "records", "WhisperTS_textgrid_files")
TTS_TEXTGRID_DIR = os.path.join(BASE_DATA_DIR, "records_microsoft", "WhisperTS_textgrid_files")

# Helper function to list available segments based on natural audio files
def get_segments():
    segments = []
    if not os.path.exists(NATURAL_AUDIO_DIR):
        logger.error(f"Natural audio directory not found: {NATURAL_AUDIO_DIR}")
        return segments
    for file in os.listdir(NATURAL_AUDIO_DIR):
        if file.lower().endswith('.wav'):
            segment = os.path.splitext(file)[0]
            # Check if corresponding TextGrid exists in natural folder
            textgrid_path = os.path.join(NATURAL_TEXTGRID_DIR, f"{segment}.TextGrid")
            if os.path.exists(textgrid_path):
                segments.append(segment)
    # Sort segments by the numeric part after "ph"
    def sort_key(seg):
        try:
            return int(seg.split("ph")[-1])
        except ValueError:
            return 0
    return sorted(segments, key=sort_key)

@app.get("/segments")
async def list_segments():
    segments = get_segments()
    if not segments:
        raise HTTPException(status_code=404, detail="No segments found.")
    return {"segments": segments}

# Helper functions for audio processing
def parse_textgrid(textgrid_path, tier_name="words"):
    """Parse TextGrid file to extract word intervals and labels."""
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        tier = next((t for t in tg.tiers if t.name.lower() == tier_name.lower()), None)
        if tier is None:
            raise ValueError(f"Tier '{tier_name}' not found in {textgrid_path}")
        return [(i.minTime, i.maxTime, i.mark.strip()) for i in tier.intervals]
    except Exception as e:
        logger.error(f"Error parsing TextGrid {textgrid_path}: {e}")
        return []

def compute_spectrogram(audio, sr, n_fft=1024, hop_length=256):
    """Compute spectrogram in dB."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)

def compute_pitch(audio, sr, fmin=60.0, fmax=2000.0, hop_length=256):
    """Compute pitch contour using PYIN."""
    f0, _, _ = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
    time_f0 = np.arange(len(f0)) * hop_length / sr
    return time_f0, f0

def sanitize_array(arr):
    """Replace NaN, Inf, -Inf with JSON-compliant values."""
    return np.where(np.isnan(arr) | np.isinf(arr), None, arr).tolist()

# Serve audio files with segment as a parameter
@app.get("/audio/{audio_type}/{segment}")
async def get_audio(audio_type: str, segment: str):
    if audio_type == "natural":
        path = os.path.join(NATURAL_AUDIO_DIR, f"{segment}.wav")
    elif audio_type == "tts":
        path = os.path.join(TTS_AUDIO_DIR, f"{segment}.wav")
    else:
        raise HTTPException(status_code=404, detail="Audio type not found")
    if not os.path.exists(path):
        logger.error(f"Audio file not found: {path}")
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")
    logger.info(f"Serving audio file: {path}")
    return FileResponse(path, media_type="audio/wav")

# Serve plot data with segment as a parameter
@app.get("/plot_data/{audio_type}/{segment}")
async def get_plot_data(audio_type: str, segment: str):
    if audio_type == "natural":
        wav_path = os.path.join(NATURAL_AUDIO_DIR, f"{segment}.wav")
        textgrid_path = os.path.join(NATURAL_TEXTGRID_DIR, f"{segment}.TextGrid")
    elif audio_type == "tts":
        wav_path = os.path.join(TTS_AUDIO_DIR, f"{segment}.wav")
        textgrid_path = os.path.join(TTS_TEXTGRID_DIR, f"{segment}.TextGrid")
    else:
        raise HTTPException(status_code=404, detail="Audio type not found")

    if not os.path.exists(wav_path):
        logger.error(f"Audio file not found: {wav_path}")
        raise HTTPException(status_code=404, detail=f"Audio file not found: {wav_path}")
    if not os.path.exists(textgrid_path):
        logger.error(f"TextGrid file not found: {textgrid_path}")
        raise HTTPException(status_code=404, detail=f"TextGrid file not found: {textgrid_path}")

    # Load audio
    audio, sr = librosa.load(wav_path, sr=None)
    time = np.linspace(0, len(audio) / sr, num=len(audio))

    # Downsample waveform for efficiency
    downsample_factor = 100
    audio_downsampled = audio[::downsample_factor]
    time_downsampled = time[::downsample_factor]

    # Compute spectrogram
    D = compute_spectrogram(audio, sr)
    # Downsample less aggressively for higher detail (factor 2)
    D_downsampled = D[:, ::2]
    time_spec = librosa.times_like(D, sr=sr, hop_length=256)[::2]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    # Compute pitch
    time_f0, f0 = compute_pitch(audio, sr)

    # Parse word intervals
    intervals = parse_textgrid(textgrid_path)

    # Sanitize data to handle NaN and Inf
    logger.info(f"Sanitizing data for {audio_type} segment {segment}")
    data = {
        "waveform": {
            "time": sanitize_array(time_downsampled),
            "audio": sanitize_array(audio_downsampled)
        },
        "spectrogram": {
            "D": [sanitize_array(row) for row in D_downsampled],
            "time": sanitize_array(time_spec),
            "freqs": sanitize_array(freqs)
        },
        "pitch": {
            "time": sanitize_array(time_f0),
            "f0": sanitize_array(f0)
        },
        "intervals": intervals
    }

    logger.info(f"Serving plot data for {audio_type} segment {segment}")
    return data

# Serve the main page
@app.get("/")
async def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"Index file not found: {index_path}")
        raise HTTPException(status_code=404, detail="Index file not found")
    return FileResponse(index_path)

# Handle favicon request with 204 No Content
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)
