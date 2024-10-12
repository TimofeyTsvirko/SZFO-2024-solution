import json
import os
import sys
from pathlib import Path

from rnnoise_wrapper import RNNoise
from vosk import KaldiRecognizer, Model

PROJECT_DPATH = Path(__file__).resolve().parents[1]
DATA_DPATH = PROJECT_DPATH / "data"
MODELS_DPATH = PROJECT_DPATH / "models"

model_path = MODELS_DPATH / "vosk-model-small-ru-0.22"
if not os.path.exists(model_path):
    print("Model not found at:", model_path)
    sys.exit(1)

model = Model(str(model_path))
denoiser = RNNoise()

audio = denoiser.read_wav(DATA_DPATH / "voice" / "test.wav")
denoised_audio = denoiser.filter(audio)

recognizer = KaldiRecognizer(model, denoised_audio.frame_rate)

transcription = ""

chunk_size = 4000
for start in range(0, len(denoised_audio), chunk_size):
    end = min(start + chunk_size, len(denoised_audio))
    chunk = denoised_audio[start:end]

    raw_data = chunk.raw_data

    if recognizer.AcceptWaveform(raw_data):
        result = recognizer.Result()
        transcription += json.loads(result)["text"] + " "

transcription += json.loads(recognizer.FinalResult())["text"]

print("Transcription:", transcription)
