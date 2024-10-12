from pathlib import Path

from rnnoise_wrapper import RNNoise

PROJECT_DPATH = Path(__file__).resolve().parents[1]
DATA_DPATH = PROJECT_DPATH / "data"

denoiser = RNNoise()

audio = denoiser.read_wav(DATA_DPATH / "voice" / "test.wav")
denoised_audio = denoiser.filter(audio)
denoiser.write_wav(DATA_DPATH / "voice" / "test_denoised.wav", denoised_audio)
