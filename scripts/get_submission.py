from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from szfo_2024_solution.voice2text import VoskASR

PROJECT_DPATH = Path(__file__).resolve().parents[1]
DATA_DPATH = PROJECT_DPATH / "data"
MODELS_DPATH = PROJECT_DPATH / "models"


class Predictor:
    """Class for your model's predictions.

    You are free to add your own properties and methods
    or modify existing ones, but the output submission
    structure must be identical to the one presented.

    Examples:
        >>> python -m get_submission --src input_dir --dst output_dir
    """

    def __init__(self):
        self.model = VoskASR(
            MODELS_DPATH / "vosk-model-small-ru-0.22",
            frame_rate=16000,
        )

    def __call__(self, audio_path: str):
        audio_path = Path(audio_path)
        self.model.read_audio(audio_path)
        self.model.denoise_audio()
        recognized_text = self.model.recognize_audio()
        pred_id = self.model.get_pred_label
        pred_attribute = self.model.get_pred_attr

        result = {
            "audio": audio_path.name,  # Audio file base name
            "text": recognized_text,  # Predicted text
            "label": pred_id,  # Text class
            "attribute": pred_attribute,  # Predicted attribute (if any, or -1)
        }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get submission.")
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the source audio files.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output submission.",
    )
    args = parser.parse_args()
    predictor = Predictor()

    results = []
    for audio_path in os.listdir(args.src):
        result = predictor(os.path.join(args.src, audio_path))
        results.append(result)

    with open(
        os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile)
