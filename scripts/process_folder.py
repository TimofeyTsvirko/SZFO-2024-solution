import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from szfo_2024_solution.voice2text import MetricsCalculator, VoskASR

PROJECT_DPATH = Path(__file__).resolve().parents[1]
DATA_DPATH = PROJECT_DPATH / "data"
MODELS_DPATH = PROJECT_DPATH / "models"


@click.command()
@click.option(
    "--voices-dpath",
    prompt="Enter voices directory path",
    help="Voices directory path.",
)
@click.option("--voice-ext", default="wav", help="Voices' files extension (mp3, wav).")
@click.option("--frame-rate", default=16000, help="Files framerate (16k, 48k).")
@click.option("--annotation-dpath", help="Annotation (json) filepath.")
def greet(voices_dpath, voice_ext, frame_rate, annotation_dpath):
    voices_dpath = Path(voices_dpath)

    annotations_df = None
    if annotation_dpath is not None:
        with open(annotation_dpath) as f:
            annotation_data = json.load(f)
        annotations_df = pd.DataFrame(annotation_data)

    model = VoskASR(MODELS_DPATH / "vosk-model-small-ru-0.22", frame_rate=frame_rate)

    recognized_data = []
    for voice_dpath in tqdm(list(voices_dpath.glob(f"*.{voice_ext}"))):
        model.read_audio(voice_dpath)
        model.denoise_audio()
        recognized_text = model.recognize_audio()
        pred_id = model.get_pred_label
        pred_attribute = model.get_pred_attr
        recognized_data.append(
            [
                voice_dpath.name,
                recognized_text,
                pred_id,
                pred_attribute,
            ]
        )
    recognized_df = pd.DataFrame(
        recognized_data,
        columns=["audio_filepath", "recognized_text", "pred_label", "pred_attribute"],
    )

    if annotations_df is not None:
        recognized_df = pd.merge(recognized_df, annotations_df, on="audio_filepath")
        recognized_df["WER"] = recognized_df.apply(
            lambda x: MetricsCalculator.wer(
                test=x["text"],
                pred=x["recognized_text"],
            ),
            axis=1,
        )

    output_path = DATA_DPATH / "results"
    output_path.mkdir(parents=True, exist_ok=True)

    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    recognized_df.to_csv(
        output_path / f"voices_dpath_{formatted_datetime}.csv",
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    greet()
