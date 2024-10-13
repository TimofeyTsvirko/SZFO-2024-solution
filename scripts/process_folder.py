import json
import os
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import psutil
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
@click.option("--to-json", is_flag=True, help="Save results to json.")
def greet(voices_dpath, voice_ext, frame_rate, annotation_dpath, to_json):
    voices_dpath = Path(voices_dpath)

    annotations_df = None
    if annotation_dpath is not None:
        with open(annotation_dpath) as f:
            annotation_data = json.load(f)
        annotations_df = pd.DataFrame(annotation_data)

    model = VoskASR(MODELS_DPATH / "vosk-model-small-ru-0.22", frame_rate=frame_rate)

    recognized_data = []

    # latency and peak RAM usage
    latency_list = []
    process = psutil.Process(os.getpid())
    peak_memory = process.memory_info().rss

    json_data = []

    for voice_dpath in tqdm(list(voices_dpath.glob(f"*.{voice_ext}"))):
        model.read_audio(voice_dpath)

        start_time = time.time()

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

        end_time = time.time()
        latency = end_time - start_time
        latency_list.append(latency)

        peak_memory = max(peak_memory, process.memory_info().rss)

        if to_json:
            json_data.append(model.get_submission_result)

    recognized_df = pd.DataFrame(
        recognized_data,
        columns=["audio_filepath", "recognized_text", "pred_label", "pred_attribute"],
    )

    print("Latency:", np.mean(latency_list))
    print("Peak RAM:", peak_memory / (1024 * 1024), "MB")

    if annotations_df is not None:
        recognized_df = pd.merge(recognized_df, annotations_df, on="audio_filepath")
        recognized_df["WER"] = recognized_df.apply(
            lambda x: MetricsCalculator.wer(
                test=x["text"],
                pred=x["recognized_text"],
            ),
            axis=1,
        )
        print("------------------")
        print("WER:", recognized_df["WER"].mean())
        f1_weighted = MetricsCalculator.f1_w(
            recognized_df["label"],
            recognized_df["pred_label"],
        )
        print("F1-weighted:", f1_weighted)

    output_path = DATA_DPATH / "results"
    output_path.mkdir(parents=True, exist_ok=True)

    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_name = f"voices_dpath_{formatted_datetime}"

    if to_json:
        with open(output_path / f"{output_name}.json", "w") as f:
            json.dump(json_data, f)
    else:
        recognized_df.to_csv(
            output_path / f"{output_name}.csv",
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    greet()
