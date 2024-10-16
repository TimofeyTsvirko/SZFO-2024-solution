import json
import os
import random
import sys
from pathlib import Path

import nltk
import pymorphy2
from rnnoise_wrapper import RNNoise
from sklearn.metrics import f1_score
from vosk import KaldiRecognizer, Model

from .label2id import Numbers, _id2label, _label2id


class VoskASR:
    def __init__(self, model_path: Path, chunk_size=4000, frame_rate=16000):
        if not os.path.exists(model_path):
            print("Model not found at:", model_path)
            sys.exit(1)
        model = Model(str(model_path))
        self.recognizer = KaldiRecognizer(model, frame_rate)
        self.denoiser = RNNoise()
        self.chunk_size = chunk_size
        self.morph = pymorphy2.MorphAnalyzer()
        self.labels_with_attr = [4, 10]

    def read_audio(self, audio_path: Path) -> None:
        self.audio = self.denoiser.read_wav(audio_path)
        self.audio_name = audio_path.name

    def denoise_audio(self) -> None:
        self.audio = self.denoiser.filter(self.audio)

    def recognize_audio(self) -> str:
        transcription = ""

        for start in range(0, len(self.audio), self.chunk_size):
            end = min(start + self.chunk_size, len(self.audio))
            chunk = self.audio[start:end]

            raw_data = chunk.raw_data

            if self.recognizer.AcceptWaveform(raw_data):
                result = self.recognizer.Result()
                transcription += json.loads(result)["text"] + " "

        transcription += json.loads(self.recognizer.FinalResult())["text"]
        self.transcription = transcription
        self.attribute = self.text_to_number(transcription)
        self.label = self.predict_label_from_text(self.transcription)
        return transcription

    @property
    def get_transcription(self):
        return self.transcription

    @property
    def get_pred_label(self):
        if self.label not in self.labels_with_attr and self.attribute != -1:
            self.label = random.choice(self.labels_with_attr)
        return self.label

    @property
    def get_pred_attr(self):
        if self.label in self.labels_with_attr:
            return self.attribute
        else:
            return -1

    def predict_label_from_text(self, text):
        return self.find_best_match(text)

    @staticmethod
    def label2id(label: str) -> int:
        return _label2id[label]

    @staticmethod
    def id2label(id: int) -> str:
        return _id2label[id]

    def text_to_number(self, text):
        words = text.lower().split()
        total = 0
        current = 0

        for word in words:
            parsed = self.morph.parse(word)[0]
            lemma = parsed.normal_form

            if lemma in Numbers.units:
                current += Numbers.units[lemma]
            elif lemma in Numbers.tens:
                current += Numbers.tens[lemma]
            elif lemma in Numbers.hundreds:
                current += Numbers.hundreds[lemma]
                current = 0
            else:
                pass
        total += current
        if total == 0:
            return -1
        return total

    @property
    def get_submission_result(self):
        return {
            "audio": self.audio_name,
            "text": self.transcription,
            "label": self.label,
            "attribute": self.attribute,
        }

    @staticmethod
    def find_best_match(text):
        text = text.lower()

        scores = {}
        for phrase, id in _label2id.items():
            phrase_words = phrase.split()
            presence = [1 if word in text else 0 for word in phrase_words]

            average_score = sum(presence) / len(phrase_words)

            if average_score not in scores:
                scores[average_score] = []
            scores[average_score].append((phrase, id, len(phrase_words)))

        # Сортировка по убыванию среднего значения, затем по количеству слов
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: (
                -x[0],
                -max(len(item[0]) for item in x[1]),
            ),
        )

        # Выбор случайного варианта при дубликатах
        best_matches = []
        for avg_score, phrases in sorted_scores:
            if not best_matches or best_matches[0][0] == avg_score:
                best_matches.extend(phrases)
            else:
                break

        if best_matches:
            max_length = max(len(item[0].split()) for item in best_matches)
            best_matches = [
                item for item in best_matches if len(item[0].split()) == max_length
            ]  # noqa: E501

        # Возвращаем случайный вариант из лучших совпадений
        if best_matches:
            return random.choice(best_matches)[1]
        else:
            return random.choice(range(len(_label2id.items())))


class MetricsCalculator:
    def __init__(self):
        pass

    def wer(test, pred):
        test_words = test.split()
        pred_words = pred.split()
        matcher = nltk.edit_distance(test_words, pred_words)
        return matcher / len(test_words)

    def f1_w(test, pred):
        return f1_score(test, pred, average="weighted")
