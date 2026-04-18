import os
import pandas as pd
import torch
from datasets import load_from_disk
from jiwer import wer, cer
from tqdm import tqdm
from transformers import pipeline

from scripts.utils import load_yaml, normalize_for_wer, ensure_dir

cfg = load_yaml("configs/data_config.yaml")
model_id = "openai/whisper-medium"

device = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    chunk_length_s=30,
    device=device,
)

test_ds = load_from_disk("data/manifests/hf_splits")["test"]

records = []
for idx, ex in enumerate(tqdm(test_ds, desc="Zero-shot eval")):
    audio = ex[cfg["audio_column"]]
    ref = ex[cfg["text_column"]]

    pred = asr(
        {"array": audio["array"], "sampling_rate": audio["sampling_rate"]},
        generate_kwargs={"language": "english", "task": "transcribe"},
    )["text"]

    records.append({
        "utterance_id": idx,
        "reference": ref,
        "prediction": pred,
        "reference_norm": normalize_for_wer(ref),
        "prediction_norm": normalize_for_wer(pred),
        "duration": ex[cfg["duration_column"]],
    })

baseline_df = pd.DataFrame(records)
baseline_df["wer_raw"] = baseline_df.apply(lambda r: wer(r["reference"], r["prediction"]), axis=1)
baseline_df["cer_raw"] = baseline_df.apply(lambda r: cer(r["reference"], r["prediction"]), axis=1)
baseline_df["wer_norm"] = baseline_df.apply(lambda r: wer(r["reference_norm"], r["prediction_norm"]), axis=1)
baseline_df["cer_norm"] = baseline_df.apply(lambda r: cer(r["reference_norm"], r["prediction_norm"]), axis=1)

ensure_dir("outputs/baseline")
baseline_df.to_csv("outputs/baseline/zero_shot_predictions.csv", index=False)

summary = {
    "mean_wer_raw": float(baseline_df["wer_raw"].mean()),
    "mean_cer_raw": float(baseline_df["cer_raw"].mean()),
    "mean_wer_norm": float(baseline_df["wer_norm"].mean()),
    "mean_cer_norm": float(baseline_df["cer_norm"].mean()),
}
print(summary)