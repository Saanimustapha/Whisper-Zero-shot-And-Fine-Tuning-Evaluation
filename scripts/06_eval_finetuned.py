import pandas as pd
import torch
from datasets import load_from_disk
from jiwer import wer, cer
from tqdm import tqdm
from transformers import pipeline

from scripts.utils import load_yaml, normalize_for_wer, ensure_dir

cfg = load_yaml("configs/data_config.yaml")
train_cfg = load_yaml("configs/train_config.yaml")

model_dir = train_cfg["output_dir"]
device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model=model_dir,
    tokenizer=model_dir,
    feature_extractor=model_dir,
    chunk_length_s=30,
    device=device,
)

test_ds = load_from_disk("data/manifests/hf_splits")["test"]

records = []
for idx, ex in enumerate(tqdm(test_ds, desc="Fine-tuned eval")):
    audio = ex[cfg["audio_column"]]
    ref = ex[cfg["text_column"]]

    pred = asr(
        {"array": audio["array"], "sampling_rate": audio["sampling_rate"]},
        generate_kwargs={"language": "english", "task": "transcribe"},
    )["text"]

    records.append({
        "utterance_id": idx,
        "reference": ref,
        "prediction_ft": pred,
        "reference_norm": normalize_for_wer(ref),
        "prediction_ft_norm": normalize_for_wer(pred),
        "duration": ex[cfg["duration_column"]],
    })

ft_df = pd.DataFrame(records)
ft_df["wer_ft_raw"] = ft_df.apply(lambda r: wer(r["reference"], r["prediction_ft"]), axis=1)
ft_df["cer_ft_raw"] = ft_df.apply(lambda r: cer(r["reference"], r["prediction_ft"]), axis=1)
ft_df["wer_ft_norm"] = ft_df.apply(lambda r: wer(r["reference_norm"], r["prediction_ft_norm"]), axis=1)
ft_df["cer_ft_norm"] = ft_df.apply(lambda r: cer(r["reference_norm"], r["prediction_ft_norm"]), axis=1)

ensure_dir("outputs/finetuned")
ft_df.to_csv("outputs/finetuned/finetuned_predictions.csv", index=False)

summary = {
    "mean_wer_ft_raw": float(ft_df["wer_ft_raw"].mean()),
    "mean_cer_ft_raw": float(ft_df["cer_ft_raw"].mean()),
    "mean_wer_ft_norm": float(ft_df["wer_ft_norm"].mean()),
    "mean_cer_ft_norm": float(ft_df["cer_ft_norm"].mean()),
    "n_test": int(len(ft_df)),
}
print(summary)
