from datasets import load_dataset
import pandas as pd
from scripts.utils import load_yaml, ensure_dir

cfg = load_yaml("configs/data_config.yaml")

dataset = load_dataset(cfg["dataset_name"])
train = dataset["train"]

print(train)
print("Columns:", train.column_names)
print("First example text:", train[0][cfg["text_column"]])
print("First example duration:", train[0][cfg["duration_column"]])
print("First example sampling rate:", train[0][cfg["audio_column"]]["sampling_rate"])

sample_n = 1000
rows = []
for i in range(min(sample_n, len(train))):
    ex = train[i]
    rows.append({
        "idx": i,
        "duration": ex[cfg["duration_column"]],
        "text_len": len(ex[cfg["text_column"]].split()),
    })

df = pd.DataFrame(rows)
ensure_dir("outputs/analysis")
df.describe().to_csv("outputs/analysis/inspection_summary.csv")
print(df.describe())