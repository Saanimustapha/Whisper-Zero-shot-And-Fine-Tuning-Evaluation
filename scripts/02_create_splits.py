from datasets import load_dataset, DatasetDict
from scripts.utils import load_yaml, set_seed

cfg = load_yaml("configs/data_config.yaml")
set_seed(cfg["seed"])

raw = load_dataset(cfg["dataset_name"])["train"]

# Basic duration filtering for tractable training and cleaner evaluation
raw = raw.filter(
    lambda x: cfg["min_duration"] <= x[cfg["duration_column"]] <= cfg["max_duration_eval"],
    num_proc=cfg["num_proc"],
)

# First carve out test
split1 = raw.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])
train_val = split1["train"]
test = split1["test"]

# Then carve out validation from remaining train
val_ratio_adjusted = cfg["val_size"] / (1.0 - cfg["test_size"])
split2 = train_val.train_test_split(test_size=val_ratio_adjusted, seed=cfg["seed"])
train = split2["train"]
validation = split2["test"]

splits = DatasetDict({
    "train": train,
    "validation": validation,
    "test": test,
})

splits.save_to_disk("data/manifests/hf_splits")
print(splits)
for name, ds in splits.items():
    print(name, len(ds))