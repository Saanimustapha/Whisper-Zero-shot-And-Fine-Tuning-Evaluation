from datasets import load_from_disk, Audio
from transformers import WhisperProcessor
from scripts.utils import load_yaml

cfg = load_yaml("configs/data_config.yaml")
train_cfg = load_yaml("configs/train_config.yaml")

processor = WhisperProcessor.from_pretrained(
    train_cfg["model_id"],
    language=train_cfg["language"],
    task=train_cfg["task"],
)

splits = load_from_disk("data/manifests/hf_splits")
splits = splits.cast_column(cfg["audio_column"], Audio(sampling_rate=cfg["sample_rate"]))

# Tighten training duration for efficiency
train_ds = splits["train"].filter(
    lambda x: x[cfg["duration_column"]] <= cfg["max_duration_train"]
)
validation_ds = splits["validation"]
test_ds = splits["test"]


def prepare_batch(batch):
    audio = batch[cfg["audio_column"]]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch[cfg["text_column"]]).input_ids
    return batch

train_ds = train_ds.map(
    prepare_batch,
    remove_columns=train_ds.column_names,
    num_proc=1,
)
validation_ds = validation_ds.map(
    prepare_batch,
    remove_columns=validation_ds.column_names,
    num_proc=1,
)
test_ds = test_ds.map(
    prepare_batch,
    remove_columns=test_ds.column_names,
    num_proc=1,
)

prepared = {
    "train": train_ds,
    "validation": validation_ds,
    "test": test_ds,
}

train_ds.save_to_disk("data/manifests/prepared_train")
validation_ds.save_to_disk("data/manifests/prepared_validation")
test_ds.save_to_disk("data/manifests/prepared_test")
print("Prepared splits saved.")