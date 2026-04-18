import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from scripts.utils import load_yaml, normalize_for_wer, set_seed


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str_norm = [normalize_for_wer(x) for x in pred_str]
    label_str_norm = [normalize_for_wer(x) for x in label_str]

    wer_score = 100 * wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
    return {"wer": wer_score}


train_cfg = load_yaml("configs/train_config.yaml")
set_seed(train_cfg["seed"])

processor = WhisperProcessor.from_pretrained(
    train_cfg["model_id"],
    language=train_cfg["language"],
    task=train_cfg["task"],
)

model = WhisperForConditionalGeneration.from_pretrained(train_cfg["model_id"])
model.generation_config.language = train_cfg["language"]
model.generation_config.task = train_cfg["task"]
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

lora_config = LoraConfig(
    r=train_cfg["lora_r"],
    lora_alpha=train_cfg["lora_alpha"],
    lora_dropout=train_cfg["lora_dropout"],
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_ds = load_from_disk("data/manifests/prepared_train")
validation_ds = load_from_disk("data/manifests/prepared_validation")

wer_metric = evaluate.load("wer")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir=train_cfg["output_dir"],
    per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
    gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    learning_rate=train_cfg["learning_rate"],
    warmup_steps=train_cfg["warmup_steps"],
    max_steps=train_cfg["max_steps"],
    gradient_checkpointing=train_cfg["gradient_checkpointing"],
    fp16=train_cfg["fp16"],
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=train_cfg["generation_max_length"],
    save_steps=train_cfg["save_steps"],
    eval_steps=train_cfg["eval_steps"],
    logging_steps=train_cfg["logging_steps"],
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=validation_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model(train_cfg["output_dir"])
processor.save_pretrained(train_cfg["output_dir"])
