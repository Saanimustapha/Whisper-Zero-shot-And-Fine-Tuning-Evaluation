import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

model_id = "openai/whisper-medium"

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

print("Loaded model:", model_id)
print("Parameter dtype:", next(model.parameters()).dtype)