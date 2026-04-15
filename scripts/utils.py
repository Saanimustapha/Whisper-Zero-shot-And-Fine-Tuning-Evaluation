import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from jiwer import Compose, ExpandCommonEnglishContractions, RemoveMultipleSpaces, Strip, ToLowerCase, RemovePunctuation


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


RAW_TO_NORM = Compose([
    ToLowerCase(),
    ExpandCommonEnglishContractions(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def normalize_for_wer(text: str) -> str:
    text = normalize_text(text)
    return RAW_TO_NORM(text)


def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)