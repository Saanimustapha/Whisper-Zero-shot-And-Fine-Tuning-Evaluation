import pandas as pd
from scripts.utils import ensure_dir

baseline = pd.read_csv("outputs/baseline/zero_shot_predictions.csv")
ft = pd.read_csv("outputs/finetuned/finetuned_predictions.csv")

merged = baseline.merge(
    ft[["utterance_id", "prediction_ft", "prediction_ft_norm", "wer_ft_raw", "wer_ft_norm"]],
    on="utterance_id",
    how="inner",
)

merged["delta_wer_raw"] = merged["wer_raw"] - merged["wer_ft_raw"]
merged["delta_wer_norm"] = merged["wer_norm"] - merged["wer_ft_norm"]

# Sample a manageable human-rating subset
rating_pack = merged.sample(n=min(300, len(merged)), random_state=42).copy()
rating_pack["human_rating"] = ""
rating_pack["notes"] = ""

ensure_dir("data/human_ratings")
rating_pack.to_csv("data/human_ratings/rating_pack.csv", index=False)
print("Saved data/human_ratings/rating_pack.csv")