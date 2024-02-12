import pandas as pd
from pathlib import Path


def collector(row):
    total_speakers = row["speaker_id"].count()
    min_sessions = row["session_id"].min()
    max_sessions = row["session_id"].max()
    average_sessions = row["session_id"].mean()
    data = {
        "total_speakers": total_speakers,
        "min_sessions": min_sessions,
        "max_sessions": max_sessions,
        "average_sessions": average_sessions,
    }
    return pd.Series(data)


location = "/home/divr_benchmark/storage/data/torgo/"
files = Path(location).rglob("prompts/*.txt")
all_prompts = []
for file_name in files:
    with open(file_name, "r") as file:
        line = file.readline().strip()
        prompt = line.lower().replace('"', "").replace("'", "")
    [speaker_id, session_id, _] = (
        str(file_name).removeprefix(location).split("/", maxsplit=2)
    )
    all_prompts.append(
        {
            "prompt": prompt,
            "speaker_id": speaker_id,
            "session_id": session_id,
        }
    )

curdir = Path(__file__).parent.resolve()
gdf = (
    pd.DataFrame(all_prompts)
    .groupby(["prompt", "speaker_id"])
    .count()
    .reset_index()
    .groupby("prompt")[["prompt", "speaker_id", "session_id"]]
    .apply(func=collector)
    .reset_index()
    .sort_values(by=["total_speakers", "prompt"], ascending=[False, True])
).to_csv(f"{curdir}/prompt_stats.csv", index=False)
