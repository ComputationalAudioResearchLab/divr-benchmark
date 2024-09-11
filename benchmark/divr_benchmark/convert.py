import yaml
import json

fp = "/home/workspace/benchmark/divr_benchmark/tasks/v1/streams/0/train.yml"

with open(fp, "r") as f:
    data = yaml.full_load(f)
    new_data = []
    for key, value in data.items():
        age = value["age"]
        audio_keys = value["audio_keys"]
        audio_key: str = audio_keys[0]
        gender = value["gender"]
        label = value["label"]
        if audio_key.endswith("a_n.wav"):
            new_data += [(age, gender, label, audio_key)]
    print(json.dumps(new_data).replace("[", "(").replace("]", ")"))