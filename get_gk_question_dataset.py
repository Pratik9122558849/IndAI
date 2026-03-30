from datasets import load_dataset
import json

dataset = load_dataset("squad", split="train")

with open("gk_dataset.jsonl", "w") as f:
    for item in dataset:
        q = item["question"]
        a = item["answers"]["text"][0]

        data = {
            "input": q,
            "output": a
        }

        f.write(json.dumps(data) + "\n")