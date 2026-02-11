import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()

name = args.model.replace("model_", "").replace(".pkl", "")

# Generate dummy metrics
with open(f"metrics_{name}.json", "w") as f:
    json.dump({"accuracy": random.random(), "loss": random.random()}, f)

# Generate dummy confusion matrix
with open(f"confusion_{name}.csv", "w") as f:
    f.write("actual,predicted\n")
    for _ in range(100):
        f.write(f"{random.randint(0,1)},{random.randint(0,1)}\n")
