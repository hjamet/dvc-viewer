import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--name", type=str)
args = parser.parse_args()

print(f"Training {args.name} for {args.epochs} epochs")
with open(f"model_{args.name}.pkl", "w") as f:
    f.write(f"Model {args.name}")
