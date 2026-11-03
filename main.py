import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from CIRE import CIRE


def main():
    parser = argparse.ArgumentParser(description="Run CIRE on example question/response pairs.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("example.json"))
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=True).to(args.device)
    scorer = CIRE(model, tokenizer, layer=args.layer)
    samples = json.loads(args.data.read_text(encoding="utf-8"))
    for sample in samples:
        prompt = f"Question: {sample['question']}\nAnswer:"
        scores = scorer.score(prompt, " " + sample["response"].strip())
        print(json.dumps({**sample, **scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
