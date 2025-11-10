import fire
import os
from transformers import AutoTokenizer
from dataclasses import dataclass

@dataclass
class Args:
    model_name: str
    model_path: str

if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name)
    tokenizer.save_pretrained(args.model_path)
