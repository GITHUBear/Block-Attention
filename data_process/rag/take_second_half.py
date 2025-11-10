import os
import fire
import json
import random
from dataclasses import dataclass


from typing import Any, Dict, List, Union, Optional

@dataclass
class Args:
    dir: str

def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, "w", encoding="utf-8") as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    input_file = os.path.join(args.dir, "final_v2_rag.train")
    output_file = os.path.join(args.dir, "final_v3_rag.train")

    ds = load_jsonline(fp=input_file)
    write_jsonline(fp=output_file, obj=ds[10000:])