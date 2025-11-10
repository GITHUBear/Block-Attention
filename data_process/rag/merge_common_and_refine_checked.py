import os
import fire
import json
import random
from dataclasses import dataclass


from typing import Any, Dict, List, Union, Optional

@dataclass
class Args:
    dir: str
    n_samples: int
    refine_ratio: float

def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, "w", encoding="utf-8") as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    
if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    random.seed(42)
    refine_input_file = os.path.join(args.dir, "refine_gened_checked_dataset")
    other_input_file = os.path.join(args.dir, "all_gened_dataset")
    refine = load_jsonline(fp=refine_input_file)
    other = load_jsonline(fp=other_input_file)
    print(f"refine: {len(refine)}")
    print(f"other: {len(other)}")

    refine_sample = int(len(refine) * args.refine_ratio)
    refine = random.sample(refine, k=refine_sample)
    print(f"new refine: {len(refine)}")
    other = random.sample(other, k=args.n_samples-refine_sample)
    print(f"new other: {len(other)}")

    refine.extend(other)
    print(f"new total: {len(refine)}")
    random.shuffle(refine)
    write_jsonline(fp=os.path.join(args.dir, "final_v2_dataset"), obj=refine)
