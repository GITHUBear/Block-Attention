import os
import fire
import json
import requests
import random
from tqdm import tqdm
from dataclasses import dataclass


from typing import Any, Dict, List, Union, Optional, TypedDict

Document = TypedDict("Document", {"title": str, "text": str, "score": float})

SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": List[str],
    "question": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs,
    "documents": List[Document]
})

@dataclass
class Args:
    input_file: str
    output_dir: str
    run_name: str
    num_samples: int
    ignore_special_token: bool = False
    without_context: bool = False

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
    ds: List[SFTDataInstance] = load_jsonline(fp=args.input_file)
    ds = random.sample(population=ds, k=args.num_samples)
    print(f"without context: {args.without_context}")
    
    for data in tqdm(ds):
        if args.without_context:
            prompt = f"<|im_start|>system\nYou are an intelligent AI assistant.<|im_end|>\n<|im_start|>user\nAnswer the question based on your knowledge. Only give me the answer and do not output any other words.\nQuestion: {data['question']}\n<|im_end|>\n<|im_start|>assistant\n"
            r = requests.post(
                url="http://127.0.0.1:12345/generate",
                data=json.dumps({"blocks": [prompt]}),
                headers={"Content-Type": "application/json"}
            )
        else:
            if data["prompt"][0].startswith("[Block-Attention]") and args.ignore_special_token:
                data["prompt"][0] = data["prompt"][0][len("[Block-Attention]"):]
            r = requests.post(
                url="http://127.0.0.1:12345/generate",
                data=json.dumps({"blocks": data["prompt"]}),
                headers={"Content-Type": "application/json"}
            )
        data["generated"] = r.json()["generated"]
    
    os.system(f"mkdir -p {args.output_dir}")
    write_jsonline(fp=os.path.join(args.output_dir, args.run_name), obj=ds)
    print("FINISH")
