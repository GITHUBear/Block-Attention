import re
import os
import json
import argparse

import random
import requests

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, TypedDict

from concurrent.futures import ThreadPoolExecutor, as_completed

Document = TypedDict("Document", {"title": str, "text": str, "score": float})

SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": str,
    "question": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs,
    "documents": List[Document]
})

SFTDataInstanceWithConv = TypedDict("SFTDataInstanceWithConv", {
    "system": str,
    "user": str,
    "ins": SFTDataInstance
})

@dataclass
class BuildArgs:
    dir: str

API_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def process_instance(ins: SFTDataInstanceWithConv) -> SFTDataInstance:
    messages = [
        {"role": "system", "content": ins["system"]},
        {"role": "user", "content": ins["user"]}
    ]
    request_data = {
        "model": "/data/shanhaikang.shk/model/modelscope/models/Qwen/Qwen2.5-32B-Instruct",
        "messages": messages,
        "temperature": 0.7,
    }
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=request_data)
        
        if response.status_code != 200:
            print(f"请求失败，状态码：{response.status_code}")
            print(response.text)
            exit(1)

        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        assert content is not None
        ins["ins"]["generated"] = content

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        exit(1)
    return ins["ins"]

def process_file(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        tqa_instances = [json.loads(i) for i in f]
    # tqa_instances = tqa_instances[:1]

    dataset: List[SFTDataInstance] = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_idx = {
            executor.submit(process_instance, tqa_instances[i]): i
            for i in range(len(tqa_instances))
        }

        for future in tqdm(
            as_completed(future_to_idx),
            desc="Process TQA: ",
            total=len(tqa_instances),
            unit="instance"
        ):
            try:
                result = future.result()
                dataset.append(result)
            except Exception as e:
                idx = future_to_idx[future]
                print(f"处理第 {idx} 个实例时出错: {e}")
                exit(1)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i in dataset:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    return BuildArgs(dir=args.dir)


if __name__ == '__main__':
    args = parse_args()
    random.seed(42)
    process_file(
        input_file=os.path.join(args.dir, "tqa_train", "dataset"), output_file=os.path.join(args.dir, "tqa_train", "gened_dataset")
    )
