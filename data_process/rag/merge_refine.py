import re
import os
import json
import argparse

import random
import requests

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, TypedDict, Optional

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

@dataclass
class BuildArgs:
    dir: str

API_ENDPOINT2 = "http://127.0.0.1:8001/v1/chat/completions"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def llm_match(prediction: str, ground_truths: List[str], question: str) -> float:
    prompt = f"你的任务是判断回答是否与标准答案列表中的某项等价，等价则输出yes否则输出no即可，不要额外输出。\n问题：{question}\n标准答案：{ground_truths}\n回答：{prediction}"
    messages = [
        {"role": "user", "content": prompt}
    ]
    request_data = {
        "model": "/data/shanhaikang.shk/model/modelscope/models/Qwen/Qwen2.5-32B-Instruct",
        "messages": messages,
        "temperature": 0.7,
    }
    try:
        response = requests.post(API_ENDPOINT2, headers=headers, json=request_data)
        
        if response.status_code != 200:
            print(f"请求失败，状态码：{response.status_code}")
            print(response.text)
            exit(1)

        result = response.json()
        content:str = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        content = content.lower()
        # print(f"llm judge: {content}")
        
        if "yes" in content:
            return 1.0
        return 0.0

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        exit(1)

def process_instance(ins: SFTDataInstance) -> Optional[SFTDataInstance]:
    if llm_match(prediction=ins["generated"], ground_truths=ins["answers"], question=ins["question"]) == 1.0:
        return ins
    return None

def process_file(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        instances = [json.loads(i) for i in f]

    print(f"origin len: {len(instances)}")
    dataset: List[SFTDataInstance] = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_idx = {
            executor.submit(process_instance, instances[i]): i
            for i in range(len(instances))
        }

        for future in tqdm(
            as_completed(future_to_idx),
            desc="Process: ",
            total=len(instances),
            unit="instance"
        ):
            try:
                result = future.result()
                if result:
                    dataset.append(result)
            except Exception as e:
                idx = future_to_idx[future]
                print(f"处理第 {idx} 个实例时出错: {e}")
                exit(1)
    
    print(f"total: {len(dataset)}")
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
        input_file=os.path.join(args.dir, "refine_gened_dataset"),
        output_file=os.path.join(args.dir, "refine_gened_checked_dataset")
    )
