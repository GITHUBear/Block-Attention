import os
import json

import fire
import regex
import string
import statistics
import requests

from torch.ao.quantization.fx.utils import all_node_args_except_first
from tqdm import tqdm
from dataclasses import dataclass

from typing import Any, Dict, List, TypedDict

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

API_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]


@dataclass
class EvalArgs:
    input: str


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def best_subspan_em(prediction: str, ground_truths: List[str], question: str) -> float:
    normalized_prediction = normalize_answer(prediction)

    if isinstance(ground_truths[0], List):
        ground_truths = ground_truths[0]
    if isinstance(ground_truths, str):
        normalized_ground_truth = normalize_answer(ground_truths)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
        return 0.0
    else:
        for ground_truth in ground_truths:
            normalized_ground_truth = normalize_answer(ground_truth)
            if normalized_ground_truth.lower() in normalized_prediction.lower():
                return 1.0
        return 0.0

def llm_match(prediction: str, ground_truths: List[str], question: str) -> float:
    prompt = f"你的任务是判断一个回答是否与标准答案等价，等价则输出yes否则输出no即可，不要额外输出。\n问题：{question}\n标准答案：{ground_truths}\n回答：{prediction}"
    messages = [
        {"role": "user", "content": prompt}
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

METRICS = [(best_subspan_em, "best_subspan_em"),(llm_match, "llm_match")]

def get_metrics_for_example(example: SFTDataInstance):
    gold_answers = example["answers"]
    model_answer = example["generated"].split("<|end_of_text|>")[0].split("<|im_end|>")[0].split("<|eot_id|>")[0]
    question = example["question"]

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers, question=question)
    return example_metrics, example


def main(args: EvalArgs):
    if os.path.isfile(args.input):
        all_examples: List[SFTDataInstance] = load_jsonline(fp=args.input)
    else:
        all_examples: List[SFTDataInstance] = []
        for f_name in os.listdir(args.input):
            fp = os.path.join(args.input, f_name)
            all_examples.extend(load_jsonline(fp=fp))

    all_example_metrics = []
    for example in tqdm(all_examples, total=len(all_examples), desc="Eval: "):
        all_example_metrics.append(get_metrics_for_example(example=example))

    print("All Examples: ", len(all_example_metrics))
    for _, metric in METRICS:
        average = statistics.mean(em[metric]  for em, _ in all_example_metrics)
        print(f"{metric}: {average}")



if __name__ == '__main__':
    args: EvalArgs = fire.Fire(component=EvalArgs)
    main(args=args)