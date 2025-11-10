import sys

sys.path.append(".")

import re
import json
import fire
import random

from tqdm import tqdm
from bson.objectid import ObjectId
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Any, Set, Dict, List, Tuple, Union, Optional, Literal, NamedTuple, TypedDict, Callable

from data_process.tulu3.define import SFTInstanceWithChunks, SFTInstanceWithChunksV2, SFTInputs


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


# def replace_prompt_to_tulu_format(prompt: str) -> str:
#     prompt = prompt.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<|user|>\n")
#     prompt = prompt.replace("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "\n\n")
#     prompt = prompt.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "\n<|assistant|>\n")
#     return prompt


def make_blocks(prompt: str) -> Tuple[List[str], str]:
    blocks: List[str] = [
        "<|im_start|>system\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    ]
    assert prompt.startswith(blocks[0]), json.dumps({
        "prompt": prompt, "blocks": blocks[0]}, ensure_ascii=False, indent=4
    )
    content = prompt[len(blocks[0]):]

    pos = content.find("<|im_end|>") + len("<|im_end|>")
    documents = content[:pos]
    instruction_ans_response = content[pos:]

    pos = documents.find("\n- Title:")
    while pos != -1:
        doc = documents[:pos + 1]
        blocks.append(doc)
        documents = documents[pos + 1:]
        pos = documents.find("\n- Title:")
    assert documents.startswith("- Title:") and documents.endswith("<|im_end|>"), documents
    blocks.append(documents[:-len("<|im_end|>")])
    instruction_ans_response = "<|im_end|>" + instruction_ans_response

    assert instruction_ans_response.startswith(
        "<|im_end|>\n<|im_start|>user\n"
        "Please write a high-quality answer for the given question using only the provided search documents"
    )
    blocks = [b for b in blocks if b != ""]

    # 为了和 Tulu 的 chat_template 对齐，进行如下的处理：
    assert "<|im_start|>system\n" in blocks[0], blocks[0]
    # blocks[0] = blocks[0].replace(
    #     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    #     "<|user|>\n"
    # )
    assert "<|im_end|>\n<|im_start|>user\n" in instruction_ans_response, instruction_ans_response
    # instruction_ans_response = instruction_ans_response.replace(
    #     "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "\n\n"
    # )
    assert "<|im_end|>\n<|im_start|>assistant\n" in instruction_ans_response, instruction_ans_response
    # instruction_ans_response = instruction_ans_response.replace(
    #     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "\n<|assistant|>\n"
    # )
    return blocks, instruction_ans_response


def to_train_data(ins: Dict[str, Any]) -> List[SFTInstanceWithChunks]:
    blocks, instruct_and_response = make_blocks(prompt=ins["prompt"])
    blocks.append(instruct_and_response)

    # tulu_prompt = replace_prompt_to_tulu_format(prompt=ins["prompt"])
    assert ins["prompt"] == "".join(blocks)
    if not blocks[-2].endswith("\n"):
        blocks[-2] += "\n"
    new_prompt = "".join(blocks)

    response: str = ins["generated"]
    # if response.endswith("<|eot_id|"):
    #     response = response[:-len("<|eot_id|")] + "<|end_of_text|>"
    # else:
    response = response + tokenizer.eos_token

    prompt_input_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    input_ids = prompt_input_ids + response_ids
    labels = [-100] * len(prompt_input_ids) + response_ids
    sft_inputs = SFTInputs(input_ids=input_ids, labels=labels)

    block_ids, block_tokens = [], []
    for b in blocks:
        _ids = tokenizer.encode(b, add_special_tokens=False)
        block_ids.extend(_ids)
        block_tokens.append(len(_ids))
    block_input_ids = block_ids + response_ids
    block_labels = [-100] * len(block_ids) + response_ids
    block_inputs = SFTInputs(input_ids=block_input_ids, labels=block_labels)

    return [
        SFTInstanceWithChunks(
            uuid=ObjectId(),
            tulu_uuid=ObjectId(),
            prompt=new_prompt,
            response=response,
            inputs=sft_inputs,
            chunks=blocks,
            block_tokens=block_tokens,
            response_tokens=len(response_ids),
            block_inputs=block_inputs,
            train_block=True
        )
    ]

def to_train_data_v2(ins: Dict[str, Any]) -> List[SFTInstanceWithChunksV2]:
    simple_user_prompt = f"<|im_end|>\n<|im_start|>user\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\nQuestion: {ins['question']}<|im_end|>\n<|im_start|>assistant\n"
    blocks, instruct_and_response = make_blocks(prompt=ins["prompt"])
    if not blocks[-1].endswith("\n"):
        blocks[-1] += "\n"
    blocks.append(instruct_and_response)
    blocks2 = blocks[:-1] + [simple_user_prompt]
    
    response: str = ins["generated"]
    response = response + tokenizer.eos_token

    answer = ins["answers"][0]
    answer = answer + tokenizer.eos_token

    prompt_input_ids = tokenizer.encode(ins["prompt"], add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    input_ids = prompt_input_ids + response_ids
    labels = [-100] * len(prompt_input_ids) + response_ids
    sft_inputs = SFTInputs(input_ids=input_ids, labels=labels)
    
    simple_prompt = "".join(blocks2)
    simple_prompt_input_ids = tokenizer.encode(simple_prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    simple_input_ids = simple_prompt_input_ids + answer_ids
    simple_labels = [-100] * len(simple_prompt_input_ids) + answer_ids
    simple_sft_inputs = SFTInputs(input_ids=simple_input_ids, labels=simple_labels)

    block_ids, block_tokens = [], []
    for b in blocks:
        _ids = tokenizer.encode(b, add_special_tokens=False)
        block_ids.extend(_ids)
        block_tokens.append(len(_ids))
    block_input_ids = block_ids + response_ids
    block_labels = [-100] * len(block_ids) + response_ids
    block_inputs = SFTInputs(input_ids=block_input_ids, labels=block_labels)

    simple_block_ids, simple_block_tokens = [], []
    for b in blocks2:
        _simple_ids = tokenizer.encode(b, add_special_tokens=False)
        simple_block_ids.extend(_simple_ids)
        simple_block_tokens.append(len(_simple_ids))
    simple_block_input_ids = simple_block_ids + answer_ids
    simple_block_labels = [-100] * len(simple_block_ids) + answer_ids
    simple_block_inputs = SFTInputs(input_ids=simple_block_input_ids, labels=simple_block_labels)

    return [
        SFTInstanceWithChunksV2(
            uuid=ObjectId(),
            tulu_uuid=ObjectId(),
            prompt=ins["prompt"],
            response=response,
            inputs=sft_inputs,
            chunks=blocks,
            block_tokens=block_tokens,
            response_tokens=len(response_ids),
            block_inputs=block_inputs,

            last_chunk=simple_user_prompt,
            inputs2=simple_sft_inputs,
            answer=answer,
            block_inputs2=simple_block_inputs,
            block_tokens2=simple_block_tokens,
            response_tokens2=len(answer_ids),

            train_block=True
        )
    ]

@dataclass
class Args:
    input: str = "datahub/rag/tqa_2wiki_p20k"
    output: str = "datahub/rag/block.train"
    model_name: str = "meta-llama/LLama-3.1-8B"
    mode: int = 1


def process(args: Args):
    dataset = load_jsonline(fp=args.input)
    # dataset = dataset[:1]
    if args.mode == 1:
        dataset = [to_train_data(ins=i)[0].to_dict() for i in tqdm(dataset, total=len(dataset), desc='Convert: ')]
    else:
        print("use v2")
        dataset = [to_train_data_v2(ins=i)[0].to_dict() for i in tqdm(dataset, total=len(dataset), desc='Convert: ')]
    write_jsonline(obj=dataset, fp=args.output)


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        use_fast=False
    )
    process(args=args)
