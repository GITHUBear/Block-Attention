import json
import fire
import torch
from flask_cors import CORS
from flask import Flask, request

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict, Union

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, Qwen2ForCausalLM, Qwen2Config

from transformers import (
    AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
)

SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs
})

app = Flask(__name__)
CORS(app, supports_credentials=True)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

def apply_pkv_rotary_position_embeddings(pkv: DynamicCache, emb: Qwen2RotaryEmbedding) -> DynamicCache:
    device = pkv.layers[0].keys.device
    emb.to(device=device)
    position_ids = torch.arange(start=0, end=pkv.layers[0].keys.size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.layers[0].keys.size(0), 1])
    cos, sin = emb(x=pkv.layers[0].keys, position_ids=position_ids)
    for i in range(0, len(pkv.layers)):
        pkv.layers[i].keys = apply_rotary_pos_emb(
            k=pkv.layers[i].keys, cos=cos, sin=sin, position_ids=position_ids
        )
    return pkv

def apply_pkv_rerotary_position_embeddings(pkv: DynamicCache, emb: Qwen2RotaryEmbedding) -> DynamicCache:
    device = pkv.layers[0].keys.device
    emb.to(device=device)
    position_ids = torch.arange(start=0, end=pkv.layers[0].keys.size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.layers[0].keys.size(0), 1])
    cos, sin = emb(x=pkv.layers[0].keys, position_ids=position_ids)
    for i in range(0, len(pkv.layers)):
        pkv.layers[i].keys = apply_rotary_pos_emb(
            k=pkv.layers[i].keys, cos=cos, sin=-sin, position_ids=position_ids
        )
    return pkv

@torch.no_grad()
def build_block_past_key_values(
        blocks: List[str], instruction: str, tokenizer: PreTrainedTokenizer, model: Qwen2ForCausalLM,
        emb: Qwen2RotaryEmbedding, num_local_attention_blocks: int
) -> Tuple[Optional[List[DynamicCache]], torch.Tensor]:
    if len(blocks) > num_local_attention_blocks:
        instruction = "".join(blocks[num_local_attention_blocks:]) + instruction
        blocks = blocks[:num_local_attention_blocks]

    if num_local_attention_blocks == 0:
        instruction = "".join(blocks) + instruction
        blocks = []

    # print(f"Prompt | num local attention blocks: {num_local_attention_blocks}\n")
    # print(json.dumps({
    #     "blocks": blocks,
    #     "instruction_ans_response": instruction,
    # }, ensure_ascii=False, indent=4))

    caches: List[DynamicCache] = []
    input_ids = None
    for b_idx, block in enumerate(blocks):
        block_input_ids = torch.tensor(
            data=[tokenizer.encode(block, add_special_tokens=False)],
            dtype=torch.int64,
            device=model.device
        )
        if b_idx == 0:
            input_ids = block_input_ids
        else:
            input_ids = torch.cat(tensors=[input_ids, block_input_ids], dim=-1)

        output: CausalLMOutputWithPast = model(
            input_ids=block_input_ids, use_cache=True, past_key_values=DynamicCache(), return_dict=True
        )
        # output.past_key_values.layers[0].keys
        pkv = apply_pkv_rerotary_position_embeddings(pkv=output.past_key_values, emb=emb)
        caches.append(pkv)

    response_input_ids = torch.tensor(
        data=[tokenizer.encode(instruction, add_special_tokens=False)],
        dtype=torch.int64,
        device=model.device
    )
    if input_ids is None:
        return None, response_input_ids
    input_ids = torch.cat(tensors=[input_ids, response_input_ids], dim=-1)
    return caches, input_ids

def merge_and_rotary_past_key_values(pkvs: List[DynamicCache], emb: Qwen2RotaryEmbedding) -> DynamicCache:
    cache = pkvs[0]
    for l_idx in range(0, len(cache.layers)):
        cache.layers[l_idx].keys = torch.cat(
            tensors=[cache.layers[l_idx].keys] + [pkvs[b_idx].layers[l_idx].keys for b_idx in range(1, len(pkvs))],
            dim=-2
        )
        cache.layers[l_idx].values = torch.cat(
            tensors=[cache.layers[l_idx].values] + [pkvs[b_idx].layers[l_idx].values for b_idx in range(1, len(pkvs))],
            dim=-2
        )
    cache = apply_pkv_rotary_position_embeddings(pkv=cache, emb=emb)
    return cache

@torch.no_grad()
def block_generate(
        blocks: List[str], instruction: str, generation_config: GenerationConfig, model: Qwen2ForCausalLM,
        emb: Qwen2RotaryEmbedding, tokenizer: PreTrainedTokenizer, num_local_attention_blocks: int
) -> str:
    past_key_values, input_ids = build_block_past_key_values(
        blocks=blocks, instruction=instruction, tokenizer=tokenizer, model=model, emb=emb,
        num_local_attention_blocks=num_local_attention_blocks,
    )
    if past_key_values is not None:
        past_key_values = merge_and_rotary_past_key_values(pkvs=past_key_values, emb=emb)
    input_length = input_ids.size(-1)

    outputs = model.generate(
        input_ids=input_ids, generation_config=generation_config, past_key_values=past_key_values,
        use_cache=True, eos_token_id=[tokenizer.eos_token_id], tokenizer=tokenizer
    )
    return tokenizer.decode(token_ids=outputs[0][input_length:].tolist())

@app.route('/generate', methods=['POST'])
def _block_generate():
    form = request.get_json()
    if remove_special_token:
        form["blocks"][0] = form["blocks"][0][len("[Block-Attention]"):]
    # form["blocks"][-1] = form["blocks"][-1].replace(
    #     "Answer the question based on the given passages. Only give me the answer and do not output any other words.", 
    #     "Please write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant)."
    # )
    # print(form["blocks"])
    generated = block_generate(
        blocks=form["blocks"][:-1],
        instruction=form["blocks"][-1],
        generation_config=generation_config,
        model=model,
        emb=emb,
        tokenizer=tokenizer,
        num_local_attention_blocks=form.get("num_local_attention_blocks", 10000),
    )
    print("generated: ", generated)
    return {"ret": 0, "generated": generated, "message": ""}

@dataclass
class Args:
    model: str
    port: int
    dtype: str
    remove_special_token: bool = False

if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    remove_special_token = args.remove_special_token
    print(f"remove_special_token {remove_special_token}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )
    model.eval()
    config: Qwen2Config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model)
    emb:Qwen2RotaryEmbedding = Qwen2RotaryEmbedding(config=config).to(device=model.device)
    emb.eval()

    generation_config = GenerationConfig(
        do_sample=False,
        temperature=0.7,
        repetition_penalty=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048,
        stop_strings=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>", "Question:"]
    )
    app.run(host="0.0.0.0", port=args.port)


