import json
import fire
import torch
from flask_cors import CORS
from flask import Flask, request

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict, Union

# from transformers.cache_utils import DynamicCache
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM

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

@torch.no_grad()
def simple_generate(
        prompt: str, generation_config: GenerationConfig, model,
        tokenizer: PreTrainedTokenizer
) -> str:
    # past_key_values, input_ids = build_block_past_key_values(
    #     blocks=blocks, instruction=instruction, tokenizer=tokenizer, model=model, emb=emb,
    #     num_local_attention_blocks=num_local_attention_blocks,
    # )
    input_ids = torch.tensor(
        data=[tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.int64,
        device=model.device
    )
    input_length = input_ids.size(-1)

    outputs = model.generate(
        input_ids=input_ids, generation_config=generation_config,
        use_cache=True, eos_token_id=[tokenizer.eos_token_id], tokenizer=tokenizer
    )
    return tokenizer.decode(token_ids=outputs[0][input_length:].tolist())

@app.route('/generate', methods=['POST'])
def _generate():
    form = request.get_json()
    prompt = "".join(form["blocks"])
    if remove_special_token:
        prompt = prompt[len("[Block-Attention]"):]
    generated = simple_generate(
        prompt=prompt,
        generation_config=generation_config,
        model=model,
        tokenizer=tokenizer,
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
        device_map="cuda:1",
        attn_implementation="flash_attention_2"
    )
    model.eval()
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model)

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
