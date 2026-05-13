import argparse
import json
import time
from pathlib import Path

from common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_PATH,
    OUTPUT_DIR,
    encode,
    extract_json,
    generate,
    get_prompts,
    load_dataset,
    prepare_messages,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot NER inference")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--data", default="devel")
    parser.add_argument("--output", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    return parser.parse_args()


def resolved_generation_config(args):
    temperature = 0.0 if args.temperature is None else args.temperature
    top_p = 1.0 if args.top_p is None else args.top_p
    top_k = 50 if args.top_k is None else args.top_k

    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": args.max_new_tokens,
    }


def load_model(model_path, load_in_4bit=False):
    model_path = Path(model_path)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Model not found in {model_path}.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model_kwargs = {"device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def default_output_name(data_path, prompt_name, generation_config):
    temp = generation_config["temperature"]
    top_p = generation_config["top_p"]
    top_k = generation_config["top_k"]
    max_new_tokens = generation_config["max_new_tokens"]
    name = (
        f"ZS.{data_path.stem}.{prompt_name}.t{temp}.p{top_p}.k{top_k}.m{max_new_tokens}.json"
    )
    return OUTPUT_DIR / name


def main():
    args = parse_args()
    generation_config = resolved_generation_config(args)
    prompts = get_prompts(args.prompt_file)
    examples, data_path = load_dataset(args.data)

    if args.limit is not None:
        selected = examples[args.start : args.start + args.limit]
    else:
        selected = examples[args.start :]

    output_path = (
        Path(args.output)
        if args.output
        else default_output_name(data_path, prompts["name"], generation_config)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("========= NER ZERO SHOT =========")
    print(f"model_path={args.model_path}")
    print(f"prompt_file={prompts['path']}")
    print(f"data={data_path}")
    print(f"generation={generation_config}")
    print(f"examples={len(selected)}")

    t0 = time.time()
    model, tokenizer = load_model(args.model_path, args.load_in_4bit)
    print(f"Model loading took {time.time() - t0:.1f} seconds")

    processed = []
    t0 = time.time()
    for index, example in enumerate(selected, start=args.start):
        print(f"Processing example {index}: {example.get('id', index)}", flush=True)
        messages = prepare_messages(prompts, example)
        model_inputs = encode(tokenizer, messages)
        gen_text = generate(model, tokenizer, model_inputs, generation_config)

        result = dict(example)
        result["prediction"] = extract_json(gen_text)
        if args.keep_raw:
            result["raw_generation"] = gen_text
        processed.append(result)

    with open(output_path, "w", encoding="utf-8") as output_fd:
        json.dump(processed, output_fd, indent=3, ensure_ascii=False)

    elapsed = time.time() - t0
    print("Done")
    print(f"Processed {len(processed)} examples in {elapsed:.1f} seconds")
    if processed:
        print(f"{elapsed / len(processed):.2f} sec/example")
    print(f"Output: {output_path}")

    del model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
