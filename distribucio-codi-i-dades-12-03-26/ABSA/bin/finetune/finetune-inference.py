import argparse
import json
import time
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from common import (
    DEFAULT_MODEL_PATH,
    OUTPUT_DIR,
    encode,
    extract_json,
    generate,
    get_prompts,
    load_dataset,
    prepare_messages,
    normalize_prediction,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned ABSA model")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to fine-tuned LoRA weights. Auto-detect if not specified:\n"
             "  Tries FT.fewshot.weights first (preferred), then FT.simple.weights"
    )
    parser.add_argument(
        "--use-fewshot-weights",
        action="store_true",
        default=True,
        help="Prefer fewshot-trained weights if available (default: True)"
    )
    parser.add_argument(
        "--use-simple-weights",
        action="store_true",
        help="Prefer simple (non-fewshot) trained weights"
    )
    parser.add_argument(
        "--prompt-file",
        default=str(Path(__file__).resolve().parents[2] / "prompts" / "absa_v6.json"),
        help="Prompt file for inference (default: absa_v6.json, Optuna-optimized)"
    )
    parser.add_argument(
        "--data",
        default="devel",
        help="Dataset to run inference on (default: devel)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: outputs/FT.devel.json)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.68,
        help="Generation temperature (default: 0.68, from Optuna best trial)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.72,
        help="Top-p sampling (default: 0.72, from Optuna best trial)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (default: 20, from Optuna best trial)"
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.8,
        help="Presence penalty (default: 1.8, from Optuna best trial)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (default: None, process all)"
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw generation output (default: False)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model quantized in 4 bits"
    )
    return parser.parse_args()


# ------------ load model and tokenizer -----------------
def load_model(weights_path, prompt_path, load_in_4bit=False):
    t0 = time.time()
    MODEL_PATH = str(DEFAULT_MODEL_PATH)

    model_kwargs = {"device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # load base model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    
    # load fine-tuned LoRA weights
    model = PeftModel.from_pretrained(model, str(weights_path))
    model.eval()
                                                 
    # load tokenizer      
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Model loading took {time.time()-t0:.1f} seconds")
    return model, tokenizer

    
############ MAIN ##################

args = parse_args()

# Auto-detect weights if not specified
if args.weights is None:
    weights_path = None
    if args.use_simple_weights:
        # Prefer simple weights
        weights_path = OUTPUT_DIR / "FT.simple.weights"
    else:
        # Prefer fewshot weights
        weights_path = OUTPUT_DIR / "FT.fewshot.weights"
        if not weights_path.exists():
            # Fallback to simple if fewshot doesn't exist
            weights_path = OUTPUT_DIR / "FT.simple.weights"
    
    if not weights_path.exists():
        # Last resort: check for old default name
        weights_path = OUTPUT_DIR / "FT.weights"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No fine-tuned weights found. Tried:\n"
                f"  - {OUTPUT_DIR / 'FT.fewshot.weights'}\n"
                f"  - {OUTPUT_DIR / 'FT.simple.weights'}\n"
                f"  - {OUTPUT_DIR / 'FT.weights'}"
            )
else:
    weights_path = Path(args.weights)

if not weights_path.exists():
    raise FileNotFoundError(f"Fine-tuned weights not found at: {weights_path}")

print("========= FT INFERENCE =========")
print(f"weights: {weights_path}")
print(f"prompt_file: {args.prompt_file}")
print(f"data: {args.data}")
print(f"generation: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, pp={args.presence_penalty}")

# load model and tokenizer
model, tokenizer = load_model(weights_path, args.prompt_file, load_in_4bit=args.load_in_4bit)

# load prompts
prompts = get_prompts(args.prompt_file)
print(f"Loaded prompt: {prompts['name']}")

# load dataset for inference
examples, data_path = load_dataset(args.data)
if args.limit:
    examples = examples[:args.limit]
print(f"Loaded {len(examples)} examples from {data_path}")

# generation config from Optuna best trial parameters
generation_config = {
    "temperature": args.temperature,
    "top_p": args.top_p,
    "top_k": args.top_k,
    "presence_penalty": args.presence_penalty,
    "max_new_tokens": args.max_new_tokens,
}

# analyze each example
t0 = time.time()
processed = []
for i, ex in enumerate(examples):
    if i % 10 == 0:
        print(f"Processing example {i}", flush=True)
    
    # prepare sequence of messages for this example
    messages = prepare_messages(prompts, ex)    
    # tokenize and encode
    input_ids = encode(tokenizer, messages)
    # call model to generate response            
    gen_text = generate(model, tokenizer, input_ids, generation_config)
    # extract json from response
    prediction = extract_json(gen_text)
    
    result = dict(ex)
    result["prediction"] = prediction
    result["prediction_normalized"] = normalize_prediction(prediction)
    if args.keep_raw:
        result["raw_generation"] = gen_text
    processed.append(result)

elapsed = time.time() - t0
print(f"\nProcessed {len(processed)} examples in {elapsed:.1f} seconds ({elapsed/len(processed):.2f} sec/example)")

# save output
if args.output:
    outfname = Path(args.output)
else:
    outfname = OUTPUT_DIR / f"FT.{data_path.stem}.json"
outfname.parent.mkdir(parents=True, exist_ok=True)

with open(outfname, "w", encoding="utf-8") as of:
    json.dump(processed, of, indent=3, ensure_ascii=False)

print(f"Output saved to: {outfname}")

# clean up gpu
del model
torch.cuda.empty_cache() 
