import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common import (
    DEFAULT_MODEL_PATH,
    OUTPUT_DIR,
    encode,
    extract_json,
    generate,
    get_prompts,
    load_dataset,
    prepare_messages,
)

# ------------ load model and tokenizer -----------------
def load_model(weightdir):
    t0 = time.time()
    MODEL_PATH = str(DEFAULT_MODEL_PATH)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # load fine-tuned weights                                             
    model = PeftModel.from_pretrained(model, str(weightdir))
    # set inference model
    model.eval()
                                                 
    # load tokenizer      
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Model loading took {time.time()-t0:.1f} seconds")
    return model, tokenizer

    
############ MAIN ##################

weights = OUTPUT_DIR / "FT.weights"
print(f"========= FT inference === WEIGHTS={weights}")

# load model and tokenizer
model, tokenizer = load_model(weights)

# load prompts
prompts = get_prompts()

# load test/devel dataset
examples, _ = load_dataset("devel")

generation_config = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": None,
    "max_new_tokens": 512,
}

# analyze each example
t0 = time.time()
for i,ex in enumerate(examples):
    print(f"*** Processing example {i}", flush=True)
    # prepare sequence of messages for this example
    messages = prepare_messages(prompts, ex)    
    # create example prompt, tokenize, and encode it into tokens
    input_ids = encode(tokenizer, messages)
    # call model to generate response            
    gen_text = generate(model, tokenizer, input_ids, generation_config)
    # extract json from response
    examples[i]["prediction"] = extract_json(gen_text)

print("Done")
print(f"Processed {len(examples)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(examples):.2f} sec/example)")

# save output
outfname = OUTPUT_DIR / "FT.out.json"
outfname.parent.mkdir(parents=True, exist_ok=True)
with open(outfname, "w", encoding="utf-8") as of :
    json.dump(examples, of, indent=3, ensure_ascii=False)

# clean up gpu
del model
torch.cuda.empty_cache() 


