import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import get_prompts, load_dataset, prepare_messages, encode, generate, extract_json

# ------------ load model and tokenizer -----------------
def load_model():
    t0 = time.time()

    MODEL_PATH = f"/scratch/nas/1/PDI/mml0/llama32B3/snapshots/snap0"

    # load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                                 dtype=torch.bfloat16,
                                                 device_map="auto")
    # set inference mode                                             
    model.eval()

    # load tokenizer      
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loading took {time.time()-t0:.1f} seconds")
    return model, tokenizer


############## main ###################


print(f"========= ZERO SHOT ===  ")

# load model and tokenizer
model, tokenizer = load_model()

# get system and user prompts
prompts = get_prompts()

# load data to process
examples = load_dataset("devel")

# annotate each example in devel data
t0 = time.time()
for i,ex in enumerate(examples):
    print(f"Processing example {i}", flush=True)
    # prepare sequence of messages for this example
    messages = prepare_messages(prompts, ex)    
    # create example prompt, tokenize, and encode it into tokens
    input_ids = encode(tokenizer, messages)
    # call model to generate response            
    gen_text = generate(model, tokenizer, input_ids)
    # extract json from response
    examples[i]["prediction"] = extract_json(gen_text)

print("Done")
print(f"Processed {len(examples)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(examples):.2f} sec/example)")

# save output
outfname = f"ZS.out.json"
with open(outfname, "w") as of :
    json.dump(examples, of, indent=3, ensure_ascii=False)

# clean up gpu
del model
torch.cuda.empty_cache() 

