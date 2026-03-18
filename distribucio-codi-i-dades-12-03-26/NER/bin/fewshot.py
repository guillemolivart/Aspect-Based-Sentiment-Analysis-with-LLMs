import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import get_prompts, load_dataset, encode, generate, extract_json

# ------------ check command line and get arguments -----------------
def get_arguments():
    # get command line arguments and check validity
    # more arguments can be added if needed
    
    if not len(sys.argv)==2 or not sys.argv[1].isdigit():
        print(f"Usage:  {sys.argv[0]} num_few_shot")
        sys.exit(1)

    args = {"num_few_shot" : int(sys.argv[1]) }
    return args


# ------------ load model and tokenizer -----------------
def load_model():
    t0 = time.time()

    MODEL_PATH = f"/scratch/nas/1/PDI/mml0/llama32B3/snapshots/snap0"

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                                 dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()

    # load tokenizer and set template.        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loading took {time.time()-t0:.1f} seconds")
    return model, tokenizer


# ------------ select given number of examples ----------------
def get_few_shots(train, num_few_shot) :
    # select which examples will be shown to the model
    # We can select the first N, or random N, or some pre-chosen N, ...
    return train[:num_few_shot]

    
# ------------ prepare prompt messages for an example ----------------
def prepare_messages(prompts, shots, question) :
    
    # First, system message
    messages = [{"role": "system", "content": prompts["sysprompt"]}]

    # then, num_few_shot examples of expected interaction user-assistant
    for example in shots :
        # add the example text (user part)
        messages.append({"role": "user",
                         "content": prompts["usrprompt"]
                                    + "\nTEXT: "
                                    + example['text']})
        # add the expected answer (assistant part)
        messages.append({"role": "assistant",
                         "content": json.dumps(example['gold'])})
  
    # finally add the text we want the anwer for, so the model will complete the response
    messages.append({"role": "user", 
                     "content": prompts["usrprompt"]
                                + "\nTEXT: "
                                + question['text']})
    return messages


############## main ###################


# get command line arguments
args = get_arguments()

print(f"========= FEW SHOT === SHOTS={args['num_few_shot']} ")

# load model and tokenizer
model, tokenizer = load_model()

# get system and user prompts
prompts = get_prompts()
# load train dataset, to get the few shot examples
train = load_dataset("train")
# select number of example shots
shots = get_few_shots(train, args["num_few_shot"])

# load devel data
examples = load_dataset("devel")
# annotate each example in devel data
t0 = time.time()
for i,ex in enumerate(examples):
    print(f"Processing example {i}", flush=True)
    # prepare sequence of messages for this example
    messages = prepare_messages(prompts, shots, ex)    
    # create example prompt, tokenize, and encode it into tokens
    input_ids = encode(tokenizer, messages)
    # call model to generate response            
    gen_text = generate(model, tokenizer, input_ids)
    # extract json from response
    examples[i]["prediction"] = extract_json(gen_text)

print("Done")
print(f"Processed {len(examples)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(examples):.2f} sec/example)")

# save output
outfname = f"FS-{args['num_few_shot']}.out.json"
with open(outfname, "w") as of :
    json.dump(examples, of, indent=3, ensure_ascii=False)

# clean up gpu
del model
torch.cuda.empty_cache() 

