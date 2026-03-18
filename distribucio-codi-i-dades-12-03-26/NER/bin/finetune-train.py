import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

# ------------ load model and tokenizer -----------------
def load_model():
    t0 = time.time()

    MODEL_PATH = f"/scratch/nas/1/PDI/mml0/llama32B3/snapshots/snap0"

    # load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                                 dtype=torch.bfloat16,
                                                 #low_cpu_mem_usage=True,
                                                 device_map=None) # Can not load to GPU yet
        
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    # Add LoRa fine-tunable layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Now, after adding PEFT layers, we can load to GPU.
    model = model.to("cuda")
    
    print(f"Model loading took {time.time()-t0:.1f} seconds")
    return model, tokenizer


# ------------ get system and user prompts -----------------
def get_prompts():
    # we can set the prompts manually, or better, load them from a given file,
    # so we can experiment with different prompt combinations without changing the code
    prompts = {}
    prompts["sysprompt"] = "\n".join(["You are an journalist assistant in a news agency and your mission is to read news items and extract the list of entities mentioned in it.",
                   "Mentioned entities may belong to one of the following types:",
                   "",
                   "  - Person (PER): The name of a person",
                   "  - Location (LOC): The name of a country, city, region, etc.",
                   "  - Organization (ORG): The name of a company, government agency, foundation, bank, ministery, etc."
                  ])
                  
    prompts["usrprompt"] = "\n".join(["Given the following text, extract entities, classified by their type (PER, LOC, ORG)",
               "Provide the output as a json dictionary, with a key for each appearing entity type, and a list of names as value (e.g. {\"PER\":[\"John Smith\"], \"LOC\":[\"USA\", \"Berlin\"]} )",
	       "",
               "It is important that you take into account the following constraints:",
               "  - If an entity is mentioned twice with the same words, extract it only once.",
               "  - If an entity is mentioned twice with the different words (e.g. John Smith and Mr. Smith), extract it in both cases.",
               "  - Only the three types in the above list are valid. DO NOT introduce new ones.",
               "  - DO NOT INCLUDE in the json entity types for which no entity is mentioned.",
               "  - Produce just the json, do not add further explanations."
              ])

    return prompts


# ------------ load dataset (either "train" or "devel") -----------------
def load_dataset(data) :
    with open(f"dataset/{data}.json") as f : 
        examples = json.load(f)
                
    return examples


# ------------ tokenize dataset in batches of appropriate size -----------------
def tokenize_dataset(tokenizer, dataset, prompts) :

    # prepare to create tokenized and encoded version of the dataset 
	newDS = {"input_ids": [],
             "labels": []}
    
	for example in dataset :
        # prepare messages for that example
		msg = [{"role": "system", "content": prompts["sysprompt"]},
		       {"role": "user", "content": prompts["usrprompt"] + "\nTEXT: " + example["text"]},
		       {"role": "assistant", "content": json.dumps(example["gold"])}
		      ]

        # convert messages to a whole text prompt
		text = tokenizer.apply_chat_template(msg, tokenize=False)

        # tokenize and encode text
		tokens = tokenizer(text,
		                   truncation=True,
		                   max_length=512,
		                   padding="max_length"
		                  )

        # mark padding tokens with -100 so the trainer ignores them
		labels = [-100 if tk == tokenizer.pad_token_id else tk for tk in tokens["input_ids"]]

	    # add example to new dataset
		newDS["input_ids"].append(tokens["input_ids"])
		newDS["labels"].append(labels)

    # create and return tokenized+encoded dataset
	return Dataset.from_dict(newDS)


# ------------ tokenize dataset in batches of appropriate size -----------------
def create_trainer(model, train_dataset, val_dataset, outputdir) :
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=outputdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_accumulation_steps=4,
        fp16=False,
        bf16=True,
        learning_rate=2e-5,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        logging_strategy="epoch",
        label_names=["labels"]
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        train_dataset=train_dataset
    )
    return trainer

############## MAIN ################


print(f"========= FINE TUNE == ")

# load model and tokenizer
model, tokenizer = load_model()

# load prompts
prompts = get_prompts()

# load and tokenize datasets
t0 = time.time()
train_dataset = load_dataset("train")
train_dataset = tokenize_dataset(tokenizer, train_dataset, prompts)
val_dataset = load_dataset("devel")
val_dataset = tokenize_dataset(tokenizer, val_dataset, prompts)
print(f"Dataset loading took {time.time()-t0:.1f} seconds")

# create trainer for fine tuning
THISDIR = os.path.dirname(__file__)
outputdir = os.path.join(THISDIR, "FT.weights")
trainer = create_trainer(model, train_dataset, val_dataset, outputdir) 

# Fine-tune the model
t0 = time.time()
trainer.train()
print(f"Training took {time.time()-t0:.1f} seconds")

# Save the fine-tuned model weighs (in outdir)
trainer.save_model()

print("Fine-tuning complete!")


