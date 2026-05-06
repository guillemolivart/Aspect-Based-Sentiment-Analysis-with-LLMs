import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from common import DEFAULT_MODEL_PATH, OUTPUT_DIR, get_prompts, load_dataset as load_absa_dataset, prepare_messages

# ------------ load model and tokenizer -----------------
def load_model():
    t0 = time.time()

    MODEL_PATH = str(DEFAULT_MODEL_PATH)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        device_map=None,  # Can not load to GPU yet
    )
        
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


# ------------ tokenize dataset in batches of appropriate size -----------------
def tokenize_dataset(tokenizer, dataset, prompts) :

    # prepare to create tokenized and encoded version of the dataset 
	newDS = {"input_ids": [],
             "labels": []}
    
	for example in dataset :
        # prepare messages for that example
		msg = prepare_messages(prompts, example) + [{"role": "assistant", "content": json.dumps(example["gold"], ensure_ascii=False)}]

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
train_dataset, _ = load_absa_dataset("train")
train_dataset = tokenize_dataset(tokenizer, train_dataset, prompts)
val_dataset, _ = load_absa_dataset("devel")
val_dataset = tokenize_dataset(tokenizer, val_dataset, prompts)
print(f"Dataset loading took {time.time()-t0:.1f} seconds")

# create trainer for fine tuning
outputdir = OUTPUT_DIR / "FT.weights"
outputdir.mkdir(parents=True, exist_ok=True)
trainer = create_trainer(model, train_dataset, val_dataset, outputdir) 

# Fine-tune the model
t0 = time.time()
trainer.train()
print(f"Training took {time.time()-t0:.1f} seconds")

# Save the fine-tuned model weighs (in outdir)
trainer.save_model()

print("Fine-tuning complete!")


