import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from common import ABSA_DIR, ASPECTS, DEFAULT_MODEL_PATH, OUTPUT_DIR, POLARITIES, get_prompts, load_dataset as load_absa_dataset, render_template


# ========== ABSA-MMR SELECTION CONSTANTS ==========
DEFAULT_EMBEDDING_MODEL_PATH = ABSA_DIR / "embedding_model" / "qwen3_embedding_0_6b"

HARD_ASPECTS = {
    "restaurant_prices",
    "food_prices",
    "food_style_options",
    "drinks_quality",
    "drinks_prices",
    "drinks_style_options",
    "location",
}

ASPECT_CUES = {
    "restaurant_general": [
        "recommend", "recommended", "return", "again", "favorite", "overall",
        "experience", "worth", "disappoint", "volver", "repetir", "recomend",
    ],
    "restaurant_prices": [
        "expensive", "cheap", "overpriced", "affordable", "value", "prices",
        "caro", "cara", "barato", "precio", "precios", "calidad precio",
    ],
    "food_quality": [
        "food", "dish", "taste", "tasty", "delicious", "fresh", "bland", "salty",
        "comida", "cocina", "plato", "sabor", "sabroso", "rico", "delicioso",
    ],
    "food_prices": [
        "food price", "menu price", "dish price", "portion price", "bargain",
        "precio", "menu", "plato caro",
    ],
    "food_style_options": [
        "menu", "selection", "variety", "options", "portion", "portions", "limited",
        "carta", "variedad", "opciones", "racion", "raciones", "abundante",
    ],
    "drinks_quality": [
        "drink", "drinks", "wine", "beer", "cocktail", "coffee", "vino", "vinos",
        "cerveza", "cafe", "copa", "sangria",
    ],
    "drinks_prices": [
        "drink price", "wine price", "water", "bottle", "precio del vino",
        "vino caro", "botella",
    ],
    "drinks_style_options": [
        "wine list", "drink selection", "beer selection", "carta de vinos",
        "bodega", "bebidas", "vinos",
    ],
    "ambience": [
        "ambience", "atmosphere", "decor", "noise", "noisy", "music", "cozy",
        "romantic", "crowded", "clean", "ambiente", "decoracion", "ruido",
    ],
    "service": [
        "service", "staff", "waiter", "waitress", "attention", "slow", "friendly",
        "servicio", "personal", "camarero", "atencion", "trato", "rapido",
    ],
    "location": [
        "location", "view", "views", "parking", "neighborhood", "access", "terrace",
        "ubicacion", "vistas", "aparcamiento", "terraza", "centrico",
    ],
}

LAMBDA_BY_K = {1: 1.00, 2: 0.88, 4: 0.80, 6: 0.76, 8: 0.72, 10: 0.70, 12: 0.68}

BASE_SCORE_WEIGHTS = {
    "semantic": 0.72,
    "same_language": 0.06,
    "aspect_cue_coverage": 0.14,
    "rare_or_hard_label_helpfulness": 0.05,
    "length_label_count_fit": 0.03,
}


# ========== EMBEDDING & SCORING FUNCTIONS ==========
def load_embedding_model():
    """Load Qwen embedding model for semantic similarity."""
    print(f"Loading embedding model from {DEFAULT_EMBEDDING_MODEL_PATH}")
    embedding_model = AutoModel.from_pretrained(
        str(DEFAULT_EMBEDDING_MODEL_PATH),
        trust_remote_code=True,
        device_map="auto",
    )
    embedding_tokenizer = AutoTokenizer.from_pretrained(
        str(DEFAULT_EMBEDDING_MODEL_PATH),
        trust_remote_code=True,
    )
    embedding_model.eval()
    return embedding_model, embedding_tokenizer


def embed_texts(embedding_model, embedding_tokenizer, texts):
    """Embed multiple texts using Qwen embedding model."""
    embeddings = []
    with torch.no_grad():
        for text in texts:
            # Truncate long texts
            if len(text) > 8000:
                text = text[:8000]
            # Use the model's encode method if available, otherwise manual embedding
            if hasattr(embedding_model, 'encode'):
                emb = embedding_model.encode(text, convert_to_numpy=False)
            else:
                # Fallback: tokenize and forward pass with embedding model's tokenizer
                inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                # Move inputs to same device as model
                inputs = {k: v.to(embedding_model.device) for k, v in inputs.items()}
                outputs = embedding_model(**inputs, output_hidden_states=True)
                # Use last hidden state with mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    emb = outputs.pooler_output[0]
                else:
                    # Mean pooling over last hidden state
                    emb = outputs.last_hidden_state.mean(dim=1)[0]
            
            if isinstance(emb, np.ndarray):
                emb = torch.tensor(emb, dtype=torch.float32, device="cpu")
            elif not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32, device="cpu")
            else:
                # Move to CPU if it's on GPU (for consistency in processing)
                emb = emb.cpu().float()
            
            embeddings.append(emb)
    return embeddings


def build_mmr_cache(examples, example_embeddings):
    """Precompute reusable metadata for ABSA-MMR scoring."""
    embedding_rows = []
    for emb in example_embeddings:
        if isinstance(emb, torch.Tensor):
            embedding_rows.append(emb.detach().cpu().float().view(-1))
        elif isinstance(emb, np.ndarray):
            embedding_rows.append(torch.tensor(emb, dtype=torch.float32).view(-1))
        else:
            embedding_rows.append(torch.tensor(emb, dtype=torch.float32).view(-1))

    embedding_matrix = torch.stack(embedding_rows)
    embedding_matrix = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)

    return {
        "examples": examples,
        "embeddings": embedding_matrix,
        "languages": [ex.get("language", "unknown") for ex in examples],
        "lengths": np.array([len(ex.get("text", "")) for ex in examples], dtype=np.float32),
        "label_counts": np.array([len(ex.get("gold", {})) for ex in examples], dtype=np.float32),
        "hard_bonus": np.array([
            calculate_hard_label_bonus(ex.get("gold", {})) for ex in examples
        ], dtype=np.float32),
        "cue_lists": [
            tuple(
                cue
                for aspect in ex.get("gold", {})
                for cue in ASPECT_CUES.get(aspect, [])
            )
            for ex in examples
        ],
        "cue_scale": np.array([
            max(1, len({cue for aspect in ex.get("gold", {}) for cue in ASPECT_CUES.get(aspect, [])}))
            for ex in examples
        ], dtype=np.float32),
    }


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    if isinstance(emb1, np.ndarray):
        emb1 = torch.tensor(emb1, dtype=torch.float32)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.tensor(emb2, dtype=torch.float32)
    
    cos_sim = torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()
    return (cos_sim + 1.0) / 2.0  # Normalize to [0, 1]


def build_user_message(prompts, example):
    """Render only the user turn for a chat example."""
    values = {
        "text": example["text"],
        "language": example.get("language", "unknown"),
        "aspects": ", ".join(ASPECTS),
        "polarities": ", ".join(POLARITIES),
    }
    return {"role": "user", "content": render_template(prompts["user"], values)}


def calculate_aspect_cue_overlap(target_text_lower, gold_aspects, aspect_cues):
    """Calculate overlap between target text and candidate's gold aspects."""
    if not gold_aspects:
        return 0.0
    
    overlap_score = 0.0
    for aspect in gold_aspects:
        if aspect in aspect_cues:
            cues = aspect_cues[aspect]
            for cue in cues:
                if cue in target_text_lower:
                    overlap_score += 1.0
    
    max_cues = max(len(cues) for cues in aspect_cues.values())
    return min(overlap_score / (len(gold_aspects) * max_cues), 1.0)


def calculate_hard_label_bonus(gold_aspects):
    """Bonus for candidates with hard-to-detect aspects."""
    hard_count = sum(1 for asp in gold_aspects if asp in HARD_ASPECTS)
    return hard_count / max(1, len(gold_aspects)) if gold_aspects else 0.0


def calculate_length_label_fit(target_ex, candidate_ex):
    """Fitness based on review length and label count similarity."""
    target_len = len(target_ex.get("text", ""))
    target_labels = len(target_ex.get("gold", {}))
    
    cand_len = len(candidate_ex.get("text", ""))
    cand_labels = len(candidate_ex.get("gold", {}))
    
    len_diff = abs(target_len - cand_len) / (target_len + 1)
    label_diff = abs(target_labels - cand_labels) / (target_labels + cand_labels + 1)
    
    return 1.0 - (len_diff + label_diff) / 2.0


def select_absa_mmr(
    target_example,
    target_embedding,
    candidate_indices,
    mmr_cache,
    k=8,
    prefilter_size=200,
):
    """
    ABSA-aware MMR selection: semantic similarity + diversity + hard examples.
    
    Returns K best training examples for the target.
    """
    target_text_lower = target_example.get("text", "").lower()
    target_lang = target_example.get("language", "unknown")
    target_len = len(target_example.get("text", ""))
    target_labels = len(target_example.get("gold", {}))

    if not candidate_indices:
        return []

    target_embedding = target_embedding.detach().cpu().float().view(-1)
    target_embedding = torch.nn.functional.normalize(target_embedding, p=2, dim=0)

    candidate_embeddings = mmr_cache["embeddings"][candidate_indices]
    similarities = torch.matmul(candidate_embeddings, target_embedding).cpu().numpy()

    if len(candidate_indices) > prefilter_size:
        top_local = np.ascontiguousarray(np.argsort(similarities)[::-1][:prefilter_size])
        candidate_indices = [candidate_indices[i] for i in top_local]
        candidate_embeddings = candidate_embeddings[top_local]
        similarities = similarities[top_local]

    same_lang = np.fromiter(
        (1.0 if mmr_cache["languages"][idx] == target_lang else 0.0 for idx in candidate_indices),
        dtype=np.float32,
        count=len(candidate_indices),
    )
    hard_bonus = mmr_cache["hard_bonus"][candidate_indices]
    candidate_lengths = mmr_cache["lengths"][candidate_indices]
    candidate_label_counts = mmr_cache["label_counts"][candidate_indices]
    length_fit = 1.0 - (
        (
            np.abs(target_len - candidate_lengths) / (target_len + 1.0)
            + np.abs(target_labels - candidate_label_counts) / (target_labels + candidate_label_counts + 1.0)
        ) / 2.0
    )

    aspect_coverage = np.fromiter(
        (
            sum(cue in target_text_lower for cue in mmr_cache["cue_lists"][idx])
            / mmr_cache["cue_scale"][idx]
            for idx in candidate_indices
        ),
        dtype=np.float32,
        count=len(candidate_indices),
    )

    base_scores = (
        0.72 * similarities
        + 0.06 * same_lang
        + 0.14 * aspect_coverage
        + 0.05 * hard_bonus
        + 0.03 * length_fit
    )

    top_base_local = np.ascontiguousarray(np.argsort(base_scores)[::-1][: min(50, len(candidate_indices))])
    candidate_indices = [candidate_indices[i] for i in top_base_local]
    candidate_embeddings = candidate_embeddings[top_base_local]
    similarities = similarities[top_base_local]
    base_scores = base_scores[top_base_local]

    lambda_k = LAMBDA_BY_K.get(k, 0.72)
    similarity_matrix = torch.matmul(candidate_embeddings, candidate_embeddings.T).cpu().numpy()
    selected_local = []
    candidate_pool = list(range(len(candidate_indices)))

    while len(selected_local) < k and candidate_pool:
        best_local = None
        best_score = -float("inf")

        for local_idx in list(candidate_pool):
            diversity_penalty = 0.0
            if selected_local:
                max_similarity = float(similarity_matrix[local_idx, selected_local].max())
                if max_similarity > 0.92 and len(candidate_pool) > k - len(selected_local):
                    continue
                diversity_penalty = (1 - lambda_k) * max_similarity

            final_score = lambda_k * float(base_scores[local_idx]) - diversity_penalty
            if final_score > best_score:
                best_score = final_score
                best_local = local_idx

        if best_local is None:
            break

        selected_local.append(best_local)
        candidate_pool.remove(best_local)

    selected_indices = [candidate_indices[i] for i in selected_local]
    selected_sorted = sorted(
        selected_indices,
        key=lambda i: float(torch.matmul(mmr_cache["embeddings"][i], target_embedding).item()),
        reverse=False,
    )
    return [mmr_cache["examples"][i] for i in selected_sorted]


# ========== LOAD MODEL AND TOKENIZER ==========
def load_model(load_in_4bit=False, fp16=False):
    t0 = time.time()

    MODEL_PATH = str(DEFAULT_MODEL_PATH)

    torch_dtype = torch.float16 if fp16 else torch.bfloat16
    model_kwargs = {
        "device_map": "auto" if load_in_4bit else None,
        "torch_dtype": torch_dtype,
    }
    if load_in_4bit:
        model_kwargs.pop("torch_dtype", None)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    # load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    model.config.use_cache = False

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

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

    # Keep memory use down during training.
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    print(f"Model loading took {time.time()-t0:.1f} seconds")
    return model, tokenizer


# ------------ parse command-line arguments -----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-2B for ABSA with LoRA")
    parser.add_argument(
        "prompt_file",
        nargs="?",
        default=str(ABSA_DIR / "prompts" / "absa_v6.json"),
        help="Path to prompt file (default: absa_v6.json)"
    )
    parser.add_argument(
        "dataset_file",
        nargs="?",
        default="train",
        help="Training dataset file or dataset name (default: train, or train+synth.json)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for LoRA fine-tuning (default: 1e-5, conservative for LoRA)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3, LoRA needs fewer epochs than full FT)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Gradient accumulation steps per device (default: 4)"
    )
    parser.add_argument(
        "--per-device-train-batch",
        type=int,
        default=1,
        help="Per-device train batch size (default: 1)"
    )
    parser.add_argument(
        "--per-device-eval-batch",
        type=int,
        default=1,
        help="Per-device eval batch size (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save fine-tuned weights (auto-derived if omitted)"
    )
    parser.add_argument(
        "--use-fewshot",
        action="store_true",
        default=True,
        help="Use ABSA-MMR few-shot augmentation during training (default: True)"
    )
    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="Disable few-shot augmentation (trains without in-context examples)"
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        default=8,
        help="Number of few-shot examples used when --use-fewshot is enabled"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load base model quantized in 4 bits"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 instead of BF16"
    )
    return parser.parse_args()


# ------------ tokenize dataset WITHOUT in-context examples (baseline) -----------------
def tokenize_dataset_simple(tokenizer, dataset, prompts):
    """
    Simple tokenization without few-shot augmentation.
    Baseline for comparing with few-shot enhanced training.
    """
    newDS = {"input_ids": [], "labels": []}
    
    for idx, example in enumerate(dataset):
        if idx % 100 == 0:
            print(f"Processing example {idx}/{len(dataset)}...", flush=True)
        
        # Simple: just target example without few-shot
        messages = [{"role": "system", "content": prompts["system"]}]
        
        # Add target example
        messages.append(build_user_message(prompts, example))
        
        # Add assistant response (gold)
        messages.append({
            "role": "assistant",
            "content": json.dumps(example.get("gold", {}), ensure_ascii=False)
        })
        
        # Tokenize
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Mark padding as -100 (ignored by trainer)
        labels = [-100 if tk == tokenizer.pad_token_id else tk for tk in tokens["input_ids"]]
        
        newDS["input_ids"].append(tokens["input_ids"])
        newDS["labels"].append(labels)
    
    return Dataset.from_dict(newDS)


# ------------ tokenize dataset WITH in-context examples from ABSA-MMR selection -----------------
def tokenize_dataset_with_fewshot(tokenizer, dataset, prompts, embedding_model, embedding_tokenizer, k_shots=8):
    """
    Tokenize dataset augmented with best few-shot examples per item.
    Uses ABSA-MMR selection to find K best training examples for context.
    """
    print(f"\nEmbedding {len(dataset)} training examples for MMR selection...")
    t0 = time.time()
    
    # Pre-embed all training examples once and reuse them for every target
    train_embeddings = embed_texts(embedding_model, embedding_tokenizer, [ex["text"] for ex in dataset])
    mmr_cache = build_mmr_cache(dataset, train_embeddings)
    target_embeddings = train_embeddings
    print(f"Embedding took {time.time()-t0:.1f} seconds")
    
    # Tokenize with in-context examples
    newDS = {"input_ids": [], "labels": []}
    
    for idx, target_example in enumerate(dataset):
        if idx % 50 == 0:
            print(f"Processing example {idx}/{len(dataset)} for tokenization...", flush=True)
        
        # Select K best examples using ABSA-MMR (excluding self)
        candidate_indices = [i for i in range(len(dataset)) if i != idx]
        selected_shots = select_absa_mmr(
            target_example,
            target_embeddings[idx],
            candidate_indices,
            mmr_cache,
            k=k_shots,
        )
        
        # Build few-shot messages
        messages = [{"role": "system", "content": prompts["system"]}]
        
        # Add few-shot examples
        for shot_ex in selected_shots:
            messages.append(build_user_message(prompts, shot_ex))
            # Add the assistant response (gold)
            messages.append({
                "role": "assistant",
                "content": json.dumps(shot_ex.get("gold", {}), ensure_ascii=False)
            })
        
        # Add target example as user query
        messages.append(build_user_message(prompts, target_example))
        
        # Add assistant response (gold)
        messages.append({
            "role": "assistant",
            "content": json.dumps(target_example.get("gold", {}), ensure_ascii=False)
        })
        
        # Tokenize
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=2048,  # Longer context for few-shot
            padding="max_length"
        )
        
        # Mark padding as -100 (ignored by trainer)
        labels = [-100 if tk == tokenizer.pad_token_id else tk for tk in tokens["input_ids"]]
        
        newDS["input_ids"].append(tokens["input_ids"])
        newDS["labels"].append(labels)
    
    return Dataset.from_dict(newDS)


# ------------ create trainer with optimized hyperparameters -----------------
def create_trainer(model, train_dataset, val_dataset, outputdir, args):
    # Configure training arguments optimized for LoRA fine-tuning
    training_args = TrainingArguments(
        output_dir=outputdir,
        per_device_train_batch_size=args.per_device_train_batch,
        per_device_eval_batch_size=args.per_device_eval_batch,
        gradient_accumulation_steps=args.batch_size,
        eval_accumulation_steps=2,
        fp16=args.fp16,
        bf16=not args.fp16,
        learning_rate=args.learning_rate,  # Conservative for LoRA
        num_train_epochs=args.num_epochs,  # Fewer epochs than full FT
        eval_strategy="epoch",
        gradient_checkpointing=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy="epoch",
        logging_strategy="epoch",
        label_names=["labels"],
        seed=args.seed,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
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

args = parse_args()

# Handle fewshot flag
use_fewshot = args.use_fewshot and not args.no_fewshot
fewshot_label = "WITH FEWSHOT" if use_fewshot else "WITHOUT FEWSHOT"

prompt_file = Path(args.prompt_file)
dataset_file = args.dataset_file

# Update output dir based on fewshot preference and dataset name
if args.output_dir is None:
    dataset_stem = Path(dataset_file).stem if Path(dataset_file).suffix else dataset_file
    suffix = "fewshot" if use_fewshot else "simple"
    output_dir = OUTPUT_DIR / f"FT.{dataset_stem}.{suffix}.weights"
else:
    output_dir = Path(args.output_dir)

print("========= FINE TUNE ABSA " + fewshot_label + " =========")
print(f"prompt_file: {prompt_file}")
print(f"dataset_file: {dataset_file}")
print(f"learning_rate: {args.learning_rate}")
print(f"num_epochs: {args.num_epochs}")
print(f"batch_size (gradient_accumulation): {args.batch_size}")
print(f"per_device_train_batch: {args.per_device_train_batch}")
print(f"per_device_eval_batch: {args.per_device_eval_batch}")
print(f"seed: {args.seed}")
print(f"output_dir: {output_dir}")
print(f"use_fewshot: {use_fewshot}")
print(f"load_in_4bit: {args.load_in_4bit}")

# load model and tokenizer
model, tokenizer = load_model(load_in_4bit=args.load_in_4bit, fp16=args.fp16)

# load prompts
prompts = get_prompts(prompt_file)
print(f"Loaded prompt: {prompts['name']}")

# Load training data
t0 = time.time()
train_examples, _ = load_absa_dataset(dataset_file)
print(f"Train dataset size: {len(train_examples)} examples")

# Load validation data
val_examples, _ = load_absa_dataset("devel")
print(f"Validation dataset size: {len(val_examples)} examples")

if use_fewshot:
    # ===== TRAINING WITH FEW-SHOT =====
    print("\n" + "="*60)
    print("TOKENIZING WITH ABSA-MMR FEW-SHOT AUGMENTATION")
    print("="*60)
    
    # Load embedding model for ABSA-MMR selection
    embedding_model, embedding_tokenizer = load_embedding_model()
    
    # Tokenize train with few-shot examples (K=8 is optimal)
    print("Tokenizing training data with few-shot examples...")
    t_tok = time.time()
    train_dataset = tokenize_dataset_with_fewshot(
        tokenizer, train_examples, prompts, embedding_model, embedding_tokenizer, k_shots=args.k_shots
    )
    print(f"Train tokenization took {time.time()-t_tok:.1f} seconds")
    
else:
    # ===== TRAINING WITHOUT FEW-SHOT =====
    print("\n" + "="*60)
    print("TOKENIZING WITHOUT FEW-SHOT (BASELINE)")
    print("="*60)
    
    # Simple tokenization
    print("Tokenizing training data (simple, no few-shot)...")
    t_tok = time.time()
    train_dataset = tokenize_dataset_simple(tokenizer, train_examples, prompts)
    print(f"Train tokenization took {time.time()-t_tok:.1f} seconds")

# Tokenize validation WITHOUT few-shot (always for faster evaluation)
print("\n" + "="*60)
print("LOADING VALIDATION DATA (no few-shot)")
print("="*60)
print("Tokenizing validation set...")
t0 = time.time()
val_dataset_dict = {"input_ids": [], "labels": []}
for idx, example in enumerate(val_examples):
    if idx % 50 == 0:
        print(f"Processing val example {idx}/{len(val_examples)}", flush=True)
    
    msg = [{"role": "system", "content": prompts["system"]}]
    msg.append(build_user_message(prompts, example))
    msg.append({
        "role": "assistant",
        "content": json.dumps(example.get("gold", {}), ensure_ascii=False)
    })
    
    text = tokenizer.apply_chat_template(msg, tokenize=False)
    tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    labels = [-100 if tk == tokenizer.pad_token_id else tk for tk in tokens["input_ids"]]
    
    val_dataset_dict["input_ids"].append(tokens["input_ids"])
    val_dataset_dict["labels"].append(labels)

val_dataset = Dataset.from_dict(val_dataset_dict)
print(f"Validation tokenization took {time.time()-t0:.1f} seconds")

# create trainer for fine tuning
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
output_dir.mkdir(parents=True, exist_ok=True)
trainer = create_trainer(model, train_dataset, val_dataset, output_dir, args)

# Fine-tune the model
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"Training took {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

# Save the fine-tuned model weights
trainer.save_model()
print(f"\nFine-tuning complete!")
print(f"Weights saved to: {output_dir}")
print(f"Config: prompt={prompts['name']}, lr={args.learning_rate}, epochs={args.num_epochs}, {fewshot_label}")

# Clean up
if use_fewshot:
    del embedding_model
del model
torch.cuda.empty_cache()
