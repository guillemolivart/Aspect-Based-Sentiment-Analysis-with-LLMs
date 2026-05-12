import argparse
import hashlib
import json
import math
import random
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    ABSA_DIR,
    ASPECTS,
    DEFAULT_MODEL_PATH,
    OUTPUT_DIR,
    _json_candidates,
    encode,
    get_prompts,
    load_dataset,
    normalize_prediction,
    render_template,
    strip_thinking,
)
from zeroshot import load_model


"""
Few-shot runner for the ABSA project.

The defaults use the best current non-thinking Qwen generation region from the
hyperparameter study. The main experimental variable here should be the
in-context example policy, not another broad generation sweep.
"""

DEFAULT_FEWSHOT_PROMPT_PATH = ABSA_DIR / "prompts" / "absa_v6.json"
DEFAULT_EMBEDDING_MODEL_PATH = (
    ABSA_DIR / "embedding_model" / "qwen3_embedding_0_6b"
)
FEWSHOT_OUTPUT_DIR = OUTPUT_DIR / "fewshot"
FEWSHOT_CACHE_DIR = FEWSHOT_OUTPUT_DIR / "cache"

GENERATION_DEFAULTS = {
    "temperature": 0.68,
    "top_p": 0.72,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.8,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

METHODS = [
    "random_fixed",
    "random_dynamic",
    "dense_topk",
    "absa_mmr",
    "manual_fixed_hard",
    "hard_mix",
]

LAMBDA_BY_K = {
    1: 1.00,
    2: 0.88,
    4: 0.80,
    6: 0.76,
    8: 0.72,
    10: 0.70,
    12: 0.68,
}

BASE_SCORE_WEIGHTS = {
    "semantic": 0.72,
    "same_language": 0.06,
    "aspect_cue_coverage": 0.14,
    "rare_or_hard_label_helpfulness": 0.05,
    "length_label_count_fit": 0.03,
}

MANUAL_HARD_BANK = [
    {
        "id": "1225162",
        "notes": "Broad positive example covering food, ambience, service, prices, drinks style, and drinks prices.",
    },
    {
        "id": "1632445",
        "notes": "Separates good food/service/ambience from limited menu and high prices.",
    },
    {
        "id": "1212346",
        "notes": "Food conflict plus negative service, menu/amount, and food prices.",
    },
    {
        "id": "es_balmes_rossello_12_LauraRamosMartinez_2015-02-23",
        "notes": "Spanish neutral/conflict example with service and ambience complications.",
    },
    {
        "id": "es_cafe_kafka_40_Kadulillo_2012-03-03",
        "notes": "Short Spanish example with positive ambience, food prices, and drinks quality.",
    },
    {
        "id": "es_l_olive_77_Mathews_2007-12-07",
        "notes": "Separates food prices, drinks prices, drinks style, ambience, and neutral restaurant general.",
    },
    {
        "id": "744478",
        "notes": "Positive location and both food/restaurant prices.",
    },
    {
        "id": "es_cafe_casa_lletres_9_RicardoSantolariaPerez_2015-01-06",
        "notes": "Legitimate neutral labels and negative portion/value.",
    },
    {
        "id": "1459569",
        "notes": "English negative service, drinks prices, neutral food quality, and negative restaurant general.",
    },
    {
        "id": "es_doble-uno-zaragoza_comment-4729",
        "notes": "Spanish example with restaurant conflict, service conflict, drinks style negative, restaurant prices negative, and ambience positive.",
    },
    {
        "id": "es_puerto-de-santa-maria-zaragoza_comment-802",
        "notes": "Empty-gold question/comment example for reducing over-labeling at larger K.",
    },
    {
        "id": "es_luis-candelas-zaragoza_comment-4486",
        "notes": "Spanish drinks quality negative while the overall meal remains positive.",
    },
]

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
        "recommend",
        "recommended",
        "return",
        "again",
        "favorite",
        "overall",
        "experience",
        "worth",
        "disappoint",
        "volver",
        "repetir",
        "recomend",
        "experiencia",
        "favorito",
        "merece",
        "decepcion",
    ],
    "restaurant_prices": [
        "expensive",
        "cheap",
        "overpriced",
        "affordable",
        "value",
        "worth the money",
        "prices",
        "pricey",
        "caro",
        "cara",
        "barato",
        "precio",
        "precios",
        "calidad precio",
        "relacion calidad",
        "calidad-precio",
        "astronomico",
    ],
    "food_quality": [
        "food",
        "dish",
        "dishes",
        "taste",
        "tasty",
        "delicious",
        "fresh",
        "bland",
        "salty",
        "cooked",
        "pizza",
        "sushi",
        "meat",
        "comida",
        "cocina",
        "plato",
        "platos",
        "sabor",
        "sabroso",
        "rico",
        "delicioso",
        "fresco",
        "carne",
        "pescado",
        "arroz",
    ],
    "food_prices": [
        "food price",
        "menu price",
        "dish price",
        "portion price",
        "bargain",
        "precio del menu",
        "precio de la comida",
        "racion",
        "raciones",
        "menu",
        "plato caro",
        "platos caros",
        "calidad precio",
        "relacion calidad",
    ],
    "food_style_options": [
        "menu",
        "selection",
        "variety",
        "options",
        "portion",
        "portions",
        "large",
        "small",
        "presentation",
        "limited",
        "choice",
        "choices",
        "carta",
        "variedad",
        "opciones",
        "racion",
        "raciones",
        "cantidad",
        "presentacion",
        "abundante",
        "escaso",
        "escasa",
        "completo",
    ],
    "drinks_quality": [
        "drink",
        "drinks",
        "wine",
        "beer",
        "cocktail",
        "coffee",
        "margarita",
        "martini",
        "vino",
        "vinos",
        "cerveza",
        "cafe",
        "copa",
        "copas",
        "combinado",
        "combinados",
        "sangria",
    ],
    "drinks_prices": [
        "drink price",
        "wine price",
        "water was",
        "bottle",
        "bottles",
        "voss",
        "precio del vino",
        "vinos caros",
        "vino caro",
        "agua",
        "botella",
        "botellas",
        "cafe a",
        "zumo",
    ],
    "drinks_style_options": [
        "wine list",
        "drink selection",
        "beer selection",
        "beverage",
        "bar",
        "carta de vinos",
        "seleccion de vinos",
        "bodega",
        "bebidas",
        "vinos",
        "copas",
    ],
    "ambience": [
        "ambience",
        "atmosphere",
        "decor",
        "noise",
        "noisy",
        "music",
        "cozy",
        "romantic",
        "crowded",
        "space",
        "clean",
        "local",
        "ambiente",
        "decoracion",
        "ruido",
        "ruidoso",
        "musica",
        "acogedor",
        "romantico",
        "limpio",
        "espacio",
        "mesa",
        "mesas",
    ],
    "service": [
        "service",
        "staff",
        "waiter",
        "waitress",
        "hostess",
        "server",
        "served",
        "attention",
        "rude",
        "slow",
        "friendly",
        "servicio",
        "personal",
        "camarero",
        "camareros",
        "atencion",
        "trato",
        "amable",
        "lento",
        "rapido",
        "reserva",
    ],
    "location": [
        "location",
        "view",
        "views",
        "parking",
        "neighborhood",
        "access",
        "terrace",
        "outside",
        "situacion",
        "ubicacion",
        "vistas",
        "parking",
        "aparcamiento",
        "terraza",
        "centrico",
        "barrio",
    ],
}

POLARITY_CUES = {
    "positive": [
        "excellent",
        "great",
        "good",
        "delicious",
        "amazing",
        "wonderful",
        "friendly",
        "recommend",
        "excelente",
        "bueno",
        "buena",
        "rico",
        "rica",
        "genial",
        "estupendo",
        "recomiendo",
    ],
    "negative": [
        "bad",
        "terrible",
        "slow",
        "rude",
        "overpriced",
        "disappoint",
        "bland",
        "salty",
        "malo",
        "mala",
        "lento",
        "caro",
        "decepcion",
        "escaso",
        "ruido",
        "mal",
    ],
    "neutral": [
        "average",
        "ok",
        "okay",
        "fine",
        "normal",
        "regular",
        "correct",
        "correcto",
        "aceptable",
        "pasable",
        "sin mas",
        "no esta mal",
    ],
    "conflict": [
        "but",
        "although",
        "however",
        "though",
        "except",
        "pero",
        "aunque",
        "sin embargo",
        "salvo",
        "excepto",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-shot ABSA inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--embedding-model-path", default=str(DEFAULT_EMBEDDING_MODEL_PATH))
    parser.add_argument("--prompt-file", default=str(DEFAULT_FEWSHOT_PROMPT_PATH))
    parser.add_argument("--data", default="devel")
    parser.add_argument("--train-data", default="train")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieve-candidates", type=int, default=50)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--redundancy-threshold", type=float, default=0.92)
    parser.add_argument("--hard-mix-manual-count", type=int, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=GENERATION_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=GENERATION_DEFAULTS["top_p"])
    parser.add_argument("--top-k", type=int, default=GENERATION_DEFAULTS["top_k"])
    parser.add_argument("--min-p", type=float, default=GENERATION_DEFAULTS["min_p"])
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=GENERATION_DEFAULTS["presence_penalty"],
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=GENERATION_DEFAULTS["repetition_penalty"],
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=GENERATION_DEFAULTS["max_new_tokens"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--no-demo-text", action="store_true")
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Store rendered chat messages for each example. Useful for audits, but large.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Rewrite the output JSON every N processed examples. Use 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the output JSON already exists, skip target ids already present there.",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=None,
        help="Override the K-dependent ABSA-MMR relevance/diversity tradeoff.",
    )
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def resolved_generation_config(args):
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower()


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def flatten(value, prefix=""):
    if isinstance(value, (str, int, float, bool)):
        return [prefix + ":" + str(value)]
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten(item, prefix))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for key, item in value.items():
            flattened.extend(flatten(item, prefix + f".{key}"))
        return flattened
    return []


def counts(prediction, gold):
    pred = flatten(prediction)
    expected = flatten(gold)
    ok = sum(1 for item in pred if item in expected)
    return len(pred), len(expected), ok


def prf(predicted, expected, ok):
    precision = 100.0 * ok / predicted if predicted else 0.0
    recall = 100.0 * ok / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def extract_json_with_status(gen_text):
    search_spaces = [strip_thinking(gen_text), gen_text]
    for text in search_spaces:
        candidates = list(_json_candidates(text))
        for candidate in reversed(candidates):
            prediction = normalize_prediction(candidate)
            if prediction or candidate == {}:
                return prediction, True
    return {}, False


def prompt_values(example):
    return {
        "text": example["text"],
        "language": example.get("language", "unknown"),
        "aspects": ", ".join(ASPECTS),
        "polarities": ", ".join(["positive", "negative", "neutral", "conflict"]),
    }


def build_demo_block(demos):
    if not demos:
        return ""

    parts = [
        "Few-shot training examples:",
        "Use these examples only as labeling guidance. Do not copy labels unless the target review explicitly supports them.",
    ]
    for rank, demo in enumerate(demos, start=1):
        parts.extend(
            [
                f"Example {rank}:",
                f"Review language: {demo.get('language', 'unknown')}",
                "Review text:",
                demo["text"],
                "Expected JSON:",
                compact_json(demo.get("gold", {})),
                "",
            ]
        )
    parts.append("Now analyze the target review.")
    return "\n".join(parts).strip()


def prepare_fewshot_messages(prompts, target, demos):
    values = prompt_values(target)
    system = render_template(prompts["system"], values)
    user_template = prompts["user"]
    demo_block = build_demo_block(demos)

    if demo_block and "\nReview language:" in user_template:
        user_template = user_template.replace(
            "\nReview language:",
            "\n\n" + demo_block + "\n\nReview language:",
            1,
        )
    elif demo_block:
        user_template = demo_block + "\n\n" + user_template

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_template(user_template, values)},
    ]


def default_output_name(data_path, prompt_name, method, k, generation_config, thinking, seed):
    mode = "think" if thinking else "nothink"
    top_k = generation_config["top_k"]
    top_k_part = "none" if top_k is None else str(top_k)
    name = (
        f"FS.{data_path.stem}.{prompt_name}.{method}.k{k}.seed{seed}.{mode}"
        f".t{generation_config['temperature']}.p{generation_config['top_p']}"
        f".tk{top_k_part}.mp{generation_config['min_p']}"
        f".pp{generation_config['presence_penalty']}"
        f".rp{generation_config['repetition_penalty']}"
        f".m{generation_config['max_new_tokens']}.json"
    )
    return FEWSHOT_OUTPUT_DIR / name


def output_path_from_args(args, data_path, prompt_name, generation_config):
    if args.output:
        return Path(args.output)
    if args.output_prefix:
        return FEWSHOT_OUTPUT_DIR / f"{args.output_prefix}.json"
    return default_output_name(
        data_path,
        prompt_name,
        args.method,
        args.k,
        generation_config,
        args.thinking,
        args.seed,
    )


def stable_train_hash(examples):
    payload = "\n".join(
        f"{example['id']}\t{example.get('language', '')}\t{example['text']}"
        for example in examples
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_embedding_model(model_path):
    model_path = Path(model_path)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Embedding model not found in {model_path}. "
            "Run notebooks/download_qwen_embedding_model.ipynb first."
        )

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        model = AutoModel.from_pretrained(model_path, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModel.from_pretrained(model_path, **kwargs)

    model.eval()
    return model, tokenizer


def last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def query_instruct(text):
    task = (
        "Given a restaurant review in English or Spanish, retrieve training reviews "
        "that are useful as in-context examples for extracting all supported "
        "aspect-polarity labels as strict JSON."
    )
    return f"Instruct: {task}\nQuery: {text}"


def embedding_text(example, is_query=False):
    text = (
        f"Review language: {example.get('language', 'unknown')}\n"
        f"Review text: {example['text']}"
    )
    if is_query:
        return query_instruct(text)
    return text


def encode_texts_for_embeddings(
    model,
    tokenizer,
    texts,
    batch_size=16,
    max_length=512,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    vectors = []
    device = model.device
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        vectors.append(embeddings.detach().float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def default_embedding_cache_path(args, train_examples):
    model_name = Path(args.embedding_model_path).name
    train_hash = stable_train_hash(train_examples)
    return FEWSHOT_CACHE_DIR / f"{model_name}.{args.train_data}.{train_hash}.npz"


def load_or_build_train_embeddings(args, train_examples):
    import numpy as np

    cache_path = Path(args.embedding_cache) if args.embedding_cache else default_embedding_cache_path(args, train_examples)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = [example["id"] for example in train_examples]

    if cache_path.exists() and not args.force_recompute_embeddings:
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["ids"].tolist()
        if cached_ids == expected_ids:
            print(f"Loaded train embeddings from {cache_path}")
            return cached["embeddings"], cache_path
        print(f"Ignoring stale embedding cache: {cache_path}")

    print("Loading embedding model")
    t0 = time.time()
    embedding_model, embedding_tokenizer = load_embedding_model(args.embedding_model_path)
    print(f"Embedding model loading took {time.time() - t0:.1f} seconds")

    texts = [embedding_text(example, is_query=False) for example in train_examples]
    t0 = time.time()
    embeddings = encode_texts_for_embeddings(
        embedding_model,
        embedding_tokenizer,
        texts,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
    )
    print(f"Encoded {len(texts)} train examples in {time.time() - t0:.1f} seconds")

    np.savez_compressed(
        cache_path,
        ids=np.array(expected_ids),
        embeddings=embeddings,
    )
    print(f"Saved train embeddings to {cache_path}")

    del embedding_model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings, cache_path


def load_query_embedding_model(args):
    print("Loading embedding model for query encoding")
    t0 = time.time()
    model, tokenizer = load_embedding_model(args.embedding_model_path)
    print(f"Embedding model loading took {time.time() - t0:.1f} seconds")
    return model, tokenizer


def detect_aspect_cues(text):
    normalized = normalize_text(text)
    cues = set()
    for aspect, terms in ASPECT_CUES.items():
        if any(term in normalized for term in terms):
            cues.add(aspect)
    return cues


def detect_polarity_cues(text):
    normalized = normalize_text(text)
    cues = set()
    for polarity, terms in POLARITY_CUES.items():
        if any(term in normalized for term in terms):
            cues.add(polarity)
    return cues


def is_question_or_comment(example):
    normalized = normalize_text(example["text"])
    question_markers = ["?", "hola", "gracias", "alguien", "puede", "duda", "un saludo"]
    return any(marker in normalized for marker in question_markers)


def minmax_normalize(values):
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def aspect_cue_coverage(query_cues, candidate):
    if not query_cues:
        return 0.0
    candidate_aspects = set(candidate.get("gold", {}).keys())
    return len(query_cues & candidate_aspects) / len(query_cues)


def rare_or_hard_label_helpfulness(candidate, query_cues, polarity_cues):
    gold = candidate.get("gold", {})
    if not gold:
        return 0.0

    score = 0.0
    for aspect, polarity in gold.items():
        if aspect in HARD_ASPECTS:
            score += 1.0
        if aspect in query_cues:
            score += 0.5
        if polarity in {"neutral", "conflict"}:
            score += 0.8
        if polarity in polarity_cues:
            score += 0.3
    return min(1.0, score / max(1.0, len(gold) * 1.3))


def length_label_count_fit(query, candidate, query_cues):
    query_words = max(1, len(query["text"].split()))
    candidate_words = max(1, len(candidate["text"].split()))
    length_ratio = abs(math.log(query_words / candidate_words))
    length_score = max(0.0, 1.0 - length_ratio / math.log(4.0))

    estimated_labels = max(1, min(6, len(query_cues) or 3))
    candidate_labels = len(candidate.get("gold", {}))
    label_gap = abs(candidate_labels - estimated_labels)
    label_score = max(0.0, 1.0 - label_gap / max(candidate_labels, estimated_labels, 1))
    return 0.5 * length_score + 0.5 * label_score


def lambda_for_k(k, override=None):
    if override is not None:
        if not 0.0 <= override <= 1.0:
            raise ValueError("--mmr-lambda must be between 0 and 1")
        return override
    if k in LAMBDA_BY_K:
        return LAMBDA_BY_K[k]
    if k <= 1:
        return 1.0
    if k <= 3:
        return 0.84
    if k <= 5:
        return 0.78
    if k <= 7:
        return 0.74
    if k <= 9:
        return 0.71
    return 0.68


def compute_absa_base_scores(
    query,
    candidates,
    candidate_sims,
):
    query_cues = detect_aspect_cues(query["text"])
    polarity_cues = detect_polarity_cues(query["text"])
    normalized_sims = minmax_normalize(candidate_sims)
    scores = []

    for candidate, semantic in zip(candidates, normalized_sims):
        same_language = 1.0 if candidate.get("language") == query.get("language") else 0.0
        score = (
            BASE_SCORE_WEIGHTS["semantic"] * semantic
            + BASE_SCORE_WEIGHTS["same_language"] * same_language
            + BASE_SCORE_WEIGHTS["aspect_cue_coverage"] * aspect_cue_coverage(query_cues, candidate)
            + BASE_SCORE_WEIGHTS["rare_or_hard_label_helpfulness"]
            * rare_or_hard_label_helpfulness(candidate, query_cues, polarity_cues)
            + BASE_SCORE_WEIGHTS["length_label_count_fit"]
            * length_label_count_fit(query, candidate, query_cues)
        )
        scores.append(score)
    return scores


def top_candidate_indices(query, train_examples, train_embeddings, query_embedding, n):
    import numpy as np

    similarities = train_embeddings @ query_embedding
    order = np.argsort(-similarities)
    selected = []
    for idx in order:
        idx = int(idx)
        if train_examples[idx]["id"] == query.get("id"):
            continue
        selected.append(idx)
        if len(selected) >= n:
            break
    return selected, similarities


def select_dense_topk(query, train_examples, train_embeddings, query_embedding, k, args):
    if k <= 0:
        return []
    candidate_indices, similarities = top_candidate_indices(
        query,
        train_examples,
        train_embeddings,
        query_embedding,
        max(k, args.retrieve_candidates),
    )
    selected = candidate_indices[:k]
    selected.sort(key=lambda idx: similarities[idx])
    return [
        {
            "example": train_examples[idx],
            "selection": {
                "method": "dense_topk",
                "semantic_similarity": float(similarities[idx]),
            },
        }
        for idx in selected
    ]


def select_absa_mmr(
    query,
    train_examples,
    train_embeddings,
    query_embedding,
    k,
    args,
    exclude_ids=None,
):
    if k <= 0:
        return []

    import numpy as np

    exclude_ids = set(exclude_ids or [])
    candidate_indices, similarities = top_candidate_indices(
        query,
        train_examples,
        train_embeddings,
        query_embedding,
        max(args.retrieve_candidates, k),
    )
    candidate_indices = [
        idx for idx in candidate_indices if train_examples[idx]["id"] not in exclude_ids
    ]
    candidates = [train_examples[idx] for idx in candidate_indices]
    candidate_sims = [float(similarities[idx]) for idx in candidate_indices]
    base_scores = compute_absa_base_scores(query, candidates, candidate_sims)

    selected = []
    selected_indices = []
    available = list(range(len(candidate_indices)))
    lambda_k = lambda_for_k(k, args.mmr_lambda)

    while available and len(selected) < k:
        best_position = None
        best_score = None
        best_redundancy = 0.0

        for position in available:
            train_idx = candidate_indices[position]
            redundancy = 0.0
            if selected_indices:
                redundancy = max(
                    float(train_embeddings[train_idx] @ train_embeddings[selected_idx])
                    for selected_idx in selected_indices
                )
            if (
                selected_indices
                and redundancy > args.redundancy_threshold
                and len(available) > (k - len(selected))
            ):
                continue

            score = lambda_k * base_scores[position] - (1.0 - lambda_k) * max(0.0, redundancy)
            if best_score is None or score > best_score:
                best_score = score
                best_position = position
                best_redundancy = redundancy

        if best_position is None:
            best_position = available[0]
            train_idx = candidate_indices[best_position]
            if selected_indices:
                best_redundancy = max(
                    float(train_embeddings[train_idx] @ train_embeddings[selected_idx])
                    for selected_idx in selected_indices
                )
            best_score = lambda_k * base_scores[best_position] - (
                1.0 - lambda_k
            ) * max(0.0, best_redundancy)

        train_idx = candidate_indices[best_position]
        selected_indices.append(train_idx)
        selected.append(
            {
                "example": train_examples[train_idx],
                "selection": {
                    "method": "absa_mmr",
                    "semantic_similarity": float(similarities[train_idx]),
                    "base_score": float(base_scores[best_position]),
                    "mmr_score": float(best_score),
                    "redundancy": float(best_redundancy),
                    "lambda": float(lambda_k),
                },
            }
        )
        available.remove(best_position)

    selected.sort(key=lambda item: item["selection"]["semantic_similarity"])
    return selected


def language_balanced_random_examples(train_examples, k, seed):
    rng = random.Random(seed)
    by_language = defaultdict(list)
    for example in train_examples:
        by_language[example.get("language", "unknown")].append(example)
    for examples in by_language.values():
        rng.shuffle(examples)

    languages = sorted(by_language, key=lambda lang: len(by_language[lang]), reverse=True)
    selected = []
    cursor = {language: 0 for language in languages}
    while len(selected) < k and languages:
        progressed = False
        for language in languages:
            examples = by_language[language]
            if cursor[language] < len(examples):
                selected.append(examples[cursor[language]])
                cursor[language] += 1
                progressed = True
                if len(selected) >= k:
                    break
        if not progressed:
            break
    return selected


def select_random_fixed(query, fixed_examples, k):
    selected = [example for example in fixed_examples if example["id"] != query.get("id")]
    return [
        {
            "example": example,
            "selection": {"method": "random_fixed"},
        }
        for example in selected[:k]
    ]


def select_random_dynamic(query, train_examples, k, seed, index):
    rng = random.Random(seed + index * 1009)
    pool = [example for example in train_examples if example["id"] != query.get("id")]
    selected = rng.sample(pool, min(k, len(pool)))
    return [
        {
            "example": example,
            "selection": {"method": "random_dynamic"},
        }
        for example in selected
    ]


def manual_bank_examples(train_examples):
    by_id = {example["id"]: example for example in train_examples}
    missing = [entry["id"] for entry in MANUAL_HARD_BANK if entry["id"] not in by_id]
    if missing:
        raise ValueError(f"Missing manual hard bank examples in train data: {missing}")
    examples = []
    for entry in MANUAL_HARD_BANK:
        example = dict(by_id[entry["id"]])
        example["_manual_notes"] = entry["notes"]
        examples.append(example)
    return examples


def select_manual_fixed_hard(query, manual_examples, k):
    selected = [example for example in manual_examples if example["id"] != query.get("id")]
    if k > len(selected):
        raise ValueError(f"manual_fixed_hard supports at most {len(selected)} examples")
    return [
        {
            "example": example,
            "selection": {
                "method": "manual_fixed_hard",
                "manual_notes": example.get("_manual_notes", ""),
            },
        }
        for example in selected[:k]
    ]


def manual_hard_score(query, example):
    query_cues = detect_aspect_cues(query["text"])
    polarity_cues = detect_polarity_cues(query["text"])
    gold = example.get("gold", {})
    score = 0.0
    score += 2.0 * len(query_cues & set(gold.keys()))
    score += 0.5 if example.get("language") == query.get("language") else 0.0
    score += sum(0.5 for polarity in gold.values() if polarity in polarity_cues)
    score += sum(0.4 for aspect in gold if aspect in HARD_ASPECTS)
    score += sum(0.3 for polarity in gold.values() if polarity in {"neutral", "conflict"})
    if not gold and is_question_or_comment(query):
        score += 5.0
    if not gold and not is_question_or_comment(query):
        score -= 2.0
    return score


def select_hard_mix(
    query,
    manual_examples,
    train_examples,
    train_embeddings,
    query_embedding,
    k,
    args,
):
    if k <= 0:
        return []

    manual_count = args.hard_mix_manual_count
    if manual_count is None:
        manual_count = 1 if k <= 4 else 2
    manual_count = min(manual_count, k, len(manual_examples))

    scored_manual = []
    for example in manual_examples:
        if example["id"] == query.get("id"):
            continue
        scored_manual.append((manual_hard_score(query, example), example))
    scored_manual.sort(key=lambda item: item[0], reverse=True)
    manual_selected = [
        {
            "example": example,
            "selection": {
                "method": "hard_mix_manual",
                "manual_score": float(score),
                "manual_notes": example.get("_manual_notes", ""),
            },
        }
        for score, example in scored_manual[:manual_count]
    ]

    exclude_ids = {item["example"]["id"] for item in manual_selected}
    retrieved = select_absa_mmr(
        query,
        train_examples,
        train_embeddings,
        query_embedding,
        k - len(manual_selected),
        args,
        exclude_ids=exclude_ids,
    )
    for item in retrieved:
        item["selection"]["method"] = "hard_mix_absa_mmr"
    return manual_selected + retrieved


def selected_demo_metadata(selected, include_text=True):
    output = []
    for rank, item in enumerate(selected, start=1):
        example = item["example"]
        record = {
            "rank": rank,
            "id": example["id"],
            "language": example.get("language"),
            "gold": example.get("gold", {}),
            "selection": item.get("selection", {}),
        }
        if include_text:
            record["text"] = example["text"]
        output.append(record)
    return output


def selected_examples_only(selected):
    return [item["example"] for item in selected]


def mean(values):
    return sum(values) / len(values) if values else None


def summarize_selection(results):
    selected_total = 0
    method_counts = Counter()
    selected_frequency = Counter()
    semantic_similarities = []
    redundancies = []
    base_scores = []
    mmr_scores = []

    for result in results:
        demos = result.get("fewshot_examples", [])
        selected_total += len(demos)
        for demo in demos:
            selected_frequency[demo["id"]] += 1
            selection = demo.get("selection", {})
            method_counts[selection.get("method", "unknown")] += 1
            if "semantic_similarity" in selection:
                semantic_similarities.append(selection["semantic_similarity"])
            if "redundancy" in selection:
                redundancies.append(selection["redundancy"])
            if "base_score" in selection:
                base_scores.append(selection["base_score"])
            if "mmr_score" in selection:
                mmr_scores.append(selection["mmr_score"])

    return {
        "selected_total": selected_total,
        "avg_k": selected_total / len(results) if results else 0.0,
        "selection_method_counts": dict(method_counts),
        "avg_semantic_similarity": mean(semantic_similarities),
        "avg_redundancy": mean(redundancies),
        "avg_base_score": mean(base_scores),
        "avg_mmr_score": mean(mmr_scores),
        "top_selected_example_ids": [
            {"id": example_id, "count": count}
            for example_id, count in selected_frequency.most_common(20)
        ],
    }


def summarize_results(results, config):
    predicted_total = expected_total = ok_total = 0
    precision_macro = recall_macro = f1_macro = 0.0
    aspect_counts = {
        aspect: Counter({"predicted": 0, "expected": 0, "ok": 0})
        for aspect in ASPECTS
    }
    polarity_counts = defaultdict(lambda: Counter({"predicted": 0, "expected": 0, "ok": 0}))
    empty_predictions = 0
    parse_ok = 0
    hit_token_limit = 0
    prompt_tokens = []
    output_tokens = []

    for example in results:
        prediction = example.get("prediction", {})
        gold = example.get("gold", {})
        predicted, expected, ok = counts(prediction, gold)
        predicted_total += predicted
        expected_total += expected
        ok_total += ok

        precision, recall, f1 = prf(predicted, expected, ok)
        precision_macro += precision
        recall_macro += recall
        f1_macro += f1

        if not prediction:
            empty_predictions += 1
        if example.get("json_parse_ok"):
            parse_ok += 1
        metadata = example.get("generation_metadata", {})
        if metadata.get("hit_token_limit"):
            hit_token_limit += 1
        if "prompt_tokens" in metadata:
            prompt_tokens.append(metadata["prompt_tokens"])
        if "output_tokens" in metadata:
            output_tokens.append(metadata["output_tokens"])

        pred_items = set(prediction.items())
        gold_items = set(gold.items())
        for aspect, polarity in pred_items:
            aspect_counts[aspect]["predicted"] += 1
            polarity_counts[polarity]["predicted"] += 1
        for aspect, polarity in gold_items:
            aspect_counts[aspect]["expected"] += 1
            polarity_counts[polarity]["expected"] += 1
        for aspect, polarity in pred_items & gold_items:
            aspect_counts[aspect]["ok"] += 1
            polarity_counts[polarity]["ok"] += 1

    n = len(results)
    if n:
        precision_macro /= n
        recall_macro /= n
        f1_macro /= n
    precision_micro, recall_micro, f1_micro = prf(
        predicted_total,
        expected_total,
        ok_total,
    )

    def metric_dict(counter):
        precision, recall, f1 = prf(
            counter["predicted"],
            counter["expected"],
            counter["ok"],
        )
        return {
            "predicted": counter["predicted"],
            "expected": counter["expected"],
            "ok": counter["ok"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    summary = {
        "config": config,
        "examples": n,
        "predicted_total": predicted_total,
        "expected_total": expected_total,
        "ok_total": ok_total,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "empty_prediction_count": empty_predictions,
        "json_parse_ok_count": parse_ok,
        "json_parse_ok_rate": 100.0 * parse_ok / n if n else 0.0,
        "hit_token_limit_count": hit_token_limit,
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "per_aspect": {aspect: metric_dict(counter) for aspect, counter in aspect_counts.items()},
        "per_polarity": {
            polarity: metric_dict(counter)
            for polarity, counter in sorted(polarity_counts.items())
        },
        "selection": summarize_selection(results),
    }
    return summary


def print_summary(summary):
    print("========= SUMMARY =========")
    print(
        f"Micro P/R/F: {summary['precision_micro']:.2f} "
        f"{summary['recall_micro']:.2f} {summary['f1_micro']:.2f}"
    )
    print(
        f"Macro P/R/F: {summary['precision_macro']:.2f} "
        f"{summary['recall_macro']:.2f} {summary['f1_macro']:.2f}"
    )
    print(
        f"Pred/Gold/OK: {summary['predicted_total']} "
        f"{summary['expected_total']} {summary['ok_total']}"
    )
    print(
        f"JSON parse ok: {summary['json_parse_ok_count']}/"
        f"{summary['examples']} ({summary['json_parse_ok_rate']:.1f}%)"
    )
    print(f"Empty predictions: {summary['empty_prediction_count']}")
    print(f"Hit token limit: {summary['hit_token_limit_count']}")
    selection = summary.get("selection", {})
    if selection:
        print(f"Average selected K: {selection['avg_k']:.2f}")


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_fd:
        json.dump(value, output_fd, indent=2, ensure_ascii=False)


def load_resume_records(output_path):
    if not output_path.exists():
        return []
    with open(output_path, encoding="utf-8") as input_fd:
        records = json.load(input_fd)
    if not isinstance(records, list):
        raise ValueError(f"Cannot resume from non-list output: {output_path}")
    return records


def should_checkpoint(args, processed):
    return (
        args.checkpoint_every
        and args.checkpoint_every > 0
        and len(processed) % args.checkpoint_every == 0
    )


def main():
    args = parse_args()
    if args.k < 0:
        raise ValueError("--k must be non-negative")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")

    generation_config = resolved_generation_config(args)
    prompts = get_prompts(args.prompt_file)
    examples, data_path = load_dataset(args.data)
    train_examples, train_path = load_dataset(args.train_data)
    manual_examples = manual_bank_examples(train_examples)

    if args.limit is not None:
        selected_targets = examples[args.start : args.start + args.limit]
    else:
        selected_targets = examples[args.start :]

    output_path = output_path_from_args(args, data_path, prompts["name"], generation_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    needs_embeddings = args.method in {"dense_topk", "absa_mmr", "hard_mix"}
    train_embeddings = None
    embedding_cache_path = None
    embedding_model = embedding_tokenizer = None
    if needs_embeddings:
        train_embeddings, embedding_cache_path = load_or_build_train_embeddings(
            args,
            train_examples,
        )
        embedding_model, embedding_tokenizer = load_query_embedding_model(args)

    fixed_random_examples = []
    if args.method == "random_fixed":
        fixed_random_examples = language_balanced_random_examples(
            train_examples,
            max(args.k, 12),
            args.seed,
        )

    config = {
        "method": args.method,
        "k": args.k,
        "seed": args.seed,
        "model_path": str(args.model_path),
        "embedding_model_path": str(args.embedding_model_path),
        "embedding_cache": str(embedding_cache_path) if embedding_cache_path else None,
        "prompt_file": prompts["path"],
        "data": str(data_path),
        "train_data": str(train_path),
        "thinking": args.thinking,
        "generation_config": generation_config,
        "retrieve_candidates": args.retrieve_candidates,
        "redundancy_threshold": args.redundancy_threshold,
        "mmr_lambda_override": args.mmr_lambda,
        "base_score_weights": BASE_SCORE_WEIGHTS,
        "lambda_by_k": LAMBDA_BY_K,
        "manual_hard_bank": MANUAL_HARD_BANK,
        "start": args.start,
        "limit": args.limit,
        "resume": args.resume,
        "checkpoint_every": args.checkpoint_every,
    }

    print("========= ABSA FEW SHOT =========")
    print(f"model_path={args.model_path}")
    print(f"embedding_model_path={args.embedding_model_path}")
    print(f"prompt_file={prompts['path']}")
    print(f"data={data_path}")
    print(f"train_data={train_path}")
    print(f"method={args.method}")
    print(f"k={args.k}")
    print(f"seed={args.seed}")
    print(f"thinking={args.thinking}")
    print(f"generation={generation_config}")
    print(f"examples={len(selected_targets)}")
    print(f"output={output_path}")

    processed = load_resume_records(output_path) if args.resume else []
    resumed_ids = {record.get("id") for record in processed}
    initial_processed_count = len(processed)
    if processed:
        print(f"Resuming from {len(processed)} existing records")

    t0 = time.time()
    model, tokenizer = load_model(args.model_path, args.load_in_4bit)
    print(f"Generator loading took {time.time() - t0:.1f} seconds")

    from common import generate_with_metadata

    t0 = time.time()
    for index, example in enumerate(selected_targets, start=args.start):
        if args.resume and example["id"] in resumed_ids:
            print(f"Skipping existing example {index}: {example['id']}", flush=True)
            continue

        print(f"Processing example {index}: {example['id']}", flush=True)
        query_embedding = None
        if needs_embeddings:
            query_embedding = encode_texts_for_embeddings(
                embedding_model,
                embedding_tokenizer,
                [embedding_text(example, is_query=True)],
                batch_size=1,
                max_length=args.embedding_max_length,
            )[0]

        if args.method == "random_fixed":
            selected = select_random_fixed(example, fixed_random_examples, args.k)
        elif args.method == "random_dynamic":
            selected = select_random_dynamic(
                example,
                train_examples,
                args.k,
                args.seed,
                index,
            )
        elif args.method == "dense_topk":
            selected = select_dense_topk(
                example,
                train_examples,
                train_embeddings,
                query_embedding,
                args.k,
                args,
            )
        elif args.method == "absa_mmr":
            selected = select_absa_mmr(
                example,
                train_examples,
                train_embeddings,
                query_embedding,
                args.k,
                args,
            )
        elif args.method == "manual_fixed_hard":
            selected = select_manual_fixed_hard(example, manual_examples, args.k)
        elif args.method == "hard_mix":
            selected = select_hard_mix(
                example,
                manual_examples,
                train_examples,
                train_embeddings,
                query_embedding,
                args.k,
                args,
            )
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        demo_examples = selected_examples_only(selected)
        messages = prepare_fewshot_messages(prompts, example, demo_examples)
        model_inputs = encode(tokenizer, messages, enable_thinking=args.thinking)
        generation = generate_with_metadata(
            model,
            tokenizer,
            model_inputs,
            generation_config,
        )
        prediction, parse_ok = extract_json_with_status(generation["text"])

        result = dict(example)
        result["prediction"] = prediction
        result["json_parse_ok"] = parse_ok
        result["fewshot_examples"] = selected_demo_metadata(
            selected,
            include_text=not args.no_demo_text,
        )
        result["generation_metadata"] = {
            "prompt_tokens": generation["prompt_tokens"],
            "output_tokens": generation["output_tokens"],
            "ended_with_eos": generation["ended_with_eos"],
            "hit_token_limit": generation["hit_token_limit"],
        }
        if args.keep_raw:
            result["raw_generation"] = generation["text"]
            result["raw_generation_with_special_tokens"] = generation[
                "text_with_special_tokens"
            ]
        if args.save_prompts:
            result["messages"] = messages
        processed.append(result)
        if should_checkpoint(args, processed):
            write_json(output_path, processed)

    elapsed = time.time() - t0
    new_processed_count = len(processed) - initial_processed_count
    config["time_sec"] = elapsed
    config["new_examples"] = new_processed_count
    config["sec_per_new_example"] = (
        elapsed / new_processed_count if new_processed_count else 0.0
    )
    summary = summarize_results(processed, config)

    write_json(output_path, processed)

    if not args.no_summary:
        summary_path = output_path.with_suffix(".summary.json")
        write_json(summary_path, summary)
        print(f"Summary: {summary_path}")

    print("Done")
    print(f"Processed {new_processed_count} new examples in {elapsed:.1f} seconds")
    print(f"Total records in output: {len(processed)}")
    if new_processed_count:
        print(f"{elapsed / new_processed_count:.2f} sec/new example")
    print(f"Output: {output_path}")
    print_summary(summary)

    del model
    if embedding_model is not None:
        del embedding_model

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
