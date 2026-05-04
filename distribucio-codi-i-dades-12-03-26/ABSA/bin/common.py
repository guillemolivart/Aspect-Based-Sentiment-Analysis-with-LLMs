import ast
import json
import re
from pathlib import Path


ABSA_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ABSA_DIR / "dataset"
DEFAULT_MODEL_PATH = ABSA_DIR / "model"
DEFAULT_PROMPT_PATH = ABSA_DIR / "prompts" / "absa_v1.json"
OUTPUT_DIR = ABSA_DIR / "outputs"

ASPECTS = [
    "restaurant_general",
    "restaurant_prices",
    "food_quality",
    "food_prices",
    "food_style_options",
    "drinks_quality",
    "drinks_prices",
    "drinks_style_options",
    "ambience",
    "service",
    "location",
]

POLARITIES = ["positive", "negative", "neutral", "conflict"]


def resolve_path(path, base=ABSA_DIR):
    path = Path(path)
    if path.is_absolute():
        return path
    return base / path


def get_prompts(prompt_file=DEFAULT_PROMPT_PATH):
    prompt_path = resolve_path(prompt_file)
    with open(prompt_path, encoding="utf-8") as prompt_fd:
        prompts = json.load(prompt_fd)

    system = prompts.get("system") or prompts.get("sysprompt")
    user = prompts.get("user") or prompts.get("usrprompt")
    if not system or not user:
        raise ValueError(f"{prompt_path} must define 'system' and 'user' prompts")

    return {
        "name": prompts.get("name", prompt_path.stem),
        "system": system,
        "user": user,
        "path": str(prompt_path),
    }


def load_dataset(data):
    data_path = Path(data)
    if not data_path.suffix:
        data_path = DATASET_DIR / f"{data}.json"
    else:
        data_path = resolve_path(data)

    with open(data_path, encoding="utf-8") as data_fd:
        examples = json.load(data_fd)
    return examples, data_path


def render_template(template, values):
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def prepare_messages(prompts, question):
    values = {
        "text": question["text"],
        "language": question.get("language", "unknown"),
        "aspects": ", ".join(ASPECTS),
        "polarities": ", ".join(POLARITIES),
    }
    return [
        {"role": "system", "content": render_template(prompts["system"], values)},
        {"role": "user", "content": render_template(prompts["user"], values)},
    ]


def encode(tokenizer, messages, enable_thinking=False):
    import torch

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    tokens = tokenizer([text], return_tensors="pt")
    return tokens.to("cuda" if torch.cuda.is_available() else "cpu")


def generate(model, tokenizer, model_inputs, generation_config):
    import torch

    kwargs = {
        "max_new_tokens": generation_config["max_new_tokens"],
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    temperature = generation_config["temperature"]
    if temperature and temperature > 0:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": generation_config["top_p"],
            }
        )
        if generation_config["top_k"] is not None:
            kwargs["top_k"] = generation_config["top_k"]
    else:
        kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **kwargs)

    prompt_len = model_inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][prompt_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def strip_thinking(text):
    if "</think>" in text:
        return text.split("</think>")[-1]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _json_candidates(text):
    cleaned = (
        text.replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .replace("“", '"')
        .replace("”", '"')
    )
    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned):
        if char not in "[{":
            continue
        fragment = cleaned[start:].lstrip()
        try:
            obj, _ = decoder.raw_decode(fragment)
            yield obj
            continue
        except json.JSONDecodeError:
            pass

        end = max(fragment.rfind("}"), fragment.rfind("]"))
        if end == -1:
            continue
        try:
            obj = ast.literal_eval(fragment[: end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(obj, (dict, list)):
            yield obj


def normalize_prediction(prediction):
    normalized = {}

    if isinstance(prediction, list):
        for item in prediction:
            if not isinstance(item, dict):
                continue
            aspect = item.get("aspect") or item.get("category")
            polarity = item.get("polarity") or item.get("sentiment")
            _add_prediction(normalized, aspect, polarity)
        return normalized

    if not isinstance(prediction, dict):
        return normalized

    for aspect, polarity in prediction.items():
        if isinstance(polarity, dict):
            polarity = polarity.get("polarity") or polarity.get("sentiment")
        _add_prediction(normalized, aspect, polarity)
    return normalized


def _add_prediction(output, aspect, polarity):
    if not isinstance(aspect, str) or not isinstance(polarity, str):
        return

    aspect = aspect.strip().lower().replace("-", "_").replace(" ", "_")
    polarity = polarity.strip().lower()
    if aspect in ASPECTS and polarity in POLARITIES:
        output[aspect] = polarity


def extract_json(gen_text):
    search_spaces = [strip_thinking(gen_text), gen_text]
    for text in search_spaces:
        candidates = list(_json_candidates(text))
        for candidate in reversed(candidates):
            prediction = normalize_prediction(candidate)
            if prediction or candidate == {}:
                return prediction
    return {}
