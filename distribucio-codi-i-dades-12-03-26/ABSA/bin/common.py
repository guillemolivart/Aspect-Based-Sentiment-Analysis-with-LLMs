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
    return generate_with_metadata(model, tokenizer, model_inputs, generation_config)["text"]


def generate_with_metadata(model, tokenizer, model_inputs, generation_config):
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
        if generation_config.get("min_p") is not None:
            kwargs["min_p"] = generation_config["min_p"]
    else:
        kwargs["do_sample"] = False

    repetition_penalty = generation_config.get("repetition_penalty")
    if repetition_penalty is not None and repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty

    presence_penalty = generation_config.get("presence_penalty")
    if presence_penalty is not None and presence_penalty != 0.0:
        from transformers import LogitsProcessorList

        kwargs["logits_processor"] = LogitsProcessorList(
            [
                PresencePenaltyLogitsProcessor(
                    presence_penalty,
                    prompt_length=model_inputs["input_ids"].shape[-1],
                )
            ]
        )

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **kwargs)

    prompt_len = model_inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][prompt_len:]
    output_token_ids = output_ids.tolist()
    eos_token_id = tokenizer.eos_token_id
    eos_token_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    ended_with_eos = bool(output_token_ids and output_token_ids[-1] in eos_token_ids)
    return {
        "text": tokenizer.decode(output_ids, skip_special_tokens=True),
        "text_with_special_tokens": tokenizer.decode(output_ids, skip_special_tokens=False),
        "prompt_tokens": prompt_len,
        "output_tokens": len(output_token_ids),
        "output_token_ids": output_token_ids,
        "ended_with_eos": ended_with_eos,
        "hit_token_limit": len(output_token_ids) >= generation_config["max_new_tokens"]
        and not ended_with_eos,
    }


class PresencePenaltyLogitsProcessor:
    def __init__(self, penalty, prompt_length):
        self.penalty = float(penalty)
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids, scores):
        import torch

        if self.penalty == 0.0:
            return scores

        for row, sequence in enumerate(input_ids):
            generated = sequence[self.prompt_length :]
            if generated.numel() == 0:
                continue
            seen_tokens = torch.unique(generated)
            scores[row, seen_tokens] -= self.penalty
        return scores


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
        for candidate in _repair_jsonish(fragment):
            try:
                obj, _ = decoder.raw_decode(candidate)
                yield obj
                continue
            except json.JSONDecodeError:
                pass

            end = max(candidate.rfind("}"), candidate.rfind("]"))
            if end == -1:
                continue
            try:
                obj = ast.literal_eval(candidate[: end + 1])
            except (SyntaxError, ValueError):
                continue
            if isinstance(obj, (dict, list)):
                yield obj


def _repair_jsonish(fragment):
    yield fragment

    repaired = re.sub(
        r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:',
        r'\1"\2":',
        fragment,
    )
    repaired = re.sub(
        r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        repaired,
    )
    repaired = re.sub(
        r'(:\s*)(positive|negative|neutral|conflict)(\s*[,}\]])',
        r'\1"\2"\3',
        repaired,
    )
    if repaired != fragment:
        yield repaired


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
