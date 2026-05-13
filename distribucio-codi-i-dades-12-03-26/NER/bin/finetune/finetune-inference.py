import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if os.path.exists(model_path + "/adapter_config.json"):
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model, tokenizer


def main():
    model_dir = sys.argv[1] if len(sys.argv)>1 else "FT.weights"
    model, tokenizer = load_model(model_dir)
    print(f"Loaded model from {model_dir}")

    # read input examples from stdin, one example per line as json
    for line in sys.stdin:
        ex = json.loads(line.strip())
        # build prompt
        prompt = ex.get("text", "")
        # encode and generate
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=256)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(text)


if __name__ == '__main__':
    main()
