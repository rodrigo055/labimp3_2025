import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 📦 Modelo Hugging Face instruct
MODEL_ID = "EleutherAI/pythia-70m"

# Cache para no recargar modelo cada vez
_model_cache = {}

def load_model():
    if MODEL_ID not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            offload_folder="./offload"
        )
        model.eval()
        _model_cache[MODEL_ID] = (tokenizer, model)
    return _model_cache[MODEL_ID]

# 🔣 Tokenización estilo LLMTIME
def normalize_and_tokenize(values, precision=3, alpha=0.99, beta=0.0):
    offset = np.percentile(values, beta * 100)
    scale = np.percentile(values - offset, alpha * 100)
    scale = scale if scale > 0 else 1
    normalized = (values - offset) / scale
    tokens = [f"{v:.{precision}f}".replace(".", "").rjust(precision + 1, "0") for v in normalized]
    return tokens, offset, scale

def tokens_to_prompt(tokens):
    return " , ".join(" ".join(list(token)) for token in tokens)

# 🔁 Decodificador robusto
def decode_completion(output_text, scale, offset):
    match = re.search(r"(?:\d\s*){3,}", output_text)
    if not match:
        return None
    digits = re.findall(r"\d", match.group())
    if not digits:
        return None
    pred_normalized = float("0." + "".join(digits[:3]))
    return pred_normalized * scale + offset

# 🎯 Función principal
def predict_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date")
    serie = df.groupby("date")["tn"].sum()

    train_vals = serie.values[:-1]
    tokens, offset, scale = normalize_and_tokenize(train_vals)
    prompt_body = tokens_to_prompt(tokens)
    prompt = (
        "The recent tn values are:\n"
        f"{prompt_body}\n"
        "What is the next tn value?\n"
        "Only respond with the number using digits, without explanation:"
    )

    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,         # ← greedy decoding
        num_return_sequences=1   # ← solo una predicción
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("📋 Prompt:\n", prompt)
    print("🧠 Raw output:\n", decoded)

    tn_pred = decode_completion(decoded, scale, offset)
    return tn_pred
