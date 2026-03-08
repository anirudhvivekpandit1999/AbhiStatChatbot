from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import re

from model import TransformerIntentClassifier, SimpleTokenizer, MAX_LEN

app = Flask(__name__)
CORS(app)

MODEL_PATH = "intent_model.pth"
MAX_LEN = 0
model = None
tokenizer = None
idx_to_intent = None


def load_model():
    global model, tokenizer, idx_to_intent, MAX_LEN

    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Run train.py first.")
        return False

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    tokenizer = SimpleTokenizer()
    tokenizer.word2idx = checkpoint["word2idx"]
    tokenizer.idx_to_word = {v: k for k, v in tokenizer.word2idx.items()}
    tokenizer.vocab_size = len(tokenizer.word2idx)

    idx_to_intent = checkpoint["idx_to_intent"]

    MAX_LEN = checkpoint["max_len"]

    model = TransformerIntentClassifier(
        tokenizer.vocab_size,
        len(idx_to_intent)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Model loaded.")
    return True

@app.route("/")
def health():
    return jsonify({"status": "running"})


@app.route("/predict-intent", methods=["POST"])
def predict_intent():
    data = request.get_json()
    text = data.get("text", "")

    with torch.no_grad():
        encoded = tokenizer.encode(text)

        if len(encoded) < MAX_LEN:
            encoded = encoded + [0] * (MAX_LEN - len(encoded))
        else:
            encoded = encoded[:MAX_LEN]

        encoded = torch.tensor([encoded])
        logits = model(encoded)
        probs = torch.sigmoid(logits)[0]

    results = []
    for i, p in enumerate(probs):
        
        results.append({
                "intent": idx_to_intent[i],
                "confidence": round(float(p), 4)
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    return jsonify({
        "text": text,
        "intents": results
    })


if __name__ == "__main__":
    if load_model():
        print("Starting server at http://127.0.0.1:8000")
        app.run(host="0.0.0.0", port=8000, debug=False)
