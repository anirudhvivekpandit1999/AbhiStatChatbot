from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
import os

from app import TransformerIntentClassifier, SimpleTokenizer, MAX_LEN

app = Flask(__name__)
CORS(app)

MODEL_PATH = "intent_model.pth"
model_loaded = False


def load_model():
    global model, tokenizer, idx_to_intent, model_loaded

    if not os.path.exists(MODEL_PATH):
        print("Train model first.")
        return

    checkpoint = torch.load(MODEL_PATH)

    tokenizer = SimpleTokenizer()
    tokenizer.word2idx = checkpoint["tokenizer"]
    tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}
    tokenizer.vocab_size = len(tokenizer.word2idx)

    idx_to_intent = checkpoint["idx_to_intent"]

    model = TransformerIntentClassifier(
        tokenizer.vocab_size,
        len(idx_to_intent)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    model_loaded = True
    print("Model loaded.")


load_model()


@app.route("/predict-intent", methods=["POST"])
def predict_intent():

    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Empty text"}), 400

    with torch.no_grad():
        encoded = torch.tensor([tokenizer.encode(text)])
        logits = model(encoded)
        probs = torch.sigmoid(logits)[0]

    intents_payload = []
    for i, p in enumerate(probs):
        confidence = float(p)
        if confidence > 0.3:
            intents_payload.append({
                "intent": idx_to_intent[i],
                "confidence": confidence
            })

    intents_payload.sort(key=lambda x: x["confidence"], reverse=True)

    # ===== ENTITY EXTRACTION =====

    entity_data = {}

    name_match = re.search(
        r"(?:call|name|rename).*?(?:excel|sheet)?\s*(?:to)?\s*([a-zA-Z0-9_ -]+)",
        text,
        re.IGNORECASE
    )
    if name_match:
        entity_data["new_sheet_name"] = name_match.group(1).strip()

    base_match = re.search(
        r"(?:select|choose|use).*?(?:sheet)?\s*(?:to)?\s*([a-zA-Z0-9_ -]+)",
        text,
        re.IGNORECASE
    )
    if base_match:
        entity_data["base_sheet"] = base_match.group(1).strip()

    return jsonify({
        "text": text,
        "top_intent": intents_payload[0]["intent"] if intents_payload else None,
        "intents": intents_payload,
        "entities": entity_data
    })


if __name__ == "__main__":
    app.run(port=8000, debug=True)
