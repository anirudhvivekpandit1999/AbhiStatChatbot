# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import TransformerLLM
import os
import traceback

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

MODEL_PATH = "chatbot_model.pkl"

chatbot = TransformerLLM(d_model=64, num_heads=4, num_layers=2, max_seq_len=50)
model_loaded = False


def try_load():
    global chatbot, model_loaded
    if os.path.exists(MODEL_PATH):
        try:
            chatbot.load_model(MODEL_PATH)
            model_loaded = True
            print("Chatbot model loaded.")
        except Exception as e:
            model_loaded = False
            print("Warning: failed to load chatbot_model.pkl:", e)
            traceback.print_exc()
    else:
        print(f"Model file {MODEL_PATH} not found. Run train.py to create it.")


try_load()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded})


@app.route("/predict-intent", methods=["POST"])
def predict_intent():
    global chatbot, model_loaded

    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        matches = chatbot.classify_intent(text)  # [(intent, confidence, matches)]

        if not matches:
            return jsonify({
                "text": text,
                "top_intent": None,
                "intents": []
            })

        # =========================
        # INTENT FILTERING RULES
        # =========================
        MIN_MATCHES = 2
        MIN_CONFIDENCE = 0.25
        RELATIVE_THRESHOLD = 0.6  # 60% of top intent confidence

        top_confidence = matches[0][1]

        filtered_intents = []

        for intent, confidence, match_count in matches:
            if match_count < MIN_MATCHES:
                continue
            if confidence < MIN_CONFIDENCE:
                continue
            if confidence < (top_confidence * RELATIVE_THRESHOLD):
                continue

            filtered_intents.append({
                "intent": intent,
                "confidence": float(confidence),
                "matches": int(match_count),
                "response": chatbot.intent_responses.get(intent)
            })

        return jsonify({
            "text": text,
            "top_intent": filtered_intents[0]["intent"] if filtered_intents else None,
            "intents": filtered_intents
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Server error processing intent",
            "details": str(e)
        }), 500


@app.route("/debug-intent", methods=["POST"])
def debug_intent():
    data = request.get_json(force=True)
    text = data.get("text", "")
    chatbot.debug_classify(text)
    return jsonify({"message": "Check server console for debug output"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
