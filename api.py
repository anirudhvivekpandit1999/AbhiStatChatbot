from ast import List
from collections import defaultdict
from typing import List
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import FeedForward, MultiHeadAttention
from app import TransformerLLM
import os
import traceback    
import re
import numpy as np
import sys


app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000","http://localhost:5173"])




class Tokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.word_count = defaultdict(int)
    
    def fit(self, texts: List[str]):
        self.word_to_idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        idx = 4
        
        for text in texts:
            words = self.tokenize_text(text)
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    idx += 1
                self.word_count[word] += 1
        
        self.vocab_size = len(self.word_to_idx)
    
    def tokenize_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\?!\.\,]', '', text)
        return text.split()
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_text(text)
        return [self.word_to_idx.get(word, 3) for word in tokens]  
    
    def decode(self, indices: List[int]) -> str:
        words = [self.idx_to_word.get(idx, "<UNK>") for idx in indices]
        return " ".join(words)
    
    def pad_sequence(self, sequence: List[int], max_len: int) -> List[int]:
        if len(sequence) > max_len:
            return sequence[:max_len]
        return sequence + [0] * (max_len - len(sequence))
    
class TransformerBlock:
    def __init__(self, d_model: int, num_heads: int = 4, d_ff: int = 256):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.gamma1 = np.ones((1, d_model))
        self.beta1 = np.zeros((1, d_model))
        self.gamma2 = np.ones((1, d_model))
        self.beta2 = np.zeros((1, d_model))
        
        self.eps = 1e-6
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return gamma * x_norm + beta
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        attn_output = self.attention.forward(x)
        x = x + attn_output
        x = self.layer_norm(x, self.gamma1, self.beta1)
        
        ff_output = self.feed_forward.forward(x)
        x = x + ff_output
        x = self.layer_norm(x, self.gamma2, self.beta2)
        
        return x



sys.modules['__main__'].Tokenizer = Tokenizer
sys.modules['__main__'].TransformerLLM = TransformerLLM
chatbot = TransformerLLM()
chatbot.load_model("chatbot_model.pkl")

MODEL_PATH = "chatbot_model.pkl"
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
    print("[DEBUG /predict-intent] chatbot :",chatbot)
    print("[DEBUG /predict-intent] model_loaded :",model_loaded)
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    print("[DEBUG /predict-intent] Received data:", data)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    print("[DEBUG /predict-intent] Processing text:", text)
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        
        matches = chatbot.classify_intent(text) 
        print(f"[DEBUG /predict-intent] Matches for '{text}':", matches)

        if not matches:
            return jsonify({
                "text": text,
                "intents": [],
                "top_intent": None
            })

        intents_payload = []

        for intent, score, match_count in matches:
            intents_payload.append({
                "intent": intent,
                "confidence": float(score),
                "matches": int(match_count),
                "response": chatbot.intent_responses.get(intent)
            })

        return jsonify({
            "text": text,
            "top_intent": intents_payload[0]["intent"],
            "intents": intents_payload
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
