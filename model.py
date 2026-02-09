# model.py
import numpy as np
import re
import math
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict

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
        return [self.word_to_idx.get(word, 3) for word in tokens]  # 3 is UNK token
    
    def decode(self, indices: List[int]) -> str:
        words = [self.idx_to_word.get(idx, "<UNK>") for idx in indices]
        return " ".join(words)
    
    def pad_sequence(self, sequence: List[int], max_len: int) -> List[int]:
        if len(sequence) > max_len:
            return sequence[:max_len]
        return sequence + [0] * (max_len - len(sequence))


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 100):
        self.d_model = d_model
        self.pos_encoding = np.zeros((max_len, d_model))
        
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                sin_coeff = math.sin(pos / (10000 ** (i / d_model)))
                cos_coeff = math.cos(pos / (10000 ** (i / d_model)))
                
                self.pos_encoding[pos, i] = sin_coeff
                if i + 1 < d_model:
                    self.pos_encoding[pos, i + 1] = cos_coeff
    
    def add_encoding(self, embeddings: np.ndarray) -> np.ndarray:
        seq_len = embeddings.shape[0]
        return embeddings + self.pos_encoding[:seq_len]


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int = 4):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.num_heads, self.d_k)
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        matmul_qk = np.matmul(Q, K.transpose(0, 2, 1))
        
        dk = float(self.d_k)
        scaled_attention = matmul_qk / math.sqrt(dk)
        
        attention_weights = np.exp(scaled_attention - np.max(scaled_attention, axis=2, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=2, keepdims=True)
        
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        
        concat_attention = self.combine_heads(scaled_attention)
        
        output = np.matmul(concat_attention, self.W_o)
        
        return output


class FeedForward:
    def __init__(self, d_model: int, d_ff: int = 256):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = self.relu(np.matmul(x, self.W1) + self.b1)
        output = np.matmul(hidden, self.W2) + self.b2
        return output


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


class TransformerLLM:
    def __init__(self):
        self.d_model = 64
        self.num_heads = 4
        self.num_layers = 2
        self.max_seq_len =50
        
        self.tokenizer = Tokenizer()
        self.embedding = None
        self.transformer_blocks = []
        self.output_projection = None
        self.positional_encoding = None
        
        self.learning_rate = 0.001
        self.is_trained = False
        self.train_questions = []
        self.train_responses = []
        
        # Intent recognition data
        self.intent_patterns = {}
        self.intent_responses = {
            "upload_file": "I'll help you upload a file. Please select your Excel spreadsheet.",
            "enter_preprocess": "Moving to preprocessing step. Let's prepare your data.",
            "select_base_sheet": "Great! I've selected this sheet as the base. What would you like to do next?",
            "name_new_sheet": "Please enter a name for your new sheet.",
            "set_row_range": "Select the row range you want to keep (start row - end row).",
            "select_x_axis": "Which column should be used for the X-axis?",
            "select_y_axis": "Which column should be used for the Y-axis?",
            "open_column_builder": "Opening formula builder. You can create custom calculated columns.",
            "add_formula_column": "Enter your formula. You can reference columns like [ColumnA] + [ColumnB]",
            "submit_sheet": "Saving your sheet configuration...",
            "go_to_results": "Taking you to the results page to review your processed data.",
            "cancel": "Canceling operation. Going back to the previous step."
        }
        
    def load_intent_data(self, intent_data: List[Dict]):
        for item in intent_data:
            intent = item['intent']
            text = item['text']
            
            if intent not in self.intent_patterns:
                self.intent_patterns[intent] = []
            
            keywords = self._extract_keywords(text)
            self.intent_patterns[intent].extend(keywords)

    def _extract_keywords(self, text: str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        stop_words = {'a','an','the','and','or','is','to','this','that','for','of','in'}
        return [w for w in words if w not in stop_words and len(w) > 2]

    def classify_intent(self, user_input: str, min_matches: int = 2) -> List[Tuple[str, float, int]]:
        user_keywords = self._extract_keywords(user_input)
        print(f"[DEBUG classify_intent] Extracted keywords from '{user_input}': {user_keywords}")

        if not user_keywords:
            return []

        intent_scores = {}
        
        

        for intent, patterns in self.intent_patterns.items():
            matches = sum(1 for keyword in user_keywords if keyword in patterns)
            print(f"[DEBUG classify_intent] Intent '{intent}' has {matches} keyword matches with patterns: {patterns}")
            score = matches / len(user_keywords) if matches > 0 else 0
            intent_scores[intent] = (score, matches)

    # ✅ Keep only intents with STRONG evidence (2+ keyword hits)
        matching_intents = [
        (intent, score, matches)
        for intent, (score, matches) in intent_scores.items()
        if matches >= min_matches
    ]

    # Sort by number of matches first, then confidence
        matching_intents.sort(key=lambda x: (x[2], x[1]), reverse=True)

        return matching_intents
    
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.intent_patterns = model_data.get("intent_patterns", {})
        self.intent_responses = model_data.get("intent_responses", {})