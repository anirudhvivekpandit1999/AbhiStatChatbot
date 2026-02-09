import numpy as np
import json
import re
import pickle
import os
from collections import defaultdict
from typing import List, Tuple, Dict
import math


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
    def __init__(self, vocab_size: int = None, d_model: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, max_seq_len: int = 50):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
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
    
    
    
    def _extract_keywords(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        
        stop_words = {'a', 'an', 'the', 'and', 'or', 'is', 'to', 'this', 'that', 'for', 'of', 'in'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def classify_intent(self, user_input: str, min_matches: int = 2) -> List[Tuple[str, float, int]]:
        user_keywords = self._extract_keywords(user_input)

        if not user_keywords:
            return []

        intent_scores = {}
        
        intent_training_data = [
    {"text": "upload file", "intent": "upload_file"},
    {"text": "upload", "intent": "upload_file"},
    {"text": "upload excel", "intent": "upload_file"},
    {"text": "upload xlsx", "intent": "upload_file"},
    {"text": "upload csv", "intent": "upload_file"},
    {"text": "upload an excel file", "intent": "upload_file"},
    {"text": "open my spreadsheet", "intent": "upload_file"},
    {"text": "choose a file to upload", "intent": "upload_file"},
    {"text": "load excel file", "intent": "upload_file"},
    {"text": "browse and upload", "intent": "upload_file"},
    {"text": "add a file", "intent": "upload_file"},
    {"text": "import data from file", "intent": "upload_file"},
    {"text": "open this file", "intent": "upload_file"},
    {"text": "drag and drop file", "intent": "upload_file"},
    {"text": "pick a file", "intent": "upload_file"},
    {"text": "select file from system", "intent": "upload_file"},
    {"text": "bring my file here", "intent": "upload_file"},
    {"text": "uplod file", "intent": "upload_file"},
    {"text": "uplod excel", "intent": "upload_file"},
    {"text": "upoad file", "intent": "upload_file"},

    {"text": "next", "intent": "enter_preprocess"},
    {"text": "next step", "intent": "enter_preprocess"},
    {"text": "go to next step", "intent": "enter_preprocess"},
    {"text": "continue", "intent": "enter_preprocess"},
    {"text": "go ahead", "intent": "enter_preprocess"},
    {"text": "move ahead", "intent": "enter_preprocess"},
    {"text": "proceed", "intent": "enter_preprocess"},
    {"text": "open preprocessing", "intent": "enter_preprocess"},
    {"text": "go to preprocessing", "intent": "enter_preprocess"},
    {"text": "start preprocessing", "intent": "enter_preprocess"},
    {"text": "create new sheet", "intent": "enter_preprocess"},
    {"text": "preprocess data", "intent": "enter_preprocess"},
    {"text": "prep data", "intent": "enter_preprocess"},
    {"text": "pre process", "intent": "enter_preprocess"},

    {"text": "select base sheet", "intent": "select_base_sheet"},
    {"text": "choose the base sheet", "intent": "select_base_sheet"},
    {"text": "use this sheet as base", "intent": "select_base_sheet"},
    {"text": "pick this sheet", "intent": "select_base_sheet"},
    {"text": "set this as the base sheet", "intent": "select_base_sheet"},
    {"text": "i want to use this sheet", "intent": "select_base_sheet"},
    {"text": "this is the sheet i want to use", "intent": "select_base_sheet"},
    {"text": "select this as base", "intent": "select_base_sheet"},
    {"text": "make this the base sheet", "intent": "select_base_sheet"},
    {"text": "base sheet is this", "intent": "select_base_sheet"},
    {"text": "base shet", "intent": "select_base_sheet"},
    {"text": "bse sheet", "intent": "select_base_sheet"},

    {"text": "name the new sheet", "intent": "name_new_sheet"},
    {"text": "set a name for the sheet", "intent": "name_new_sheet"},
    {"text": "call this sheet something else", "intent": "name_new_sheet"},
    {"text": "rename the new sheet", "intent": "name_new_sheet"},
    {"text": "give the sheet a name", "intent": "name_new_sheet"},
    {"text": "enter sheet name", "intent": "name_new_sheet"},
    {"text": "change sheet name", "intent": "name_new_sheet"},
    {"text": "sheet name", "intent": "name_new_sheet"},
    {"text": "nam sheet", "intent": "name_new_sheet"},

    {"text": "select row range", "intent": "set_row_range"},
    {"text": "trim rows", "intent": "set_row_range"},
    {"text": "filter rows by range", "intent": "set_row_range"},
    {"text": "select rows from start to end", "intent": "set_row_range"},
    {"text": "choose row range", "intent": "set_row_range"},
    {"text": "slice rows", "intent": "set_row_range"},
    {"text": "cut rows", "intent": "set_row_range"},
    {"text": "row range", "intent": "set_row_range"},
    {"text": "rows from to", "intent": "set_row_range"},
    {"text": "row rng", "intent": "set_row_range"},

    {"text": "select x axis", "intent": "select_x_axis"},
    {"text": "choose x axis", "intent": "select_x_axis"},
    {"text": "use this for x axis", "intent": "select_x_axis"},
    {"text": "set x axis column", "intent": "select_x_axis"},
    {"text": "x axis", "intent": "select_x_axis"},
    {"text": "xaxis", "intent": "select_x_axis"},
    {"text": "select x", "intent": "select_x_axis"},
    {"text": "x axix", "intent": "select_x_axis"},

    {"text": "select y axis", "intent": "select_y_axis"},
    {"text": "choose y axis", "intent": "select_y_axis"},
    {"text": "use this for y axis", "intent": "select_y_axis"},
    {"text": "set y axis column", "intent": "select_y_axis"},
    {"text": "y axis", "intent": "select_y_axis"},
    {"text": "yaxis", "intent": "select_y_axis"},
    {"text": "select y", "intent": "select_y_axis"},
    {"text": "y axix", "intent": "select_y_axis"},

    {"text": "open column builder", "intent": "open_column_builder"},
    {"text": "open formula builder", "intent": "open_column_builder"},
    {"text": "add a calculated column", "intent": "open_column_builder"},
    {"text": "create a formula column", "intent": "open_column_builder"},
    {"text": "column builder", "intent": "open_column_builder"},
    {"text": "formula editor", "intent": "open_column_builder"},
    {"text": "column bulder", "intent": "open_column_builder"},

    {"text": "add formula column", "intent": "add_formula_column"},
    {"text": "create a new calculated column", "intent": "add_formula_column"},
    {"text": "add a new derived column", "intent": "add_formula_column"},
    {"text": "add formula", "intent": "add_formula_column"},
    {"text": "apply formula", "intent": "add_formula_column"},
    {"text": "add calculated column", "intent": "add_formula_column"},
    {"text": "formula column", "intent": "add_formula_column"},
    {"text": "formla column", "intent": "add_formula_column"},

    {"text": "submit sheet", "intent": "submit_sheet"},
    {"text": "save this sheet", "intent": "submit_sheet"},
    {"text": "finish sheet creation", "intent": "submit_sheet"},
    {"text": "complete this step", "intent": "submit_sheet"},
    {"text": "submit", "intent": "submit_sheet"},
    {"text": "done", "intent": "submit_sheet"},
    {"text": "save and continue", "intent": "submit_sheet"},
    {"text": "finalize sheet", "intent": "submit_sheet"},
    {"text": "submt sheet", "intent": "submit_sheet"},

    {"text": "result", "intent": "go_to_results"},
    {"text": "results", "intent": "go_to_results"},
    {"text": "show result", "intent": "go_to_results"},
    {"text": "show results", "intent": "go_to_results"},
    {"text": "open result", "intent": "go_to_results"},
    {"text": "open results", "intent": "go_to_results"},
    {"text": "go to result", "intent": "go_to_results"},
    {"text": "go to results", "intent": "go_to_results"},
    {"text": "take me to results page", "intent": "go_to_results"},
    {"text": "see result", "intent": "go_to_results"},
    {"text": "see results", "intent": "go_to_results"},
    {"text": "view result", "intent": "go_to_results"},
    {"text": "view results", "intent": "go_to_results"},
    {"text": "output", "intent": "go_to_results"},
    {"text": "outputs", "intent": "go_to_results"},
    {"text": "analysis result", "intent": "go_to_results"},
    {"text": "analysis results", "intent": "go_to_results"},
    {"text": "reslt", "intent": "go_to_results"},
    {"text": "reslts", "intent": "go_to_results"},
    {"text": "rsults", "intent": "go_to_results"},

    {"text": "cancel", "intent": "cancel"},
    {"text": "go back", "intent": "cancel"},
    {"text": "stop this", "intent": "cancel"},
    {"text": "undo", "intent": "cancel"},
    {"text": "cancel this", "intent": "cancel"},
    {"text": "never mind", "intent": "cancel"},
    {"text": "abort", "intent": "cancel"},
    {"text": "exit", "intent": "cancel"},
    {"text": "cncel", "intent": "cancel"}
]
    
        self.load_intent_data(intent_training_data)
        
        for intent, patterns in self.intent_patterns.items():
            
            print(f"[DEBUG classify_intent] Intent '{intent}' patterns: {patterns}")
            matches = sum(1 for keyword in user_keywords if keyword in patterns)
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
    def generate_response(self, question: str, max_tokens: int = 15) -> str:
        if len(self.intent_patterns) > 0:
            matching_intents = self.classify_intent(question)
            
            if matching_intents:
                result = []
                for intent, confidence, matches in matching_intents:
                    result.append(f"{intent}|{confidence:.2f}")
                
                return ",".join(result)
        
        return "unknown|0.00"
    
    def debug_classify(self, user_input: str):
        user_keywords = self._extract_keywords(user_input)
        print(f"  Keywords extracted: {user_keywords}\n")
        
        intent_matches = {}
        for intent, patterns in self.intent_patterns.items():
            matches = [kw for kw in user_keywords if kw in patterns]
            if matches:
                intent_matches[intent] = matches
        
        if intent_matches:
            print("  Detected intents (sorted by priority):")
            # Sort by number of matches
            sorted_intents = sorted(intent_matches.items(), key=lambda x: len(x[1]), reverse=True)
            for intent, matches in sorted_intents:
                print(f"    {intent}: {len(matches)} matches - {matches}")
        else:
            print("  No intents detected")
    
    def build_model(self, vocab_size: int):
        self.vocab_size = vocab_size
        
        self.embedding = np.random.randn(vocab_size, self.d_model) * 0.01
        
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        self.transformer_blocks = [
            TransformerBlock(self.d_model, self.num_heads)
            for _ in range(self.num_layers)
        ]
        
        self.output_projection = np.random.randn(self.d_model, vocab_size) * 0.01
    
    def add_training_data(self, question: str, response: str):
        self.train_questions.append(question)
        self.train_responses.append(response)
    
    def embed_text(self, indices: List[int]) -> np.ndarray:
        embeddings = np.array([self.embedding[idx] for idx in indices])
        embeddings = self.positional_encoding.add_encoding(embeddings)
        return embeddings
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        x = self.embed_text(input_ids)
        
        for block in self.transformer_blocks:
            x = block.forward(x)
        
        logits = np.matmul(x, self.output_projection)
        
        return logits
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def compute_loss(self, logits: np.ndarray, target_ids: np.ndarray) -> float:
        probs = self.softmax(logits)
        loss = 0.0
        
        for t, target_id in enumerate(target_ids):
            if target_id > 0:
                prob = probs[t, target_id]
                if prob > 0:
                    loss -= np.log(prob)
        
        return loss / len(target_ids)
    
    def train(self, epochs: int = 10):
        if not self.train_questions:
            print("No training data provided.")
            return
        
        all_texts = self.train_questions + self.train_responses
        self.tokenizer.fit(all_texts)
        
        self.build_model(self.tokenizer.vocab_size)
        
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"Training samples: {len(self.train_questions)}")
        print(f"Model: {self.num_layers} Transformer blocks, {self.num_heads} attention heads")
        print(f"Starting training for {epochs} epochs...\n")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for q, r in zip(self.train_questions, self.train_responses):
                q_encoded = self.tokenizer.encode(q)
                q_padded = self.tokenizer.pad_sequence(q_encoded, self.max_seq_len)
                
                r_encoded = self.tokenizer.encode(r)
                r_padded = self.tokenizer.pad_sequence(r_encoded, self.max_seq_len)
                
                q_input = np.array(q_padded)
                encoder_output = self.forward(q_input)
                
                loss = self.compute_loss(encoder_output, np.array(r_padded))
                total_loss += loss
                
                self._update_weights(q_input, r_padded, encoder_output)
            
            avg_loss = total_loss / len(self.train_questions)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        print("\nTraining complete!")
    
    def _update_weights(self, inputs: np.ndarray, targets: np.ndarray, logits: np.ndarray):
        probs = self.softmax(logits)
        grad_logits = probs.copy()
        for t, target in enumerate(targets):
            if target > 0:
                grad_logits[t, target] -= 1.0
        
        embedding_output = self.embed_text(inputs)
        grad_W_out = np.matmul(embedding_output.T, grad_logits)
        self.output_projection -= self.learning_rate * grad_W_out
    

    
    def save_model(self, filepath: str):
        model_data = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'tokenizer': self.tokenizer,
            'embedding': self.embedding,
            'output_projection': self.output_projection,
            'transformer_blocks': self.transformer_blocks,
            'is_trained': self.is_trained,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.d_model = model_data['d_model']
        self.num_heads = model_data['num_heads']
        self.num_layers = model_data['num_layers']
        self.max_seq_len = model_data['max_seq_len']
        self.tokenizer = model_data['tokenizer']
        self.embedding = model_data['embedding']
        self.output_projection = model_data['output_projection']
        self.transformer_blocks = model_data['transformer_blocks']
        self.is_trained = model_data['is_trained']
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        print(f"Model loaded from {filepath}")


def main():
    
    chatbot = TransformerLLM(d_model=64, num_heads=4, num_layers=2, max_seq_len=50)
    
    intent_training_data = [
    {"text": "upload file", "intent": "upload_file"},
    {"text": "upload", "intent": "upload_file"},
    {"text": "upload excel", "intent": "upload_file"},
    {"text": "upload xlsx", "intent": "upload_file"},
    {"text": "upload csv", "intent": "upload_file"},
    {"text": "upload an excel file", "intent": "upload_file"},
    {"text": "open my spreadsheet", "intent": "upload_file"},
    {"text": "choose a file to upload", "intent": "upload_file"},
    {"text": "load excel file", "intent": "upload_file"},
    {"text": "browse and upload", "intent": "upload_file"},
    {"text": "add a file", "intent": "upload_file"},
    {"text": "import data from file", "intent": "upload_file"},
    {"text": "open this file", "intent": "upload_file"},
    {"text": "drag and drop file", "intent": "upload_file"},
    {"text": "pick a file", "intent": "upload_file"},
    {"text": "select file from system", "intent": "upload_file"},
    {"text": "bring my file here", "intent": "upload_file"},
    {"text": "uplod file", "intent": "upload_file"},
    {"text": "uplod excel", "intent": "upload_file"},
    {"text": "upoad file", "intent": "upload_file"},

    {"text": "next", "intent": "enter_preprocess"},
    {"text": "next step", "intent": "enter_preprocess"},
    {"text": "go to next step", "intent": "enter_preprocess"},
    {"text": "continue", "intent": "enter_preprocess"},
    {"text": "go ahead", "intent": "enter_preprocess"},
    {"text": "move ahead", "intent": "enter_preprocess"},
    {"text": "proceed", "intent": "enter_preprocess"},
    {"text": "open preprocessing", "intent": "enter_preprocess"},
    {"text": "go to preprocessing", "intent": "enter_preprocess"},
    {"text": "start preprocessing", "intent": "enter_preprocess"},
    {"text": "create new sheet", "intent": "enter_preprocess"},
    {"text": "preprocess data", "intent": "enter_preprocess"},
    {"text": "prep data", "intent": "enter_preprocess"},
    {"text": "pre process", "intent": "enter_preprocess"},

    {"text": "select base sheet", "intent": "select_base_sheet"},
    {"text": "choose the base sheet", "intent": "select_base_sheet"},
    {"text": "use this sheet as base", "intent": "select_base_sheet"},
    {"text": "pick this sheet", "intent": "select_base_sheet"},
    {"text": "set this as the base sheet", "intent": "select_base_sheet"},
    {"text": "i want to use this sheet", "intent": "select_base_sheet"},
    {"text": "this is the sheet i want to use", "intent": "select_base_sheet"},
    {"text": "select this as base", "intent": "select_base_sheet"},
    {"text": "make this the base sheet", "intent": "select_base_sheet"},
    {"text": "base sheet is this", "intent": "select_base_sheet"},
    {"text": "base shet", "intent": "select_base_sheet"},
    {"text": "bse sheet", "intent": "select_base_sheet"},

    {"text": "name the new sheet", "intent": "name_new_sheet"},
    {"text": "set a name for the sheet", "intent": "name_new_sheet"},
    {"text": "call this sheet something else", "intent": "name_new_sheet"},
    {"text": "rename the new sheet", "intent": "name_new_sheet"},
    {"text": "give the sheet a name", "intent": "name_new_sheet"},
    {"text": "enter sheet name", "intent": "name_new_sheet"},
    {"text": "change sheet name", "intent": "name_new_sheet"},
    {"text": "sheet name", "intent": "name_new_sheet"},
    {"text": "nam sheet", "intent": "name_new_sheet"},

    {"text": "select row range", "intent": "set_row_range"},
    {"text": "trim rows", "intent": "set_row_range"},
    {"text": "filter rows by range", "intent": "set_row_range"},
    {"text": "select rows from start to end", "intent": "set_row_range"},
    {"text": "choose row range", "intent": "set_row_range"},
    {"text": "slice rows", "intent": "set_row_range"},
    {"text": "cut rows", "intent": "set_row_range"},
    {"text": "row range", "intent": "set_row_range"},
    {"text": "rows from to", "intent": "set_row_range"},
    {"text": "row rng", "intent": "set_row_range"},

    {"text": "select x axis", "intent": "select_x_axis"},
    {"text": "choose x axis", "intent": "select_x_axis"},
    {"text": "use this for x axis", "intent": "select_x_axis"},
    {"text": "set x axis column", "intent": "select_x_axis"},
    {"text": "x axis", "intent": "select_x_axis"},
    {"text": "xaxis", "intent": "select_x_axis"},
    {"text": "select x", "intent": "select_x_axis"},
    {"text": "x axix", "intent": "select_x_axis"},

    {"text": "select y axis", "intent": "select_y_axis"},
    {"text": "choose y axis", "intent": "select_y_axis"},
    {"text": "use this for y axis", "intent": "select_y_axis"},
    {"text": "set y axis column", "intent": "select_y_axis"},
    {"text": "y axis", "intent": "select_y_axis"},
    {"text": "yaxis", "intent": "select_y_axis"},
    {"text": "select y", "intent": "select_y_axis"},
    {"text": "y axix", "intent": "select_y_axis"},

    {"text": "open column builder", "intent": "open_column_builder"},
    {"text": "open formula builder", "intent": "open_column_builder"},
    {"text": "add a calculated column", "intent": "open_column_builder"},
    {"text": "create a formula column", "intent": "open_column_builder"},
    {"text": "column builder", "intent": "open_column_builder"},
    {"text": "formula editor", "intent": "open_column_builder"},
    {"text": "column bulder", "intent": "open_column_builder"},

    {"text": "add formula column", "intent": "add_formula_column"},
    {"text": "create a new calculated column", "intent": "add_formula_column"},
    {"text": "add a new derived column", "intent": "add_formula_column"},
    {"text": "add formula", "intent": "add_formula_column"},
    {"text": "apply formula", "intent": "add_formula_column"},
    {"text": "add calculated column", "intent": "add_formula_column"},
    {"text": "formula column", "intent": "add_formula_column"},
    {"text": "formla column", "intent": "add_formula_column"},

    {"text": "submit sheet", "intent": "submit_sheet"},
    {"text": "save this sheet", "intent": "submit_sheet"},
    {"text": "finish sheet creation", "intent": "submit_sheet"},
    {"text": "complete this step", "intent": "submit_sheet"},
    {"text": "submit", "intent": "submit_sheet"},
    {"text": "done", "intent": "submit_sheet"},
    {"text": "save and continue", "intent": "submit_sheet"},
    {"text": "finalize sheet", "intent": "submit_sheet"},
    {"text": "submt sheet", "intent": "submit_sheet"},

    {"text": "result", "intent": "go_to_results"},
    {"text": "results", "intent": "go_to_results"},
    {"text": "show result", "intent": "go_to_results"},
    {"text": "show results", "intent": "go_to_results"},
    {"text": "open result", "intent": "go_to_results"},
    {"text": "open results", "intent": "go_to_results"},
    {"text": "go to result", "intent": "go_to_results"},
    {"text": "go to results", "intent": "go_to_results"},
    {"text": "take me to results page", "intent": "go_to_results"},
    {"text": "see result", "intent": "go_to_results"},
    {"text": "see results", "intent": "go_to_results"},
    {"text": "view result", "intent": "go_to_results"},
    {"text": "view results", "intent": "go_to_results"},
    {"text": "output", "intent": "go_to_results"},
    {"text": "outputs", "intent": "go_to_results"},
    {"text": "analysis result", "intent": "go_to_results"},
    {"text": "analysis results", "intent": "go_to_results"},
    {"text": "reslt", "intent": "go_to_results"},
    {"text": "reslts", "intent": "go_to_results"},
    {"text": "rsults", "intent": "go_to_results"},

    {"text": "cancel", "intent": "cancel"},
    {"text": "go back", "intent": "cancel"},
    {"text": "stop this", "intent": "cancel"},
    {"text": "undo", "intent": "cancel"},
    {"text": "cancel this", "intent": "cancel"},
    {"text": "never mind", "intent": "cancel"},
    {"text": "abort", "intent": "cancel"},
    {"text": "exit", "intent": "cancel"},
    {"text": "cncel", "intent": "cancel"}
]
    
    chatbot.load_intent_data(intent_training_data)
    
    training_data = [
        ("Hello", "Hi there! How can I help you today?"),
        ("How are you", "I'm doing great, thank you for asking!"),
        ("What is your name", "I'm a transformer based LLM chatbot built from scratch"),
        ("What can you do", "I can answer questions and have conversations with you"),
        ("Tell me a joke", "Why did the transformer go to school? To improve its attention!"),
        ("Goodbye", "Goodbye! It was nice talking to you"),
        ("Thanks", "You're welcome! Happy to help"),
    ]
    
    print("=" * 70)
    print("Transformer-based Language Model with Intent Recognition")
    print("=" * 70)
    print("Architecture: 2-layer Transformer with 4-head Self-Attention + Intent Classifier")
    print("=" * 70 + "\n")
    
    for question, response in training_data:
        chatbot.add_training_data(question, response)
    
    print("Training on general conversation data...")
    chatbot.train(epochs=10)
    
    chatbot.save_model("chatbot_model.pkl")
    
    print("\n" + "=" * 70)
    print("Chat Interface - Type 'exit' to quit, 'debug' to see intent matching")
    print("=" * 70)
    print("I can recognize intents and respond accordingly.")
    print("Example: 'I want to upload a file and start preprocessing'\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'debug':
            debug_input = input("Enter text to debug: ").strip()
            print(f"Debugging: '{debug_input}'")
            chatbot.debug_classify(debug_input)
            print()
            continue
        
        if not user_input:
            continue
        
        response = chatbot.generate_response(user_input)
        print(f"ChatBot: {response}\n")


if __name__ == "__main__":
    main()
