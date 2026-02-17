import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re

# -----------------------------
# TOKENIZER
# -----------------------------
class Tokenizer:
    def __init__(self):
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def fit(self, texts):
        for text in texts:
            words = self.clean(text).split()
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text, max_len=20):
        words = self.clean(text).split()
        tokens = [self.word_to_idx.get(w, 1) for w in words]
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        return tokens

    def clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text


# -----------------------------
# TRANSFORMER MODEL
# -----------------------------
class TransformerIntentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


# -----------------------------
# TRAINING DATA
# -----------------------------
intent_data = [
    ("upload file", ["upload_file"]),
    ("upload excel", ["upload_file"]),
    ("create new sheet", ["create_new_sheet"]),
    ("make a new sheet", ["create_new_sheet"]),
    ("rename sheet", ["set_pre_sheet_name"]),
    ("set sheet name", ["set_pre_sheet_name"]),
    ("set base sheet", ["set_base_sheet"]),
    ("upload file and create new sheet", ["upload_file", "create_new_sheet"]),
    ("upload and rename sheet", ["upload_file", "set_pre_sheet_name"]),
]

all_intents = sorted(list(set(intent for _, intents in intent_data for intent in intents)))
intent_to_idx = {intent: i for i, intent in enumerate(all_intents)}
idx_to_intent = {i: intent for intent, i in intent_to_idx.items()}

# -----------------------------
# PREPARE DATA
# -----------------------------
tokenizer = Tokenizer()
texts = [text for text, _ in intent_data]
tokenizer.fit(texts)

X = []
Y = []

for text, intents in intent_data:
    X.append(tokenizer.encode(text))
    label = np.zeros(len(all_intents))
    for intent in intents:
        label[intent_to_idx[intent]] = 1
    Y.append(label)

X = torch.tensor(X)
Y = torch.tensor(Y, dtype=torch.float32)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = TransformerIntentModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=4,
    hidden_dim=128,
    num_classes=len(all_intents)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model...")
for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

print("Training complete!\n")


# -----------------------------
# MULTI-INTENT PREDICTION
# -----------------------------
def predict_intents(text, threshold=0.5):
    model.eval()
    tokens = torch.tensor([tokenizer.encode(text)])

    with torch.no_grad():
        outputs = model(tokens)
        probs = torch.sigmoid(outputs)[0]

    detected = []
    for i, prob in enumerate(probs):
        if prob.item() >= threshold:
            detected.append((idx_to_intent[i], round(prob.item(), 4)))

    detected.sort(key=lambda x: x[1], reverse=True)
    return detected


# -----------------------------
# CHATBOT LOOP
# -----------------------------
print("🤖 Multi-Intent Chatbot Ready!")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye 👋")
        break

    intents = predict_intents(user_input)

    if not intents:
        print("Bot: Sorry, I didn't understand that.\n")
    else:
        print("Bot detected:")
        for intent, confidence in intents:
            print(f" - {intent} ({confidence})")
        print()
