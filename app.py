import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import random

# -----------------------------
# TOKENIZER
# -----------------------------
class Tokenizer:
    def __init__(self):
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def fit(self, texts):
        for text in texts:
            words = self.clean(text).split()
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text, max_len=40):
        words = self.clean(text).split()
        tokens = [self.word_to_idx.get(w, 1) for w in words]
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        return tokens


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
            batch_first=True,
            dropout=0.1
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


# -----------------------------
# INTENT DATA (EXPANDED)
# -----------------------------
intent_data = [

    # Single intents
    ("upload excel file", ["upload_file"]),
    ("upload a csv file", ["upload_file"]),
    ("please upload the file", ["upload_file"]),

    ("create new sheet", ["create_new_sheet"]),
    ("make a sheet", ["create_new_sheet"]),
    ("add another sheet", ["create_new_sheet"]),

    ("name the new sheet sales data", ["name_new_sheet"]),
    ("rename the sheet", ["name_new_sheet"]),
    ("call the new sheet report", ["name_new_sheet"]),

    ("set this as base sheet", ["set_base_sheet"]),
    ("make this the base sheet", ["set_base_sheet"]),
    ("select this sheet as base", ["set_base_sheet"]),

    ("set preprocessing sheet name", ["set_pre_sheet_name"]),
    ("set pre sheet name cleaned data", ["set_pre_sheet_name"]),
    ("define preprocessing sheet name", ["set_pre_sheet_name"]),

    # ✅ NEW INTENT — Post Sheet Name
    ("set postprocessing sheet name", ["set_post_sheet_name"]),
    ("set post sheet name final output", ["set_post_sheet_name"]),
    ("define postprocessing sheet name", ["set_post_sheet_name"]),

    # Multi-intent combinations (IMPORTANT)
    ("upload file and create new sheet", ["upload_file", "create_new_sheet"]),
    ("create and name new sheet", ["create_new_sheet", "name_new_sheet"]),
    ("upload file create sheet name sheet", ["upload_file", "create_new_sheet", "name_new_sheet"]),
    ("upload file create new sheet name new sheet set base sheet", 
        ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet"]),
    ("upload file create new sheet name new sheet set base sheet set pre sheet name", 
        ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name"]),

    # ✅ Updated Full Combination Including Post Sheet
    ("upload file create new sheet name new sheet set base sheet set pre sheet name set post sheet name",
        ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name"]),
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
    embed_dim=128,
    num_heads=4,
    hidden_dim=256,
    num_classes=len(all_intents)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model...")

for epoch in range(800):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

print("Training complete!\n")

# -----------------------------
# MULTI-INTENT PREDICTION
# -----------------------------
def predict_intents(text, threshold=0.4):
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
