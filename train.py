import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import SimpleTokenizer, TransformerIntentClassifier, MAX_LEN

MODEL_PATH = "intent_model.pth"

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

all_intents = sorted(list(set(i for _, intents in intent_data for i in intents)))
intent_to_idx = {intent: i for i, intent in enumerate(all_intents)}
idx_to_intent = {i: intent for intent, i in intent_to_idx.items()}

tokenizer = SimpleTokenizer()
tokenizer.fit([text for text, _ in intent_data])

X = []
Y = []

for text, intents in intent_data:
    X.append(tokenizer.encode(text))
    label = np.zeros(len(all_intents))
    for intent in intents:
        label[intent_to_idx[intent]] = 1
    Y.append(label)

X = torch.tensor(np.array(X))
Y = torch.tensor(np.array(Y), dtype=torch.float32)

model = TransformerIntentClassifier(tokenizer.vocab_size, len(all_intents))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model...")

for epoch in range(600):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

print("Training complete!")

torch.save({
    "model_state": model.state_dict(),
    "tokenizer": tokenizer.word2idx,
    "idx_to_intent": idx_to_intent
}, MODEL_PATH)

print("Model saved as intent_model.pth")
