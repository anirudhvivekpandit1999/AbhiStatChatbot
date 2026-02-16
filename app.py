import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import List, Dict
import re

MAX_LEN = 16
D_MODEL = 128
HEADS = 4
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 50
LR = 0.001


# ================= TOKENIZER =================

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.vocab_size = 2

    def fit(self, texts: List[str]):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    idx = self.vocab_size
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    self.vocab_size += 1

    def encode(self, text: str):
        words = re.findall(r"\b\w+\b", text.lower())
        tokens = [
            self.word2idx.get(word, self.word2idx["<unk>"])
            for word in words
        ]
        tokens = tokens[:MAX_LEN]
        tokens += [0] * (MAX_LEN - len(tokens))
        return tokens


# ================= POSITIONAL ENCODING =================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


# ================= MODEL =================

class TransformerIntentClassifier(nn.Module):
    def __init__(self, vocab_size, num_intents):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.positional = PositionalEncoding(D_MODEL, MAX_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=HEADS,
            dim_feedforward=256,
            dropout=DROPOUT,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=LAYERS
        )

        self.classifier = nn.Linear(D_MODEL, num_intents)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ================= TRAINING =================

def train_model(intent_data: List[Dict]):

    texts = [item["text"] for item in intent_data]

    all_intents = set()
    for item in intent_data:
        all_intents.add(item["intent"])

    intents = sorted(list(all_intents))

    intent_to_idx = {intent: i for i, intent in enumerate(intents)}
    idx_to_intent = {i: intent for intent, i in intent_to_idx.items()}

    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)

    X = torch.tensor([tokenizer.encode(t) for t in texts])
    y = torch.zeros(len(intent_data), len(intents))

    for i, item in enumerate(intent_data):
        y[i][intent_to_idx[item["intent"]]] = 1

    model = TransformerIntentClassifier(tokenizer.vocab_size, len(intents))

    class_counts = y.sum(dim=0)
    pos_weight = (len(y) - class_counts) / (class_counts + 1e-5)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model, tokenizer, idx_to_intent


# ================= TRAIN DATA =================

intent_training_data = [ {"text": "upload file", "intent": "upload_file"}, {"text": "upload", "intent": "upload_file"}, {"text": "upload excel", "intent": "upload_file"}, {"text": "upload xlsx", "intent": "upload_file"}, {"text": "upload csv", "intent": "upload_file"}, {"text": "upload an excel file", "intent": "upload_file"}, {"text": "open my spreadsheet", "intent": "upload_file"}, {"text": "choose a file to upload", "intent": "upload_file"}, {"text": "load excel file", "intent": "upload_file"}, {"text": "browse and upload", "intent": "upload_file"}, {"text": "add a file", "intent": "upload_file"}, {"text": "import data from file", "intent": "upload_file"}, {"text": "open this file", "intent": "upload_file"}, {"text": "drag and drop file", "intent": "upload_file"}, {"text": "pick a file", "intent": "upload_file"}, {"text": "select file from system", "intent": "upload_file"}, {"text": "bring my file here", "intent": "upload_file"}, {"text": "uplod file", "intent": "upload_file"}, {"text": "uplod excel", "intent": "upload_file"}, {"text": "upoad file", "intent": "upload_file"}, {"text": "next", "intent": "enter_preprocess"}, {"text": "next step", "intent": "enter_preprocess"}, {"text": "go to next step", "intent": "enter_preprocess"}, {"text": "continue", "intent": "enter_preprocess"}, {"text": "go ahead", "intent": "enter_preprocess"}, {"text": "move ahead", "intent": "enter_preprocess"}, {"text": "proceed", "intent": "enter_preprocess"}, {"text": "open preprocessing", "intent": "enter_preprocess"}, {"text": "go to preprocessing", "intent": "enter_preprocess"}, {"text": "start preprocessing", "intent": "enter_preprocess"}, {"text": "create new sheet", "intent": "enter_preprocess"}, {"text": "preprocess data", "intent": "enter_preprocess"}, {"text": "prep data", "intent": "enter_preprocess"}, {"text": "pre process", "intent": "enter_preprocess"}, {"text": "select base sheet", "intent": "select_base_sheet"}, {"text": "choose the base sheet", "intent": "select_base_sheet"}, {"text": "use this sheet as base", "intent": "select_base_sheet"}, {"text": "pick this sheet", "intent": "select_base_sheet"}, {"text": "set this as the base sheet", "intent": "select_base_sheet"}, {"text": "i want to use this sheet", "intent": "select_base_sheet"}, {"text": "this is the sheet i want to use", "intent": "select_base_sheet"}, {"text": "select this as base", "intent": "select_base_sheet"}, {"text": "make this the base sheet", "intent": "select_base_sheet"}, {"text": "base sheet is this", "intent": "select_base_sheet"}, {"text": "base shet", "intent": "select_base_sheet"}, {"text": "bse sheet", "intent": "select_base_sheet"}, {"text": "name the new sheet", "intent": "name_new_sheet"}, {"text": "set a name for the sheet", "intent": "name_new_sheet"}, {"text": "call this sheet something else", "intent": "name_new_sheet"}, {"text": "rename the new sheet", "intent": "name_new_sheet"}, {"text": "give the sheet a name", "intent": "name_new_sheet"}, {"text": "enter sheet name", "intent": "name_new_sheet"}, {"text": "change sheet name", "intent": "name_new_sheet"}, {"text": "sheet name", "intent": "name_new_sheet"}, {"text": "nam sheet", "intent": "name_new_sheet"}, {"text": "select row range", "intent": "set_row_range"}, {"text": "trim rows", "intent": "set_row_range"}, {"text": "filter rows by range", "intent": "set_row_range"}, {"text": "select rows from start to end", "intent": "set_row_range"}, {"text": "choose row range", "intent": "set_row_range"}, {"text": "slice rows", "intent": "set_row_range"}, {"text": "cut rows", "intent": "set_row_range"}, {"text": "row range", "intent": "set_row_range"}, {"text": "rows from to", "intent": "set_row_range"}, {"text": "row rng", "intent": "set_row_range"}, {"text": "select x axis", "intent": "select_x_axis"}, {"text": "choose x axis", "intent": "select_x_axis"}, {"text": "use this for x axis", "intent": "select_x_axis"}, {"text": "set x axis column", "intent": "select_x_axis"}, {"text": "x axis", "intent": "select_x_axis"}, {"text": "xaxis", "intent": "select_x_axis"}, {"text": "select x", "intent": "select_x_axis"}, {"text": "x axix", "intent": "select_x_axis"}, {"text": "select y axis", "intent": "select_y_axis"}, {"text": "choose y axis", "intent": "select_y_axis"}, {"text": "use this for y axis", "intent": "select_y_axis"}, {"text": "set y axis column", "intent": "select_y_axis"}, {"text": "y axis", "intent": "select_y_axis"}, {"text": "yaxis", "intent": "select_y_axis"}, {"text": "select y", "intent": "select_y_axis"}, {"text": "y axix", "intent": "select_y_axis"}, {"text": "open column builder", "intent": "open_column_builder"}, {"text": "open formula builder", "intent": "open_column_builder"}, {"text": "add a calculated column", "intent": "open_column_builder"}, {"text": "create a formula column", "intent": "open_column_builder"}, {"text": "column builder", "intent": "open_column_builder"}, {"text": "formula editor", "intent": "open_column_builder"}, {"text": "column bulder", "intent": "open_column_builder"}, {"text": "add formula column", "intent": "add_formula_column"}, {"text": "create a new calculated column", "intent": "add_formula_column"}, {"text": "add a new derived column", "intent": "add_formula_column"}, {"text": "add formula", "intent": "add_formula_column"}, {"text": "apply formula", "intent": "add_formula_column"}, {"text": "add calculated column", "intent": "add_formula_column"}, {"text": "formula column", "intent": "add_formula_column"}, {"text": "formla column", "intent": "add_formula_column"}, {"text": "submit sheet", "intent": "submit_sheet"}, {"text": "save this sheet", "intent": "submit_sheet"}, {"text": "finish sheet creation", "intent": "submit_sheet"}, {"text": "complete this step", "intent": "submit_sheet"}, {"text": "submit", "intent": "submit_sheet"}, {"text": "done", "intent": "submit_sheet"}, {"text": "save and continue", "intent": "submit_sheet"}, {"text": "finalize sheet", "intent": "submit_sheet"}, {"text": "submt sheet", "intent": "submit_sheet"}, {"text": "result", "intent": "go_to_results"}, {"text": "results", "intent": "go_to_results"}, {"text": "show result", "intent": "go_to_results"}, {"text": "show results", "intent": "go_to_results"}, {"text": "open result", "intent": "go_to_results"}, {"text": "open results", "intent": "go_to_results"}, {"text": "go to result", "intent": "go_to_results"}, {"text": "go to results", "intent": "go_to_results"}, {"text": "take me to results page", "intent": "go_to_results"}, {"text": "see result", "intent": "go_to_results"}, {"text": "see results", "intent": "go_to_results"}, {"text": "view result", "intent": "go_to_results"}, {"text": "view results", "intent": "go_to_results"}, {"text": "output", "intent": "go_to_results"}, {"text": "outputs", "intent": "go_to_results"}, {"text": "analysis result", "intent": "go_to_results"}, {"text": "analysis results", "intent": "go_to_results"}, {"text": "reslt", "intent": "go_to_results"}, {"text": "reslts", "intent": "go_to_results"}, {"text": "rsults", "intent": "go_to_results"}, {"text": "cancel", "intent": "cancel"}, {"text": "go back", "intent": "cancel"}, {"text": "stop this", "intent": "cancel"}, {"text": "undo", "intent": "cancel"}, {"text": "cancel this", "intent": "cancel"}, {"text": "never mind", "intent": "cancel"}, {"text": "abort", "intent": "cancel"}, {"text": "exit", "intent": "cancel"}, {"text": "cncel", "intent": "cancel"}, { "text": "create new excel", "intent": "create_new_excel" }, { "text": "create a new spreadsheet", "intent": "create_new_excel" }, { "text": "make a new excel file", "intent": "create_new_excel" }, { "text": "start a new sheet", "intent": "create_new_excel" }, { "text": "new excel sheet", "intent": "create_new_excel" }, { "text": "new spreadsheet", "intent": "create_new_excel" }, { "text": "add new excel", "intent": "create_new_excel" }, { "text": "start a new excel file", "intent": "create_new_excel" }, { "text": "open a new sheet", "intent": "create_new_excel" }, { "text": "create excel", "intent": "create_new_excel" }, { "text": "new file", "intent": "create_new_excel" }, { "text": "make new spreadsheet", "intent": "create_new_excel" }, { "text": "create new xlsx", "intent": "create_new_excel" }, { "text": "new xlsx file", "intent": "create_new_excel" }, { "text": "creat new excel", "intent": "create_new_excel" }, { "text": "create nwe excel", "intent": "create_new_excel" }, { "text": "new exel sheet", "intent": "create_new_excel" }, { "text": "start nwe sheet", "intent": "create_new_excel" } ]



# ================= SAVE MODEL =================

if __name__ == "__main__":

    model, tokenizer, idx_to_intent = train_model(intent_training_data)

    torch.save({
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.word2idx,
        "idx_to_intent": idx_to_intent
    }, "intent_model.pth")

    print("Model saved as intent_model.pth")
