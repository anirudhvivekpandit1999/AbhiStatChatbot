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
    ("open column builder", ["open_column_builder"]),
("open the column builder", ["open_column_builder"]),
("launch column builder", ["open_column_builder"]),
("start column builder", ["open_column_builder"]),
("show column builder", ["open_column_builder"]),
("go to column builder", ["open_column_builder"]),
("open column editor", ["open_column_builder"]),
("open column configuration", ["open_column_builder"]),
("open column settings", ["open_column_builder"]),
("I want to edit columns", ["open_column_builder"]),
("let me modify columns", ["open_column_builder"]),
("take me to column builder", ["open_column_builder"]),
("open column builder and set x axis", 
 ["open_column_builder", "set_x_axis"]),

("open column builder and set y axis", 
 ["open_column_builder", "set_y_axis"]),

("launch column builder and set x axis to date", 
 ["open_column_builder", "set_x_axis"]),

("open column builder and set y axis to revenue", 
 ["open_column_builder", "set_y_axis"]),
("upload file and open column builder", 
 ["upload_file", "open_column_builder"]),

("create new sheet and open column builder", 
 ["create_new_sheet", "open_column_builder"]),

("upload file create sheet open column builder", 
 ["upload_file", "create_new_sheet", "open_column_builder"]),

("create sheet name it report and open column builder", 
 ["create_new_sheet", "name_new_sheet", "open_column_builder"]),
("upload file create new sheet name it report set base sheet set pre sheet name cleaned data set post sheet name final output set x axis date set y axis revenue open column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),
("please upload the file, create a new sheet, call it sales report, make it the base sheet, define preprocessing sheet cleaned data, define postprocessing sheet final output, use month as x axis, revenue as y axis, and then open the column builder",
 ["upload_file",
  "create_new_sheet",
  "name_new_sheet",
  "set_base_sheet",
  "set_pre_sheet_name",
  "set_post_sheet_name",
  "set_x_axis",
  "set_y_axis",
  "open_column_builder"]),

    ("set x axis", ["set_x_axis"]),
("set the x axis", ["set_x_axis"]),
("define x axis", ["set_x_axis"]),
("set x axis to date", ["set_x_axis"]),
("make x axis revenue", ["set_x_axis"]),
("use month as x axis", ["set_x_axis"]),
("choose sales as x axis", ["set_x_axis"]),
("x axis should be profit", ["set_x_axis"]),
("set column date as x axis", ["set_x_axis"]),
("upload file create new sheet name new sheet set base sheet set pre sheet name set post sheet name set x axis",
 ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name", "set_x_axis"]),

("upload file, create sheet, name it, set base, set pre sheet, set post sheet, and set x axis to date",
 ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name", "set_x_axis"]),

("please upload file, create sheet, name it report, make it base, define preprocessing sheet, define postprocessing sheet, and set x axis to revenue",
 ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name", "set_x_axis"]),
# ✅ NEW INTENT — Set Y Axis
("set y axis", ["set_y_axis"]),
("set the y axis", ["set_y_axis"]),
("define y axis", ["set_y_axis"]),
("set y axis to revenue", ["set_y_axis"]),
("make y axis profit", ["set_y_axis"]),
("use sales as y axis", ["set_y_axis"]),
("choose quantity as y axis", ["set_y_axis"]),
("y axis should be growth", ["set_y_axis"]),
("set column revenue as y axis", ["set_y_axis"]),
("upload file create new sheet name new sheet set base sheet set pre sheet name set post sheet name set x axis set y axis",
 ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet",
  "set_pre_sheet_name", "set_post_sheet_name", "set_x_axis", "set_y_axis"]),
("upload file, create sheet, name it, set base sheet, set preprocessing sheet, set postprocessing sheet, set x axis to date, and set y axis to revenue",
 ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet",
  "set_pre_sheet_name", "set_post_sheet_name", "set_x_axis", "set_y_axis"]),
("please upload file, make a new sheet, call it report, select it as base, define pre sheet name cleaned data, define post sheet name final output, use month as x axis and profit as y axis",
 ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet",
  "set_pre_sheet_name", "set_post_sheet_name", "set_x_axis", "set_y_axis"]),

("upload file create sheet name it sales set base set pre sheet set post sheet set x axis date set y axis revenue",
 ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet",
  "set_pre_sheet_name", "set_post_sheet_name", "set_x_axis", "set_y_axis"]),



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
    ("please upload the file and create a new sheet", ["upload_file", "create_new_sheet"]),
("create new sheet after uploading file", ["create_new_sheet", "upload_file"]),
("make a sheet and upload file first", ["create_new_sheet", "upload_file"]),
("name new sheet after creating it", ["create_new_sheet", "name_new_sheet"]),
("create sheet then call it report", ["create_new_sheet", "name_new_sheet"]),
("upload file and create new sheet then name it", ["upload_file", "create_new_sheet", "name_new_sheet"]),
("create sheet, name it, and upload file", ["create_new_sheet", "name_new_sheet", "upload_file"]),
("upload a file, create a sheet, and call the new sheet sales data", ["upload_file", "create_new_sheet", "name_new_sheet"]),
("create sheet, upload file, and name sheet report", ["create_new_sheet", "upload_file", "name_new_sheet"]),
("upload file, create new sheet, name it, and set as base sheet", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet"]),
("create sheet, name it, upload file, and select this sheet as base", ["create_new_sheet", "name_new_sheet", "upload_file", "set_base_sheet"]),
("upload file, make new sheet, call it report, and make this the base sheet", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet"]),
("upload file, create new sheet, name it, set base sheet, and set preprocessing sheet name", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name"]),
("create and name new sheet, upload file, make it base, define pre sheet name", ["create_new_sheet", "name_new_sheet", "upload_file", "set_base_sheet", "set_pre_sheet_name"]),
("upload file, create sheet, name sheet, select as base sheet, set pre sheet name cleaned data", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name"]),
("upload file, create new sheet, name it, set base sheet, set pre sheet name, and set post sheet name", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name"]),
("please upload file, create and name sheet, make it base, define pre sheet, define post sheet", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name"]),
("upload file, make new sheet, call it, select as base, set preprocessing name, set postprocessing name", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name"]),
("upload file, create sheet, name it, set base, set pre sheet cleaned data, set post sheet final output", ["upload_file", "create_new_sheet", "name_new_sheet", "set_base_sheet", "set_pre_sheet_name", "set_post_sheet_name"]),


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
