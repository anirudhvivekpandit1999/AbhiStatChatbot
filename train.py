# train.py
from model import TransformerLLM
import json

def main():
    chatbot = TransformerLLM(d_model=64, num_heads=4, num_layers=2, max_seq_len=50)
    
    intent_training_data = [
        {"text": "upload file", "intent": "upload_file"},
        {"text": "upload an excel file", "intent": "upload_file"},
        {"text": "open my spreadsheet", "intent": "upload_file"},
        {"text": "choose a file to upload", "intent": "upload_file"},
        {"text": "load excel file", "intent": "upload_file"},
        {"text": "browse and upload", "intent": "upload_file"},
        {"text": "add a file", "intent": "upload_file"},
        {"text": "import data from file", "intent": "upload_file"},
        {"text": "open this file", "intent": "upload_file"},
        
        {"text": "next", "intent": "enter_preprocess"},
        {"text": "go to next step", "intent": "enter_preprocess"},
        {"text": "continue", "intent": "enter_preprocess"},
        {"text": "open preprocessing", "intent": "enter_preprocess"},
        {"text": "go to preprocessing", "intent": "enter_preprocess"},
        {"text": "start preprocessing", "intent": "enter_preprocess"},
        {"text": "create new sheet", "intent": "enter_preprocess"},
        
        {"text": "select base sheet", "intent": "select_base_sheet"},
        {"text": "choose the base sheet", "intent": "select_base_sheet"},
        {"text": "use this sheet as base", "intent": "select_base_sheet"},
        {"text": "pick this sheet", "intent": "select_base_sheet"},
        {"text": "set this as the base sheet", "intent": "select_base_sheet"},
        {"text": "i want to use this sheet", "intent": "select_base_sheet"},
        {"text": "this is the sheet i want to use", "intent": "select_base_sheet"},
        
        {"text": "name the new sheet", "intent": "name_new_sheet"},
        {"text": "set a name for the sheet", "intent": "name_new_sheet"},
        {"text": "call this sheet something else", "intent": "name_new_sheet"},
        {"text": "rename the new sheet", "intent": "name_new_sheet"},
        
        {"text": "select row range", "intent": "set_row_range"},
        {"text": "trim rows", "intent": "set_row_range"},
        {"text": "filter rows by range", "intent": "set_row_range"},
        {"text": "select rows from start to end", "intent": "set_row_range"},
        
        {"text": "select x axis", "intent": "select_x_axis"},
        {"text": "choose x axis", "intent": "select_x_axis"},
        {"text": "use this for x axis", "intent": "select_x_axis"},
        {"text": "set x axis column", "intent": "select_x_axis"},
        
        {"text": "select y axis", "intent": "select_y_axis"},
        {"text": "choose y axis", "intent": "select_y_axis"},
        {"text": "use this for y axis", "intent": "select_y_axis"},
        {"text": "set y axis column", "intent": "select_y_axis"},
        
        {"text": "open column builder", "intent": "open_column_builder"},
        {"text": "open formula builder", "intent": "open_column_builder"},
        {"text": "add a calculated column", "intent": "open_column_builder"},
        {"text": "create a formula column", "intent": "open_column_builder"},
        
        {"text": "add formula column", "intent": "add_formula_column"},
        {"text": "create a new calculated column", "intent": "add_formula_column"},
        {"text": "add a new derived column", "intent": "add_formula_column"},
        
        {"text": "submit sheet", "intent": "submit_sheet"},
        {"text": "save this sheet", "intent": "submit_sheet"},
        {"text": "finish sheet creation", "intent": "submit_sheet"},
        {"text": "complete this step", "intent": "submit_sheet"},
        
        {"text": "go to results", "intent": "go_to_results"},
        {"text": "open data check", "intent": "go_to_results"},
        {"text": "show result", "intent": "go_to_results"},
        {"text": "take me to result page", "intent": "go_to_results"},
        
        {"text": "cancel", "intent": "cancel"},
        {"text": "go back", "intent": "cancel"},
        {"text": "stop this", "intent": "cancel"},
        {"text": "undo", "intent": "cancel"},
    ]
    
    chatbot.load_intent_data(intent_training_data)
    
    # small conversational training set (optional)
    training_data = [
        ("Hello", "Hi there! How can I help you today?"),
        ("How are you", "I'm doing great, thank you for asking!"),
        ("What is your name", "I'm a transformer based LLM chatbot built from scratch"),
        ("What can you do", "I can answer questions and have conversations with you"),
        ("Tell me a joke", "Why did the transformer go to school? To improve its attention!"),
        ("Goodbye", "Goodbye! It was nice talking to you"),
        ("Thanks", "You're welcome! Happy to help"),
    ]
    
    for question, response in training_data:
        chatbot.add_training_data(question, response)
    
    print("Training on general conversation data...")
    chatbot.train(epochs=10)
    
    # Save to file
    chatbot.save_model("chatbot_model.pkl")
    print("Training & save complete. Model file: chatbot_model.pkl")

if __name__ == "__main__":
    main()
