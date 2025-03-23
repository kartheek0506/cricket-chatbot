import os
import torch
import pickle
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from train_bot import ChatbotModel

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths
MODEL_DIR = os.path.join(BASE_DIR, "../models")
DATA_DIR = os.path.join(BASE_DIR, "../data")
CSV_PATH = os.path.join(DATA_DIR, "cricketfaqs.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pth")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Ensure models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load trained components safely
try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    # Load trained model
    model = ChatbotModel(input_size=vectorizer.transform([""]).shape[1], hidden_size=64, output_size=len(label_encoder.classes_))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

except FileNotFoundError as e:
    print(f"Missing file: {e}. Please train the model first.")
    exit()

# Load dataset safely
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")

    # Standardize column names
    expected_columns = {"question": "question", "Question": "question", "answer": "answer", "Answer": "answer"}
    df.rename(columns={col: expected_columns[col] for col in df.columns if col in expected_columns}, inplace=True)

    # Ensure necessary columns exist
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV file must have 'question' and 'answer' columns!")

except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Define chatbot response function
def get_response(user_input):
    X_input = vectorizer.transform([user_input]).toarray()
    with torch.no_grad():
        output = model(torch.tensor(X_input, dtype=torch.float32))
    predicted_index = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_index])[0]

# Gradio interface
def chat_interface(user_input):
    if not user_input.strip():
        return "Please enter a valid question."
    
    response = get_response(user_input)
    return response

# Launch Gradio app
iface = gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs="text",
    title="Cricket Chatbot",
    description="Ask any question related to cricket, and the chatbot will answer!",
)

if __name__ == "__main__":
    iface.launch()
