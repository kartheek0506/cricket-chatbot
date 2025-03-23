import torch
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from train_bot import ChatbotModel

# ✅ Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")
DATA_DIR = os.path.join(BASE_DIR, "../data")

# ✅ Ensure all required files exist before proceeding
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
model_path = os.path.join(MODEL_DIR, "chatbot_model.pth")
csv_path = os.path.join(DATA_DIR, "cricketfaqs.csv")

for file_path in [vectorizer_path, label_encoder_path, model_path, csv_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: Required file not found -> {file_path}")

# ✅ Load trained components safely
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# ✅ Load trained chatbot model
input_size = vectorizer.transform([""]).shape[1]
model = ChatbotModel(input_size=input_size, hidden_size=64, output_size=len(label_encoder.classes_))
model.load_state_dict(torch.load(model_path))
model.eval()

# ✅ Load dataset safely with error handling
try:
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")  # Skip problematic lines
except Exception as e:
    raise RuntimeError(f" Error reading CSV file: {e}")

# ✅ Debugging: Print first few rows to check data format
print("\n First 5 rows of CSV data:\n", df.head())

# ✅ Standardize column names (case-insensitive handling)
expected_columns = {"question": "question", "Question": "question", "answer": "answer", "Answer": "answer"}
df.rename(columns={col: expected_columns[col] for col in df.columns if col in expected_columns}, inplace=True)

# ✅ Validate required columns
if "question" not in df.columns or "answer" not in df.columns:
    raise KeyError("Error: CSV file must contain 'question' and 'answer' columns.")

questions = df["question"].values
answers = df["answer"].values

# ✅ Function to get bot response
def get_response(user_input):
    X_input = vectorizer.transform([user_input]).toarray()
    with torch.no_grad():
        output = model(torch.tensor(X_input, dtype=torch.float32))
    predicted_index = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_index])[0]

# ✅ Chat Loop
print("\n Chatbot is ready! Type 'exit' to stop.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print(" Exiting chatbot. Have a great day!")
        break
    response = get_response(user_input)
    print("Bot:", response)
