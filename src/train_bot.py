import os
import pandas as pd
import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn, optim
import numpy as np

# ✅ Ensure the "models" directory exists before saving files
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

# ✅ Ensure the "data" directory exists before loading data
data_path = os.path.join(os.getcwd(), "data", "cricketfaqs.csv")

# ✅ Read the CSV file correctly (fix encoding issues)
# ✅ Read CSV file & Strip Column Names to Remove Spaces
try:
    df = pd.read_csv(data_path, encoding="utf-8", sep=",", on_bad_lines="skip")
    df.columns = df.columns.str.strip()  # ✅ Remove unwanted spaces in column names
    print("CSV file loaded successfully with columns:", df.columns.tolist())

    # ✅ Check if required columns exist
    if "question" not in df.columns or "answer" not in df.columns:
        print("Error: CSV file must have 'question' and 'answer' columns.")
        exit()

except FileNotFoundError:
    print(f"Error: The file '{data_path}' was not found. Make sure it exists!")
    exit()
except pd.errors.ParserError:
    print("Error: CSV file has formatting issues. Check for missing or extra columns.")
    exit()


# ✅ Data Preprocessing
questions = df["question"].astype(str).tolist()
answers = df["answer"].astype(str).tolist()

# ✅ Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions).toarray()
y = np.array(answers)

# ✅ Save vectorizer
vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"Vectorizer saved successfully at: {vectorizer_path}")

# ✅ Define Neural Network Model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = X.shape[1]
hidden_size = 64
output_size = y.shape[0]  # Output size should match the number of possible answers

model = ChatbotModel(input_size, hidden_size, output_size)

# ✅ Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert labels to indices
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y_tensor = torch.tensor(y_encoded, dtype=torch.long)
X_tensor = torch.tensor(X, dtype=torch.float32)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ✅ Save trained model
model_path = os.path.join(models_dir, "chatbot_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at: {model_path}")

# ✅ Save label encoder
encoder_path = os.path.join(models_dir, "label_encoder.pkl")
with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Label Encoder saved successfully at: {encoder_path}")
