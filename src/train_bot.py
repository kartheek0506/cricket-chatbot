import os
import pandas as pd
import logging
from src.utils.preprocessing import clean_text

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Load and Prepare Data
# -----------------------------
def load_data():
    try:
        # Get project root directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Construct dataset path
        data_path = os.path.join(BASE_DIR, "data", "cricketfaqs.csv")

        logging.info(f"Loading dataset from: {data_path}")

        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        # Load CSV file
        df = pd.read_csv(data_path)

        logging.info("Dataset loaded successfully")

        # Validate required columns
        if 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("CSV must contain 'question' and 'answer' columns")

        # Clean questions
        df['question'] = df['question'].apply(clean_text)

        # Remove empty rows
        df = df.dropna(subset=['question', 'answer'])

        # Convert to lists
        questions = df['question'].tolist()
        answers = df['answer'].tolist()

        logging.info(f"Loaded {len(questions)} question-answer pairs")

        return questions, answers

    except Exception as e:
        logging.error(f"Error in load_data: {str(e)}")
        raise e