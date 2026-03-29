from src.train_bot import load_data
from src.utils.preprocessing import clean_text
from src.utils.similarity import SimilarityMatcher
import logging

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Load Data (One-Time)
# -----------------------------
try:
    questions, answers = load_data()
    matcher = SimilarityMatcher(questions)
    logging.info("Chatbot initialized successfully.")
except Exception as e:
    logging.error(f"Error during initialization: {str(e)}")
    raise e


# -----------------------------
# Chatbot Response Function
# -----------------------------
def get_response(user_input: str) -> str:
    try:
        logging.info(f"User Input: {user_input}")

        # -----------------------------
        # Edge Case: Empty Input
        # -----------------------------
        if not user_input or not user_input.strip():
            logging.warning("Empty input received")
            return "Please enter a valid question."

        # -----------------------------
        # Preprocessing
        # -----------------------------
        cleaned_input = clean_text(user_input)

        # -----------------------------
        # Similarity Matching
        # -----------------------------
        best_index, score = matcher.find_best_match(cleaned_input)

        logging.info(f"Best Match Index: {best_index}")
        logging.info(f"Similarity Score: {score}")

        # -----------------------------
        # Confidence Threshold
        # -----------------------------
        THRESHOLD = 0.3

        if score < THRESHOLD:
            logging.warning("Low confidence response triggered")
            return "I'm not sure about that. Try asking something related to cricket."

        # -----------------------------
        # Return Best Answer
        # -----------------------------
        response = answers[best_index]

        return response

    except Exception as e:
        logging.error(f"Runtime error: {str(e)}")
        return "Something went wrong. Please try again."