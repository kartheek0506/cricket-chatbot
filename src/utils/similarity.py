from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatcher:
    def __init__(self, questions):
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(questions)
        self.questions = questions

    def find_best_match(self, user_input):
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.question_vectors)

        best_index = similarities.argmax()
        best_score = similarities[0][best_index]

        return best_index, best_score