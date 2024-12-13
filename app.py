from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import questions_answers  # Updated for better modularity
import nltk

nltk.download('punkt')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Corrected template name

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.form['question']
    answer = find_best_match(user_question)
    return render_template('index.html', question=user_question, response=answer)  # Pass question and response

def find_best_match(user_question):
    questions = list(questions_answers.keys())
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    vectors = tfidf_vectorizer.fit_transform(questions + [user_question])
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    best_match_index = cosine_similarities.argmax()

    # Improved matching logic
    if cosine_similarities[best_match_index] > 0.5:
        return questions_answers[questions[best_match_index]]
    return "Sorry, I couldn't understand your question. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
