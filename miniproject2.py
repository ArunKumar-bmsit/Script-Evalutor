import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Improved Text Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize text with a more refined tokenizer that excludes punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(filtered_tokens)  # Return filtered text

# Using Sentence-BERT for Sentence Embeddings
def get_sbert_embeddings(sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Pre-trained SBERT model
    embeddings = model.encode(sentences)  # Get embeddings for all sentences
    return np.array(embeddings)

# Evaluate Answers using cosine similarity and scaled scores
def evaluate_answers(answer_sheet, evaluation_sheet, marks):
    # Preprocess the answer sheet and evaluation sheet
    answer_sheet_processed = [preprocess_text(answer) for answer in answer_sheet]
    evaluation_sheet_processed = [preprocess_text(evaluation) for evaluation in evaluation_sheet]

    # Get embeddings for both answer and evaluation sheets
    all_sentences = answer_sheet_processed + evaluation_sheet_processed
    embeddings = get_sbert_embeddings(all_sentences)

    # Split embeddings into answer and evaluation embeddings
    answer_embeddings = embeddings[:len(answer_sheet)]
    evaluation_embeddings = embeddings[len(answer_sheet):]

    # Calculate cosine similarity between each answer and its corresponding evaluation
    scores = []
    for answer, eval_vec in zip(answer_embeddings, evaluation_embeddings):
        similarity = cosine_similarity([answer], [eval_vec])[0][0]
        scores.append(similarity)

    # Scale scores to marks
    final_scores = [round(score * marks, 2) for score in scores]

    # Calculate and return average score
    average_score = round(np.mean(final_scores), 2)

    return average_score

# Example input
answer_sheet = [
    "The quick brown fox jumps over the lazy dog."
]
evaluation_sheet = [
    "A quick brown fox leaped over a lazy dog."
]
marks = 100

# Evaluate the answers and get an aggregated score
average_score = evaluate_answers(answer_sheet, evaluation_sheet, marks)
print(f"Average Score: {average_score}")
