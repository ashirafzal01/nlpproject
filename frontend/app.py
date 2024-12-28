from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)
print("Flask app initialized")

# Load pre-trained model and vectorizer
model = joblib.load("./global_models_and_dataset/news_category_model.pkl")
print("Global matrix loaded")

vectorizer = joblib.load("./global_models_and_dataset/tfidf_vectorizer.pkl")
print("Global vectorizer loaded")

df = None

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def recommend_articles(input_content, predicted_category):

    global df

    # Load the dataset for the predicted category
    category_df = pd.read_csv(f'./categorized_models_and_datasets/cat_datasets/category_{predicted_category}.csv')
    df = category_df

    # Load the pre-saved TF-IDF matrix for the predicted category
    category_tfidf_matrix = joblib.load(f'./categorized_models_and_datasets/cat_tfidf_matrix/category_{predicted_category}_tfidf.pkl')

    category_tfidf_vector = joblib.load(f'./categorized_models_and_datasets/cat_tfidf_vect/category_{predicted_category}_tfidf_vectorizer.pkl')

    input_vector_new = category_tfidf_vector.transform([input_content])

    # Calculate cosine similarity between the input vector and the category's TF-IDF matrix
    similarities = cosine_similarity(input_vector_new, category_tfidf_matrix)

    # Get the indices of the top N most similar articles
    top_indices = np.argsort(similarities.flatten())[-3:][::-1]

    # Fetch the recommended articles based on top indices
    recommended_articles = category_df.iloc[top_indices]

    recommendations = []
    for i, (index1, row) in enumerate(recommended_articles.iterrows(), 1):
        truncated_content = ' '.join(row['text'].split('\n')[:3]) + '...'
        recommendations.append({
            "name": f"Article {i}",  # Add the display name
            "content": truncated_content,
            # "category": row['target'],
            "category": predicted_category,
            "full_content": row['text'],
            "original_index": index1  # Include the original index
        })

    return recommendations


@app.route('/styles.css')
def styles():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'styles.css')


@app.route("/", methods=["GET"])
def index():
    # Render the index.html template when a GET request is made to the root URL
    return render_template('index.html')


@app.route('/article/<int:index>', methods=["GET"])
def article(index):
    name = request.args.get('name', f"Article {index + 1}")  # Default to "Article X" if name is missing
    if 0 <= index < len(df):
        article_data = df.iloc[index].to_dict()
        article_data['name'] = name  # Use the passed name
        article_data['full_content'] = article_data['text']  # Pass the full content
        return render_template('article.html', article=article_data)
    else:
        return render_template('article.html', error="Article not found")
        
@app.route('/get_article/<int:index>', methods=["GET"])
def get_article(index):
    if 0 <= index < len(df):  # Use df instead of articles
        article_data = df.iloc[index].to_dict()  # Fetch directly from df
        return jsonify({
            "text": article_data["text"],
            "target": article_data["target"],
        })
    else:
        return jsonify({"error": "Article not found"}), 404


@app.route("/categorize", methods=["POST"])
def categorize():

    data = request.json  # Use JSON for better data handling
    title = data.get("title", "")
    content = data.get("content", "")

    # if not title or not content:
    if not content:
        # return jsonify({"error": "Title and content are required."}), 400
        return jsonify({"error": "Content is required."}), 400

    
    # Preprocess the input text
    combined_text = title + " " + content
    cleaned_input_text = preprocess_text(combined_text)
    input_vector = vectorizer.transform([cleaned_input_text])
    predicted_category = model.predict(input_vector)[0]
    print("Prdicted category:", predicted_category)

    # Recommend articles
    recommendations = recommend_articles(cleaned_input_text, predicted_category)

    return jsonify({
        "category": predicted_category,
        "recommendations": recommendations
    })

    data = request.json  # Use JSON for better data handling
    title = data.get("title", "")
    content = data.get("content", "")

    # if not title or not content:
    if not content:
        return jsonify({"error": "Content is required."}), 400

    combined_text = title + " " + content

    # Predict category
    text_vector = vectorizer.transform([combined_text])
    predicted_category = model.predict(text_vector)[0]

    # Recommend articles
    recommendations = recommend_articles(combined_text)

    return jsonify({
        "category": predicted_category,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    print("\nInitializing")
    app.run(debug=True)