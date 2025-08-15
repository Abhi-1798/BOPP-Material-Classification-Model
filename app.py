import os
import pickle
import re
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. SETUP AND INITIALIZATION ---

# Create the Flask web server
app = Flask(__name__)

# --- Download NLTK data during the build process ---
# This ensures that when Render builds your app, it has the necessary data.
print("Downloading NLTK data...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)
print("NLTK data is ready.")


# --- 2. LOAD THE SAVED MODEL PIPELINE ---

# Define the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'nlp_model_pipeline.pkl')

# Load the model from the pickle file
try:
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    print("Model pipeline loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    pipeline = None # Set to None if loading fails

# Extract components for easier access
vectorizer_mat = pipeline['vectorizer_material'] if pipeline else None
model_mat = pipeline['model_material'] if pipeline else None
vectorizer_sec = pipeline['vectorizer_secondary'] if pipeline else None
model_sec = pipeline['model_secondary'] if pipeline else None


# --- 3. RECREATE THE PREPROCESSING FUNCTION ---
# This MUST be identical to the function used for training.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized_tokens)


# --- 4. CREATE THE API ENDPOINT ---

@app.route("/predict", methods=["POST"])
def predict():
    """Receives a JSON request with a 'description' and returns model predictions."""
    if not pipeline:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({"error": "Invalid input. 'description' key is required."}), 400

    description = data['description']

    # --- Prediction Logic ---
    cleaned_desc = preprocess_text(description)
    vectorized_desc_mat = vectorizer_mat.transform([cleaned_desc])
    predicted_material = model_mat.predict(vectorized_desc_mat)[0]

    if predicted_material == 'Others':
        mat, typ, cat = 'Others', 'Others', 'Others'
    elif predicted_material == 'BOPP':
        if model_sec:
            vectorized_desc_sec = vectorizer_sec.transform([cleaned_desc])
            predicted_secondary = model_sec.predict(vectorized_desc_sec)
            mat, typ, cat = predicted_material, predicted_secondary[0][0], predicted_secondary[0][1]
        else:
            mat, typ, cat = predicted_material, "Not Available", "Not Available"
    else:
        mat, typ, cat = predicted_material, "Not Available", "Not Available"

    response = {
        "input_description": description,
        "predicted_material": mat,
        "predicted_type": typ,
        "predicted_category": cat
    }
    
    return jsonify(response)

# Health check endpoint for Render
@app.route("/")
def index():
    return "NLP Prediction API is running!", 200
