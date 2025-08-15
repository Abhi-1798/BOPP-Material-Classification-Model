import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from io import StringIO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Product Classification Engine",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NLTK DATA DOWNLOAD ---
# Download necessary NLTK data if not already present
@st.cache_data
def download_nltk_data():
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

download_nltk_data()

# --- MODEL LOADING ---
# Use caching to load the model only once
@st.cache_resource
def load_model_pipeline():
    """Loads the saved model pipeline from the pickle file."""
    model_path = os.path.join(os.path.dirname(__file__), 'nlp_model_pipeline.pkl')
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please make sure it's in the same directory as app.py.")
        return None

pipeline = load_model_pipeline()
if pipeline:
    vectorizer_mat = pipeline['vectorizer_material']
    model_mat = pipeline['model_material']
    vectorizer_sec = pipeline['vectorizer_secondary']
    model_sec = pipeline['model_secondary']

# --- PREPROCESSING FUNCTION ---
@st.cache_data
def preprocess_text(text):
    """Cleans and prepares text data for modeling."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized_tokens)

# --- PREDICTION FUNCTION ---
def predict(description):
    """Predicts material, type, and category for a single description."""
    if not pipeline:
        return "Model not loaded", "", ""
        
    cleaned_desc = preprocess_text(description)
    vectorized_desc_mat = vectorizer_mat.transform([cleaned_desc])
    predicted_material = model_mat.predict(vectorized_desc_mat)[0]

    if predicted_material == 'Others':
        return 'Others', 'Others', 'Others'
    elif predicted_material == 'BOPP':
        if model_sec:
            vectorized_desc_sec = vectorizer_sec.transform([cleaned_desc])
            predicted_secondary = model_sec.predict(vectorized_desc_sec)
            return predicted_material, predicted_secondary[0][0], predicted_secondary[0][1]
        else:
            return predicted_material, "Not Available", "Not Available"
    return predicted_material, "Not Available", "Not Available"

# --- UI LAYOUT ---
st.title("📦 Product Classification Engine")
st.markdown("This tool uses a trained NLP model to automatically classify products based on their description. Try a single prediction or upload a CSV for batch processing.")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["**Single Prediction**", "**Batch Prediction (CSV)**"])


# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.header("Classify a Single Product")
    
    # Input text area
    user_input = st.text_area("Enter Product Description Here:", height=150, placeholder="e.g., BOPP FILMS 50TTOPSL TRANSPARENT")

    if st.button("Classify Product", type="primary"):
        if user_input.strip() and pipeline:
            with st.spinner("🔍 Analyzing description..."):
                material, p_type, category = predict(user_input)
            
            st.success("✅ Classification Complete!")
            st.subheader("Results:")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Material", material)
            col2.metric("Predicted Type", p_type)
            col3.metric("Predicted Category", category)
        elif not pipeline:
            st.error("Cannot classify: Model is not loaded.")
        else:
            st.warning("Please enter a product description.")

# --- TAB 2: BATCH PREDICTION ---
with tab2:
    st.header("Classify a Batch of Products")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            
            if 'Product Description' not in df.columns:
                st.error("Error: The CSV file must contain a column named 'Product Description'.")
            else:
                st.success(f"File '{uploaded_file.name}' uploaded successfully. Found {len(df)} rows.")
                
                if st.button("Process File", type="primary"):
                    with st.spinner(f"Processing {len(df)} rows... This may take a moment."):
                        
                        results = []
                        # Progress bar
                        progress_bar = st.progress(0)
                        
                        for index, row in df.iterrows():
                            description = row['Product Description']
                            material, p_type, category = predict(description)
                            results.append({
                                'Product Description': description,
                                'Predicted_Material': material,
                                'Predicted_Type': p_type,
                                'Predicted_Category': category
                            })
                            # Update progress
                            progress_bar.progress((index + 1) / len(df))
                    
                    st.success("🎉 Batch processing complete!")
                    
                    result_df = pd.DataFrame(results)
                    
                    st.subheader("Results Preview:")
                    st.dataframe(result_df.head())
                    
                    # Provide a download link for the results
                    csv_output = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_output,
                        file_name='classified_products.csv',
                        mime='text/csv',
                    )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# --- SIDEBAR ---
st.sidebar.header("About")
st.sidebar.info(
    "This application is a demonstration of a multi-label text classification pipeline. "
    "The model is trained to predict a product's Material, Type, and Category from its description."
)
st.sidebar.header("Model Components")
if pipeline:
    st.sidebar.markdown("- **Material Classifier**: Predicts 'BOPP' vs 'Others'.")
    if model_sec:
        st.sidebar.markdown("- **Secondary Classifier**: Predicts 'Type' and 'Category' for BOPP products.")
    else:
        st.sidebar.warning("Secondary model not loaded.")
else:
    st.sidebar.error("Model pipeline not loaded.")
