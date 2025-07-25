import streamlit as st
import pandas as pd
from transformers import pipeline # Using the standard Python transformers library
import io

# For Keyword Extraction (TF-IDF) and Keyword Search
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import re # For text cleaning

# For Semantic Search
from sentence_transformers import SentenceTransformer, util


# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger') # Needed for pos_tag
except LookupError:
    nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

# --- Configuration ---
# Define available ZSC models
AVAILABLE_ZSC_MODELS = {
    "DeBERTa-v3-xsmall (MoritzLaurer)": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    "XtremeDistil-l6-h256 (MoritzLaurer)": "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
    "DistilBERT-base-uncased-MNLI (Typeform)": "typeform/distilbert-base-uncased-mnli",
    "BART-large-MNLI (Facebook)": "facebook/bart-large-mnli"
}

# Define available NER models
AVAILABLE_NER_MODELS = {
    "BERT-base-NER (dslim)": "dslim/bert-base-NER",
    "BERT-large-cased-finetuned-conll03-english (dbmdz)": "dbmdz/bert-large-cased-finetuned-conll03-english"
}

# Define available Sentiment Analysis models
AVAILABLE_SENTIMENT_MODELS = {
    "DistilBERT-Sentiment (sst-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT-base-uncased-sentiment (cardiffnlp)": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}

# Define available Summarization models
AVAILABLE_SUMMARIZATION_MODELS = {
    "BART-large-CNN (facebook)": "facebook/bart-large-cnn",
    "T5-small (google)": "t5-small",
    "DistilBART-CNN-12-6 (sshleifer)": "sshleifer/distilbart-cnn-12-6"
}

# Define available Semantic Search models (Sentence Transformers)
AVAILABLE_SEMANTIC_MODELS = {
    "all-MiniLM-L6-v2 (fast)": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (balanced)": "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1 (Q&A optimized)": "multi-qa-MiniLM-L6-cos-v1"
}


MAX_TEXT_LENGTH = 512 # Max tokens for the model, adjust if needed. Long texts will be truncated.

# --- Helper Functions ---

@st.cache_resource
def load_zsc_pipeline(model_id: str):
    """
    Loads the Zero-Shot Classification pipeline from Hugging Face.
    This function is cached to avoid reloading the model on every rerun.
    """
    st.info(f"Loading Zero-Shot Classification model: {model_id}. This may take a moment on first run as the model downloads to the server.")
    try:
        classifier = pipeline("zero-shot-classification", model=model_id)
        st.success("ZSC Model loaded successfully!")
        return classifier
    except Exception as e:
        st.error(f"Error loading ZSC model: {e}. Please ensure you have the 'transformers' library installed and an internet connection on the server.")
        st.stop()

@st.cache_resource
def load_ner_pipeline(model_id: str):
    """
    Loads the Named Entity Recognition pipeline from Hugging Face.
    This function is cached to avoid reloading the model on every rerun.
    """
    st.info(f"Loading Named Entity Recognition model: {model_id}. This may take a moment on first run as the model downloads to the server.")
    try:
        ner_model = pipeline("ner", model=model_id, aggregation_strategy="simple") # simple strategy aggregates tokens into entities
        st.success("NER Model loaded successfully!")
        return ner_model
    except Exception as e:
        st.error(f"Error loading NER model: {e}. Please ensure you have the 'transformers' library installed and an internet connection on the server.")
        st.stop()

@st.cache_resource
def load_sentiment_pipeline(model_id: str):
    """
    Loads the Sentiment Analysis pipeline from Hugging Face.
    This function is cached to avoid reloading the model on every rerun.
    """
    st.info(f"Loading Sentiment Analysis model: {model_id}. This may take a moment on first run as the model downloads to the server.")
    try:
        sentiment_model = pipeline("sentiment-analysis", model=model_id)
        st.success("Sentiment Model loaded successfully!")
        return sentiment_model
    except Exception as e:
        st.error(f"Error loading Sentiment Analysis model: {e}. Please ensure you have the 'transformers' library installed and an internet connection on the server.")
        st.stop()

@st.cache_resource
def load_summarization_pipeline(model_id: str):
    """
    Loads the Summarization pipeline from Hugging Face.
    This function is cached to avoid reloading the model on every rerun.
    """
    st.info(f"Loading Summarization model: {model_id}. This may take a moment on first run as the model downloads to the server.")
    try:
        summarizer = pipeline("summarization", model=model_id)
        st.success("Summarization Model loaded successfully!")
        return summarizer
    except Exception as e:
        st.error(f"Error loading Summarization model: {e}. Please ensure you have the 'transformers' library installed and an internet connection on the server.")
        st.stop()

@st.cache_resource
def load_semantic_model(model_id: str):
    """
    Loads a SentenceTransformer model for semantic search.
    """
    st.info(f"Loading Semantic Search model: {model_id}. This may take a moment on first run as the model downloads to the server.")
    try:
        semantic_model = SentenceTransformer(model_id)
        st.success("Semantic Search Model loaded successfully!")
        return semantic_model
    except Exception as e:
        st.error(f"Error loading Semantic Search model: {e}. Please ensure you have the 'sentence-transformers' library installed and an internet connection on the server.")
        st.stop()


@st.cache_resource
def get_tfidf_vectorizer(ngram_range=(1, 1)):
    """
    Initializes and caches a TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=ngram_range)
    return vectorizer

@st.cache_resource
def get_lemmatizer():
    """
    Initializes and caches a WordNetLemmatizer.
    """
    return WordNetLemmatizer()

def get_wordnet_pos(tag):
    """Map NLTK POS tags to WordNetLemmatizer POS tags."""
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN # Default to noun if no clear tag

def preprocess_text_for_search(text):
    """
    Cleans, tokenizes, lowercases, and lemmatizes the input text.
    Removes non-alphanumeric characters but keeps spaces for tokenization.
    """
    if not isinstance(text, str):
        return []
    
    # Remove special characters and numbers, keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())
    
    lemmatized_tokens = []
    # Perform POS tagging to improve lemmatization accuracy
    tagged_tokens = nltk.pos_tag(tokens) # This line is new

    for word, tag in tagged_tokens:
        w_pos = get_wordnet_pos(tag)
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=w_pos)) # Use pos argument
    
    return lemmatized_tokens
    
def improved_keyword_search(text_to_search, keywords_list):
    """
    Performs a keyword search on the given text, leveraging lemmatization
    for both the text and the keywords, and supports multi-word phrases.

    Args:
        text_to_search (str): The main text to search within.
        keywords_list (list): A list of keywords or phrases to search for.
                              Each element can be a single word or a phrase
                              (e.g., ["apple", "banana", "try hard"]).

    Returns:
        bool: True if any keyword/phrase is found (after lemmatization),
              False otherwise.
    """
    # 1. Preprocess the main text
    lemmatized_text_words = preprocess_text_for_search(text_to_search)
    # print(f"Lemmatized Text: {lemmatized_text_words}") # For debugging

    # 2. Iterate through each keyword/phrase provided
    for keyword_phrase in keywords_list:
        # Preprocess the current keyword phrase
        lemmatized_keyword_phrase_words = preprocess_text_for_search(keyword_phrase)
        # print(f"Lemmatized Keyword Phrase '{keyword_phrase}': {lemmatized_keyword_phrase_words}") # For debugging

        # If the keyword phrase is empty after preprocessing (e.g., just punctuation), skip it
        if not lemmatized_keyword_phrase_words:
            continue

        # 3. Check if the lemmatized keyword phrase (sequence of words)
        #    exists as a sub-sequence in the lemmatized text.
        #    This is similar to "protein alignment" for word sequences.
        phrase_len = len(lemmatized_keyword_phrase_words)
        text_len = len(lemmatized_text_words)

        for i in range(text_len - phrase_len + 1):
            # Check if the slice of the text matches the keyword phrase
            if lemmatized_text_words[i:i + phrase_len] == lemmatized_keyword_phrase_words:
                return True # Found a match!

    return False # No match found after checking all keywords
    
def __preprocess_text_for_search__(text):
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes text for keyword search.
    """
    if not isinstance(text, str):
        return []

    lemmatizer = get_lemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation and special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


def classify_text(classifier, text_to_classify, candidate_labels, threshold: float):
    """
    Performs zero-shot classification on a single text string, returning all labels
    and scores sorted by confidence, and optionally filtered by a threshold.
    Truncates the text if it's too long for the model.
    """
    if not text_to_classify or not isinstance(text_to_classify, str):
        return "N/A", 0.0, []

    if len(text_to_classify.split()) > MAX_TEXT_LENGTH:
        text_to_classify = " ".join(text_to_classify.split()[:MAX_TEXT_LENGTH])

    try:
        result = classifier(text_to_classify, candidate_labels, multi_label=True)

        all_labels_with_scores = []
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold:
                all_labels_with_scores.append({"label": label, "score": score})

        formatted_all_labels = ", ".join([f"{item['label']}: {item['score']:.2f}" for item in all_labels_with_scores])

        predicted_label = all_labels_with_scores[0]['label'] if all_labels_with_scores else 'N/A'
        confidence_score = all_labels_with_scores[0]['score'] if all_labels_with_scores else 0.0

        return predicted_label, confidence_score, formatted_all_labels
    except Exception as e:
        st.error(f"Error classifying text: {e}")
        return "Error", 0.0, "Error during classification"

def extract_entities(ner_model, text_to_analyze):
    """
    Performs Named Entity Recognition on a single text string.
    Returns a dictionary of entity types to lists of extracted entities.
    """
    if not text_to_analyze or not isinstance(text_to_analyze, str):
        return {}

    if len(text_to_analyze.split()) > MAX_TEXT_LENGTH:
        text_to_analyze = " ".join(text_to_analyze.split()[:MAX_TEXT_LENGTH])

    try:
        entities = ner_model(text_to_analyze)

        # Initialize dictionary to hold entities by type
        extracted_by_type = {
            "PER": [], # Person
            "ORG": [], # Organization
            "LOC": [], # Location
            "MISC": [] # Miscellaneous
        }

        for entity in entities:
            entity_type = entity['entity_group']
            entity_text = entity['word']
            if entity_type in extracted_by_type:
                extracted_by_type[entity_type].append(entity_text)
            else:
                # Handle other entity types if the model provides them, or add to MISC
                extracted_by_type["MISC"].append(entity_text)

        # Remove duplicates and return
        return {k: list(set(v)) for k, v in extracted_by_type.items()}

    except Exception as e:
        st.error(f"Error extracting entities: {e}")
        return {"Error": ["Error during NER"]}

def analyze_sentiment(sentiment_model, text_to_analyze):
    """
    Performs sentiment analysis on a single text string.
    Returns the sentiment label (e.g., 'POSITIVE', 'NEGATIVE') and its score.
    """
    if not text_to_analyze or not isinstance(text_to_analyze, str):
        return "N/A", 0.0

    if len(text_to_analyze.split()) > MAX_TEXT_LENGTH:
        text_to_analyze = " ".join(text_to_analyze.split()[:MAX_TEXT_LENGTH])

    try:
        result = sentiment_model(text_to_analyze)
        # The sentiment-analysis pipeline typically returns a list of dicts, e.g.,
        # [{'label': 'POSITIVE', 'score': 0.999}]
        if result and len(result) > 0:
            return result[0]['label'], result[0]['score']
        return "N/A", 0.0
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return "Error", 0.0

def perform_summarization(summarizer, text_to_summarize, min_len, max_len):
    """
    Performs summarization on a single text string.
    Returns the generated summary.
    """
    if not text_to_summarize or not isinstance(text_to_summarize, str):
        return "N/A"

    # Summarization models can handle longer texts, their tokenizers will truncate if too long.
    # The min_length and max_length parameters control the output summary length.
    try:
        # Ensure text is not empty after stripping
        if not text_to_summarize.strip():
            return "Empty Text"

        summary = summarizer(text_to_summarize, min_length=min_len, max_length=max_len, do_sample=False)
        if summary and len(summary) > 0:
            return summary[0]['summary_text']
        return "No summary generated."
    except Exception as e:
        st.error(f"Error performing summarization: {e}")
        return "Error during summarization."


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Unstructured Text Data Analyzer")

st.title("📄 Unstructured Text Data Analyzer")
st.markdown("""
This application provides quick and easy analysis of your text data using Artificial Intelligence and Machine Learning tools.
You can perform **Zero-Shot Classification (ZSC)** to categorize text into custom labels or **Named Entity Recognition (NER)** to extract key entities. You can also perform **Sentiment Analysis** to gauge the emotional tone of your text, and **Summarization** to generate concise versions of your documents. Finally, **Keyword Extraction (TF-IDF)** helps identify the most important words and phrases.
""")

# --- Sidebar for Data Upload ---
st.sidebar.header("Data & Model Configuration")

st.sidebar.subheader("1. Upload Text Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file containing your text data", type=["csv", "xlsx"])

data_df = pd.DataFrame()
text_columns_available = [] # Columns available for selection in ZSC/NER tabs

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data_df = pd.read_csv(uploaded_file)
        else: # .xlsx
            data_df = pd.read_excel(uploaded_file)

        st.sidebar.success("File uploaded successfully!")
        st.sidebar.write(f"Loaded {data_df.shape[0]} rows and {data_df.shape[1]} columns.")

        # Identify potential text columns for selection in tabs
        text_columns_available = data_df.select_dtypes(include=['object', 'string']).columns.tolist()
        if not text_columns_available:
            st.sidebar.warning("No text (string/object) columns detected in your data. Please ensure your text data is in a column with a string data type.")
            text_columns_available = data_df.columns.tolist() # Allow selection of any column if no text detected

    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}. Please ensure it's a valid CSV or Excel format.")
        st.stop()
else:
    st.sidebar.info("Please upload your text data to begin analysis.")


# --- Main Content Area (Tabs) ---
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Data Preview", "Zero-Shot Classification (ZSC)", "Named Entity Recognition (NER)", "Sentiment Analysis", "Summarization", "Keyword Extraction (TF-IDF)", "Keyword Search", "Semantic Search"])

# --- Tab 0: Data Preview ---
with tab0:
    st.header("Data Preview")
    st.markdown("""
    This tab provides a quick overview of your uploaded dataset.
    """)
    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to preview it.")
    else:
        st.subheader("Your Data:")
        st.dataframe(data_df)
        st.subheader("Data Information:")
        # Display basic info like column names, non-null counts, dtypes
        buffer = io.StringIO()
        data_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.subheader("Descriptive Statistics (Numeric Columns):")
        st.dataframe(data_df.describe())


# --- Tab 1: Zero-Shot Classification ---
with tab1:
    st.header("Zero-Shot Classification (ZSC)")
    st.markdown("""
    Categorize your text data into custom labels without needing pre-labeled examples.
    The model understands the meaning of your text and categories to make predictions.
    """)

    # ZSC Model Selection moved to tab
    st.subheader("Select ZSC Model")
    selected_zsc_model_name = st.selectbox(
        "Choose a pre-trained Zero-Shot Classification model:",
        options=list(AVAILABLE_ZSC_MODELS.keys()),
        index=0,
        key="zsc_model_selector", # Unique key for this selectbox
        help="Smaller models (e.g., xsmall, XtremeDistil) are faster but might be slightly less accurate. Larger models (e.g., BART-large) offer higher accuracy but are slower."
    )
    selected_zsc_model_id = AVAILABLE_ZSC_MODELS[selected_zsc_model_name]
    classifier = load_zsc_pipeline(selected_zsc_model_id) # Load model here

    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to use ZSC.")
    elif not text_columns_available:
        st.warning("No suitable text columns found in your data for ZSC. Please check your data or upload a file with text columns.")
    else:
        # User selects text columns for ZSC
        default_selected_zsc_cols = [col for col in ['text', 'article', 'description', 'comment', 'body', 'content'] if col in text_columns_available]
        if not default_selected_zsc_cols and text_columns_available:
            default_selected_zsc_cols = [text_columns_available[0]]

        selected_zsc_text_cols = st.multiselect(
            "Select one or more columns containing text for ZSC:",
            text_columns_available,
            default=default_selected_zsc_cols,
            key="zsc_text_cols_selector",
            help="If multiple columns are selected, their text content will be concatenated for classification."
        )

        if not selected_zsc_text_cols:
            st.warning("Please select at least one text column for Zero-Shot Classification.")
        else:
            # --- Define Categories for ZSC ---
            st.subheader("Define Your Categories")
            st.markdown("""
            You can define your categories below, or **upload a custom CSV/Excel file** to overwrite them.
            The file should contain columns: `candidate_label`, `candidate_description`, `keywords`.
            """)

            custom_labels_file = st.file_uploader("Upload custom categories (CSV or Excel)", type=["csv", "xlsx"], key="custom_labels_uploader_zsc")

            categories_data = []
            if custom_labels_file is not None:
                try:
                    if custom_labels_file.name.endswith('.csv'):
                        custom_df = pd.read_csv(custom_labels_file)
                    else:
                        custom_df = pd.read_excel(custom_labels_file)

                    required_cols = ["candidate_label", "candidate_description", "keywords"]
                    if not all(col in custom_df.columns for col in required_cols):
                        st.error(f"Uploaded file must contain all required columns: {', '.join(required_cols)}")
                        st.stop()

                    categories_data = custom_df.to_dict('records')
                    st.success("Custom categories loaded successfully!")
                    st.info("The table below is now populated with your uploaded categories.")

                except Exception as e:
                    st.error(f"Error reading custom labels file: {e}. Please ensure it's a valid CSV or Excel format with the correct columns.")
                    st.stop()
            else:
                default_categories = [
                    {"candidate_label": "Positive Sentiment", "candidate_description": "Expresses joy, approval, or positive feelings.", "keywords": "happy, good, excellent, love, great, positive"},
                    {"candidate_label": "Negative Sentiment", "candidate_description": "Expresses sadness, anger, disappointment, or criticism.", "keywords": "bad, terrible, hate, awful, negative, difficult"},
                    {"candidate_label": "Product Feature Request", "candidate_description": "Suggests new features or improvements for a product.", "keywords": "feature, add, new, improve, suggestion, request"},
                    {"candidate_label": "Bug Report", "candidate_description": "Describes a technical issue or malfunction.", "keywords": "bug, error, crash, broken, issue, problem"},
                    {"candidate_label": "Customer Support Query", "candidate_description": "Asks for help or information from customer service.", "keywords": "help, support, question, how to, contact, assistance"},
                    {"candidate_label": "General Inquiry", "candidate_description": "A general question or comment not fitting other categories.", "keywords": "general, question, comment, info, about"},
                ]
                categories_data = default_categories

            categories_df = st.data_editor(
                pd.DataFrame(categories_data),
                num_rows="dynamic",
                column_config={
                    "candidate_label": st.column_config.TextColumn("Category Label", help="The label the ZSC model will use for classification."),
                    "candidate_description": st.column_config.TextColumn("Description", help="A detailed description of this category."),
                    "keywords": st.column_config.TextColumn("Keywords", help="Comma-separated keywords relevant to this category (for your reference)."),
                },
                hide_index=True,
                key="categories_editor_zsc"
            )

            candidate_labels = categories_df["candidate_label"].tolist() if not categories_df.empty else []

            if not candidate_labels:
                st.warning("Please define at least one category label to proceed with ZSC.")
            else:
                # --- Analyze Text for ZSC ---
                st.subheader("Analyze Text for Categories")

                user_confidence_threshold = st.slider(
                    "Adjust Confidence Threshold for Multi-Label Output:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.01,
                    step=0.01,
                    help="Only labels with a confidence score equal to or above this threshold will be included in the 'All Predicted Labels' column. Higher values mean fewer, but potentially more precise, labels."
                )

                if st.button("Run ZSC Classification"):
                    st.info("Running Zero-Shot Classification... This might take a while depending on the number of rows and your server's performance.")

                    zsc_results = []
                    progress_bar = st.progress(0)
                    total_rows = len(data_df)

                    for i, row in data_df.iterrows():
                        combined_text = " ".join([
                            str(row[col]) for col in selected_zsc_text_cols if pd.notna(row[col])
                        ])

                        predicted_label, confidence_score, all_labels_formatted = classify_text(
                            classifier, combined_text, candidate_labels, threshold=user_confidence_threshold
                        )

                        row_result = row.to_dict()
                        row_result["ZSC_Combined_Text"] = combined_text
                        row_result["ZSC_Top_Label"] = predicted_label
                        row_result["ZSC_Top_Confidence"] = f"{confidence_score:.2f}"
                        row_result["ZSC_All_Labels_Scores"] = all_labels_formatted

                        zsc_results.append(row_result)
                        progress_bar.progress((i + 1) / total_rows)

                    zsc_results_df = pd.DataFrame(zsc_results)
                    st.subheader("ZSC Classification Results")
                    st.dataframe(zsc_results_df)

                    csv_output_zsc = zsc_results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download ZSC Results as CSV",
                        data=csv_output_zsc,
                        file_name="zsc_classification_results.csv",
                        mime="text/csv",
                    )

                    st.subheader("ZSC Top Category Distribution")
                    label_counts_zsc = zsc_results_df["ZSC_Top_Label"].value_counts().reset_index()
                    label_counts_zsc.columns = ["Category Label", "Count"]
                    st.bar_chart(label_counts_zsc.set_index("Category Label"))

# --- Tab 2: Named Entity Recognition ---
with tab2:
    st.header("Named Entity Recognition (NER)")
    st.markdown("""
    Extract key entities (like Persons, Organizations, Locations, Miscellaneous) from your text data.
    """)

    # NER Model Selection moved to tab
    st.subheader("Select NER Model")
    selected_ner_model_name = st.selectbox(
        "Choose a pre-trained Named Entity Recognition model:",
        options=list(AVAILABLE_NER_MODELS.keys()),
        index=0,
        key="ner_model_selector", # Unique key for this selectbox
        help="Models like 'BERT-base-NER' are general-purpose. Larger models might offer more entity types or higher accuracy."
    )
    selected_ner_model_id = AVAILABLE_NER_MODELS[selected_ner_model_name]
    ner_model = load_ner_pipeline(selected_ner_model_id) # Load model here

    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to use NER.")
    elif not text_columns_available:
        st.warning("No suitable text columns found in your data for NER. Please check your data or upload a file with text columns.")
    else:
        # User selects text columns for NER
        default_selected_ner_cols = [col for col in ['text', 'article', 'description', 'comment', 'body', 'content'] if col in text_columns_available]
        if not default_selected_ner_cols and text_columns_available:
            default_selected_ner_cols = [text_columns_available[0]]

        selected_ner_text_cols = st.multiselect(
            "Select one or more columns containing text for NER:",
            text_columns_available,
            default=default_selected_ner_cols,
            key="ner_text_cols_selector",
            help="If multiple columns are selected, their text content will be concatenated for NER."
        )

        if not selected_ner_text_cols:
            st.warning("Please select at least one text column for Named Entity Recognition.")
        else:
            if st.button("Run NER"):
                st.info("Running Named Entity Recognition... This might take a while depending on the number of rows and your server's performance.")

                ner_results_list = []
                progress_bar_ner = st.progress(0)
                total_rows_ner = len(data_df)

                for i, row in data_df.iterrows():
                    combined_text_ner = " ".join([
                        str(row[col]) for col in selected_ner_text_cols if pd.notna(row[col])
                    ])

                    extracted_entities = extract_entities(ner_model, combined_text_ner)

                    row_result_ner = row.to_dict()
                    row_result_ner["NER_Combined_Text"] = combined_text_ner

                    # Add extracted entities as new columns
                    for entity_type, entities_list in extracted_entities.items():
                        row_result_ner[f"NER_Entities_{entity_type}"] = ", ".join(entities_list) if entities_list else "N/A"

                    ner_results_list.append(row_result_ner)
                    progress_bar_ner.progress((i + 1) / total_rows_ner)

                ner_results_df = pd.DataFrame(ner_results_list)
                st.subheader("NER Results")
                st.dataframe(ner_results_df)

                csv_output_ner = ner_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download NER Results as CSV",
                    data=csv_output_ner,
                    file_name="ner_results.csv",
                    mime="text/csv",
                )

                # Optional: Display top entities found
                st.subheader("Top Extracted Entities")
                all_extracted = {
                    "PER": [], "ORG": [], "LOC": [], "MISC": []
                }
                for row_dict in ner_results_list:
                    for entity_type in all_extracted.keys():
                        entities_str = row_dict.get(f"NER_Entities_{entity_type}", "")
                        if entities_str and entities_str != "N/A":
                            all_extracted[entity_type].extend([e.strip() for e in entities_str.split(',')])

                for entity_type, entities in all_extracted.items():
                    if entities:
                        st.write(f"**Top {entity_type} Entities:**")
                        entity_counts = pd.Series(entities).value_counts().head(10)
                        st.dataframe(entity_counts)
                    else:
                        st.write(f"No {entity_type} entities found.")

# --- Tab 3: Sentiment Analysis ---
with tab3:
    st.header("Sentiment Analysis")
    st.markdown("""
    Determine the emotional tone of your text (e.g., Positive, Negative, Neutral).
    """)

    # Sentiment Model Selection moved to tab
    st.subheader("Select Sentiment Analysis Model")
    selected_sentiment_model_name = st.selectbox(
        "Choose a pre-trained Sentiment Analysis model:",
        options=list(AVAILABLE_SENTIMENT_MODELS.keys()),
        index=0,
        key="sentiment_model_selector", # Unique key for this selectbox
        help="Models are generally fine-tuned for specific sentiment tasks (e.g., movie reviews, social media)."
    )
    selected_sentiment_model_id = AVAILABLE_SENTIMENT_MODELS[selected_sentiment_model_name]
    sentiment_model = load_sentiment_pipeline(selected_sentiment_model_id) # Load model here

    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to use Sentiment Analysis.")
    elif not text_columns_available:
        st.warning("No suitable text columns found in your data for Sentiment Analysis. Please check your data or upload a file with text columns.")
    else:
        # User selects text columns for Sentiment Analysis
        default_selected_sentiment_cols = [col for col in ['text', 'comment', 'review', 'feedback', 'description', 'body', 'content'] if col in text_columns_available]
        if not default_selected_sentiment_cols and text_columns_available:
            default_selected_sentiment_cols = [text_columns_available[0]]

        selected_sentiment_text_cols = st.multiselect(
            "Select one or more columns containing text for Sentiment Analysis:",
            text_columns_available,
            default=default_selected_sentiment_cols,
            key="sentiment_text_cols_selector",
            help="If multiple columns are selected, their text content will be concatenated for analysis."
        )

        if not selected_sentiment_text_cols:
            st.warning("Please select at least one text column for Sentiment Analysis.")
        else:
            if st.button("Run Sentiment Analysis"):
                st.info("Running Sentiment Analysis... This might take a while depending on the number of rows and your server's performance.")

                sentiment_results_list = []
                progress_bar_sentiment = st.progress(0)
                total_rows_sentiment = len(data_df)

                for i, row in data_df.iterrows():
                    combined_text_sentiment = " ".join([
                        str(row[col]) for col in selected_sentiment_text_cols if pd.notna(row[col])
                    ])

                    sentiment_label, sentiment_score = analyze_sentiment(sentiment_model, combined_text_sentiment)

                    row_result_sentiment = row.to_dict()
                    row_result_sentiment["SA_Combined_Text"] = combined_text_sentiment
                    row_result_sentiment["SA_Label"] = sentiment_label
                    row_result_sentiment["SA_Confidence"] = f"{sentiment_score:.2f}"

                    sentiment_results_list.append(row_result_sentiment)
                    progress_bar_sentiment.progress((i + 1) / total_rows_sentiment)

                sentiment_results_df = pd.DataFrame(sentiment_results_list)
                st.subheader("Sentiment Analysis Results")
                st.dataframe(sentiment_results_df)

                csv_output_sentiment = sentiment_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Sentiment Results as CSV",
                    data=csv_output_sentiment,
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )

                st.subheader("Sentiment Distribution")
                sentiment_counts = sentiment_results_df["SA_Label"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment Label", "Count"]
                st.bar_chart(sentiment_counts.set_index("Sentiment Label"))

# --- Tab 4: Summarization ---
with tab4:
    st.header("Summarization")
    st.markdown("""
    Generate concise summaries from your longer text documents.
    """)

    # Summarization Model Selection moved to tab
    st.subheader("Select Summarization Model")
    selected_summarization_model_name = st.selectbox(
        "Choose a pre-trained Summarization model:",
        options=list(AVAILABLE_SUMMARIZATION_MODELS.keys()),
        index=0,
        key="summarization_model_selector", # Unique key for this selectbox
        help="Models like 'BART-large-CNN' are generally good for news-style summarization. 'T5-small' is faster but summaries may be less fluent."
    )
    selected_summarization_model_id = AVAILABLE_SUMMARIZATION_MODELS[selected_summarization_model_name]
    summarizer = load_summarization_pipeline(selected_summarization_model_id) # Load model here

    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to use Summarization.")
    elif not text_columns_available:
        st.warning("No suitable text columns found in your data for Summarization. Please check your data or upload a file with text columns.")
    else:
        # User selects text columns for Summarization
        default_selected_summarization_cols = [col for col in ['text', 'article', 'document', 'body', 'content'] if col in text_columns_available]
        if not default_selected_summarization_cols and text_columns_available:
            default_selected_summarization_cols = [text_columns_available[0]]

        selected_summarization_text_cols = st.multiselect(
            "Select one or more columns containing text for Summarization:",
            text_columns_available,
            default=default_selected_summarization_cols,
            key="summarization_text_cols_selector",
            help="If multiple columns are selected, their text content will be concatenated for summarization."
        )

        if not selected_summarization_text_cols:
            st.warning("Please select at least one text column for Summarization.")
        else:
            st.subheader("Summary Length Options:")
            min_summary_length = st.slider(
                "Minimum Summary Length (tokens):",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="The minimum number of tokens (words/subwords) in the generated summary."
            )
            max_summary_length = st.slider(
                "Maximum Summary Length (tokens):",
                min_value=50,
                max_value=200,
                value=150,
                step=5,
                help="The maximum number of tokens (words/subwords) in the generated summary."
            )
            if min_summary_length >= max_summary_length:
                st.error("Minimum summary length must be less than maximum summary length.")
                st.stop()

            if st.button("Run Summarization"):
                st.info("Running Summarization... This might take a while depending on the number of rows and your server's performance.")

                summarization_results_list = []
                progress_bar_summarization = st.progress(0)
                total_rows_summarization = len(data_df)

                for i, row in data_df.iterrows():
                    combined_text_summarization = " ".join([
                        str(row[col]) for col in selected_summarization_text_cols if pd.notna(row[col])
                    ])

                    summary_text = perform_summarization(
                        summarizer, combined_text_summarization, min_summary_length, max_summary_length
                    )

                    row_result_summarization = row.to_dict()
                    row_result_summarization["SUM_Combined_Text"] = combined_text_summarization
                    row_result_summarization["SUM_Summary"] = summary_text

                    summarization_results_list.append(row_result_summarization)
                    progress_bar_summarization.progress((i + 1) / total_rows_summarization)

                summarization_results_df = pd.DataFrame(summarization_results_list)
                st.subheader("Summarization Results")
                st.dataframe(summarization_results_df)

                csv_output_summarization = summarization_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Summarization Results as CSV",
                    data=csv_output_summarization,
                    file_name="summarization_results.csv",
                    mime="text/csv",
                )

# --- Tab 5: Keyword Extraction (TF-IDF) ---
with tab5:
    st.header("Keyword Extraction (TF-IDF)")
    st.markdown("""
    Identify the most important words and phrases in your text data using TF-IDF (Term Frequency-Inverse Document Frequency).
    This helps in understanding the main topics and themes.
    """)

    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to use Keyword Extraction.")
    elif not text_columns_available:
        st.warning("No suitable text columns found in your data for Keyword Extraction. Please check your data or upload a file with text columns.")
    else:
        # User selects text columns for Keyword Extraction
        default_selected_keyword_cols = [col for col in ['text', 'article', 'description', 'comment', 'body', 'content'] if col in text_columns_available]
        if not default_selected_keyword_cols and text_columns_available:
            default_selected_keyword_cols = [text_columns_available[0]]

        selected_keyword_text_cols = st.multiselect(
            "Select one or more columns containing text for Keyword Extraction:",
            text_columns_available,
            default=default_selected_keyword_cols,
            key="keyword_text_cols_selector",
            help="If multiple columns are selected, their text content will be concatenated for analysis."
        )

        if not selected_keyword_text_cols:
            st.warning("Please select at least one text column for Keyword Extraction.")
        else:
            st.subheader("Keyword Extraction Options:")
            num_top_keywords = st.slider(
                "Number of Top Keywords per Document:",
                min_value=3,
                max_value=20,
                value=5,
                step=1,
                help="The maximum number of top TF-IDF keywords to extract for each document."
            )
            ngram_range_option = st.selectbox(
                "N-gram Range:",
                options=["1-gram (single words)", "1-2-grams (single words and two-word phrases)", "1-3-grams (single words, two-word, and three-word phrases)"],
                index=0,
                help="Select the length of word sequences to consider as keywords."
            )
            ngram_map = {
                "1-gram (single words)": (1, 1),
                "1-2-grams (single words and two-word phrases)": (1, 2),
                "1-3-grams (single words, two-word, and three-word phrases)": (1, 3)
            }
            selected_ngram_range = ngram_map[ngram_range_option]

            if st.button("Run Keyword Extraction"):
                st.info("Running Keyword Extraction... This might take a while depending on the number of rows.")

                keyword_results_list = []
                combined_texts_for_tfidf = []

                # First, combine texts for TF-IDF Vectorizer
                for i, row in data_df.iterrows():
                    combined_text_kw = " ".join([
                        str(row[col]) for col in selected_keyword_text_cols if pd.notna(row[col])
                    ])
                    combined_texts_for_tfidf.append(combined_text_kw)

                # Initialize and fit TF-IDF Vectorizer
                try:
                    tfidf_vectorizer = get_tfidf_vectorizer(ngram_range=selected_ngram_range)
                    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts_for_tfidf)
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                except Exception as e:
                    st.error(f"Error initializing or fitting TF-IDF Vectorizer: {e}")
                    st.stop()

                progress_bar_kw = st.progress(0)
                total_rows_kw = len(data_df)

                for i, row in data_df.iterrows():
                    combined_text_kw = combined_texts_for_tfidf[i] # Use the already combined text

                    # Get top TF-IDF keywords for the current document
                    row_tfidf_scores = tfidf_matrix[i].toarray().flatten()
                    top_indices = row_tfidf_scores.argsort()[-num_top_keywords:][::-1]
                    top_keywords = [feature_names[idx] for idx in top_indices if row_tfidf_scores[idx] > 0] # Only include if score > 0

                    row_result_kw = row.to_dict()
                    row_result_kw["KW_Combined_Text"] = combined_text_kw
                    row_result_kw["TFIDF_Keywords"] = ", ".join(top_keywords) if top_keywords else "N/A"

                    keyword_results_list.append(row_result_kw)
                    progress_bar_kw.progress((i + 1) / total_rows_kw)

                keyword_results_df = pd.DataFrame(keyword_results_list)
                st.subheader("Keyword Extraction Results (TF-IDF)")
                st.dataframe(keyword_results_df)

                csv_output_kw = keyword_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Keyword Results as CSV",
                    data=csv_output_kw,
                    file_name="keyword_results.csv",
                    mime="text/csv",
                )

                # Display overall top keywords
                st.subheader("Overall Top Keywords (TF-IDF)")
                if tfidf_matrix.shape[0] > 0:
                    # Sum TF-IDF scores across all documents for each feature
                    overall_tfidf_scores = tfidf_matrix.sum(axis=0).A1
                    overall_top_indices = overall_tfidf_scores.argsort()[-20:][::-1] # Top 20 overall
                    overall_top_keywords = [feature_names[idx] for idx in overall_top_indices if overall_tfidf_scores[idx] > 0]

                    if overall_top_keywords:
                        st.write("These are the most important keywords across your entire dataset:")
                        st.write(", ".join(overall_top_keywords))
                    else:
                        st.info("No significant keywords found across the dataset.")
                else:
                    st.info("No data to extract overall keywords from.")

# --- Tab 6: Keyword Search ---
with tab6:
    st.header("Keyword Search")
    st.markdown("""
    Search for specific keywords within your text data after applying text cleaning and lemmatization.
    """)

    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to use Keyword Search.")
    elif not text_columns_available:
        st.warning("No suitable text columns found in your data for Keyword Search. Please check your data or upload a file with text columns.")
    else:
        st.subheader("2. Enter Keywords to Search")
        keywords_input = st.text_input(
            "Enter keywords (comma-separated):",
            placeholder="e.g., product, bug, customer service",
            help="Enter words or phrases you want to search for. They will be cleaned and lemmatized before searching."
        )

        st.subheader("3. Select Text Columns to Search Within")
        default_selected_search_cols = [col for col in ['text', 'article', 'description', 'comment', 'body', 'content'] if col in text_columns_available]
        if not default_selected_search_cols and text_columns_available:
            default_selected_search_cols = [text_columns_available[0]]

        selected_search_text_cols = st.multiselect(
            "Select one or more columns to search:",
            text_columns_available,
            default=default_selected_search_cols,
            key="search_text_cols_selector",
            help="If multiple columns are selected, their text content will be combined for the search."
        )

        st.subheader("4. Choose Search Logic")
        search_logic = st.radio(
            "Match Logic:",
            ("Match ANY of the keywords (OR)", "Match ALL of the keywords (AND)"),
            index=0,
            help="Choose whether a row should match if it contains ANY of your keywords, or ALL of them."
        )

        if st.button("Run Keyword Search"):
            if not keywords_input:
                st.warning("Please enter at least one keyword to search.")
            elif not selected_search_text_cols:
                st.warning("Please select at least one text column to search within.")
            else:
                st.info("Running Keyword Search... This might take a while depending on the number of rows.")

                search_keywords_raw = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
                if not search_keywords_raw:
                    st.warning("No valid keywords entered after splitting by comma.")
                    st.stop()

                # --- MODIFICATION START ---
                # Display the lemmatized version of the phrases being searched
                display_keywords = []
                for kw_phrase in search_keywords_raw:
                    lemmatized_phrase_words = preprocess_text_for_search(kw_phrase)
                    if lemmatized_phrase_words: # Only add if not empty after preprocessing
                        display_keywords.append(" ".join(lemmatized_phrase_words))

                if not display_keywords:
                    st.warning("Keywords became empty after preprocessing (e.g., only stopwords or punctuation were entered). Please try different keywords.")
                    st.stop()

                st.write(f"Searching for processed keywords: **{', '.join(display_keywords)}**")
                # --- MODIFICATION END ---


                search_results_list = []
                progress_bar_search = st.progress(0)
                total_rows_search = len(data_df)

                for i, row in data_df.iterrows():
                    combined_text_search = " ".join([
                        str(row[col]) for col in selected_search_text_cols if pd.notna(row[col])
                    ])

                    match = False
                    if search_logic == "Match ANY of the keywords (OR)":
                        # The improved_keyword_search function naturally handles "ANY"
                        # when given a list of keywords/phrases.
                        match = improved_keyword_search(combined_text_search, search_keywords_raw)
                    else: # Match ALL of the keywords (AND)
                        # To match ALL, we need to ensure each individual keyword phrase is present.
                        # We do this by iterating through each keyword and checking if
                        # improved_keyword_search finds it when searched individually.
                        all_keywords_found = True
                        for single_keyword_phrase in search_keywords_raw:
                            if not improved_keyword_search(combined_text_search, [single_keyword_phrase]):
                                all_keywords_found = False
                                break # No need to check further if one is missing
                        match = all_keywords_found

                    row_result_search = row.to_dict()
                    row_result_search["KS_Combined_Text"] = combined_text_search
                    row_result_search["KS_Match"] = match

                    search_results_list.append(row_result_search)
                    progress_bar_search.progress((i + 1) / total_rows_search)

                search_results_df = pd.DataFrame(search_results_list)
                st.subheader("Keyword Search Results")
                st.dataframe(search_results_df)

                csv_output_search = search_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Keyword Search Results as CSV",
                    data=csv_output_search,
                    file_name="keyword_search_results.csv",
                    mime="text/csv",
                )

                matched_rows_count = search_results_df["KS_Match"].sum()
                st.info(f"**Summary:** {matched_rows_count} out of {total_rows_search} rows matched your search criteria.")

# --- Tab 7: Semantic Search ---
with tab7:
    st.header("Semantic Search")
    st.markdown("""
    Find text passages that are semantically similar to your query, even if they don't share exact keywords.
    This search understands the meaning and context of your words.
    """)

    # Semantic Search Model Selection
    st.subheader("Select Semantic Search Model")
    selected_semantic_model_name = st.selectbox(
        "Choose a pre-trained Sentence Transformer model:",
        options=list(AVAILABLE_SEMANTIC_MODELS.keys()),
        index=0,
        key="semantic_model_selector",
        help="Models like 'all-MiniLM-L6-v2' are fast and good general-purpose models. 'all-mpnet-base-v2' is larger and more accurate. 'multi-qa-MiniLM-L6-cos-v1' is optimized for question-answering."
    )
    selected_semantic_model_id = AVAILABLE_SEMANTIC_MODELS[selected_semantic_model_name]
    semantic_model = load_semantic_model(selected_semantic_model_id) # Load model here

    if uploaded_file is None:
        st.info("Please upload text data in the sidebar to use Semantic Search.")
    elif not text_columns_available:
        st.warning("No suitable text columns found in your data for Semantic Search. Please check your data or upload a file with text columns.")
    else:
        st.subheader("2. Enter Your Search Query")
        query_text = st.text_area(
            "Enter your search query:",
            placeholder="e.g., finding new business collaborations",
            help="Enter a sentence or phrase. The model will find documents with similar meaning."
        )

        st.subheader("3. Select Text Columns to Search Within")
        default_selected_semantic_cols = [col for col in ['text', 'article', 'description', 'comment', 'body', 'content'] if col in text_columns_available]
        if not default_selected_semantic_cols and text_columns_available:
            default_selected_semantic_cols = [text_columns_available[0]]

        selected_semantic_text_cols = st.multiselect(
            "Select one or more columns to search:",
            text_columns_available,
            default=default_selected_semantic_cols,
            key="semantic_text_cols_selector",
            help="If multiple columns are selected, their text content will be combined for the search."
        )

        st.subheader("4. Set Similarity Threshold")
        similarity_threshold = st.slider(
            "Minimum Similarity Score (Cosine Similarity):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only documents with a semantic similarity score above this threshold will be shown. Higher values mean more relevant but fewer results."
        )

        if st.button("Run Semantic Search"):
            if not query_text:
                st.warning("Please enter a search query.")
            elif not selected_semantic_text_cols:
                st.warning("Please select at least one text column to search within.")
            else:
                st.info("Running Semantic Search... This might take a while depending on the number of rows and model size.")

                # Generate embedding for the query
                query_embedding = semantic_model.encode(query_text, convert_to_tensor=True)

                semantic_results_list = []
                combined_texts_for_semantic = []

                # First, combine texts for document embeddings
                for i, row in data_df.iterrows():
                    combined_text_sem = " ".join([
                        str(row[col]) for col in selected_semantic_text_cols if pd.notna(row[col])
                    ])
                    combined_texts_for_semantic.append(combined_text_sem)

                # Generate embeddings for all documents
                document_embeddings = semantic_model.encode(combined_texts_for_semantic, convert_to_tensor=True)

                progress_bar_semantic = st.progress(0)
                total_rows_semantic = len(data_df)

                # Calculate cosine similarity
                cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0] # Get scores for the single query

                for i, row in data_df.iterrows():
                    score = cosine_scores[i].item() # Convert tensor to Python float

                    if score >= similarity_threshold:
                        row_result_semantic = row.to_dict()
                        row_result_semantic["SS_Combined_Text"] = combined_texts_for_semantic[i]
                        row_result_semantic["SS_Similarity_Score"] = f"{score:.4f}"
                        semantic_results_list.append(row_result_semantic)

                    progress_bar_semantic.progress((i + 1) / total_rows_semantic)

                semantic_results_df = pd.DataFrame(semantic_results_list)

                if not semantic_results_df.empty:
                    # Sort by similarity score
                    semantic_results_df["SS_Similarity_Score"] = pd.to_numeric(semantic_results_df["SS_Similarity_Score"])
                    semantic_results_df = semantic_results_df.sort_values(by="SS_Similarity_Score", ascending=False).reset_index(drop=True)

                    st.subheader("Semantic Search Results (Sorted by Similarity)")
                    st.dataframe(semantic_results_df)

                    csv_output_semantic = semantic_results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Semantic Search Results as CSV",
                        data=csv_output_semantic,
                        file_name="semantic_search_results.csv",
                        mime="text/csv",
                    )
                    st.info(f"**Summary:** {len(semantic_results_df)} rows matched with a similarity score of {similarity_threshold} or higher.")
                else:
                    st.info(f"No documents found matching your query with a similarity score of {similarity_threshold} or higher.")


st.markdown("---")
st.markdown("""
**Note on Model Usage:**
The AI/ML models (Zero-Shot Classification, Named Entity Recognition, Sentiment Analysis, Summarization, Keyword Extraction, and Semantic Search) used in this application are pre-trained. They leverage vast knowledge gained during their training on large text datasets to understand context and perform their tasks without requiring you to provide labeled examples for your specific categories or entities.

If you need to adapt these models to very specific nuances of your data, you would typically explore techniques such as:
* **Refining Labels/Descriptions (for ZSC):** Experiment with different phrasings for your category labels and descriptions to improve classification accuracy.
* **Prompt Engineering (for ZSC):** Craft more elaborate "prompts" or "hypotheses" that the ZSC model evaluates.
* **Fine-tuning (Advanced):** For highly specialized tasks or domains, you might consider fine-tuning a pre-trained model on a small, custom-labeled dataset. This is a more advanced process and typically involves separate training steps outside this application.
""")
