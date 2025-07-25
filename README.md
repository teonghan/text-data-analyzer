# 📊 Text Analyzer App

A **Streamlit** app for analyzing open-ended textual survey responses using modern NLP techniques including:
- Sentence embeddings
- TF-IDF vectorization
- Cosine similarity
- Grouping by themes or categories

## ✨ Features

- 🔍 Analyze similarities between survey responses
- 📁 Upload your own Excel dataset
- 📌 Select specific columns to compare
- 🧠 Automatically computes sentence embeddings
- 📈 Displays most and least similar response pairs
- 💬 Helps you identify clusters of meaning

## 🚀 Getting Started

### Option 1: Run with Conda (recommended)

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/textanalyzer.git
   cd textanalyzer
   ```

2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate textanalyzer
   ```

3. Launch the app:
   ```bash
   streamlit run app.py
   ```

### Option 2: Use the Installer

- For **Windows**: double-click `installer_windows.bat`
- For **macOS**: run `installer.sh` in Terminal

A desktop shortcut will be created for you (with a custom icon if available).

## 🧾 Usage Guide

1. Launch the app
2. Upload an Excel file (`.xlsx`) containing open-text responses
3. Select the columns to analyze (text column and optional category column)
4. Choose parameters like:
   - Similarity method: Sentence Embedding or TF-IDF
   - Number of top similar/dissimilar pairs
5. Click **Analyze**
6. Review similarity tables and explore highlighted pairs

## 🧠 How It Works

- Uses **`sentence-transformers`** to generate high-dimensional embeddings
- Optionally uses **TF-IDF** vectors for faster, simpler analysis
- Computes **cosine similarity** between every pair of sentences
- Optionally segments results by group/category column

## 📂 Folder Structure

```
textanalyzer/
├── app.py
├── environment.yml
├── installer.sh                  # macOS setup script
├── installer_windows.bat         # Windows setup script
├── run_app.sh / run_app.bat      # Launchers
├── icon.ico / icon.icns          # Optional shortcut icon
└── README.md
```

## 📦 Dependencies

Installed via `environment.yml`:
- streamlit
- pandas
- transformers
- scikit-learn
- nltk
- sentence-transformers
- openpyxl

## 📃 License

MIT License — feel free to modify and use for personal or institutional surveys.
