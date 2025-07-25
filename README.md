# 📊 Text Analyzer App

A **Streamlit** app for analyzing textual data using modern AI & NLP techniques including:
- Semantic analysis
- Named Entity Recognition
- Zero-shot classification

## ✨ Features

- 📁 Upload your own Excel dataset
- 📌 Select specific columns to analyze
- 💬 Helps you identify clusters of meaning

## 🚀 Getting Started

### Option 1: Run with Conda (recommended)

1. Clone this repo:
   ```bash
   git clone https://github.com/teonghan/text-data-analyzer.git
   cd text-data-analyzer
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

- For **Windows**: double-click `installer-windows.bat`
- For **macOS**: run `installer-macos-universal.sh` in Terminal

A desktop shortcut will be created for you (with a custom icon if available).

## 🧾 Usage Guide

1. Launch the app
2. Upload an Excel file (`.xlsx`) containing text data
3. Choose the function to analyze

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
