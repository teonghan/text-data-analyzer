# 🧠 Text Analyzer App

A modern **Streamlit** application to analyze your unstructured text data with powerful NLP and AI models — no coding required.

Try it online 👉 *(Add your deployed link here)*

---

## 🔍 Overview

The **Text Analyzer App** helps you quickly understand and analyze text data using:

- 🧠 **Zero-Shot Classification (ZSC)** — Automatically label text into custom categories
- 🏷️ **Named Entity Recognition (NER)** — Extract people, locations, organizations, and more
- ❤️ **Sentiment Analysis** — Understand emotions in your text
- ✂️ **Summarization** — Generate concise summaries from long text
- 🔑 **Keyword Extraction (TF-IDF)** — Find most important terms
- 🔍 **Keyword Search** — Advanced search using lemmatized phrases
- 🤝 **Semantic Search** — Find similar meanings even if words differ

You can upload `.csv` or `.xlsx` files, choose columns, and analyze in multiple ways.

---

## ✨ Key Features

- 📁 Upload Excel/CSV datasets
- 🔄 Select columns for analysis
- 📋 Supports multiple transformer models from Hugging Face
- 🚀 Fast and intuitive tabbed interface for:
  - Zero-shot classification
  - Entity extraction
  - Sentiment scoring
  - Document summarization
  - Keyword mining (TF-IDF)
  - Smart phrase search (lemmatized)
  - Semantic vector-based search

---

## 🛠 Installation

### Option 1: One-Click macOS Installer

```bash
bash installer-macos-universal.sh
```

This will:
- Detect your Mac chip (Intel/Apple Silicon)
- Install Miniforge & dependencies
- Create a desktop shortcut to launch the app

---

### Option 2: One-Click Windows Installer

```powershell
Right-click → Run with PowerShell → installer-windows.ps1
```

This will:
- Detect your Anaconda or Miniconda installation
- Create (or update) the `textanalyzer` Conda environment from `__environment__.yml`
- Generate a launcher (`start-streamlit-app.ps1`)
- Create a desktop shortcut (`Start Text Analyzer App`) to launch the app
- Generate a clean uninstaller (`uninstall-streamlit-app.ps1`)

> ⚠️ Make sure Anaconda or Miniconda is already installed.

---

### Option 3: Manual Setup (Cross-Platform)

1. Clone the repository:

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

---

## 🧾 Usage Guide

1. Start the app via the shortcut or terminal
2. Upload your dataset (.xlsx or .csv)
3. Navigate tabs to analyze using your selected columns
4. Download results as `.csv`

---

## 📦 Dependencies

Installed via `environment.yml`:

- `streamlit`
- `pandas`
- `transformers`
- `scikit-learn`
- `nltk`
- `sentence-transformers`
- `openpyxl`

---

## 📃 License

MIT License — Free to use and adapt for surveys, reviews, research, and more.

---

Let your data speak 🔍 — no coding, just insights.
