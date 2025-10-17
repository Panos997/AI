README.txt
==========
Mini Semantic Search Graph (OpenAI + Python)

DESCRIPTION
-----------
This project is a small experimental app that builds a semantic search
engine using OpenAI embeddings and a lightweight FAISS HNSW index.

It lets you:
- enter a phrase, a full article, or a .txt file,
- optionally summarize it with GPT-4o or GPT-4o-mini before searching,
- and retrieve the most relevant keywords from your dataset.

CONTENTS
--------
- app.py ............ main interactive app
- requirements.txt .. dependencies
- experiments/ ...... folder with my Colab tests and sample experiments
                      (no company data included)

SETUP
-----
1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   python app.py

3. When prompted:
   - Enter your OPENAI_API_KEY and GEMINI_API_KEY
   - Provide the path to your CSV file with keywords
     (example: data/keywords.csv)

CSV FORMAT
----------
keyword
Πανελλήνιες
Προετοιμασία Πανελληνίων
Εκπαιδευτικά Νέα

NOTES
-----
- No real company data is included in this repository.
- The experiments folder contains only example runs and testing code.
- Summarization before search is optional but improves results on long texts.

