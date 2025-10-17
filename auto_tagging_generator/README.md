# 🧠 AI Auto Tagging Generator

An **AI-powered tagging tool** that automatically generates relevant tags for article or text content using **Google’s Gemini GenAI API**.  
Ideal for **content management**, **SEO**, and **automated classification** workflows.

---

## 🚀 Overview

This project reads a text or article and produces a list of relevant tags using the `gemini-2.5-flash` model.  
It includes:
- A Python client for interacting with the Gemini API  
- A lightweight UI for testing tag generation  
- A ready-to-run Colab setup  

---

## ⚙️ How It Works

1. Initializes a Gemini client with your API key.  
2. Sends the article text to the model with a tagging prompt.  
3. Receives structured JSON containing tag suggestions.  
4. Cleans and deduplicates the tags before returning them.

---

## 📂 Project Files

| File | Description |
|------|--------------|
| **Code.ipynb** | Jupyter/Colab notebook for quick testing and demos. |
| **genai_client.py** | Core script for initializing the Gemini client, validating keys, and generating tags. |
| **requirements.txt** | List of dependencies required for the app. |
| **styles.py** | Contains color, layout, and font settings for the UI. |
| **ui.py** | Simple UI layer that runs the tag generator interactively. |

---

## 🧠 Example Usage (Python)

```python
from genai_client import init_client, generate_tags

client = init_client("YOUR_GEMINI_API_KEY")
tags = generate_tags(client, "The European Union introduced new AI regulations.")
print(tags)
