# Dressify AI - Fashion Shopping Assistance

Dressify AI is an advanced fashion search and recommendation platform powered by AI. It combines visual similarity, natural language processing, and style reasoning to provide a premium shopping experience.

# Interface Example:

## image 1:
<img width="1558" height="1177" alt="image" src="https://github.com/user-attachments/assets/90d01884-be34-4566-a8d6-4bdfd8053bb0" />

## image 2:
<img width="1534" height="1174" alt="image" src="https://github.com/user-attachments/assets/5159bc45-58d3-4f6a-9ccb-a8370f1842dc" />


## ✨ Features

- 📸 **Visual Search:** Upload an image of any fashion item to find visually similar products in the catalog using Fashion-CLIP.
- 🔍 **Hybrid Search:** Combine text and images to refine your search (e.g., upload a shirt and type "blue" to find blue variations).
- 🎨 **Style Matching:** Ask the AI to find items that "match" or "fit" your uploaded image. The system uses color theory and style reasoning to suggest complementary items.
- 👔 **Gender-Aware Search:** Automatic detection of target gender from queries, with support for men, women, and unisex fashion.
- ⚖️ **Outfit Recommendations:** Get complete look suggestions (tops, bottoms, shoes, accessories, and watches) for specific occasions like "weddings" or "casual summer".
- 🤖 **Multi-Agent Architecture:** Powered by a sophisticated LangGraph workflow with specialized agents for validation, classification, and execution.

## 🚀 Technology Stack

- **ML Models:** Fashion-CLIP (Visual/Text Embeddings), Gemini Pro (Reasoning & Query Generation), Flan-T5 (Classification).
- **Backend:** FastAPI, Python.
- **Workflow:** LangGraph.
- **Vector DB:** FAISS (Fast Approximate Nearest Neighbor Search).
- **Frontend:** Responsive HTML5/Vanilla CSS UI with Glassmorphism aesthetics.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Enas888/Dressify_AI_Fashion_Shopping_Assistance_version1.git
cd Dressify_AI_Fashion_Shopping_Assistance_version1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_actual_api_key_here"
```

### 4. Running the Application
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
Visit `http://127.0.0.1:8000` in your browser.

## 📁 Project Structure

- `main.py`: Entry point and FastAPI routes.
- `agents/`: Core logic for AI agents and LangGraph workflow.
- `models/`: Model initialization and the Jupyter notebook research.
- `helpers/`: Utility functions for embeddings, data loading, and FAISS.
- `assets/`: Dataset (CSV) and pre-built FAISS index.

## 📜 License
This project is for educational and portfolio purposes. Data used is from the Fashion Product Images dataset.
