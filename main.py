# main.py - FINAL WORKING VERSION
import os
import sys
import warnings
from typing import Optional
from PIL import Image
import io
import numpy as np

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------
# Config
# -----------------------
from config import Config

# -----------------------
# Gemini API
# -----------------------
from google import genai
from google.genai import types

api_key = Config.get_gemini_api_key()
client = genai.Client(api_key=api_key)
MODEL = Config.GEMINI_MODEL
print(f"✅ Gemini API configured: {MODEL}")

# -----------------------
# Load FAISS & Models
# -----------------------
from helpers.faiss_loader import load_saved_index
from helpers.image_loader import get_or_download_image

# This loads FAISS and initial metadata
faiss_index, metadata_df = load_saved_index()

# Load models
from models.models import clip_model, clip_processor, classifier_llm, generator_llm

# Load workflow
from agents.workflow import workflow_app

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()
app.mount("/images", StaticFiles(directory=Config.IMAGES_PATH), name="images")

# -----------------------
# Load and fix dataset
# -----------------------
print("\n📊 Loading dataset and resolving local image paths...")

# Load styles.csv ONLY
df = pd.read_csv(Config.STYLES_CSV, on_bad_lines="skip")
df.columns = df.columns.str.strip()

# Ensure ID is string
df["image_id"] = df["id"].astype(str)

# Build LOCAL image paths (NO DOWNLOADS)
def resolve_local_image(image_id: str):
    path = os.path.join(Config.IMAGES_PATH, f"{image_id}.jpg")
    return path if os.path.exists(path) else None

df["image_path"] = df["image_id"].apply(resolve_local_image)

# Keep only rows with existing images
df_with_images = df[df["image_path"].notna()].copy()

print(f"✅ Total rows: {len(df)}")
print(f"✅ Rows with local images: {len(df_with_images)}")
print(f"📁 Image cache path: {Config.IMAGES_PATH}")


def build_dynamic_vocabulary(dataframe):
    print("\n🔧 Building dynamic vocabulary from local dataset...")

    # -------------------------
    # Article Types (shirts → shirt)
    # -------------------------
    if "articleType" in dataframe.columns:
        items = (
            dataframe["articleType"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.replace("-", " ", regex=False)
            .str.replace("_", " ", regex=False)
            .str.strip()
            .unique()
        )

        # Blocklist for noisy terms
        blocklist = {"football", "skin", "water", "bottle", "perfume", "mascara", "lipstick", "nail", "polish", "toy", "game"}
        
        for item in items:
            # Skip blocklisted items or items containing them if they are single words
            if item in blocklist:
                continue
                
            Config.DYNAMIC_FASHION_ITEMS.add(item)

            # plural → singular
            if item.endswith("s") and len(item) > 3:
                Config.DYNAMIC_FASHION_ITEMS.add(item[:-1])

        print(f"   ✓ {len(items)} article types detected")

    # -------------------------
    # Colors (navy blues → navy blue)
    # -------------------------
    if "baseColour" in dataframe.columns:
        colors = (
            dataframe["baseColour"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
        )

        for color in colors:
            Config.DYNAMIC_COLORS.add(color)

            if color.endswith("s") and len(color) > 3:
                Config.DYNAMIC_COLORS.add(color[:-1])

        print(f"   ✓ {len(colors)} colors detected")

    # -------------------------
    # Genders
    # -------------------------
    if "gender" in dataframe.columns:
        genders = (
            dataframe["gender"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
        )
        Config.DYNAMIC_GENDERS.update(genders)
        print(f"   ✓ {len(genders)} genders detected")

    # -------------------------
    # Brands (ONLY if column exists)
    # -------------------------
    if "brandName" in dataframe.columns:
        brands = (
            dataframe["brandName"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
        )
        Config.DYNAMIC_BRANDS.update(brands)
        print(f"   ✓ {len(brands)} brands detected")

    print("\n📌 Vocabulary Summary:")
    print(f"   Items   : {len(Config.DYNAMIC_FASHION_ITEMS)}")
    print(f"   Colors  : {len(Config.DYNAMIC_COLORS)}")
    print(f"   Genders : {len(Config.DYNAMIC_GENDERS)}")
    print(f"   Brands  : {len(Config.DYNAMIC_BRANDS)}")


build_dynamic_vocabulary(df_with_images)
print(df_with_images[["id", "image_path"]].head())


# -----------------------
# Align Global Reference
# -----------------------
from helpers import faiss_loader
# The metadata returned by load_saved_index() is ALREADY synchronized with faiss_index.
# We should ONLY update it if we are sure it doesn't break indexing.
# Instead of replacing it entirely, let's just ensure it's available.
faiss_loader.metadata_df = metadata_df
# ===============================
# FIX PRICE COLUMN (CRITICAL)
# ===============================
metadata_df["price"] = (
    metadata_df["price"]
    .replace("N/A", 0)
    .replace("", 0)
    .fillna(0)
)

metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce").fillna(0)

valid_images = (metadata_df['source_path'] != '').sum() if 'source_path' in metadata_df.columns else 0
print(f"✅ Metadata fixed: {len(metadata_df)} items, {valid_images} with images")

if 'gender' in metadata_df.columns:
    print(f"   Gender distribution: {metadata_df['gender'].value_counts().to_dict()}")

print("\n✅ All systems ready!")
print(f"   FAISS vectors: {faiss_index.ntotal}")
print(f"   Metadata items: {len(metadata_df)}")
print(f"   Valid images: {valid_images}")
print(metadata_df.columns.tolist())

# -----------------------
# Process query
# -----------------------
def process_query(user_text: Optional[str] = None, user_image: Optional[UploadFile] = None):
    try:
        img_path = None

        if user_image:
            contents = user_image.file.read()
            img_path = "temp_upload.jpg"
            Image.open(io.BytesIO(contents)).convert("RGB").save(img_path)
            user_image.file.seek(0)

        state = {
            "user_input": user_text or "",
            "image_input": img_path,
            "image_embedding": None,
            "is_fashion_image": None,
            "image_validation_reason": None,
            "image_description": None,
            "text_query": None,
            "intent": None,
            "intent_class": None,
            "messages": [],
            "final_response": None,
            "next_agent": None,
            "debug_info": {},
            "search_queries": [],
            "search_results_data": [],
            "query_categories": [],
            "intent_type": None,
            "search_mode": None,
            "detected_gender": None,
            "gender_source": None,
        }

        result = workflow_app.invoke(state)
        
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except:
                pass
        
        # Sanitize result for JSON
        if isinstance(result, dict):
            # Remove non-serializable or large internal fields
            result.pop('image_embedding', None)
            
            # Helper to convert numpy to list
            def clean_obj(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: clean_obj(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean_obj(i) for i in obj]
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
                
            result = clean_obj(result)

        return result

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error:\n{error_trace}")
        return {
            "error": str(e),
            "error_trace": error_trace,
            "final_response": f"❌ Error: {str(e)}"
        }

# -----------------------
# Routes
# -----------------------
@app.post("/query/text")
def query_text(user_text: str = Form(...)):
    try:
        result = process_query(user_text, None)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500
        )

@app.post("/query/image")
def query_image(
    user_text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        if image:
            print(f"   Image filename: {image.filename}, Content-Type: {image.content_type}")
        result = process_query(user_text, image)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"❌ CRITICAL ERROR in /query/image:\n{trace}")
        return JSONResponse(
            content={"error": str(e), "trace": trace},
            status_code=500
        )

@app.get("/debug/metadata")
def debug_metadata():
    """Check metadata structure"""
    sample = metadata_df.head(5).to_dict('records')
    return JSONResponse({
        "total": len(metadata_df),
        "columns": metadata_df.columns.tolist(),
        "sample": sample,
        "faiss_size": faiss_index.ntotal,
        "valid_images": (metadata_df['source_path'] != '').sum() if 'source_path' in metadata_df.columns else 0
    })

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dressify AI - Fashion Search</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                margin-bottom: 30px;
                text-align: center;
                font-size: 2.5em;
            }
            .search-box {
                background: #f8f9fa;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
            }
            input[type="text"] {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 16px;
                margin-bottom: 15px;
                transition: border 0.3s;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
            }
            input[type="file"] {
                margin-bottom: 20px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 10px;
                font-size: 18px;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .loading {
                text-align: center;
                padding: 40px;
                font-size: 24px;
                color: #667eea;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                margin-top: 30px;
            }
            .result-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .category-section {
                margin-bottom: 40px;
            }
            .category-title {
                font-size: 24px;
                color: #667eea;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
            }
            .results-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .result-card {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
                transition: transform 0.3s, box-shadow 0.3s;
                cursor: pointer;
            }
            .result-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                border-color: #667eea;
            }
            .result-image {
                width: 100%;
                height: 300px;
                object-fit: cover;
                background: #f5f5f5;
            }
            .result-info {
                padding: 15px;
            }
            .result-title {
                font-weight: bold;
                color: #333;
                margin-bottom: 8px;
                font-size: 16px;
                line-height: 1.4;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            .result-detail {
                color: #666;
                font-size: 14px;
                margin: 5px 0;
            }
            .result-score {
                background: #667eea;
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 12px;
                display: inline-block;
                margin-top: 8px;
            }
            .back-button {
                margin-top: 20px;
                background: #6c757d;
            }
            .error {
                background: #fff3cd;
                border: 2px solid #ffc107;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .debug-info {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                font-size: 12px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👗 Dressify AI</h1>
            
            <div class="search-box">
                <form id="searchForm">
                    <input 
                        type="text" 
                        name="user_text" 
                        id="user_text" 
                        placeholder="Search for fashion items (e.g., 'blue shirt for men', 'wedding outfit')"
                    >
                    
                    <input 
                        type="file" 
                        name="image" 
                        id="image" 
                        accept="image/*"
                    >
                    
                    <button type="submit" id="submitBtn">🔍 Search</button>
                </form>
            </div>
            
            <div id="resultsContainer"></div>
        </div>

        <script>
            document.getElementById('searchForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const hasImage = document.getElementById('image').files.length > 0;
                const hasText = document.getElementById('user_text').value.trim() !== '';
                
                if (!hasImage && !hasText) {
                    alert('Please enter text or upload an image');
                    return;
                }
                
                const resultsContainer = document.getElementById('resultsContainer');
                const submitBtn = document.getElementById('submitBtn');
                
                submitBtn.disabled = true;
                resultsContainer.innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>🔍 Searching fashion items...</p>
                    </div>
                `;
                
                const endpoint = hasImage ? '/query/image' : '/query/text';
                
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error('Server error');
                    }
                    
                    const data = await response.json();
                    displayResults(data);
                    
                } catch (error) {
                    resultsContainer.innerHTML = `
                        <div class="error">
                            <h3>❌ Error</h3>
                            <p>${error.message}</p>
                        </div>
                        <button class="back-button" onclick="location.reload()">← Try Again</button>
                    `;
                } finally {
                    submitBtn.disabled = false;
                }
            };
            
            function displayResults(data) {
                const container = document.getElementById('resultsContainer');
                
                if (data.error) {
                    container.innerHTML = `
                        <div class="error">
                            <h3>❌ Error</h3>
                            <p>${data.error}</p>
                            <pre style="margin-top: 10px; font-size: 12px; overflow: auto;">${data.error_trace || ''}</pre>
                        </div>
                        <button class="back-button" onclick="location.reload()">← Try Again</button>
                    `;
                    return;
                }
                
                let html = '<div class="results">';
                
                // Header with response
                html += `<div class="result-header">`;
                html += `<h2>${data.final_response || 'Results'}</h2>`;
                if (data.messages && data.messages.length > 0) {
                    html += `<p style="margin-top: 10px; opacity: 0.9;">`;
                    html += data.messages.slice(-3).join(' • ');
                    html += `</p>`;
                }
                html += `</div>`;
                
                // Display grouped results
                if (data.search_results_data && data.search_results_data.length > 0) {
                    // Group by category
                    const categoryGroups = {};
                    data.search_results_data.forEach(group => {
                        const cat = group.category || 'general';
                        if (!categoryGroups[cat]) {
                            categoryGroups[cat] = [];
                        }
                        categoryGroups[cat].push(group);
                    });
                    
                    // Category emojis
                    const categoryEmojis = {
                        'top': '👕',
                        'bottom': '👖',
                        'footwear': '👟',
                        'accessories': '👜',
                        'watches': '⌚',
                        'general': '🔍',
                        'similar': '🎨'
                    };
                    
                    // Display each category
                    const categoryOrder = ['top', 'bottom', 'footwear', 'accessories', 'watches', 'similar', 'general'];
                    
                    categoryOrder.forEach(catName => {
                        if (categoryGroups[catName]) {
                            const emoji = categoryEmojis[catName] || '📦';
                            html += `<div class="category-section">`;
                            html += `<h3 class="category-title">${emoji} ${catName.toUpperCase()}</h3>`;
                            
                            categoryGroups[catName].forEach(group => {
                                if (group.items && group.items.length > 0) {
                                    if (group.query_text) {
                                        html += `<p style="margin-bottom: 15px; color: #666;"><strong>Query:</strong> ${group.query_text}</p>`;
                                    }
                                    
                                    html += `<div class="results-grid">`;
                                    
                                    group.items.forEach(item => {
                                        // Robust image path construction
                                        let imageUrl = '';
                                        if (item.id) {
                                           imageUrl = `/images/${item.id}.jpg`;
                                        } else if (item.thumbnail_url) {
                                           imageUrl = item.thumbnail_url;
                                        }
                                        
                                        // Debug log for first item
                                        if (imageUrl && !window.debuggedFirstImage) {
                                            console.log('Debug Image URL:', imageUrl, 'Item:', item);
                                            window.debuggedFirstImage = true;
                                        }
                                        
                                        const score = item.score || item.similarity || 0;
                                        
                                        html += `
                                            <div class="result-card">
                                                ${imageUrl ? `<img src="${imageUrl}" class="result-image" alt="${item.title || 'Product'}" loading="lazy" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'200\\' height=\\'200\\'%3E%3Crect fill=\\'%23ddd\\' width=\\'200\\' height=\\'200\\'/%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' fill=\\'%23999\\' dy=\\'.3em\\'%3ENo Image%3C/text%3E%3C/svg%3E'; console.error('Failed to load image:', '${imageUrl}');">` : ''}
                                                <div class="result-info">
                                                    <div class="result-title">${item.title || item.product_name || 'Product'}</div>
                                                    <div class="result-detail">Type: ${item.article_type || item.articleType || 'N/A'}</div>
                                                    <div class="result-detail">Color: ${item.color || item.base_color || item.baseColour || 'N/A'}</div>
                                                    <div class="result-detail">Gender: ${item.gender || 'N/A'}</div>
                                                    ${item.brand && item.brand !== 'Unknown' ? `<div class="result-detail">Brand: ${item.brand}</div>` : ''}
                                                    ${item.price && item.price > 0 ? `<div class="result-detail">Price: ₹${item.price}</div>` : ''}
                                                    <span class="result-score">Match: ${(score * 100).toFixed(1)}%</span>
                                                </div>
                                            </div>
                                        `;
                                    });
                                    
                                    html += `</div>`;
                                }
                            });
                            
                            html += `</div>`;
                        }
                    });
                } else {
                    html += '<p style="color: #666; margin-top: 20px; text-align: center; font-size: 18px;">No results found. Try a different search!</p>';
                }
                
                html += '</div>';
                html += '<button class="back-button" onclick="location.reload()">← New Search</button>';
                
                // Debug info (optional)
                if (data.debug_info && Object.keys(data.debug_info).length > 0) {
                    html += `<div class="debug-info">`;
                    html += `<strong>Debug Info:</strong><br>`;
                    html += `<pre>${JSON.stringify(data.debug_info, null, 2)}</pre>`;
                    html += `</div>`;
                }
                
                container.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


