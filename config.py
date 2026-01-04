import os, json, warnings
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Dict, Union
import operator
warnings.filterwarnings('ignore')

from langgraph.graph import StateGraph, END
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel
import torch
import faiss

# Gemini imports
from google import genai
from google.genai import types


# config/Config.py
import os
import torch

class Config:
    DATA_PATH = "/home/enooo/Documents/Dresiffy/assets"

    STYLES_CSV = os.path.join(DATA_PATH, "styles.csv")
    IMAGES_CSV = os.path.join(DATA_PATH, "images.csv")

    # Image cache (downloaded images)
    IMAGES_PATH = os.path.join(DATA_PATH, "image_cache")
    os.makedirs(IMAGES_PATH, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dynamic vocab containers
    DYNAMIC_FASHION_ITEMS = set()
    DYNAMIC_COLORS = set()
    DYNAMIC_BRANDS = set()
    DYNAMIC_GENDERS = set()
    
    # Models
    CLASSIFIER_MODEL = "google/flan-t5-large"
    GENERATOR_MODEL = "google/flan-t5-base"
    CLIP_MODEL = "patrickjohncyh/fashion-clip"
    
    
    # LLM Settings
    CLASSIFIER_TEMP = 0.1
    GENERATOR_TEMP = 0.7
    MAX_TOKENS_CLASSIFIER = 20
    MAX_TOKENS_GENERATOR = 150
    
    # Image Validation Thresholds
    FASHION_SCORE_THRESHOLD = 0.20
    NON_FASHION_THRESHOLD = 0.30
    HIGH_CONFIDENCE_THRESHOLD = 0.70
    SIMILARITY_WARNING_THRESHOLD = 0.25
    
    # Text Intent Classification
    TEXT_FASHION_THRESHOLD = 0.3
    TEXT_SEARCH_THRESHOLD = 0.5
    
    # Gemini API
    _GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEYS = [k.strip() for k in _GEMINI_API_KEYS_STR.split(",") if k.strip()]
    GEMINI_MODEL = "gemini-2.0-flash-lite"
    
    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Get a random Gemini API key from the pool."""
        if not cls.GEMINI_API_KEYS:
            return ""
        import random
        return random.choice(cls.GEMINI_API_KEYS)

    # FIXED: Hybrid Search Weights
    TEXT_WEIGHT = 0.60  # 60% for text when both present
    IMAGE_WEIGHT = 0.40  # 40% for image when both present
    IMAGE_ONLY_WEIGHT = 1.0  # 100% for image-only search
    TEXT_ONLY_WEIGHT = 1.0  # 100% for text-only search
    
    # Search Settings
    FAISS_MAX_ITEMS = 44419
    SEARCH_DEFAULT_K = 5
    INDEX_BATCH_SIZE = 1000
    
    # Dynamic Vocabulary (populated at runtime)
    DYNAMIC_FASHION_ITEMS = set()
    DYNAMIC_COLORS = set()
    DYNAMIC_BRANDS = set()
    DYNAMIC_GENDERS = set()

config = Config()
