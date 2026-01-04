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
from config import Config


from models.models import clip_model, clip_processor

def get_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(Config.DEVICE)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        embedding = features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype('float32')

def get_text_embedding(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(Config.DEVICE)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        embedding = features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype('float32')

