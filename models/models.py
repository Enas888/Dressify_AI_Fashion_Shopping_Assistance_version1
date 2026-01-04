import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import Config
from pathlib import Path
from typing import Optional
from PIL import Image

import pandas as pd
import numpy as np
import torch
import faiss

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

# Gemini imports
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, END
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel
from langchain_huggingface import HuggingFacePipeline
from config import Config

print("\n🎨 Loading Fashion-CLIP...")
clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL).to(Config.DEVICE)
clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)
print("✅ Fashion-CLIP loaded")

# CELL 4: Initialize LLMs
# ================================================================================
def safe_invoke(llm, prompt: str) -> str:
    try:
        return str(llm.invoke(prompt)).strip().replace("</s>", "")
    except:
        return "ERROR"

def initialize_llms():
    print("\n🔧 Initializing LLMs...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.CLASSIFIER_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(Config.CLASSIFIER_MODEL).to(Config.DEVICE)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, 
                       device=0 if Config.DEVICE=="cuda" else -1, 
                       max_new_tokens=Config.MAX_TOKENS_CLASSIFIER, temperature=Config.CLASSIFIER_TEMP)
        classifier_llm = HuggingFacePipeline(pipeline=pipe)
        print(f"✅ Classifier: {Config.CLASSIFIER_MODEL}")
    except Exception as e:
        print(f"⚠️ Classifier failed: {e}")
        classifier_llm = None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.GENERATOR_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(Config.GENERATOR_MODEL).to(Config.DEVICE)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer,
                       device=0 if Config.DEVICE=="cuda" else -1,
                       max_new_tokens=Config.MAX_TOKENS_GENERATOR, temperature=Config.GENERATOR_TEMP)
        generator_llm = HuggingFacePipeline(pipeline=pipe)
        print(f"✅ Generator: {Config.GENERATOR_MODEL}")
    except Exception as e:
        print(f"⚠️ Generator failed: {e}")
        generator_llm = None
    
    return classifier_llm, generator_llm

classifier_llm, generator_llm = initialize_llms()

