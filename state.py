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

class AgentState(TypedDict):
    user_input: str
    image_input: Optional[str]
    image_embedding: Optional[np.ndarray]  # NEW: Store image embedding
    is_fashion_image: Optional[bool]
    image_validation_reason: Optional[str]
    image_description: Optional[str]
    text_query: Optional[str]
    intent: Optional[str]
    intent_class: Optional[str]
    messages: Annotated[List[str], operator.add]
    final_response: Optional[str]
    next_agent: Optional[str]
    debug_info: Optional[dict]
    search_queries: Optional[List[str]]
    search_results_data: Optional[List[dict]]
    query_categories: Optional[List[str]]
    intent_type: Optional[str]
    search_mode: Optional[str]  # NEW: 'image_only', 'text_only', 'hybrid'
    detected_gender: Optional[str]  # NEW: Track detected gender
    gender_source: Optional[str]  # NEW: Track how gender was detected (LLM/rule/both/none)