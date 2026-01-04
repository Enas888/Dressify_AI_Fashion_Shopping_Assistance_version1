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
from state import AgentState

from agents.agents import (
    image_fashion_validator_agent,
    image_to_description_agent,
    non_relevant_image_agent,
    intent_classifier_agent,
    welcome_agent,
    non_relevant_agent,
    smart_query_understanding_agent,
    search_executor_agent,
    
)
from models.models import initialize_llms
classifier_llm, generator_llm = initialize_llms()

def route_agent(state: AgentState) -> str:
    return state.get('next_agent', 'end')

workflow = StateGraph(AgentState)

workflow.add_node("image_fashion_validator", image_fashion_validator_agent)
workflow.add_node("image_to_description", image_to_description_agent)
workflow.add_node("non_relevant_image_agent", non_relevant_image_agent)
workflow.add_node("intent_classifier", intent_classifier_agent)
workflow.add_node("welcome_agent", welcome_agent)
workflow.add_node("non_relevant_agent", non_relevant_agent)
workflow.add_node("smart_query_understanding", smart_query_understanding_agent)
workflow.add_node("search_executor", search_executor_agent)

workflow.set_entry_point("image_fashion_validator")

workflow.add_conditional_edges(
    "image_fashion_validator", 
    route_agent, 
    {
        "image_to_description": "image_to_description",
        "non_relevant_image_agent": "non_relevant_image_agent", 
        "intent_classifier": "intent_classifier",
        "end": END
    }
)

workflow.add_conditional_edges("non_relevant_image_agent", route_agent, {"end": END})
workflow.add_conditional_edges("image_to_description", route_agent, {"intent_classifier": "intent_classifier", "end": END})
workflow.add_conditional_edges(
    "intent_classifier",
    route_agent, 
    {
        "welcome_agent": "welcome_agent",
        "non_relevant_agent": "non_relevant_agent", 
        "fashion_classifier": "smart_query_understanding",
        "end": END
    }
)
workflow.add_conditional_edges("welcome_agent", route_agent, {"end": END})
workflow.add_conditional_edges("non_relevant_agent", route_agent, {"end": END})
workflow.add_edge("smart_query_understanding", "search_executor")
workflow.add_edge("search_executor", END)

workflow_app = workflow.compile()
print("✅ Workflow ready with FIXED image search")
