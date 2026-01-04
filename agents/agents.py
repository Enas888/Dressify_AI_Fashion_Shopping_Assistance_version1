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
from config import Config
from helpers.embeddings import get_image_embedding, get_text_embedding
from models.models import safe_invoke

"""
Fashion search agents

Each agent receives state, processes it, and returns updated state.
"""
from typing import TypedDict, Annotated, List, Optional, Dict, Union, Any
import numpy as np
import os, json, warnings
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Dict, Union
import operator


# Import shared resources
from models.models import clip_model, clip_processor, classifier_llm, generator_llm, safe_invoke
from helpers.faiss_loader import load_saved_index

# Load FAISS index
faiss_index, metadata_df = load_saved_index()

print("\n🎨 Loading Fashion-CLIP...")
clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL).to(Config.DEVICE)
clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)
print("✅ Fashion-CLIP loaded")

def image_fashion_validator_agent(state: AgentState) -> AgentState:
    """Validates if uploaded image is fashion-related and extracts embedding"""
    if not state.get('image_input'):
        state['is_fashion_image'] = None
        state['image_embedding'] = None
        state['text_query'] = state.get('user_input', '').strip() or "hello"
        state['messages'].append("⭐ No image - text mode")
        state['next_agent'] = 'intent_classifier'
        return state
    
    try:
        image = Image.open(state['image_input']).convert('RGB')
        
        # Pass PIL Image object
        img_embedding = get_image_embedding(image)
        state['image_embedding'] = img_embedding
        
        inputs = clip_processor(images=image, return_tensors="pt").to(Config.DEVICE)
        
        # Dynamic fashion categories from dataset
        fashion_cats = list(Config.DYNAMIC_FASHION_ITEMS)[:30]
        generic_fashion = ["clothing", "fashion", "apparel", "footwear", "accessory", 
                          "garment", "outfit", "attire", "wear"]
        fashion_cats = list(set(fashion_cats + generic_fashion))
        
        # Non-fashion categories
        non_fashion_cats = ["animal", "car", "vehicle", "building", "architecture", 
                           "food", "meal", "landscape", "nature", "plant", "tree",
                           "electronics", "furniture", "tool", "instrument"]
        
        all_cats = fashion_cats + non_fashion_cats
        
        # CLIP Classification
        text_inputs = clip_processor(text=all_cats, return_tensors="pt", padding=True).to(Config.DEVICE)
        
        with torch.no_grad():
            img_feat = clip_model.get_image_features(**inputs)
            txt_feat = clip_model.get_text_features(**text_inputs)
            sim = (img_feat @ txt_feat.T).softmax(dim=-1)
            top_idx = sim[0].topk(10).indices.cpu().numpy()
            top_scores = sim[0].topk(10).values.cpu().numpy()
            top_cats = [all_cats[i] for i in top_idx]
        
        # Calculate fashion vs non-fashion scores
        f_score = sum(float(top_scores[i]) for i, c in enumerate(top_cats) if c in fashion_cats)
        nf_score = sum(float(top_scores[i]) for i, c in enumerate(top_cats) if c in non_fashion_cats)
        
        is_fashion = f_score > Config.FASHION_SCORE_THRESHOLD and f_score > nf_score
        
        top_3_cats = [f"{top_cats[i]} ({top_scores[i]:.2f})" for i in range(min(3, len(top_cats)))]
        reason = f"F={f_score:.2f}, NF={nf_score:.2f} | Top: {', '.join(top_3_cats)}"
        
        state['is_fashion_image'] = is_fashion
        state['image_validation_reason'] = reason
        state['debug_info'] = {
            'fashion_score': float(f_score),
            'non_fashion_score': float(nf_score),
            'top_predictions': top_3_cats,
            'fashion_categories_used': len(fashion_cats),
            'dynamic_categories': fashion_cats[:10],
            'image_embedding_extracted': True
        }
        
        state['messages'].append(
            f"{'✅ Fashion image' if is_fashion else '❌ Non-fashion'}: {reason}"
        )
        state['next_agent'] = 'image_to_description' if is_fashion else 'non_relevant_image_agent'
        
    except Exception as e:
        state['is_fashion_image'] = False
        state['image_embedding'] = None
        state['image_validation_reason'] = f"Error: {str(e)}"
        state['messages'].append(f"⚠️ Image error: {str(e)[:50]}")
        state['next_agent'] = 'non_relevant_image_agent'
    
    return state

def image_to_description_agent(state: AgentState) -> AgentState:
    """Converts fashion image to text description (lightweight for context)"""
    user_text = state.get('user_input', '').strip()
    
    if state.get('image_input') and state.get('is_fashion_image'):
        try:
            image = Image.open(state['image_input']).convert('RGB')
            inputs = clip_processor(images=image, return_tensors="pt").to(Config.DEVICE)
            
            # Get top attributes from dataset vocabulary
            sample_items = list(Config.DYNAMIC_FASHION_ITEMS)[:50]
            sample_colors = list(Config.DYNAMIC_COLORS)[:30]
            
            attrs = sample_items + sample_colors + ["casual", "formal", "summer", "winter"]
            text_inputs = clip_processor(text=attrs, return_tensors="pt", padding=True).to(Config.DEVICE)
            
            with torch.no_grad():
                img_feat = clip_model.get_image_features(**inputs)
                txt_feat = clip_model.get_text_features(**text_inputs)
                sim = (img_feat @ txt_feat.T).softmax(dim=-1)
                top_idx = sim[0].topk(5).indices.cpu().numpy()
                detected = [attrs[i] for i in top_idx]
            
            # Build richer description
            image_desc = " ".join(detected[:5])
            state['image_description'] = image_desc
            
            if user_text:
                state['text_query'] = user_text
                state['messages'].append(f"📝 Text: '{user_text}' + Image: {image_desc}")
            else:
                state['text_query'] = image_desc  # Fallback description
                state['messages'].append(f"📸 Image query: {image_desc}")
                
        except Exception as e:
            state['text_query'] = user_text or "fashion items"
            state['messages'].append(f"⚠️ Image desc error")
    else:
        state['text_query'] = user_text or "hello"
        state['messages'].append(f"💬 Text mode")
    
    state['next_agent'] = 'intent_classifier'
    return state

def non_relevant_image_agent(state: AgentState) -> AgentState:
    """Handles non-fashion images"""
    reason = state.get('image_validation_reason', 'Image is not fashion-related')
    state['final_response'] = f"📸 **Non-Fashion Image Detected**\n\n{reason}\n\nPlease upload fashion items (clothing, shoes, accessories) for search!"
    state['next_agent'] = 'end'
    state['messages'].append("❌ Ended: Non-fashion image")
    return state

# ================================================================================
# CELL 9: INTENT CLASSIFIER
# ================================================================================

def intent_classifier_agent(state: AgentState) -> AgentState:
    """
    ENHANCED HYBRID APPROACH: Fast keyword check + Dynamic LLM verification
    """
    query = state.get('text_query', '').lower().strip()
    
    # Greeting detection
    greeting_patterns = [
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "what can you", "who are", "what do you do", "help me", "greetings"
    ]
    
    if any(query.startswith(pattern) for pattern in greeting_patterns):
        state['intent'] = 'welcome'
        state['next_agent'] = 'welcome_agent'
        state['messages'].append(f"🎯 Welcome (instant detection)")
        return state
    
    # Quick keyword scan
    matched_items = []
    matched_colors = []
    matched_brands = []
    
    for item in Config.DYNAMIC_FASHION_ITEMS:
        if item in query:
            matched_items.append(item)
    
    for color in Config.DYNAMIC_COLORS:
        if color in query:
            matched_colors.append(color)
    
    for brand in Config.DYNAMIC_BRANDS:
        if brand in query:
            matched_brands.append(brand)
    
    fashion_keywords = [
        "wear", "outfit", "style", "look", "trend", "collection", "season",
        "material", "fabric", "pattern", "design", "size", "fit", "casual",
        "formal", "party", "wedding", "sport", "summer", "winter", "vintage"
    ]
    matched_keywords = [kw for kw in fashion_keywords if kw in query]
    
    has_fashion_image = state.get('is_fashion_image') == True
    
    has_fashion_signals = (
        len(matched_items) >= 1 or
        has_fashion_image or
        len(matched_colors) >= 1 or
        len(matched_brands) >= 1 or
        len(matched_keywords) >= 1
    )
    
    if has_fashion_signals:
        state['intent'] = 'relevant_fashion'
        state['next_agent'] = 'fashion_classifier'
        
        signals = []
        if matched_items:
            signals.append(f"items={matched_items[:2]}")
        if matched_colors:
            signals.append(f"colors={matched_colors[:2]}")
        if matched_brands:
            signals.append(f"brands={matched_brands[:1]}")
        if matched_keywords:
            signals.append(f"keywords={matched_keywords[:2]}")
        
        state['messages'].append(f"🎯 Fashion detected ({', '.join(signals)})")
    
    elif classifier_llm:
        # Dynamic LLM verification
        sample_items = list(Config.DYNAMIC_FASHION_ITEMS)[:20]
        sample_colors = list(Config.DYNAMIC_COLORS)[:15]
        sample_brands = list(Config.DYNAMIC_BRANDS)[:10]
        
        categories_text = ""
        
        if sample_items:
            categories_text += f"\n- Fashion Items: {', '.join(sample_items)}"
        
        if sample_colors:
            categories_text += f"\n- Colors: {', '.join(sample_colors)}"
        
        if sample_brands:
            categories_text += f"\n- Brands: {', '.join(sample_brands)}"
        
        categories_text += f"\n- Fashion Keywords: shopping, style, outfit, wear, look, trend, material, fabric, size, fit"
        
        llm_prompt = f"""Is this query related to fashion?

Query: "{query}"

Fashion includes:{categories_text}

Answer ONLY with YES or NO.
YES if the query is about ANY of the above fashion categories.
NO if it's about other topics (food, animals, cars, sports events, general questions).

Answer:"""
        
        try:
            llm_response = safe_invoke(classifier_llm, llm_prompt).strip().upper()
            is_fashion = "YES" in llm_response
            
            if is_fashion:
                state['intent'] = 'relevant_fashion'
                state['next_agent'] = 'fashion_classifier'
                state['messages'].append(f"🎯 Fashion (LLM verified: {llm_response})")
            else:
                state['intent'] = 'non_relevant'
                state['next_agent'] = 'non_relevant_agent'
                state['messages'].append(f"🎯 Non-fashion (LLM: {llm_response})")
        
        except Exception as e:
            state['intent'] = 'non_relevant'
            state['next_agent'] = 'non_relevant_agent'
            state['messages'].append(f"🎯 Non-fashion (LLM error)")
    
    else:
        state['intent'] = 'non_relevant'
        state['next_agent'] = 'non_relevant_agent'
        state['messages'].append(f"🎯 Non-fashion (no signals, no LLM)")
    
    state['debug_info'] = state.get('debug_info', {})
    state['debug_info'].update({
        'matched_items': matched_items,
        'matched_colors': matched_colors,
        'matched_brands': matched_brands,
        'matched_keywords': matched_keywords,
        'has_fashion_image': has_fashion_image,
        'vocabulary_size': len(Config.DYNAMIC_FASHION_ITEMS),
        'total_signals': len(matched_items) + len(matched_colors) + len(matched_brands) + len(matched_keywords)
    })
    
    return state

def welcome_agent(state: AgentState) -> AgentState:
    """Welcomes user and provides examples"""
    state['final_response'] = """👋 **Welcome to Smart Fashion Search!**

**What I can do:**

🔍 **Direct Search:**
- "blue jeans" -> Find blue jeans
- "black dress" -> Find black dresses

🎨 **Matching Items:**
- "black shirt for blue pants" -> Find matching shirts
- Upload pants image + "white shirt" -> Find complementary shirts

🎯 **Complete Outfits:**
- "wedding outfit" -> Get tops, bottoms, shoes, accessories
- "casual summer look" -> Full outfit suggestions

📸 **Image Search:**
- Upload fashion image -> Find similar items (uses visual similarity)

**Try these examples to get started!**"""
    state['next_agent'] = 'end'
    state['messages'].append("✅ Ended: Welcome message")
    return state

def non_relevant_agent(state: AgentState) -> AgentState:
    """Handles non-fashion text queries"""
    state['final_response'] = "😊 **I specialize in fashion search!**\n\n**Try:**\n• 'blue shirt'\n• 'black jeans for white shirt'\n• 'wedding outfit'\n• Upload a fashion image"
    state['next_agent'] = 'end'
    state['messages'].append("❌ Ended: Non-fashion query")
    return state

# ================================================================================
# CELL 10: FIXED SMART QUERY UNDERSTANDING AGENT WITH GENDER DETECTION
# ================================================================================

def smart_query_understanding_agent(state: AgentState) -> AgentState:
    """
    ENHANCED: Gender-aware query understanding
    - Detects gender from text (rule-based + LLM)
    - If no gender specified -> searches both men and women
    - If gender specified -> searches only that gender
    """
    
    query = state.get('text_query', '').strip().lower()
    has_image = state.get('is_fashion_image') == True
    image_desc = state.get('image_description', '')
    debug_info = state.get('debug_info', {})
    
    matched_items = debug_info.get('matched_items', [])
    matched_colors = debug_info.get('matched_colors', [])
    
    # ========================================================================
    # STEP 1: GENDER DETECTION (Rule-based + LLM)
    # ========================================================================
    
    detected_gender_rule = None
    detected_gender_llm = None
    final_gender = None
    gender_source = "none"
    
    # Rule-based gender detection
    male_keywords = ['men', 'man', 'male', 'guy', 'boy', 'gentleman', 'his', 'he', 'him']
    female_keywords = ['women', 'woman', 'female', 'girl', 'lady', 'her', 'she']
    
    query_words = query.split()
    has_male = any(kw in query_words for kw in male_keywords)
    has_female = any(kw in query_words for kw in female_keywords)
    
    if has_male and not has_female:
        detected_gender_rule = "men"
        state['messages'].append(f"🚹 Rule-based: Detected MALE gender")
    elif has_female and not has_male:
        detected_gender_rule = "women"
        state['messages'].append(f"🚺 Rule-based: Detected FEMALE gender")
    elif has_male and has_female:
        detected_gender_rule = "both"
        state['messages'].append(f"⚧ Rule-based: Both genders mentioned")
    else:
        state['messages'].append(f"⚪ Rule-based: No gender detected")
    
    # LLM-based gender detection (only if needed)
    if classifier_llm and detected_gender_rule is None:
        available_genders = list(Config.DYNAMIC_GENDERS)
        
        llm_gender_prompt = f"""Analyze this fashion query and determine the target gender.

Query: "{query}"

Available genders in our catalog: {', '.join(available_genders)}

Instructions:
- If the query mentions men/man/male/boy/his/he/him -> Answer: MEN
- If the query mentions women/woman/female/girl/her/she -> Answer: WOMEN  
- If the query mentions both genders -> Answer: BOTH
- If NO gender is mentioned -> Answer: BOTH (show both genders)

Answer with ONLY ONE WORD: MEN, WOMEN, or BOTH"""

        try:
            llm_response = safe_invoke(classifier_llm, llm_gender_prompt).strip().upper()
            
            if "MEN" in llm_response and "WOMEN" not in llm_response:
                detected_gender_llm = "men"
                state['messages'].append(f"🤖 LLM: Detected MALE gender")
            elif "WOMEN" in llm_response:
                detected_gender_llm = "women"
                state['messages'].append(f"🤖 LLM: Detected FEMALE gender")
            elif "BOTH" in llm_response:
                detected_gender_llm = "both"
                state['messages'].append(f"🤖 LLM: Both genders")
            else:
                state['messages'].append(f"🤖 LLM: Unclear response -> defaulting to BOTH")
                detected_gender_llm = "both"
        
        except Exception as e:
            state['messages'].append(f"⚠️ LLM gender detection failed")
            detected_gender_llm = None
    
    # Combine rule-based + LLM results
    if detected_gender_rule and detected_gender_llm:
        # Both detected - prefer rule-based if they match, otherwise use rule
        if detected_gender_rule == detected_gender_llm:
            final_gender = detected_gender_rule
            gender_source = "both_agree"
            state['messages'].append(f"✅ Gender confirmed: {final_gender.upper()} (rule + LLM agree)")
        else:
            final_gender = detected_gender_rule  # Prefer explicit keywords
            gender_source = "rule_priority"
            state['messages'].append(f"⚖️ Gender conflict: Using rule-based ({final_gender.upper()})")
    
    elif detected_gender_rule:
        final_gender = detected_gender_rule
        gender_source = "rule_only"
        state['messages'].append(f"📏 Gender: {final_gender.upper()} (rule-based only)")
    
    elif detected_gender_llm:
        final_gender = detected_gender_llm
        gender_source = "llm_only"
        state['messages'].append(f"🤖 Gender: {final_gender.upper()} (LLM only)")
    
    else:
        final_gender = "both"
        gender_source = "default_both"
        state['messages'].append(f"🌐 No gender specified -> Showing BOTH genders")
    
    state['detected_gender'] = final_gender
    state['gender_source'] = gender_source
    
    # ========================================================================
    # STEP 2: SCENARIO DETECTION
    # ========================================================================
    
    # Scenario 1: IMAGE ONLY - Prioritize raw user intent
    user_raw = state.get('user_input', '').strip().lower()
    
    if has_image and (not user_raw or user_raw in ['similar', 'like this', 'same', ''] or len(user_raw.split()) <= 2):
        state['search_mode'] = 'image_only'
        state['search_queries'] = ['visual_search']
        state['query_categories'] = ['similar']
        state['intent_type'] = 'image_search'
        
        # KEY FIX: If user didn't specify gender in text, default to BOTH for image search
        # This prevents noisy auto-descriptions (e.g. "women tshirt") from filtering out valid matches
        if not user_raw:
             state['detected_gender'] = "both"
             state['gender_source'] = "default_image_only"
             state['messages'].append(f"🌐 Image Search: Defaulting to BOTH genders")
        
        state['messages'].append(f"🎯 Image-only search mode (100% visual similarity)")
        state['next_agent'] = 'search_executor'
        return state
    
    # Detect "Complementary/Matching" Intent
    matching_keywords = ['match', 'fit', 'with', 'go well', 'complement', 'outfit for', 'jeans for', 'shirt for']
    is_matching_request = any(kw in user_raw for kw in matching_keywords)
    
    # Scenario 4: IMAGE + TEXT (hybrid or matching)
    if has_image and query and len(query.split()) > 2:
        if is_matching_request:
            state['search_mode'] = 'hybrid'
            state['intent_type'] = 'matching'
            state['messages'].append(f"🎯 Matching mode: Finding complementary items (60% text, 40% image)")
        else:
            state['search_mode'] = 'hybrid'
            state['messages'].append(f"🎯 Hybrid mode: Text + Image (60% text, 40% image)")
    
    # Scenario 2, 3, 5: TEXT ONLY
    else:
        state['search_mode'] = 'text_only'
        state['messages'].append(f"🎯 Text-only mode")
    
    # ========================================================================
    # STEP 3: GEMINI QUERY GENERATION WITH GENDER AWARENESS
    # ========================================================================
    
    context_parts = []
    if has_image:
        context_parts.append(f"User uploaded image: {image_desc}")
    if matched_items:
        context_parts.append(f"Detected items: {', '.join(matched_items[:5])}")
    if matched_colors:
        context_parts.append(f"Detected colors: {', '.join(matched_colors[:5])}")
    
    context_parts.append(f"Target gender: {final_gender.upper()} (source: {gender_source})")
    context_str = "\n".join(context_parts)
    
    available_items = list(Config.DYNAMIC_FASHION_ITEMS)[:40]
    available_colors = list(Config.DYNAMIC_COLORS)[:25]
    available_genders = list(Config.DYNAMIC_GENDERS)
    
    system_instruction = f"""You are an Expert Fashion Search Query Generator with STYLE REASONING and GENDER AWARENESS.

**OBJECTIVE:**
Generate targeted search queries that will find the best items in our catalog based on the user's intent.

**GENDER HANDLING (MANDATORY):**
- Target gender: {final_gender.upper()}
- ALWAYS prefix every query with the gender: "men [item]" or "women [item]".
- If target is "BOTH", generate a balanced mix of "men" and "women" queries.

**SCENARIOS:**

1. **DIRECT SEARCH:** User wants a specific item (e.g., "red dress"). 
   - Generate queries for that exact item with variations in description.

2. **COMPLEMENTARY MATCHING (CRITICAL):** User wants something that "goes with" or "matches" the image (e.g., "pants for this shirt").
   - 1. Identify what item is in the image (e.g., it's a shirt).
   - 2. Identify the *requested* item (e.g., they asked for pants).
   - 3. Use color theory to pick 2-3 colors that match the image item.
   - 4. Generate queries ONLY for the *requested* item in matching colors.
   - *Example:* Image is a "red t-shirt", Query is "pants to fit this". 
     - Reasoning: Red matches well with black, navy, or beige.
     - Queries: ["women black trousers", "women navy jeans", "women beige chinos"]

3. **OUTFIT RECOMMENDATION:** User wants a full look (e.g., "wedding outfit").
   - Generate queries for: top, bottom, footwear, accessories, and ALWAYS "watches".

**OUTPUT FORMAT - STRICT JSON:**
{{
  "reasoning": "Briefly explain the style choice (e.g., 'Matching red shirt with neutral dark pants')",
  "intent": "direct_search" | "matching" | "outfit_recommendation",
  "queries": ["query1", "query2", ...],
  "categories": [...] (Required ONLY for outfit_recommendation)
}}

**RULES:**
- 2-6 words per query.
- Use only common fashion terms.
- Return ONLY valid JSON."""

    user_prompt = f"""Analyze this fashion request:
User Input: "{user_raw}"
Context (from image): {image_desc}
Detected Items: {', '.join(matched_items)}
Target Gender: {final_gender.upper()}

Generate 3-5 search queries. 
If they asked for a 'match', search for COMPLEMENTARY items, not the item in the image.
Return JSON with 'reasoning', 'intent', and 'queries'."""

    # ========================================================================
    # CALL GEMINI API
    # ========================================================================
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(response_text)
        
        # PRINT TO CONSOLE AS REQUESTED
        print(f"\n--- 🤖 Gemini Query Generator Output ---")
        print(f"User Request: {user_raw}")
        print(f"Gemini Response: {json.dumps(parsed_response, indent=2)}")
        print(f"----------------------------------------\n")
        
        intent_type = parsed_response.get('intent', 'direct_search')
        reasoning = parsed_response.get('reasoning', '')
        
        if reasoning:
            state['messages'].append(f"💡 Style Logic: {reasoning}")
        
        if intent_type == "outfit_recommendation":
            all_queries = []
            categories_info = []
            
            for cat in parsed_response.get('categories', []):
                category_name = cat.get('category', 'items')
                category_queries = cat.get('queries', [])
                
                for q in category_queries:
                    all_queries.append(q)
                    categories_info.append(category_name)
            
            state['search_queries'] = all_queries
            state['query_categories'] = categories_info
            state['intent_type'] = 'recommendation'
            state['messages'].append(f"✅ Gemini: {len(all_queries)} queries (Outfit)")
        
        else:
            queries = parsed_response.get('queries', [])
            
            # SANITY CHECK: If matching, ensure queries don't just repeat the image items or the original input
            user_raw_clean = user_raw.lower().strip()
            if has_image:
                filtered_queries = []
                img_words = set(image_desc.lower().split())
                for q in queries:
                    q_lower = q.lower().strip()
                    q_words = set(q_lower.split())
                    
                    # Skip if query is identical to original input
                    if q_lower == user_raw_clean:
                        continue
                        
                    # Skip if query is too similar to image description in matching mode
                    if intent_type == 'matching' and len(q_words - img_words) < 1:
                        continue
                        
                    filtered_queries.append(q)
                queries = filtered_queries if filtered_queries else queries
            
            if not queries:
                queries = [query if query else "fashion items"]
            
            state['search_queries'] = queries[:10]
            state['query_categories'] = ['general'] * len(state['search_queries'])
            state['intent_type'] = intent_type
            state['messages'].append(f"✅ Gemini: {len(state['search_queries'])} queries ({intent_type})")
            
            # Add to debug info for UI visibility
            state['debug_info']['generated_queries'] = queries[:10]
            state['debug_info']['gemini_intent'] = intent_type
            state['debug_info']['gemini_reasoning'] = reasoning
        
    except Exception as e:
        # ====================================================================
        # FALLBACK: Rule-based with gender awareness
        # ====================================================================
        state['messages'].append(f"⚠️ Gemini failed -> Rule-based fallback")
        state['debug_info']['gemini_error'] = str(e)
        
        gender_prefixes = []
        if final_gender == "both":
            gender_prefixes = ["men", "women"]
        elif final_gender == "men":
            gender_prefixes = ["men"]
        elif final_gender == "women":
            gender_prefixes = ["women"]
        else:
            gender_prefixes = ["men", "women"]  # Default to both
        
        if any(word in query for word in ['wedding', 'party', 'office', 'recommend', 'outfit']):
            state['search_queries'] = []
            state['query_categories'] = []
            for gender_prefix in gender_prefixes:
                state['search_queries'].extend([
                    f"{gender_prefix} formal shirt",
                    f"{gender_prefix} dress pants",
                    f"{gender_prefix} formal shoes",
                    f"{gender_prefix} leather belt",
                    f"{gender_prefix} formal watch"  # Added watches
                ])
                state['query_categories'].extend(['top', 'bottom', 'footwear', 'accessories', 'watches'])
            state['intent_type'] = 'recommendation'
        
        else:
            state['search_queries'] = [f"{gp} {query}" for gp in gender_prefixes]
            state['query_categories'] = ['general'] * len(state['search_queries'])
            state['intent_type'] = 'direct_search'
    
    state['debug_info'].update({
        'search_mode': state.get('search_mode'),
        'detected_gender': final_gender,
        'gender_source': gender_source,
        'gender_rule': detected_gender_rule,
        'gender_llm': detected_gender_llm
    })
    
    state['next_agent'] = 'search_executor'
    return state

# ================================================================================
# CELL 11: FIXED SEARCH EXECUTOR WITH PROPER IMAGE SEARCH
# ================================================================================

def search_executor_agent(state: AgentState) -> AgentState:
    """
    FIXED: Properly handles image-only, text-only, and hybrid searches with gender filtering
    """
    search_mode = state.get('search_mode', 'text_only')
    queries = state.get('search_queries', [])
    categories = state.get('query_categories', [])
    intent_type = state.get('intent_type', 'direct_search')
    image_embedding = state.get('image_embedding')
    detected_gender = state.get('detected_gender', 'both')
    
    if not queries and not image_embedding:
        state['final_response'] = "❌ No search criteria available"
        state['search_results_data'] = []
        state['next_agent'] = 'end'
        return state
    
    state['messages'].append(f"🔍 Search mode: {search_mode}, Gender: {detected_gender.upper()}")
    
    all_grouped_results = []
    
    # ========================================================================
    # IMAGE-ONLY SEARCH (100% visual similarity)
    # ========================================================================
    if search_mode == 'image_only' and image_embedding is not None:
        try:
            img_emb_normalized = image_embedding / np.linalg.norm(image_embedding)
            emb_arr = np.array([img_emb_normalized]).astype('float32')
            
            distances, indices = faiss_index.search(emb_arr, 50)  # Get more for filtering
            
            valid_items = []
            for faiss_idx, score in zip(indices[0], distances[0]):
                if faiss_idx >= len(metadata_df):
                    continue
                
                try:
                    meta = metadata_df.iloc[faiss_idx]
                    
                    # Gender filtering
                    item_gender = str(meta.get('gender', '')).lower()
                    if detected_gender == "men" and item_gender not in ["men", "male", "boys"]:
                        continue
                    elif detected_gender == "women" and item_gender not in ["women", "female", "girls"]:
                        continue
                    
                    # Get image path - try multiple column names
                    img_path = (meta.get('source_path') or 
                               meta.get('image_path') or 
                               meta.get('thumbnail_url') or 
                               '')
                    
                    # Skip if no valid image path
                    if not img_path or not os.path.exists(str(img_path)):
                        continue
                    
                    valid_items.append({
                        'id': int(meta.get('id', 0)),
                        'title': str(meta.get('title') or meta.get('product_name') or meta.get('productDisplayName') or 'Product'),
                        'brand': str(meta.get('brand') or meta.get('brandName') or 'Unknown'),
                        'price': float(meta.get('price', 0)),
                        'color': str(meta.get('color') or meta.get('base_color') or meta.get('baseColour') or 'N/A'),
                        'article_type': str(meta.get('article_type') or meta.get('articleType') or 'N/A'),
                        'snippet': str(meta.get('snippet') or meta.get('title') or ''),
                        'source_path': str(img_path),
                        'thumbnail_url': str(img_path),
                        'score': float(score),
                        'similarity': float(score),
                        'gender': item_gender,
                        'image_id': str(meta.get('image_id', meta.get('id', '')))
                    })
                    
                    if len(valid_items) >= 10:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️ Error processing item {faiss_idx}: {str(e)}")
                    continue
            
            all_grouped_results.append({
                "query_number": 1,
                "query_text": "Similar items (visual search)",
                "category": "similar",
                "items": valid_items,
                "item_count": len(valid_items)
            })
            
            state['messages'].append(f"  ✓ Visual search: {len(valid_items)} similar items")
            
        except Exception as e:
            import traceback
            print(f"  ❌ Image search error: {traceback.format_exc()}")
            state['messages'].append(f"  ⚠️ Image search failed: {str(e)[:50]}")
    
    # ========================================================================
    # TEXT-ONLY or HYBRID SEARCH WITH GENDER FILTERING
    # ========================================================================
    else:
        for idx, query_text in enumerate(queries):
            category = categories[idx] if idx < len(categories) else 'general'
            
            try:
                # Get text embedding
                text_emb = get_text_embedding(query_text)
                text_emb = text_emb / np.linalg.norm(text_emb)
                
                # HYBRID: Combine text and image embeddings
                if search_mode == 'hybrid' and image_embedding is not None:
                    img_emb_normalized = image_embedding / np.linalg.norm(image_embedding)
                    combined_emb = (Config.TEXT_WEIGHT * text_emb + 
                                   Config.IMAGE_WEIGHT * img_emb_normalized)
                    combined_emb = combined_emb / np.linalg.norm(combined_emb)
                    search_emb = combined_emb
                else:
                    search_emb = text_emb
                
                # Search FAISS
                emb_arr = np.array([search_emb]).astype('float32')
                distances, indices = faiss_index.search(emb_arr, 50)  # Get more for filtering
                
                # Collect valid results with gender filtering
                valid_items = []
                for faiss_idx, score in zip(indices[0], distances[0]):
                    if faiss_idx >= len(metadata_df):
                        continue
                    
                    try:
                        meta = metadata_df.iloc[faiss_idx]
                        
                        # Gender filtering logic
                        item_gender = str(meta.get('gender', '')).lower()
                        if detected_gender == "men" and item_gender not in ["men", "male", "boys"]:
                            continue
                        elif detected_gender == "women" and item_gender not in ["women", "female", "girls"]:
                            continue
                        # If detected_gender == "both", include all items
                        
                        # Get image path - try multiple column names
                        img_path = (meta.get('source_path') or 
                                   meta.get('image_path') or 
                                   meta.get('thumbnail_url') or 
                                   '')
                        
                        # RELAXED: Don't skip if image doesn't exist, but flag it
                        img_exists = os.path.exists(str(img_path)) if img_path else False
                        
                        valid_items.append({
                            'id': int(meta.get('id', 0)),
                            'title': str(meta.get('title') or meta.get('product_name') or meta.get('productDisplayName') or 'Product'),
                            'brand': str(meta.get('brand') or meta.get('brandName') or 'Unknown'),
                            'price': float(meta.get('price', 0)),
                            'color': str(meta.get('color') or meta.get('base_color') or meta.get('baseColour') or 'N/A'),
                            'article_type': str(meta.get('article_type') or meta.get('articleType') or 'N/A'),
                            'snippet': str(meta.get('snippet') or meta.get('title') or ''),
                            'source_path': str(img_path) if img_exists else '',
                            'thumbnail_url': str(img_path) if img_exists else '',
                            'score': float(score),
                            'similarity': float(score),
                            'gender': item_gender,
                            'image_id': str(meta.get('image_id', meta.get('id', ''))),
                            'image_exists': img_exists
                        })
                        
                        if len(valid_items) >= 10:
                            break
                            
                    except Exception as e:
                        print(f"  ⚠️ Error processing item {faiss_idx}: {str(e)}")
                        continue
                
                all_grouped_results.append({
                    "query_number": idx + 1,
                    "query_text": query_text,
                    "category": category,
                    "items": valid_items,
                    "item_count": len(valid_items),
                    "gender_filter": detected_gender
                })
                
                mode_str = "hybrid" if search_mode == 'hybrid' else "text"
                state['messages'].append(f"  ✓ Q{idx+1} ({mode_str}) [{category}]: '{query_text}' -> {len(valid_items)} items ({detected_gender})")
                
            except Exception as e:
                import traceback
                print(f"  ❌ Query {idx+1} error: {traceback.format_exc()}")
                state['messages'].append(f"  ⚠️ Query {idx+1} failed: {str(e)[:50]}")
                all_grouped_results.append({
                    "query_number": idx + 1,
                    "query_text": query_text,
                    "category": category,
                    "items": [],
                    "item_count": 0,
                    "gender_filter": detected_gender,
                    "error": str(e)
                })
    
    # ========================================================================
    # BUILD FINAL RESPONSE WITH GENDER INFO
    # ========================================================================
    
    total_items = sum(g['item_count'] for g in all_grouped_results)
    state['search_results_data'] = all_grouped_results
    
    gender_info = ""
    if detected_gender == "men":
        gender_info = "\n🚹 **Showing: Men's Fashion Only**"
    elif detected_gender == "women":
        gender_info = "\n🚺 **Showing: Women's Fashion Only**"
    elif detected_gender == "both":
        gender_info = "\n⚧ **Showing: Both Men's and Women's Fashion**"
    
    if search_mode == 'image_only':
        state['final_response'] = f"""📸 **Similar Fashion Items**

**Found {total_items} visually similar items**{gender_info}

---

**💡 Tip:** These items match the style, color, and type of your uploaded image!"""
    
    elif intent_type == 'recommendation':
        state['final_response'] = f"""✨ **Complete Outfit Recommendation**

**{len(queries)} Items Curated for Your Occasion**
**{total_items} Total Products Found**{gender_info}

---

**💡 Styling Tip:** Mix and match these pieces for a complete look!"""
        
        categories_dict = {}
        category_emojis = {
            'top': '👕',
            'bottom': '👖',
            'footwear': '👟',
            'accessories': '👜',
            'watches': '⌚'
        }
        
        for group in all_grouped_results:
            cat = group['category']
            if cat not in categories_dict:
                categories_dict[cat] = []
            categories_dict[cat].append(group)
        
        # Display in logical order
        category_order = ['top', 'bottom', 'footwear', 'accessories', 'watches']
        
        for cat_name in category_order:
            if cat_name in categories_dict:
                cat_groups = categories_dict[cat_name]
                total_cat_items = sum(g['item_count'] for g in cat_groups)
                emoji = category_emojis.get(cat_name, '📦')
                state['final_response'] += f"\n\n**{emoji} {cat_name.upper()}** ({total_cat_items} items)"
                for group in cat_groups:
                    if group['item_count'] > 0:
                        state['final_response'] += f"\n  └─ {group['query_text']}: {group['item_count']} options"
    
    elif search_mode == 'hybrid':
        intent_label = "Smart Matching" if intent_type == 'matching' else "Smart Hybrid"
        state['final_response'] = f"""🎨 **{intent_label} Results**
        
**These items complement your image and match your request!**
**{total_items} Total Matches**{gender_info}

---

**💡 Styling Advice:** I used your image's style/color to find matching items that look great together!"""
        
        for group in all_grouped_results:
            if group['item_count'] > 0:
                state['final_response'] += f"\n\n**{group['query_text']}**"
                state['final_response'] += f"\n└─ {group['item_count']} items"
    
    else:
        state['final_response'] = f"""🔍 **Search Results**

**Found {total_items} items**{gender_info}

---"""
        
        for group in all_grouped_results:
            if group['item_count'] > 0:
                state['final_response'] += f"\n\n**{group['query_text']}**"
                state['final_response'] += f"\n└─ {group['item_count']} items"
    
    state['next_agent'] = 'end'
    state['messages'].append(f"✅ Complete: {total_items} items, mode={search_mode}, gender={detected_gender}")
    
    # Add debug info
    state['debug_info']['search_debug'] = {
        'total_queries': len(queries),
        'total_results': total_items,
        'metadata_size': len(metadata_df),
        'faiss_size': faiss_index.ntotal
    }
    
    return state