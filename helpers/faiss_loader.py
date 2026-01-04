
import faiss
import pandas as pd
import numpy as np
import pickle
import os
from config import Config
from helpers.embeddings import get_image_embedding
from PIL import Image

def load_saved_index():
    index_path = os.path.join(Config.DATA_PATH, "fashion_faiss_index.faiss")
    # Try both pickle and csv for metadata, prioritizing pickle
    metadata_pickle = os.path.join(Config.DATA_PATH, "fashion_metadata.pkl")
    metadata_csv = os.path.join(Config.DATA_PATH, "fashion_metadata.csv")
    
    index = None
    metadata_df = pd.DataFrame()
    
    # Load Metadata
    if os.path.exists(metadata_pickle):
        try:
            with open(metadata_pickle, "rb") as f:
                metadata_df = pickle.load(f)
            # Fix price column
            if 'price' in metadata_df.columns:
                metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce").fillna(0)
            print(f"✅ Loaded metadata from {metadata_pickle} ({len(metadata_df)} items)")
        except Exception as e:
            print(f"⚠️ Failed to load metadata pickle: {e}")
            
    elif os.path.exists(metadata_csv):
        try:
            metadata_df = pd.read_csv(metadata_csv)
            if 'price' in metadata_df.columns:
                metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce").fillna(0)
            print(f"✅ Loaded metadata from {metadata_csv} ({len(metadata_df)} items)")
        except Exception as e:
            print(f"⚠️ Failed to load metadata csv: {e}")
            
    # Load Index
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print(f"✅ Loaded FAISS index from {index_path} ({index.ntotal} vectors)")
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}")
            
    if index is not None and not metadata_df.empty:
        return index, metadata_df
        
    print("⚠️ Saved index/metadata not found or invalid. Building new index...")
    index, metadata_df = build_new_index()
    
    if index is not None and not metadata_df.empty:
        save_index(index, metadata_df)
        
    return index, metadata_df

def save_index(index, metadata_df):
    index_path = os.path.join(Config.DATA_PATH, "fashion_faiss_index.faiss")
    metadata_pickle = os.path.join(Config.DATA_PATH, "fashion_metadata.pkl")
    
    try:
        faiss.write_index(index, index_path)
        with open(metadata_pickle, "wb") as f:
            pickle.dump(metadata_df, f)
        print(f"✅ Saved index and metadata to disk.")
    except Exception as e:
        print(f"⚠️ Failed to save index: {e}")

def build_new_index():
    if not os.path.exists(Config.STYLES_CSV):
         print("❌ Styles CSV not found. Returning empty index.")
         index = faiss.IndexFlatIP(512) 
         return index, pd.DataFrame()
         
    print(f"   Loading styles from {Config.STYLES_CSV}...")
    df = pd.read_csv(Config.STYLES_CSV, on_bad_lines='skip')
    
    # Helper to resolve path
    def resolve_path(r):
        # We assume id matches filename in IMAGES_PATH
        p = os.path.join(Config.IMAGES_PATH, f"{r['id']}.jpg")
        return p if os.path.exists(p) else None
        
    df['image_path'] = df.apply(resolve_path, axis=1)
    df = df[df['image_path'].notna()]
    
    return build_faiss_index(df)

def build_faiss_index(df_subset):
    embeddings, metadata = [], []
    
    print(f"   Processing {len(df_subset)} items for index...")
    
    # Limit for safety 
    limit = Config.FAISS_MAX_ITEMS
    
    count = 0
    for idx, row in df_subset.head(limit).iterrows():
        try:
            img_path = row['image_path']
            img = Image.open(img_path).convert('RGB')
            emb = get_image_embedding(img)
            embeddings.append(emb)
            
            # Construct metadata consistent with expected schema
            metadata.append({
                'id': row.get('id', idx),
                'title': str(row.get('productDisplayName', row.get('product_name', 'Product'))),
                'brand': str(row.get('brandName', row.get('brand', 'Unknown'))),
                'price': row.get('price', 0),
                'color': str(row.get('baseColour', row.get('color', 'N/A'))),
                'article_type': str(row.get('articleType', row.get('article_type', 'N/A'))),
                'gender': str(row.get('gender', 'N/A')),
                'image_path': img_path,
                'source_path': img_path,
                'snippet': f"{row.get('gender', '')} {row.get('articleType', '')} {row.get('baseColour', '')}"
            })
            
            count += 1
            if count % 100 == 0:
                print(f"   ... processed {count} items")
                
        except Exception as e:
            continue
            
    if not embeddings:
        print("❌ No embeddings generated.")
        index = faiss.IndexFlatIP(512)
        return index, pd.DataFrame(metadata)

    emb_array = np.array(embeddings).astype('float32')
    
    d = emb_array.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb_array)
    
    metadata_df = pd.DataFrame(metadata)
    
    print(f"✅ Built fresh index: {index.ntotal} items")
    return index, metadata_df