"""
Vector Database Module.

This module implements a Semantic Search engine for Vendor Matching.
Instead of relying on exact spelling (fuzzy match), it uses Vector Embeddings
to find vendors that are "conceptually similar".

It includes:
- An in-memory Vector Store (dictionary of numpy arrays).
- Integration with Ollama for generating embeddings.
- Cosine Similarity calculation for matching.
"""

import requests
import numpy as np
import json

# Mock Vendor Master Data
VENDORS = [
    "Amazon Web Services", "Microsoft Azure", "Google Cloud Platform", 
    "Staples Office Supplies", "WeWork Space", "Delta Airlines", "Uber Business"
]

# We will store embeddings in memory for this demo
# In production, use a Vector DB like ChromaDB or Qdrant
VENDOR_EMBEDDINGS = {}

def get_embedding(text, model="nomic-embed-text"):
    """
    Generates a vector embedding for a given text using the Ollama API.
    
    Args:
        text (str): The text to embed (e.g., a vendor name).
        model (str): The embedding model to use.
        
    Returns:
        list: A list of floats representing the vector embedding.
    """
    try:
        # Try embeddinggemma first as requested, fallback to nomic-embed-text or llama3
        url = "http://localhost:11434/api/embeddings"
        payload = {
            "model": model,
            "prompt": text
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get('embedding')
        else:
            # If 404, maybe model not found, try generic one
            if model != "all-minilm":
                 return get_embedding(text, model="all-minilm")
            return None
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def initialize_vector_db():
    """Compute embeddings for all vendors on startup"""
    print("Initializing Vector DB...")
    # Use a common embedding model. 
    # User asked for 'embeddinggemma', but 'nomic-embed-text' is more standard for Ollama.
    # We'll try 'embeddinggemma' first.
    target_model = 'embeddinggemma'
    
    for vendor in VENDORS:
        emb = get_embedding(vendor, model=target_model)
        if emb:
            VENDOR_EMBEDDINGS[vendor] = emb
        else:
            print(f"Could not embed {vendor}. Ensure '{target_model}' or 'all-minilm' is pulled.")

def smart_match_vendor(ocr_name):
    """
    Finds the conceptually closest vendor using Vector Search (Cosine Similarity).
    
    This matches "Amazon Mktp" to "Amazon Web Services" based on semantic meaning
    rather than just character overlap.
    
    Args:
        ocr_name (str): The vendor name extracted from the invoice.
        
    Returns:
        str: The matched Master Vendor Name or "New Vendor" if no close match found.
    """
    if not ocr_name:
        return "Unknown"
        
    if not VENDOR_EMBEDDINGS:
        # Lazy load if not initialized
        initialize_vector_db()
        
    if not VENDOR_EMBEDDINGS:
        return f"Unknown (DB Init Failed)"

    input_emb = get_embedding(ocr_name, model='embeddinggemma')
    if not input_emb:
        return f"Unknown (Embedding Failed)"

    best_match = None
    best_score = -1.0

    for vendor, db_emb in VENDOR_EMBEDDINGS.items():
        score = cosine_similarity(input_emb, db_emb)
        if score > best_score:
            best_score = score
            best_match = vendor
            
    # Threshold for "Conceptually Similar"
    if best_score > 0.6: 
        return best_match
    
    return f"New Vendor ({ocr_name})"
