"""
OCR Engine Module.

This module handles the Optical Character Recognition (OCR) and Information Extraction
phase of the pipeline. It leverages Vision-Language Models (VLMs) via Ollama
to transcribe images directly into structured JSON data.

It bypasses traditional OCR + Regex pipelines by using a generative approach.
"""

import requests
import json
import base64

# The system prompt defines the schema and persona for the LLM.
# It ensures the output is strict JSON for downstream processing.
SYSTEM_PROMPT = """
You are an expert accountant. Extract the following fields from the invoice image/text:
- vendor_name (string)
- invoice_number (string)
- invoice_date (YYYY-MM-DD)
- total_amount (float)
- currency (USD, EUR, etc)
- line_items (list of objects: {description, quantity, unit_price, total})

Return ONLY valid JSON. No markdown.
"""

def extract_invoice_data(image_path):
    """
    Extracts structured data from an invoice image using a local Vision LLM.
    
    This function:
    1. Encodes the image to Base64.
    2. Sends it to the local Ollama instance (running Llama 3.2 Vision or similar).
    3. Parses the LLM's response into a Python dictionary.
    
    Args:
        image_path (str): The file path to the invoice image.
        
    Returns:
        dict: The extracted invoice data (or an error dictionary).
    """
    print(f"📄 Processing {image_path}...")
    
    try:
        # Read image and convert to base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # DeepSeek-OCR via Ollama API (ensure model is pulled: ollama pull deepseek-r1 or similar vision model)
        # Note: If using pure OCR model, you get text. 
        # We use a Vision LLM here for "OCR + Extraction" in one step.
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model': 'llama3.2-vision', # Or deepseek-vl if available
                'messages': [{
                    'role': 'user',
                    'content': SYSTEM_PROMPT,
                    'images': [image_base64]
                }],
                'stream': False
            },
            timeout=30
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        # Clean response to ensure it's pure JSON
        content = response_data['message']['content'].replace("```json", "").replace("```", "").strip()
        return json.loads(content)
        
    except Exception:
        return {"error": "OCR processing failed", "file": image_path}
