"""
Invoice Processor Module.

This is the central logic core of the pipeline. It orchestrates:
1. Vendor Normalization (using Vector/Fuzzy matching).
2. Rule-Based Validation (duplicates, missing fields).
3. AI-Based Audit (fraud detection using LLMs).
4. 3-Way Matching (ERP validation).

It transforms raw OCR data into a validated, flagged business object.
"""

import pandas as pd
from rapidfuzz import process, fuzz
from datetime import datetime
import requests
import json
from vector_db import smart_match_vendor
from erp_mock import check_3_way_match

# Simulating a Master Vendor Database
KNOWN_VENDORS = ["Amazon Web Services", "Microsoft Azure", "GitHub Inc", "Staples", "WeWork"]

# Simulating a Database of processed invoices
PROCESSED_DB = pd.DataFrame(columns=["invoice_number", "vendor", "total", "date"])

AUDIT_PROMPT = """
You are a Forensic Accountant. Analyze this invoice JSON for risk.
Rules:
1. Flag if 'bank_details' changed (simulate this check).
2. Flag if vendor email is a free provider (@gmail, @yahoo) but vendor is Corporate.
3. Check if tax amount seems consistent (~10-20%).
4. Flag round number totals (e.g. $5000.00) which are rare in business.

Return JSON: {"risk_score": 0-100, "flags": ["list of warnings"]}
INVOICE DATA:
"""

def audit_invoice_with_ai(invoice_json):
    """
    Uses a Large Language Model (Llama 3) to act as a 'Forensic Accountant'.
    
    It analyzes the invoice data for subtle fraud indicators that rule-based systems miss,
    such as:
    - Inconsistent tax rates.
    - Suspicious email domains for corporate vendors.
    - Round-number totals (uncommon in B2B).
    - Changes in bank details (simulated).
    
    Args:
        invoice_json (dict): The invoice data.
        
    Returns:
        dict: A dictionary containing a 'risk_score' and a list of 'flags'.
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model': 'llama3', # or lfm2.5-thinking
                'messages': [{'role': 'user', 'content': AUDIT_PROMPT + json.dumps(invoice_json)}],
                'stream': False,
                'format': 'json' 
            }
        )
        if response.status_code == 200:
             content = response.json()['message']['content']
             return json.loads(content)
        return {"risk_score": 0, "flags": ["AI Audit Failed"]}
    except Exception as e:
        return {"risk_score": 0, "flags": [f"Audit Error: {str(e)}"]}

def match_vendor(raw_name):
    """Fuzzy matches OCR vendor name to Master Database"""
    # ... existing code ...
    if not raw_name: return "Unknown"
    
    # Returns (match, score, index)
    match = process.extractOne(raw_name, KNOWN_VENDORS, scorer=fuzz.token_sort_ratio)
    
    if match and match[1] > 80: # 80% confidence threshold
        return match[0]
    return f"New Vendor ({raw_name})"

def validate_invoice(data):
    flags = []
    
    # 1. Check for Duplicate (Business Logic)
    # A duplicate is same Vendor + Same Invoice Number
    is_duplicate = not PROCESSED_DB[
        (PROCESSED_DB['invoice_number'] == data.get('invoice_number')) & 
        (PROCESSED_DB['vendor'] == data.get('vendor_name'))
    ].empty
    
    if is_duplicate:
        flags.append("🔴 Duplicate Invoice Detected")

    # 2. Check for High Value
    if data.get('total_amount', 0) > 10000:
        flags.append("🟠 High Value - Approval Needed")

    # 3. Check Data Integrity
    if not data.get('invoice_number'):
        flags.append("🔴 Missing Invoice Number")
        
    return flags

def process_pipeline(raw_json):
    """
    Runs the full processing pipeline on raw extracted data.
    
    Steps:
    1. Vendor Normalization: Map raw OCR strings to Master Vendor records.
    2. Validation: Apply deterministic business rules.
    3. AI Audit: Apply probabilistic fraud detection.
    4. 3-Way Match: Validate against the Mock ERP.
    
    Args:
        raw_json (dict): The raw JSON output from the OCR engine.
        
    Returns:
        dict: The enriched data object with flags, scores, and standardized fields.
    """
    # 1. Vendor Matching (Try Vector first, then Fuzzy fallback if needed, but here we just use Vector)
    # The prompt asked to replace Fuzzy with Semantic matching. 
    # But we can keep both or switch. Let's switch to smart_match_vendor as primary.
    standardized_vendor = smart_match_vendor(raw_json.get('vendor_name'))
    
    # Fallback to fuzzy if Vector DB returns "Unknown" or similar and we want to try fuzzy on the local list
    if "Unknown" in standardized_vendor or "New Vendor" in standardized_vendor:
         # Try the old fuzzy match against the KNOWN_VENDORS list in this file
         fuzzy_match = match_vendor(raw_json.get('vendor_name'))
         if "New Vendor" not in fuzzy_match and "Unknown" not in fuzzy_match:
             standardized_vendor = fuzzy_match
             
    raw_json['standardized_vendor'] = standardized_vendor
    
    # 2. Run Rule-Based Validation
    raw_json['flags'] = validate_invoice(raw_json)
    
    # 3. AI Audit Guard
    audit_result = audit_invoice_with_ai(raw_json)
    raw_json['audit_risk_score'] = audit_result.get('risk_score', 0)
    raw_json['audit_flags'] = audit_result.get('flags', [])
    
    # 4. 3-Way Match
    raw_json['po_match_status'] = check_3_way_match(raw_json)
    
    return raw_json
