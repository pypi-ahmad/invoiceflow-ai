"""
InvoiceFlow AI - Streamlit Dashboard Application.

This module serves as the frontend for the Invoice Processing Pipeline.
It provides a user interface for:
1. Uploading invoice documents (PDF/Images).
2. Triggering the OCR and validation pipeline.
3. Displaying processed results and analytics.
4. Auditing and reviewing flagged invoices.

The application uses Streamlit for the UI and connects to the backend
processing modules (OCR, Validation, Vector DB).
"""

import streamlit as st
import pandas as pd
import os
from ocr_engine import extract_invoice_data
from processor import process_pipeline

# Configure the Streamlit page layout
st.set_page_config(layout="wide", page_title="InvoiceFlow AI")

st.title("🧾 Intelligent Invoice Processing Pipeline")

# Sidebar: Input
with st.sidebar:
    st.header("Input Stream")
    uploaded_files = st.file_uploader("Upload Invoices (PDF/IMG)", accept_multiple_files=True)
    run_btn = st.button("Run Batch Process")

# Main Dashboard
col1, col2 = st.columns([2, 1])

if run_btn and uploaded_files:
    results = []
    
    with st.spinner(f"Processing {len(uploaded_files)} documents with DeepSeek..."):
        for uploaded_file in uploaded_files:
            # 1. Save temp file
            bytes_data = uploaded_file.getvalue()
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(bytes_data)
            
            # 2. Pipeline Execution
            raw_data = extract_invoice_data(temp_path)
            processed_data = process_pipeline(raw_data)
            processed_data['filename'] = uploaded_file.name
            results.append(processed_data)
            
            # Cleanup
            os.remove(temp_path)

    # Convert to DataFrame for display
    df = pd.json_normalize(results)
    
    with col1:
        st.subheader("Processing Queue")
        # Color code rows based on flags
        display_cols = ['filename', 'standardized_vendor', 'total_amount', 'po_match_status', 'audit_risk_score']
        # check if columns exist
        display_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[display_cols],
            use_container_width=True
        )

    with col2:
        st.subheader("Analytics")
        total_spend = df['total_amount'].sum()
        st.metric("Total Batch Value", f"${total_spend:,.2f}")
        
        flagged_count = df[df['flags'].apply(lambda x: len(x) > 0)].shape[0]
        st.metric("Flagged for Review", flagged_count, delta_color="inverse")

    # Detailed Audit View
    st.divider()
    st.subheader("🔍 Audit & Edit")
    
    selected_invoice = st.selectbox("Select Invoice to Review", df['filename'])
    invoice_data = df[df['filename'] == selected_invoice].iloc[0]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Extracted Data")
        st.json(invoice_data.to_dict())
    with c2:
        st.subheader("Validation Rules")
        if len(invoice_data['flags']) > 0:
            st.error("Issues Detected:")
            for flag in invoice_data['flags']:
                st.write(f"- {flag}")
        else:
            st.success("✅ Clean Invoice")
            
        st.subheader("3-Way Match")
        st.info(invoice_data.get('po_match_status', 'N/A'))
        
    with c3:
        st.subheader("🤖 AI Audit Guard")
        risk = invoice_data.get('audit_risk_score', 0)
        st.metric("Risk Score", f"{risk}/100")
        
        audit_flags = invoice_data.get('audit_flags', [])
        if audit_flags:
            st.warning("AI Flags:")
            for flag in audit_flags:
                st.write(f"- {flag}")
        else:
            st.success("No AI Flags")

        st.button("Approve & Push to ERP")

else:
    st.info("Upload invoices to begin batch processing.")
