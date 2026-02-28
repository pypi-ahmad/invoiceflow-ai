"""
ERP Mock Module.

This module simulates an Enterprise Resource Planning (ERP) system or a Purchase Order (PO) database.
It is used to perform 3-Way Matching, a critical accounting control process.

Key Features:
- Mock Database of approved Purchase Orders (POS).
- Logic to cross-reference incoming invoices against approved POs.
- Variance detection to flag discrepancies in total amounts.
"""

# Mock PO Database
# In a real-world scenario, this would be a SQL database or an API connection to SAP/Oracle/NetSuite.
POS = {
    "PO-1001": {"vendor": "Amazon Web Services", "total_approved": 5000.00, "items": ["Cloud Hosting"]},
    "PO-1002": {"vendor": "Staples", "total_approved": 250.50, "items": ["Paper", "Pens"]}
}

def check_3_way_match(invoice_data):
    """
    Performs a 3-Way Match validation between the Invoice and the Purchase Order.
    
    The 3-Way Match checks:
    1. Does the PO number exist?
    2. Does the Invoice Total match the PO Total (within a small tolerance)?
    
    Args:
        invoice_data (dict): The extracted data from the invoice (JSON).
        
    Returns:
        str: A status message indicating 'Matched', 'Variance', or 'PO Not Found'.
    """
    # Try to find PO number in invoice data, case-insensitive keys
    po_num = None
    for k, v in invoice_data.items():
        if 'po' in k.lower() and 'number' in k.lower():
            po_num = v
            break
    
    # If not found, check if it's explicitly 'po_number'
    if not po_num:
        po_num = invoice_data.get('po_number')

    if not po_num or po_num not in POS:
        return "⚠️ PO Not Found"
    
    po_data = POS[po_num]
    inv_total = invoice_data.get('total_amount', 0)
    
    # Variance Check
    try:
        variance = float(inv_total) - float(po_data['total_approved'])
        if abs(variance) < 1.0:
            return "✅ Matched"
        else:
            return f"❌ Variance of ${variance:.2f}"
    except (TypeError, ValueError):
        return "⚠️ Error calculating variance"
