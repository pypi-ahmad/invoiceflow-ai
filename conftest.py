"""
Shared pytest fixtures for InvoiceFlow AI test suite.

Provides reusable mock data and configurations across all test modules.
"""

import pytest
import sys
import os
import pandas as pd

# Ensure the project root is on PYTHONPATH for imports
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Auto-reset PROCESSED_DB before every test (prevents C-01 cross-contamination)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_processed_db():
    """Clear processor.PROCESSED_DB before each test."""
    import processor
    processor.PROCESSED_DB = pd.DataFrame(columns=["invoice_number", "vendor", "total", "date"])
    yield
    processor.PROCESSED_DB = pd.DataFrame(columns=["invoice_number", "vendor", "total", "date"])


# ---------------------------------------------------------------------------
# Invoice data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_invoice_data():
    """A complete, well-formed invoice dict as would come from OCR."""
    return {
        "vendor_name": "Amazon Web Services",
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-06-15",
        "total_amount": 4999.99,
        "currency": "USD",
        "line_items": [
            {"description": "Cloud Hosting", "quantity": 1, "unit_price": 4999.99, "total": 4999.99}
        ],
    }


@pytest.fixture
def high_value_invoice_data():
    """Invoice with total_amount > 10000 threshold."""
    return {
        "vendor_name": "Microsoft Azure",
        "invoice_number": "INV-2024-002",
        "invoice_date": "2024-07-01",
        "total_amount": 15000.00,
        "currency": "USD",
        "line_items": [],
    }


@pytest.fixture
def missing_fields_invoice():
    """Invoice missing invoice_number — triggers validation flag."""
    return {
        "vendor_name": "Staples",
        "total_amount": 100.00,
        "currency": "USD",
    }


@pytest.fixture
def ocr_error_result():
    """Simulates the error dict returned when OCR fails."""
    return {"error": "Connection refused", "file": "temp_bad.png"}


@pytest.fixture
def po_matched_invoice():
    """Invoice that should match PO-1001 exactly."""
    return {
        "vendor_name": "Amazon Web Services",
        "invoice_number": "INV-2024-010",
        "invoice_date": "2024-08-01",
        "total_amount": 5000.00,
        "currency": "USD",
        "po_number": "PO-1001",
        "line_items": [],
    }


@pytest.fixture
def po_variance_invoice():
    """Invoice that matches PO-1001 but with amount variance."""
    return {
        "vendor_name": "Amazon Web Services",
        "invoice_number": "INV-2024-011",
        "invoice_date": "2024-08-02",
        "total_amount": 5500.00,
        "currency": "USD",
        "po_number": "PO-1001",
        "line_items": [],
    }


@pytest.fixture
def sample_embedding():
    """A deterministic 5-dimension embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def zero_embedding():
    """An all-zeros vector — edge case for cosine similarity."""
    return [0.0, 0.0, 0.0, 0.0, 0.0]
