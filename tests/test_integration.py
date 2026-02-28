"""
Integration tests for the InvoiceFlow AI pipeline.

These tests verify multi-module interactions with all external services mocked.
They exercise the real call-chains between modules:
  app.py → ocr_engine.py → processor.py → vector_db.py → erp_mock.py

Coverage intent:
  - End-to-end: OCR output → process_pipeline → enriched result
  - Error propagation: OCR error dict flowing through pipeline
  - Vector + fuzzy fallback chain across modules
  - 3-way match with real erp_mock logic (no mock on erp_mock)
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import vector_db
from processor import process_pipeline
from erp_mock import check_3_way_match


# ============================================================================
# End-to-end pipeline integration (OCR result → pipeline → enriched dict)
# ============================================================================

class TestEndToEndPipeline:
    """Integration: process_pipeline with real validate_invoice + real erp_mock."""

    def setup_method(self):
        vector_db.VENDOR_EMBEDDINGS.clear()

    def teardown_method(self):
        vector_db.VENDOR_EMBEDDINGS.clear()

    @patch("processor.audit_invoice_with_ai")
    @patch("processor.smart_match_vendor")
    def test_clean_invoice_with_po_match(self, mock_vendor, mock_audit):
        """Simulates a clean invoice that matches PO-1001 exactly."""
        mock_vendor.return_value = "Amazon Web Services"
        mock_audit.return_value = {"risk_score": 5, "flags": []}

        ocr_output = {
            "vendor_name": "Amazon Web Services",
            "invoice_number": "INV-INT-001",
            "invoice_date": "2024-06-15",
            "total_amount": 5000.00,
            "currency": "USD",
            "po_number": "PO-1001",
            "line_items": [{"description": "Cloud", "quantity": 1, "unit_price": 5000, "total": 5000}],
        }

        result = process_pipeline(ocr_output)

        # Vendor matched
        assert result["standardized_vendor"] == "Amazon Web Services"
        # No validation flags (under 10k, has invoice_number)
        assert result["flags"] == []
        # Low risk
        assert result["audit_risk_score"] == 5
        # PO matched via real erp_mock
        assert "Matched" in result["po_match_status"]

    @patch("processor.audit_invoice_with_ai")
    @patch("processor.smart_match_vendor")
    def test_high_value_invoice_with_variance(self, mock_vendor, mock_audit):
        """Invoice >10k with PO variance → multiple flags."""
        mock_vendor.return_value = "Amazon Web Services"
        mock_audit.return_value = {"risk_score": 60, "flags": ["Round total"]}

        ocr_output = {
            "vendor_name": "AWS",
            "invoice_number": "INV-INT-002",
            "total_amount": 15000.00,
            "po_number": "PO-1001",
        }

        result = process_pipeline(ocr_output)

        assert any("High Value" in f for f in result["flags"])
        assert result["audit_risk_score"] == 60
        # Variance: 15000 - 5000 = 10000
        assert "Variance" in result["po_match_status"]
        assert "$10000.00" in result["po_match_status"]

    @patch("processor.audit_invoice_with_ai")
    @patch("processor.smart_match_vendor")
    def test_missing_fields_and_no_po(self, mock_vendor, mock_audit):
        """Minimal invoice missing invoice_number and po_number."""
        mock_vendor.return_value = "Unknown (DB Init Failed)"
        mock_audit.return_value = {"risk_score": 80, "flags": ["Very suspicious"]}

        ocr_output = {
            "vendor_name": "Random Corp",
            "total_amount": 999.99,
        }

        result = process_pipeline(ocr_output)

        assert any("Missing Invoice Number" in f for f in result["flags"])
        assert "PO Not Found" in result["po_match_status"]
        assert result["audit_risk_score"] == 80


# ============================================================================
# OCR error propagation through pipeline
# ============================================================================

class TestErrorPropagation:
    """Integration: OCR error dict flowing through pipeline — documents Phase 2 C-02."""

    def setup_method(self):
        vector_db.VENDOR_EMBEDDINGS.clear()

    def teardown_method(self):
        vector_db.VENDOR_EMBEDDINGS.clear()

    @patch("processor.audit_invoice_with_ai")
    @patch("processor.smart_match_vendor")
    def test_ocr_error_dict_enters_pipeline(self, mock_vendor, mock_audit):
        """OCR failure dict is processed as if it were valid data.
        Documents bug C-02: no error-checking at pipeline entry."""
        mock_vendor.return_value = "Unknown"
        mock_audit.return_value = {"risk_score": 0, "flags": []}

        error_dict = {"error": "Connection refused", "file": "temp_bad.png"}

        # This should NOT crash — pipeline processes it silently
        result = process_pipeline(error_dict)

        # Pipeline adds fields even to error dicts
        assert "standardized_vendor" in result
        assert "flags" in result
        assert "audit_risk_score" in result
        assert "po_match_status" in result
        # The error key is still present
        assert result["error"] == "Connection refused"
        # Missing invoice_number is flagged
        assert any("Missing Invoice Number" in f for f in result["flags"])


# ============================================================================
# Vector → Fuzzy fallback chain (cross-module)
# ============================================================================

class TestVendorMatchingFallbackChain:
    """Integration: smart_match_vendor → match_vendor fallback."""

    def setup_method(self):
        vector_db.VENDOR_EMBEDDINGS.clear()

    def teardown_method(self):
        vector_db.VENDOR_EMBEDDINGS.clear()

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="New Vendor (Stpls)")
    def test_vector_new_vendor_fuzzy_recovers(self, mock_vector, mock_audit):
        """Vector returns 'New Vendor' → fuzzy called → matches 'Staples'."""
        ocr_output = {
            "vendor_name": "Stpls",
            "invoice_number": "INV-FB-001",
            "total_amount": 200,
        }

        with patch("processor.match_vendor", return_value="Staples") as mock_fuzzy:
            result = process_pipeline(ocr_output)
            mock_fuzzy.assert_called_once_with("Stpls")
            assert result["standardized_vendor"] == "Staples"

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Amazon Web Services")
    def test_vector_succeeds_fuzzy_not_called(self, mock_vector, mock_audit):
        """Vector match succeeds → fuzzy fallback is NOT invoked."""
        ocr_output = {
            "vendor_name": "AWS",
            "invoice_number": "INV-FB-002",
            "total_amount": 100,
        }

        with patch("processor.match_vendor") as mock_fuzzy:
            result = process_pipeline(ocr_output)
            mock_fuzzy.assert_not_called()
            assert result["standardized_vendor"] == "Amazon Web Services"


# ============================================================================
# 3-way match integration with real erp_mock
# ============================================================================

class TestThreeWayMatchIntegration:
    """Integration: process_pipeline calls real check_3_way_match from erp_mock."""

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Amazon Web Services")
    def test_real_po_matched(self, mock_vendor, mock_audit):
        """PO-1001 + total 5000 → real erp_mock returns Matched."""
        raw = {
            "vendor_name": "AWS",
            "invoice_number": "INV-3W-001",
            "total_amount": 5000.00,
            "po_number": "PO-1001",
        }
        result = process_pipeline(raw)
        assert "Matched" in result["po_match_status"]
        assert "✅" in result["po_match_status"]

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Staples Office Supplies")
    def test_real_po_1002_matched(self, mock_vendor, mock_audit):
        """PO-1002 + total 250.50 → real erp_mock returns Matched."""
        raw = {
            "vendor_name": "Staples",
            "invoice_number": "INV-3W-002",
            "total_amount": 250.50,
            "po_number": "PO-1002",
        }
        result = process_pipeline(raw)
        assert "Matched" in result["po_match_status"]

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Test Vendor")
    def test_real_po_not_found(self, mock_vendor, mock_audit):
        """No PO key → real erp_mock returns PO Not Found."""
        raw = {"vendor_name": "Test", "invoice_number": "INV-3W-003", "total_amount": 100}
        result = process_pipeline(raw)
        assert "PO Not Found" in result["po_match_status"]


# ============================================================================
# Data type edge cases across module boundaries
# ============================================================================

class TestDataTypeEdgeCases:
    """Edge cases around data types flowing between modules."""

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Unknown")
    def test_string_total_amount_through_pipeline(self, mock_vendor, mock_audit):
        """total_amount as string from OCR — M-02 fixed: float() coercion handles strings."""
        raw = {
            "vendor_name": "Test",
            "invoice_number": "INV-DT-001",
            "total_amount": "5000.00",  # String, not float
            "po_number": "PO-1001",
        }
        result = process_pipeline(raw)
        assert "po_match_status" in result
        # String "5000.00" is coerced to float 5000.0, below 10000 threshold
        assert not any("High Value" in f for f in result["flags"])

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Unknown")
    def test_none_vendor_name(self, mock_vendor, mock_audit):
        """vendor_name is None — exercise the None paths."""
        raw = {"vendor_name": None, "invoice_number": "INV-DT-002", "total_amount": 100}
        result = process_pipeline(raw)
        assert "standardized_vendor" in result


# ============================================================================
# Duplicate detection through pipeline (validates C-01 fix)
# ============================================================================

class TestDuplicateDetectionIntegration:
    """Integration: PROCESSED_DB is written to after process_pipeline,
    so second call with same invoice triggers duplicate flag."""

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Amazon Web Services")
    def test_second_call_triggers_duplicate(self, mock_vendor, mock_audit):
        """Same invoice_number + vendor_name twice → duplicate flag on second call."""
        raw = {
            "vendor_name": "Amazon Web Services",
            "invoice_number": "INV-DUP-001",
            "total_amount": 100,
        }
        result1 = process_pipeline(dict(raw))
        assert not any("Duplicate" in f for f in result1["flags"])

        result2 = process_pipeline(dict(raw))
        assert any("Duplicate" in f for f in result2["flags"])

    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Amazon Web Services")
    def test_different_invoice_no_duplicate(self, mock_vendor, mock_audit):
        """Different invoice_number → no duplicate flag."""
        raw1 = {"vendor_name": "Amazon Web Services", "invoice_number": "INV-DUP-002", "total_amount": 100}
        raw2 = {"vendor_name": "Amazon Web Services", "invoice_number": "INV-DUP-003", "total_amount": 200}
        process_pipeline(raw1)
        result2 = process_pipeline(raw2)
        assert not any("Duplicate" in f for f in result2["flags"])
