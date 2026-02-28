"""
Unit tests for erp_mock.py — check_3_way_match() and POS data.

Target: erp_mock.py, function check_3_way_match (line 20-60)
Coverage intent:
  - PO lookup via dynamic key scanning (lines 37-40)
  - Explicit po_number fallback (lines 42-43)
  - PO not found path (line 45)
  - Exact match within $1 tolerance (line 54)
  - Variance detection (line 56)
  - Non-numeric total_amount (bare except, line 58)
  - Empty dict input
  - Keys that accidentally contain 'po' and 'number' substrings
"""

import pytest
from erp_mock import check_3_way_match, POS


# ============================================================================
# POS module-level data validation
# ============================================================================

class TestPOSData:
    """Verify the mock PO database structure (erp_mock.py lines 15-17)."""

    def test_pos_contains_expected_keys(self):
        assert "PO-1001" in POS
        assert "PO-1002" in POS

    def test_po_1001_structure(self):
        po = POS["PO-1001"]
        assert po["vendor"] == "Amazon Web Services"
        assert po["total_approved"] == 5000.00
        assert isinstance(po["items"], list)

    def test_po_1002_structure(self):
        po = POS["PO-1002"]
        assert po["vendor"] == "Staples"
        assert po["total_approved"] == 250.50


# ============================================================================
# check_3_way_match — Happy Path
# ============================================================================

class TestCheck3WayMatchHappyPath:
    """Tests for successful matching logic (erp_mock.py lines 37-56)."""

    def test_exact_match_via_po_number_key(self, po_matched_invoice):
        """po_number = 'PO-1001', total_amount = 5000.00 → exact match."""
        result = check_3_way_match(po_matched_invoice)
        assert "Matched" in result
        assert "✅" in result

    def test_match_within_tolerance(self):
        """Variance < $1.00 should still be '✅ Matched'."""
        data = {"po_number": "PO-1001", "total_amount": 5000.50}
        result = check_3_way_match(data)
        assert "Matched" in result

    def test_match_at_boundary_negative(self):
        """total_amount = 4999.01 → variance = -0.99 → still matched."""
        data = {"po_number": "PO-1001", "total_amount": 4999.01}
        result = check_3_way_match(data)
        assert "Matched" in result

    def test_match_po_1002(self):
        """PO-1002 with exact amount 250.50"""
        data = {"po_number": "PO-1002", "total_amount": 250.50}
        result = check_3_way_match(data)
        assert "Matched" in result


# ============================================================================
# check_3_way_match — Variance Detection
# ============================================================================

class TestCheck3WayMatchVariance:
    """Tests for variance detection (erp_mock.py line 56)."""

    def test_positive_variance(self, po_variance_invoice):
        """total_amount 5500 vs approved 5000 → variance +500."""
        result = check_3_way_match(po_variance_invoice)
        assert "Variance" in result
        assert "❌" in result
        assert "$500.00" in result

    def test_negative_variance(self):
        """total_amount 4000 vs approved 5000 → variance -1000."""
        data = {"po_number": "PO-1001", "total_amount": 4000.00}
        result = check_3_way_match(data)
        assert "Variance" in result
        assert "-$1000.00" in result or "$-1000.00" in result

    def test_variance_at_boundary_exactly_one_dollar(self):
        """Variance of exactly $1.00 → abs(1.0) < 1.0 is False → Variance."""
        data = {"po_number": "PO-1001", "total_amount": 5001.00}
        result = check_3_way_match(data)
        assert "Variance" in result

    def test_string_total_amount_still_works(self):
        """total_amount as string '5000.00' — float() conversion in try block."""
        data = {"po_number": "PO-1001", "total_amount": "5000.00"}
        result = check_3_way_match(data)
        assert "Matched" in result


# ============================================================================
# check_3_way_match — PO Not Found
# ============================================================================

class TestCheck3WayMatchPONotFound:
    """Tests for PO-not-found paths (erp_mock.py line 45)."""

    def test_no_po_key_at_all(self, valid_invoice_data):
        """Invoice without any PO-related key → PO Not Found."""
        # valid_invoice_data has no po_number key
        result = check_3_way_match(valid_invoice_data)
        assert "PO Not Found" in result

    def test_po_number_nonexistent(self):
        """Explicit po_number that doesn't exist in POS."""
        data = {"po_number": "PO-9999", "total_amount": 100}
        result = check_3_way_match(data)
        assert "PO Not Found" in result

    def test_po_number_is_none(self):
        """po_number key exists but value is None."""
        data = {"po_number": None, "total_amount": 100}
        result = check_3_way_match(data)
        assert "PO Not Found" in result

    def test_po_number_is_empty_string(self):
        """po_number key exists but value is empty string → falsy."""
        data = {"po_number": "", "total_amount": 100}
        result = check_3_way_match(data)
        assert "PO Not Found" in result

    def test_empty_dict_input(self):
        """Completely empty dict."""
        result = check_3_way_match({})
        assert "PO Not Found" in result


# ============================================================================
# check_3_way_match — Dynamic Key Scanning
# ============================================================================

class TestCheck3WayMatchKeyScanning:
    """Tests for dynamic key iteration (erp_mock.py lines 37-40)."""

    def test_key_purchase_order_number_no_match(self):
        """Key 'purchase_order_number' does NOT contain substring 'po' consecutively.
        'p' and 'o' are separated in 'purchase_order_number', so 'po' in k.lower()
        is False. The loop at erp_mock.py L37-40 won't match. Falls through to
        explicit po_number check at L42-43 which also misses → PO Not Found."""
        data = {"purchase_order_number": "PO-1001", "total_amount": 5000.00}
        result = check_3_way_match(data)
        assert "PO Not Found" in result

    def test_key_po_ref_number(self):
        """Key 'po_ref_number' should match the scanner."""
        data = {"po_ref_number": "PO-1002", "total_amount": 250.50}
        result = check_3_way_match(data)
        assert "Matched" in result

    def test_false_positive_key_report_number(self):
        """Key 'report_number' contains 'po' (in 'report') and 'number'.
        This is a known bug (Phase 2 M-07) — the scanner will pick it up."""
        data = {"report_number": "PO-1001", "total_amount": 5000.00}
        # The bug means it WILL match — documenting current behavior
        result = check_3_way_match(data)
        assert "Matched" in result  # Documenting the false-positive behavior

    def test_first_matching_key_wins(self):
        """If multiple keys match, the first one encountered is used (break at line 40)."""
        # dict iteration order is insertion order in Python 3.7+
        data = {
            "po_number": "PO-1001",
            "export_number": "PO-9999",
            "total_amount": 5000.00,
        }
        result = check_3_way_match(data)
        assert "Matched" in result  # po_number comes first → PO-1001


# ============================================================================
# check_3_way_match — Error Handling
# ============================================================================

class TestCheck3WayMatchErrorHandling:
    """Tests for the bare except block (erp_mock.py line 58)."""

    def test_non_numeric_total_amount(self):
        """total_amount = 'not-a-number' → float() raises ValueError → caught."""
        data = {"po_number": "PO-1001", "total_amount": "not-a-number"}
        result = check_3_way_match(data)
        assert "Error calculating variance" in result

    def test_total_amount_is_list(self):
        """total_amount is a list → float() raises TypeError → caught."""
        data = {"po_number": "PO-1001", "total_amount": [5000]}
        result = check_3_way_match(data)
        assert "Error calculating variance" in result

    def test_total_amount_missing_defaults_to_zero(self):
        """No total_amount key → defaults to 0 → variance = -5000."""
        data = {"po_number": "PO-1001"}
        result = check_3_way_match(data)
        assert "Variance" in result
